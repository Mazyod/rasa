from collections import namedtuple
import copy
import json
import logging
import os
import warnings

import numpy as np
import typing
from tqdm import tqdm
from typing import Any, List, Optional, Text, Dict, Tuple, Union

import rasa.utils.io
from rasa.core import utils
from rasa.core.domain import Domain
from rasa.core.featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    LabelTokenizerSingleStateFeaturizer,
    MaxHistoryTrackerFeaturizer
)
from rasa.core.policies.policy import Policy
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.common import is_logging_disabled

import tensorflow as tf
from tensorflow.python.ops import gen_array_ops

try:
    from tensor2tensor.layers import common_attention
    from tensor2tensor.models.transformer import transformer_base, transformer_prepare_encoder, transformer_encoder
except ImportError:
    common_attention = None
    transformer_base = None
    transformer_prepare_encoder = None
    transformer_encoder = None

try:
    import cPickle as pickle
except ImportError:
    import pickle


logger = logging.getLogger(__name__)

# namedtuple for all tf session related data
SessionData = namedtuple(
    "SessionData",
    (
        "X",
        "Y",
        "slots",
        "previous_actions",
        "actions_for_Y",
        "all_Y_d",
    ),
)


class EmbeddingPolicy(Policy):
    """Recurrent Embedding Dialogue Policy (REDP)

    The policy that is used in our paper https://arxiv.org/abs/1811.11707
    """

    SUPPORTS_ONLINE_TRAINING = True

    # default properties (DOC MARKER - don't remove)
    defaults = {
        # nn architecture
        # a list of hidden layers sizes before user embed layer
        # number of hidden layers is equal to the length of this list
        "hidden_layers_sizes_a": [],
        # a list of hidden layers sizes before bot embed layer
        # number of hidden layers is equal to the length of this list
        "hidden_layers_sizes_b": [],

        "pos_encoding": "timing",  # {"timing", "emb", "custom_timing"}
        # introduce phase shift in time encodings between transformers
        # 0.5 - 0.8 works on small dataset
        "pos_max_timescale": 1.0e1,
        "max_seq_length": 256,
        "num_heads": 4,
        # number of units in rnn cell
        "rnn_size": 128,
        "num_rnn_layers": 1,
        # training parameters
        # flag if to turn on layer normalization for lstm cell
        "layer_norm": True,
        # initial and final batch sizes - batch size will be
        # linearly increased for each epoch
        "batch_size": [8, 32],
        # number of epochs
        "epochs": 1,
        # set random seed to any int to get reproducible results
        "random_seed": None,
        # embedding parameters
        # dimension size of embedding vectors
        "embed_dim": 20,
        # how similar the algorithm should try
        # to make embedding vectors for correct actions
        "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect actions
        "mu_neg": -0.2,  # should be -1.0 < ... < 1.0 for 'cosine'
        # the type of the similarity
        "similarity_type": "cosine",  # string 'cosine' or 'inner'
        # the number of incorrect actions, the algorithm will minimize
        # their similarity to the user input during training
        "num_neg": 20,
        # flag if minimize only maximum similarity over incorrect actions
        "use_max_sim_neg": True,  # flag which loss function to use
        # regularization
        # the scale of L2 regularization
        "C2": 0.001,
        # the scale of how important is to minimize the maximum similarity
        # between embeddings of different actions
        "C_emb": 0.8,
        # dropout rate for user nn
        "droprate_a": 0.0,
        # dropout rate for bot nn
        "droprate_b": 0.0,
        # dropout rate for rnn
        "droprate_rnn": 0.1,
        # attention parameters
        # flag to use attention over user input
        # as an input to rnn
        "attn_before_rnn": True,
        # flag to use attention over prev bot actions
        # and copy it to output bypassing rnn
        "attn_after_rnn": True,
        # flag to use `sparsemax` instead of `softmax` for attention
        "sparse_attention": False,  # flag to use sparsemax for probs
        # the range of allowed location-based attention shifts
        "attn_shift_range": None,  # if None, set to mean dialogue length / 2
        # visualization of accuracy
        # how often calculate train accuracy
        "evaluate_every_num_epochs": 20,  # small values may hurt performance
        # how many examples to use for calculation of train accuracy
        "evaluate_on_num_examples": 100,  # large values may hurt performance
    }

    # end default properties (DOC MARKER - don't remove)

    @staticmethod
    def _standard_featurizer(max_history=None):
        if max_history is None:
            return FullDialogueTrackerFeaturizer(LabelTokenizerSingleStateFeaturizer())
        else:
            return MaxHistoryTrackerFeaturizer(LabelTokenizerSingleStateFeaturizer(), max_history=max_history)

    @staticmethod
    def _check_t2t():
        if common_attention is None:
            raise ImportError("Please install tensor2tensor")

    def __init__(
        self,
        featurizer: Optional['FullDialogueTrackerFeaturizer'] = None,
        priority: int = 1,
        encoded_all_actions: Optional['np.ndarray'] = None,
        graph: Optional['tf.Graph'] = None,
        session: Optional['tf.Session'] = None,
        intent_placeholder: Optional['tf.Tensor'] = None,
        action_placeholder: Optional['tf.Tensor'] = None,
        slots_placeholder: Optional['tf.Tensor'] = None,
        prev_act_placeholder: Optional['tf.Tensor'] = None,
        dialogue_len: Optional['tf.Tensor'] = None,
        similarity_op: Optional['tf.Tensor'] = None,
        alignment_history: Optional['tf.Tensor'] = None,
        user_embed: Optional['tf.Tensor'] = None,
        bot_embed: Optional['tf.Tensor'] = None,
        slot_embed: Optional['tf.Tensor'] = None,
        dial_embed: Optional['tf.Tensor'] = None,
        rnn_embed: Optional['tf.Tensor'] = None,
        attn_embed: Optional['tf.Tensor'] = None,
        copy_attn_debug: Optional['tf.Tensor'] = None,
        all_time_masks: Optional['tf.Tensor'] = None,
        attention_weights=None,
        max_history: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        # check if t2t is installed
        self._check_t2t()

        if not featurizer:
            featurizer = self._standard_featurizer(max_history)
        super(EmbeddingPolicy, self).__init__(featurizer, priority)

        # flag if to use the same embeddings for user and bot
        try:
            self.share_embedding = self.featurizer.state_featurizer.use_shared_vocab
        except AttributeError:
            self.share_embedding = False

        self._load_params(**kwargs)

        # chrono initialization for forget bias
        self.characteristic_time = None

        # encode all actions with numbers
        # persist this array for prediction time
        self.encoded_all_actions = encoded_all_actions

        # tf related instances
        self.graph = graph
        self.session = session
        self.a_in = intent_placeholder
        self.b_in = action_placeholder
        self.c_in = slots_placeholder
        self.b_prev_in = prev_act_placeholder
        self._dialogue_len = dialogue_len
        self.sim_op = similarity_op

        # store attention probability distribution as
        # concatenated tensor of each attention types
        self.alignment_history = alignment_history

        # persisted embeddings
        self.user_embed = user_embed
        self.bot_embed = bot_embed
        self.slot_embed = slot_embed
        self.dial_embed = dial_embed

        self.rnn_embed = rnn_embed
        self.attn_embed = attn_embed
        self.copy_attn_debug = copy_attn_debug

        self.all_time_masks = all_time_masks
        self.attention_weights = attention_weights
        # internal tf instances
        self._train_op = None
        self._is_training = None

    # init helpers
    def _load_nn_architecture_params(self, config: Dict[Text, Any]) -> None:
        self.hidden_layer_sizes = {
            "a": config["hidden_layers_sizes_a"],
            "b": config["hidden_layers_sizes_b"],
        }

        if self.share_embedding:
            if self.hidden_layer_sizes["a"] != self.hidden_layer_sizes["b"]:
                raise ValueError(
                    "Due to sharing vocabulary "
                    "in the featurizer, embedding weights "
                    "are shared as well. "
                    "So hidden_layers_sizes_a={} should be "
                    "equal to hidden_layers_sizes_b={}"
                    "".format(
                        self.hidden_layer_sizes["a"], self.hidden_layer_sizes["b"]
                    )
                )
        self.pos_encoding = config['pos_encoding']
        self.pos_max_timescale = config['pos_max_timescale']
        self.max_seq_length = config['max_seq_length']
        self.num_heads = config['num_heads']

        self.rnn_size = config["rnn_size"]
        self.num_rnn_layers = config["num_rnn_layers"]
        self.layer_norm = config["layer_norm"]

        self.batch_size = config["batch_size"]

        self.epochs = config["epochs"]

        self.random_seed = config["random_seed"]

    def _load_embedding_params(self, config: Dict[Text, Any]) -> None:
        self.embed_dim = config["embed_dim"]
        self.mu_pos = config["mu_pos"]
        self.mu_neg = config["mu_neg"]
        self.similarity_type = config["similarity_type"]
        self.num_neg = config["num_neg"]
        self.use_max_sim_neg = config["use_max_sim_neg"]

    def _load_regularization_params(self, config: Dict[Text, Any]) -> None:
        self.C2 = config["C2"]
        self.C_emb = config["C_emb"]
        self.droprate = {
            "a": config["droprate_a"],
            "b": config["droprate_b"],
            "rnn": config["droprate_rnn"],
        }

    def _load_attn_params(self, config: Dict[Text, Any]) -> None:
        self.sparse_attention = config["sparse_attention"]
        self.attn_shift_range = config["attn_shift_range"]
        self.attn_after_rnn = config["attn_after_rnn"]
        self.attn_before_rnn = config["attn_before_rnn"]

    def is_using_attention(self):
        return self.attn_after_rnn or self.attn_before_rnn

    def _load_visual_params(self, config: Dict[Text, Any]) -> None:
        self.evaluate_every_num_epochs = config["evaluate_every_num_epochs"]
        if self.evaluate_every_num_epochs < 1:
            self.evaluate_every_num_epochs = self.epochs
        self.evaluate_on_num_examples = config["evaluate_on_num_examples"]

    def _load_params(self, **kwargs: Dict[Text, Any]) -> None:
        config = copy.deepcopy(self.defaults)
        config.update(kwargs)

        self._tf_config = self._load_tf_config(config)
        self._load_nn_architecture_params(config)
        self._load_embedding_params(config)
        self._load_regularization_params(config)
        self._load_attn_params(config)
        self._load_visual_params(config)

    # data helpers
    # noinspection PyPep8Naming
    def _create_X_slots_previous_actions(
        self, data_X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract feature vectors

        for user input (X), slots and
        previously executed actions from training data.
        """

        featurizer = self.featurizer.state_featurizer
        slot_start = featurizer.user_feature_len
        previous_start = slot_start + featurizer.slot_feature_len

        X = data_X[:, :, :slot_start]
        slots = data_X[:, :, slot_start:previous_start]
        previous_actions = data_X[:, :, previous_start:]

        return X, slots, previous_actions

    # noinspection PyPep8Naming
    @staticmethod
    def _actions_for_Y(data_Y: np.ndarray) -> np.ndarray:
        """Prepare Y data for training: extract actions indices."""
        return data_Y.argmax(axis=-1)

    # noinspection PyPep8Naming
    def _action_features_for_Y(self, actions_for_Y: np.ndarray) -> np.ndarray:
        """Prepare Y data for training: features for action labels."""

        if len(actions_for_Y.shape) == 2:
            return np.stack(
                [
                    np.stack(
                        [self.encoded_all_actions[action_idx] for action_idx in action_ids]
                    )
                    for action_ids in actions_for_Y
                ]
            )
        else:
            return np.stack(
                [
                    self.encoded_all_actions[action_idx] for action_idx in actions_for_Y
                ]
            )

    # noinspection PyPep8Naming
    def _create_all_Y_d(self, dialogue_len: int) -> np.ndarray:
        """Stack encoded_all_intents on top of each other

        to create candidates for training examples and
        to calculate training accuracy.
        """

        return np.stack([self.encoded_all_actions] * dialogue_len)

    # noinspection PyPep8Naming
    def _create_session_data(
        self, data_X: np.ndarray, data_Y: Optional[np.ndarray] = None
    ) -> SessionData:
        """Combine all tf session related data into a named tuple"""

        X, slots, previous_actions = self._create_X_slots_previous_actions(data_X)

        if data_Y is not None:
            # training time
            actions_for_Y = self._actions_for_Y(data_Y)
            Y = self._action_features_for_Y(actions_for_Y)
        else:
            # prediction time
            actions_for_Y = None
            Y = None

        # is needed to calculate train accuracy
        if isinstance(self.featurizer, FullDialogueTrackerFeaturizer):
            dial_len = X.shape[1]
        else:
            dial_len = 1
        all_Y_d = self._create_all_Y_d(dial_len)

        return SessionData(
            X=X,
            Y=Y,
            slots=slots,
            previous_actions=previous_actions,
            actions_for_Y=actions_for_Y,
            all_Y_d=all_Y_d,
        )

    @staticmethod
    def _sample_session_data(session_data: 'SessionData',
                             num_samples: int) -> 'SessionData':
        ids = np.random.permutation(len(session_data.X))[:num_samples]
        return SessionData(
            X=session_data.X[ids],
            Y=session_data.Y[ids],
            slots=session_data.slots[ids],
            previous_actions=session_data.previous_actions[ids],
            actions_for_Y=session_data.actions_for_Y[ids],
            all_Y_d=session_data.all_Y_d,
        )

    # tf helpers:
    @staticmethod
    def _create_tf_dataset(session_data: 'SessionData',
                           batch_size: Union['tf.Tensor', int]) -> 'tf.data.Dataset':
        train_dataset = tf.data.Dataset.from_tensor_slices((session_data.X,
                                                            session_data.Y,
                                                            session_data.slots,
                                                            session_data.previous_actions))
        train_dataset = train_dataset.shuffle(buffer_size=len(session_data.X))
        train_dataset = train_dataset.batch(batch_size)
        return train_dataset

    def _create_tf_nn(
        self,
        x_in: 'tf.Tensor',
        layer_sizes: List,
        droprate: float,
        layer_name_suffix: Text,
    ) -> 'tf.Tensor':
        """Create nn with hidden layers and name suffix."""

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        x = tf.nn.relu(x_in)
        for i, layer_size in enumerate(layer_sizes):
            x = tf.layers.dense(
                inputs=x,
                units=layer_size,
                activation=tf.nn.relu,
                kernel_regularizer=reg,
                name="hidden_layer_{}_{}".format(layer_name_suffix, i),
                reuse=tf.AUTO_REUSE,
            )
            x = tf.layers.dropout(x, rate=droprate, training=self._is_training)
        return x

    def _create_embed(self, x: 'tf.Tensor', layer_name_suffix: Text) -> 'tf.Tensor':
        """Create dense embedding layer with a name."""

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        embed_x = tf.layers.dense(
            inputs=x,
            units=self.embed_dim,
            activation=None,
            kernel_regularizer=reg,
            name="embed_layer_{}".format(layer_name_suffix),
            reuse=tf.AUTO_REUSE,
        )
        return embed_x

    def _create_tf_bot_embed(self, b_in: 'tf.Tensor') -> 'tf.Tensor':
        """Create embedding bot vector."""

        layer_name_suffix = "a_and_b" if self.share_embedding else "b"

        b = self._create_tf_nn(
            b_in,
            self.hidden_layer_sizes["b"],
            self.droprate["b"],
            layer_name_suffix=layer_name_suffix,
        )
        return self._create_embed(b, layer_name_suffix=layer_name_suffix)

    def _create_hparams(self):
        hparams = transformer_base()

        hparams.num_hidden_layers = self.num_rnn_layers
        hparams.hidden_size = self.rnn_size
        # it seems to be factor of 4 for transformer architectures in t2t
        hparams.filter_size = hparams.hidden_size * 4
        hparams.num_heads = self.num_heads
        hparams.relu_dropout = self.droprate["rnn"]
        hparams.pos = self.pos_encoding

        hparams.max_length = self.max_seq_length

        hparams.unidirectional_encoder = True

        hparams.self_attention_type = "dot_product_relative_v2"
        hparams.max_relative_position = 5
        hparams.add_relative_to_values = True
        return hparams

    def _create_transformer_encoder(self, a_in, c_in, b_prev_in, mask, attention_weights):
        hparams = self._create_hparams()

        x_in = tf.concat([a_in, b_prev_in, c_in], -1)

        # When not in training mode, set all forms of dropout to zero.
        for key, value in hparams.values().items():
            if key.endswith("dropout") or key == "label_smoothing":
                setattr(hparams, key, value * tf.cast(self._is_training, tf.float32))
        reg = tf.contrib.layers.l2_regularizer(self.C2)

        x = tf.layers.dense(inputs=x_in,
                            units=hparams.hidden_size,
                            use_bias=False,
                            kernel_initializer=tf.random_normal_initializer(0.0, hparams.hidden_size ** -0.5),
                            kernel_regularizer=reg,
                            name='transformer_embed_layer',
                            reuse=tf.AUTO_REUSE)

        x = tf.layers.dropout(x, rate=hparams.layer_prepostprocess_dropout, training=self._is_training)

        if hparams.multiply_embedding_mode == "sqrt_depth":
            x *= hparams.hidden_size ** 0.5

        x *= tf.expand_dims(mask, -1)

        with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE):
            (x,
             self_attention_bias,
             encoder_decoder_attention_bias
             ) = transformer_prepare_encoder(x, None, hparams)

            if hparams.pos == 'custom_timing':
                x = common_attention.add_timing_signal_1d(x, max_timescale=self.pos_max_timescale)

            x *= tf.expand_dims(mask, -1)

            x = tf.nn.dropout(x, 1.0 - hparams.layer_prepostprocess_dropout)

            attn_bias_for_padding = None
            # Otherwise the encoder will just use encoder_self_attention_bias.
            if hparams.unidirectional_encoder:
                attn_bias_for_padding = encoder_decoder_attention_bias

            x = transformer_encoder(
                x,
                self_attention_bias,
                hparams,
                nonpadding=mask,
                save_weights_to=attention_weights,
                attn_bias_for_padding=attn_bias_for_padding,
            )

            x *= tf.expand_dims(mask, -1)

            return tf.nn.relu(x)

    def _tf_sample_neg(self,
                       pos_b,
                       neg_bs=None,
                       neg_ids=None,
                       batch_size=None,
                       first_only=False
                       ) -> 'tf.Tensor':

        all_b = pos_b[tf.newaxis, :, :]
        if batch_size is None:
            batch_size = tf.shape(pos_b)[0]
        all_b = tf.tile(all_b, [batch_size, 1, 1])
        if neg_bs is None and neg_ids is None:
            return all_b

        def sample_neg_b():
            if neg_bs is not None:
                _neg_bs = neg_bs
            elif neg_ids is not None:
                _neg_bs = tf.batch_gather(all_b, neg_ids)
            else:
                raise
            return tf.concat([pos_b[:, tf.newaxis, :], _neg_bs], 1)

        if first_only:
            out_b = pos_b[:, tf.newaxis, :]
        else:
            out_b = all_b

        if neg_bs is not None:
            cond = tf.logical_and(self._is_training, tf.shape(neg_bs)[0] > 1)
        elif neg_ids is not None:
            cond = tf.logical_and(self._is_training, tf.shape(neg_ids)[0] > 1)
        else:
            raise

        return tf.cond(cond, sample_neg_b, lambda: out_b)

    def _tf_calc_iou(self,
                     b_raw,
                     neg_bs=None,
                     neg_ids=None
                     ) -> 'tf.Tensor':

        tiled_intent_raw = self._tf_sample_neg(b_raw, neg_bs=neg_bs, neg_ids=neg_ids)
        pos_b_raw = tiled_intent_raw[:, :1, :]
        neg_b_raw = tiled_intent_raw[:, 1:, :]
        intersection_b_raw = tf.minimum(neg_b_raw, pos_b_raw)
        union_b_raw = tf.maximum(neg_b_raw, pos_b_raw)

        return tf.reduce_sum(intersection_b_raw, -1) / tf.reduce_sum(union_b_raw, -1)

    def _tf_sim(
        self,
        embed_dialogue: 'tf.Tensor',
        embed_action: 'tf.Tensor',
        mask: Optional['tf.Tensor'],
    ) -> Tuple['tf.Tensor', 'tf.Tensor', 'tf.Tensor', 'tf.Tensor']:
        """Define similarity.

        This method has two roles:
        - calculate similarity between
            two embedding vectors of the same size
            and output binary mask and similarity;
        - calculate similarity with several embedded actions for the loss
            and output similarities between user input and bot actions
            and similarities between bot actions.

        They are kept in the same helper method,
        because it is necessary for them to be mathematically identical.
        """

        if self.similarity_type not in {"cosine", "inner"}:
            raise ValueError(
                "Wrong similarity type {}, "
                "should be 'cosine' or 'inner'"
                "".format(self.similarity_type)
            )

        # calculate similarity with several
        # embedded actions for the loss

        if self.similarity_type == "cosine":
            # normalize embedding vectors for cosine similarity
            embed_dialogue = tf.nn.l2_normalize(embed_dialogue, -1)
            embed_action = tf.nn.l2_normalize(embed_action, -1)

        if len(embed_dialogue.shape) == 4:
            embed_dialogue_pos = embed_dialogue[:, :, :1, :]
        else:
            embed_dialogue_pos = tf.expand_dims(embed_dialogue, -2)

        sim = tf.reduce_sum(
            embed_dialogue_pos * embed_action, -1
        ) * tf.expand_dims(mask, 2)

        sim_bot_emb = tf.reduce_sum(
            embed_action[:, :, :1, :] * embed_action[:, :, 1:, :], -1
        ) * tf.expand_dims(mask, 2)

        if len(embed_dialogue.shape) == 4:
            sim_dial_emb = tf.reduce_sum(
                embed_dialogue[:, :, :1, :] * embed_dialogue[:, :, 1:, :], -1
            ) * tf.expand_dims(mask, 2)
        else:
            sim_dial_emb = None

        if len(embed_dialogue.shape) == 4:
            sim_dial_bot_emb = tf.reduce_sum(
                embed_dialogue[:, :, :1, :] * embed_action[:, :, 1:, :], -1
            ) * tf.expand_dims(mask, 2)
        else:
            sim_dial_bot_emb = None

        # output similarities between user input and bot actions
        # and similarities between bot actions
        return sim,  sim_bot_emb, sim_dial_emb, sim_dial_bot_emb

    def _tf_loss(
        self,
        sim: 'tf.Tensor',
        sim_bot_emb: 'tf.Tensor',
        sim_dial_emb: 'tf.Tensor',
        sims_rnn_to_max: List['tf.Tensor'],
        bad_negs,
        mask: 'tf.Tensor',
        batch_bad_negs
    ) -> 'tf.Tensor':
        """Define loss."""

        # loss for maximizing similarity with correct action
        loss = tf.maximum(0., self.mu_pos - sim[:, :, 0])

        # loss for minimizing similarity with `num_neg` incorrect actions
        sim_neg = sim[:, :, 1:] + common_attention.large_compatible_negative(bad_negs.dtype) * bad_negs
        if self.use_max_sim_neg:
            # minimize only maximum similarity over incorrect actions
            max_sim_neg = tf.reduce_max(sim_neg, -1)
            loss += tf.maximum(0., self.mu_neg + max_sim_neg)
        else:
            # minimize all similarities with incorrect actions
            max_margin = tf.maximum(0., self.mu_neg + sim_neg)
            loss += tf.reduce_sum(max_margin, -1)

        # penalize max similarity between bot embeddings
        sim_bot_emb += common_attention.large_compatible_negative(bad_negs.dtype) * bad_negs
        max_sim_bot_emb = tf.maximum(0., tf.reduce_max(sim_bot_emb, -1))
        loss += max_sim_bot_emb * self.C_emb

        # penalize max similarity between dial embeddings
        if sim_dial_emb is not None:
            sim_dial_emb += common_attention.large_compatible_negative(batch_bad_negs.dtype) * batch_bad_negs
            max_sim_input_emb = tf.maximum(0., tf.reduce_max(sim_dial_emb, -1))
            loss += max_sim_input_emb * self.C_emb

        # maximize similarity returned by time attention wrapper
        for sim_to_add in sims_rnn_to_max:
            loss += tf.maximum(0.0, 1.0 - sim_to_add)

        # mask loss for different length sequences
        loss *= mask
        # average the loss over sequence length
        loss = tf.reduce_sum(loss, -1) / tf.reduce_sum(mask, 1)

        # average the loss over the batch
        loss = (
            tf.reduce_mean(loss)
            # add regularization losses
            + self._regularization_loss()
            + tf.losses.get_regularization_loss()
        )
        return loss

    def _tf_loss_2(
        self,
        sim: 'tf.Tensor',
        sim_bot_emb: 'tf.Tensor',
        sim_dial_emb: 'tf.Tensor',
        sim_dial_bot_emb,
        sims_rnn_to_max: List['tf.Tensor'],
        bad_negs,
        mask: 'tf.Tensor',
        batch_bad_negs=None,
    ) -> 'tf.Tensor':
        """Define loss."""

        all_sim = [sim[:, :, :1],
                   sim[:, :, 1:] + common_attention.large_compatible_negative(bad_negs.dtype) * bad_negs,
                   sim_bot_emb + common_attention.large_compatible_negative(bad_negs.dtype) * bad_negs,
                   ]
        if sim_dial_emb is not None:
            all_sim.append(sim_dial_emb + common_attention.large_compatible_negative(batch_bad_negs.dtype) * batch_bad_negs)

        if sim_dial_bot_emb is not None:
            all_sim.append(sim_dial_bot_emb + common_attention.large_compatible_negative(bad_negs.dtype) * bad_negs)

        logits = tf.concat(all_sim, -1)
        pos_labels = tf.ones_like(logits[:, :, :1])
        neg_labels = tf.zeros_like(logits[:, :, 1:])
        labels = tf.concat([pos_labels, neg_labels], -1)

        pred = tf.nn.softmax(logits)
        # fake_logits = tf.concat([logits[:, :, :1] - common_attention.large_compatible_negative(logits.dtype),
        #                          logits[:, :, 1:] + common_attention.large_compatible_negative(logits.dtype)], -1)

        # ones = tf.ones_like(pred[:, :, 0])
        # zeros = tf.zeros_like(pred[:, :, 0])

        # already_learned = tf.where(pred[:, :, 0] > 0.8, zeros, ones)
        already_learned = tf.pow((1 - pred[:, :, 0]) / 0.5, 4)

        loss = tf.losses.softmax_cross_entropy(labels,
                                               logits,
                                               mask * already_learned)
        # add regularization losses
        loss += tf.losses.get_regularization_loss()

        # maximize similarity returned by time attention wrapper
        add_loss = []
        for sim_to_add in sims_rnn_to_max:
            add_loss.append(tf.maximum(0.0, 1.0 - sim_to_add))

        if add_loss:
            # mask loss for different length sequences
            add_loss = sum(add_loss) * mask
            # average the loss over sequence length
            add_loss = tf.reduce_sum(add_loss, -1) / tf.reduce_sum(mask, 1)
            # average the loss over the batch
            add_loss = tf.reduce_mean(add_loss)

            loss += add_loss

        return loss

    # training methods
    def train(
        self,
        training_trackers: List['DialogueStateTracker'],
        domain: 'Domain',
        **kwargs: Any
    ) -> None:
        """Train the policy on given training trackers."""

        logger.debug("Started training embedding policy.")

        # set numpy random seed
        np.random.seed(self.random_seed)

        # dealing with training data
        training_data = self.featurize_for_training(training_trackers, domain, **kwargs)

        # encode all actions with policies' featurizer
        self.encoded_all_actions = self.featurizer.state_featurizer.create_encoded_all_actions(
            domain
        )

        # check if number of negatives is less than number of actions
        logger.debug(
            "Check if num_neg {} is smaller "
            "than number of actions {}, "
            "else set num_neg to the number of actions - 1"
            "".format(self.num_neg, domain.num_actions)
        )
        self.num_neg = min(self.num_neg, domain.num_actions - 1)

        # extract actual training data to feed to tf session
        session_data = self._create_session_data(training_data.X, training_data.y)

        self.graph = tf.Graph()

        with self.graph.as_default():
            # set random seed in tf
            tf.set_random_seed(self.random_seed)

            # allows increasing batch size
            batch_size_in = tf.placeholder(tf.int64)
            train_dataset = self._create_tf_dataset(session_data, batch_size_in)

            if self.evaluate_on_num_examples:
                eval_session_data = self._sample_session_data(session_data, self.evaluate_on_num_examples)
                eval_train_dataset = self._create_tf_dataset(eval_session_data, self.evaluate_on_num_examples)
            else:
                eval_train_dataset = None

            iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                       train_dataset.output_shapes,
                                                       output_classes=train_dataset.output_classes)

            # session data are int counts but we need a float tensors
            (self.a_in,
             self.b_in,
             self.c_in,
             self.b_prev_in) = (tf.cast(x_in, tf.float32) for x_in in iterator.get_next())

            all_actions = tf.constant(self.encoded_all_actions,
                                      dtype=tf.float32,
                                      name="all_actions")

            # dynamic variables
            self._is_training = tf.placeholder_with_default(False, shape=())
            self._dialogue_len = tf.placeholder(
                dtype=tf.int32, shape=(), name="dialogue_len"
            )

            # mask different length sequences
            # if there is at least one `-1` it should be masked
            mask = tf.sign(tf.reduce_max(self.a_in, -1) + 1)

            self.attention_weights = {}
            transformer_out = self._create_transformer_encoder(
                self.a_in, self.c_in, self.b_prev_in, mask, self.attention_weights)
            self.dial_embed = self._create_embed(transformer_out, layer_name_suffix="out")
            sims_rnn_to_max = []

            self.bot_embed = self._create_tf_bot_embed(self.b_in)
            all_actions_embed = self._create_tf_bot_embed(all_actions)

            # calculate similarities
            if isinstance(self.featurizer, MaxHistoryTrackerFeaturizer):
                # pick last action if max history is used
                self.b_in = self.b_in[:, tf.newaxis, :]
                self.bot_embed = self.bot_embed[:, tf.newaxis, :]
                self.dial_embed = self.dial_embed[:, -1:, :]
                mask = mask[:, -1:]

            b_raw = tf.reshape(self.b_in, (-1, self.b_in.shape[-1]))

            _, i, c = gen_array_ops.unique_with_counts_v2(b_raw, axis=[0])
            counts = tf.expand_dims(tf.reshape(tf.gather(tf.cast(c, tf.float32), i), (tf.shape(b_raw)[0],)), 0)
            batch_neg_ids = tf.random.categorical(tf.log((1. - tf.eye(tf.shape(b_raw)[0])/counts)), self.num_neg)

            batch_iou_bot = self._tf_calc_iou(b_raw, neg_ids=batch_neg_ids)
            batch_bad_negs = 1. - tf.nn.relu(tf.sign(1. - batch_iou_bot))
            batch_bad_negs = tf.reshape(batch_bad_negs, (tf.shape(self.dial_embed)[0],
                                                         tf.shape(self.dial_embed)[1],
                                                         -1))

            neg_ids = tf.random.categorical(tf.log(tf.ones((tf.shape(b_raw)[0], tf.shape(all_actions)[0]))), self.num_neg)

            tiled_all_actions = tf.tile(tf.expand_dims(all_actions, 0), (tf.shape(b_raw)[0], 1, 1))
            neg_bs = tf.batch_gather(tiled_all_actions, neg_ids)
            iou_bot = self._tf_calc_iou(b_raw, neg_bs)
            bad_negs = 1. - tf.nn.relu(tf.sign(1. - iou_bot))
            bad_negs = tf.reshape(bad_negs, (tf.shape(self.bot_embed)[0],
                                             tf.shape(self.bot_embed)[1],
                                             -1))

            dial_embed_flat = tf.reshape(self.dial_embed, (-1, self.dial_embed.shape[-1]))

            tiled_dial_embed = self._tf_sample_neg(dial_embed_flat, neg_ids=batch_neg_ids, first_only=True)
            tiled_dial_embed = tf.reshape(tiled_dial_embed, (tf.shape(self.dial_embed)[0],
                                                             tf.shape(self.dial_embed)[1],
                                                             -1,
                                                             self.dial_embed.shape[-1]))

            bot_embed_flat = tf.reshape(self.bot_embed, (-1, self.bot_embed.shape[-1]))
            tiled_all_actions_embed = tf.tile(tf.expand_dims(all_actions_embed, 0), (tf.shape(b_raw)[0], 1, 1))
            neg_embs = tf.batch_gather(tiled_all_actions_embed, neg_ids)
            tiled_bot_embed = self._tf_sample_neg(bot_embed_flat, neg_bs=neg_embs)
            tiled_bot_embed = tf.reshape(tiled_bot_embed, (tf.shape(self.bot_embed)[0],
                                                           tf.shape(self.bot_embed)[1],
                                                           -1,
                                                           self.bot_embed.shape[-1]))

            # self.sim_op, sim_bot_emb, sim_dial_emb = self._tf_sim(self.dial_embed, tiled_bot_embed, mask)
            self.sim_op, sim_bot_emb, sim_dial_emb, sim_dial_bot_emb = self._tf_sim(tiled_dial_embed, tiled_bot_embed, mask)

            # loss = self._tf_loss_2(self.sim_op, sim_bot_emb, sim_dial_emb, sims_rnn_to_max, bad_negs, mask)
            loss = self._tf_loss_2(self.sim_op, sim_bot_emb, sim_dial_emb, sim_dial_bot_emb, sims_rnn_to_max, bad_negs, mask, batch_bad_negs)

            # define which optimizer to use
            self._train_op = tf.train.AdamOptimizer(
                # learning_rate=0.001, epsilon=1e-16
            ).minimize(loss)

            train_init_op = iterator.make_initializer(train_dataset)
            if self.evaluate_on_num_examples:
                eval_init_op = iterator.make_initializer(eval_train_dataset)
            else:
                eval_init_op = None

            # train tensorflow graph
            self.session = tf.Session(config=self._tf_config)

            # self._train_tf(session_data, loss, mask)
            self._train_tf_dataset(train_init_op, eval_init_op, batch_size_in, loss, mask, session_data.X.shape[1])

            dialogue_len = None  # use dynamic time for rnn
            # create placeholders
            self.a_in = tf.placeholder(
                dtype=tf.float32,
                shape=(None, dialogue_len, session_data.X.shape[-1]),
                name="a",
            )
            self.b_in = tf.placeholder(
                dtype=tf.float32,
                shape=(None, dialogue_len, None, session_data.Y.shape[-1]),
                name="b",
            )
            self.c_in = tf.placeholder(
                dtype=tf.float32,
                shape=(None, dialogue_len, session_data.slots.shape[-1]),
                name="slt",
            )
            self.b_prev_in = tf.placeholder(
                dtype=tf.float32,
                shape=(None, dialogue_len, session_data.Y.shape[-1]),
                name="b_prev",
            )

            # mask different length sequences
            # if there is at least one `-1` it should be masked
            mask = tf.sign(tf.reduce_max(self.a_in, -1) + 1)

            self.attention_weights = {}
            transformer_out = self._create_transformer_encoder(
                self.a_in, self.c_in, self.b_prev_in, mask, self.attention_weights)
            self.dial_embed = self._create_embed(transformer_out, layer_name_suffix="out")

            self.bot_embed = self._create_tf_bot_embed(self.b_in)

            if isinstance(self.featurizer, MaxHistoryTrackerFeaturizer):
                self.dial_embed = self.dial_embed[:, -1:, :]

            self.sim_op, _, _, _ = self._tf_sim(self.dial_embed, self.bot_embed, mask)

            # if self.attention_weights.items():
            #     self.attention_weights = tf.concat([tf.expand_dims(t, 0)
            #                                         for name, t in self.attention_weights.items()
            #                                         if name.endswith('multihead_attention/dot_product_attention')], 0)

    # training helpers
    def _linearly_increasing_batch_size(self, epoch: int) -> int:
        """Linearly increase batch size with every epoch.

        The idea comes from https://arxiv.org/abs/1711.00489.
        """

        if not isinstance(self.batch_size, list):
            return int(self.batch_size)

        if self.epochs > 1:
            return int(
                self.batch_size[0]
                + epoch * (self.batch_size[1] - self.batch_size[0]) / (self.epochs - 1)
            )
        else:
            return int(self.batch_size[0])

    def _train_tf_dataset(self,
                          train_init_op,
                          eval_init_op,
                          batch_size_in,
                          loss: 'tf.Tensor',
                          mask,
                          dialogue_len,
                          ) -> None:
        """Train tf graph"""

        self.session.run(tf.global_variables_initializer())

        if self.evaluate_on_num_examples:
            logger.info(
                "Accuracy is updated every {} epochs"
                "".format(self.evaluate_every_num_epochs)
            )
        pbar = tqdm(range(self.epochs), desc="Epochs", disable=is_logging_disabled())

        train_acc = 0
        last_loss = 0
        for ep in pbar:

            batch_size = self._linearly_increasing_batch_size(ep)

            self.session.run(train_init_op, feed_dict={batch_size_in: batch_size})

            ep_loss = 0
            batches_per_epoch = 0
            while True:
                try:
                    _, batch_loss = self.session.run((self._train_op, loss),
                                                     feed_dict={self._is_training: True,
                                                                self._dialogue_len: dialogue_len})

                except tf.errors.OutOfRangeError:
                    break

                batches_per_epoch += 1
                ep_loss += batch_loss

            ep_loss /= batches_per_epoch

            if self.evaluate_on_num_examples and eval_init_op is not None:
                if (ep == 0 or
                        (ep + 1) % self.evaluate_every_num_epochs == 0 or
                        (ep + 1) == self.epochs):
                    train_acc = self._output_training_stat_dataset(eval_init_op, mask, dialogue_len)
                    last_loss = ep_loss

                pbar.set_postfix({
                    "loss": "{:.3f}".format(ep_loss),
                    "acc": "{:.3f}".format(train_acc)
                })
            else:
                pbar.set_postfix({
                    "loss": "{:.3f}".format(ep_loss)
                })

        if self.evaluate_on_num_examples:
            logger.info("Finished training embedding classifier, "
                        "loss={:.3f}, train accuracy={:.3f}"
                        "".format(last_loss, train_acc))

    def _output_training_stat_dataset(self, eval_init_op, mask, dialogue_len) -> np.ndarray:
        """Output training statistics"""

        self.session.run(eval_init_op)

        sim_, mask_ = self.session.run([self.sim_op, mask],
                                       feed_dict={self._is_training: False,
                                                  self._dialogue_len: dialogue_len})
        sim_ = sim_.reshape((-1, sim_.shape[-1]))
        mask_ = mask_.reshape((-1,))

        train_acc = np.sum((np.max(sim_, -1) == sim_.diagonal()) * mask_) / np.sum(mask_)

        return train_acc

    def continue_training(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs: Any
    ) -> None:
        """Continue training an already trained policy."""

        batch_size = kwargs.get("batch_size", 5)
        epochs = kwargs.get("epochs", 50)

        for _ in range(epochs):
            training_data = self._training_data_for_continue_training(
                batch_size, training_trackers, domain
            )

            session_data = self._create_session_data(training_data.X, training_data.y)

            b = self._create_batch_b(session_data.Y, session_data.actions_for_Y)

            # fit to one extra example using updated trackers
            self.session.run(
                self._train_op,
                feed_dict={
                    self.a_in: session_data.X,
                    self.b_in: b,
                    self.c_in: session_data.slots,
                    self.b_prev_in: session_data.previous_actions,
                    self._dialogue_len: session_data.X.shape[1],
                    self._is_training: True,
                },
            )

    def tf_feed_dict_for_prediction(self,
                                    tracker: DialogueStateTracker,
                                    domain: Domain) -> Dict:
        # noinspection PyPep8Naming
        data_X = self.featurizer.create_X([tracker], domain)
        session_data = self._create_session_data(data_X)
        # noinspection PyPep8Naming
        all_Y_d_x = np.stack([session_data.all_Y_d
                              for _ in range(session_data.X.shape[0])])

        return {self.a_in: session_data.X,
                self.b_in: all_Y_d_x,
                self.c_in: session_data.slots,
                self.b_prev_in: session_data.previous_actions,
                self._dialogue_len: session_data.X.shape[1]}

    def predict_action_probabilities(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        """Predict the next action the bot should take.

        Return the list of probabilities for the next actions.
        """

        if self.session is None:
            logger.error(
                "There is no trained tf.session: "
                "component is either not trained or "
                "didn't receive enough training data"
            )
            return [0.0] * domain.num_actions

        # noinspection PyPep8Naming
        data_X = self.featurizer.create_X([tracker], domain)
        session_data = self._create_session_data(data_X)
        # noinspection PyPep8Naming
        all_Y_d_x = np.stack(
            [session_data.all_Y_d for _ in range(session_data.X.shape[0])]
        )
        # self.similarity_type = 'cosine'
        # mask = tf.sign(tf.reduce_max(self.a_in, -1) + 1)
        # self.sim_op, _, _ = self._tf_sim(self.dial_embed, self.bot_embed, mask)
        _sim = self.session.run(
            self.sim_op,
            feed_dict={
                self.a_in: session_data.X,
                self.b_in: all_Y_d_x,
                self.c_in: session_data.slots,
                self.b_prev_in: session_data.previous_actions,
                self._dialogue_len: session_data.X.shape[1],
            },
        )

        # TODO assume we used inner:
        self.similarity_type = "inner"

        result = _sim[0, -1, :]
        if self.similarity_type == "cosine":
            # clip negative values to zero
            result[result < 0] = 0
        elif self.similarity_type == "inner":
            # normalize result to [0, 1] with softmax but only over 3*num_neg+1 values
            low_ids = result.argsort()[::-1][4*self.num_neg+1:]
            result[low_ids] += -np.inf
            result = np.exp(result)
            result /= np.sum(result)

        return result.tolist()

    def _persist_tensor(self, name: Text, tensor: 'tf.Tensor') -> None:
        if tensor is not None:
            self.graph.clear_collection(name)
            self.graph.add_to_collection(name, tensor)

    def persist(self, path: Text) -> None:
        """Persists the policy to a storage."""

        if self.session is None:
            warnings.warn(
                "Method `persist(...)` was called "
                "without a trained model present. "
                "Nothing to persist then!"
            )
            return

        self.featurizer.persist(path)

        meta = {"priority": self.priority}

        meta_file = os.path.join(path, "embedding_policy.json")
        utils.dump_obj_as_json_to_file(meta_file, meta)

        file_name = "tensorflow_embedding.ckpt"
        checkpoint = os.path.join(path, file_name)
        rasa.utils.io.create_directory_for_file(checkpoint)

        with self.graph.as_default():
            self._persist_tensor("intent_placeholder", self.a_in)
            self._persist_tensor("action_placeholder", self.b_in)
            self._persist_tensor("slots_placeholder", self.c_in)
            self._persist_tensor("prev_act_placeholder", self.b_prev_in)
            self._persist_tensor("dialogue_len", self._dialogue_len)

            self._persist_tensor("similarity_op", self.sim_op)

            self._persist_tensor("alignment_history", self.alignment_history)

            self._persist_tensor("user_embed", self.user_embed)
            self._persist_tensor("bot_embed", self.bot_embed)
            self._persist_tensor("slot_embed", self.slot_embed)
            self._persist_tensor("dial_embed", self.dial_embed)

            self._persist_tensor("rnn_embed", self.rnn_embed)
            self._persist_tensor("attn_embed", self.attn_embed)
            self._persist_tensor("copy_attn_debug", self.copy_attn_debug)

            self._persist_tensor("all_time_masks", self.all_time_masks)

            self._persist_tensor("attention_weights", self.attention_weights)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        encoded_actions_file = os.path.join(
            path, file_name + ".encoded_all_actions.pkl"
        )
        with open(encoded_actions_file, "wb") as f:
            pickle.dump(self.encoded_all_actions, f)

        tf_config_file = os.path.join(path, file_name + ".tf_config.pkl")
        with open(tf_config_file, "wb") as f:
            pickle.dump(self._tf_config, f)

    @staticmethod
    def load_tensor(name: Text) -> Optional['tf.Tensor']:
        tensor_list = tf.get_collection(name)
        return tensor_list[0] if tensor_list else None

    @classmethod
    def load(cls, path: Text) -> "EmbeddingPolicy":
        """Loads a policy from the storage.

            **Needs to load its featurizer**"""

        if not os.path.exists(path):
            raise Exception(
                "Failed to load dialogue model. Path {} "
                "doesn't exist".format(os.path.abspath(path))
            )

        featurizer = TrackerFeaturizer.load(path)

        file_name = "tensorflow_embedding.ckpt"
        checkpoint = os.path.join(path, file_name)

        if not os.path.exists(checkpoint + ".meta"):
            return cls(featurizer=featurizer)

        meta_file = os.path.join(path, "embedding_policy.json")
        meta = json.loads(rasa.utils.io.read_file(meta_file))

        tf_config_file = os.path.join(path, "{}.tf_config.pkl".format(file_name))

        with open(tf_config_file, "rb") as f:
            _tf_config = pickle.load(f)

        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session(config=_tf_config)
            saver = tf.train.import_meta_graph(checkpoint + ".meta")

            saver.restore(sess, checkpoint)

            a_in = cls.load_tensor("intent_placeholder")
            b_in = cls.load_tensor("action_placeholder")
            c_in = cls.load_tensor("slots_placeholder")
            b_prev_in = cls.load_tensor("prev_act_placeholder")
            dialogue_len = cls.load_tensor("dialogue_len")

            sim_op = cls.load_tensor("similarity_op")

            alignment_history = cls.load_tensor("alignment_history")

            user_embed = cls.load_tensor("user_embed")
            bot_embed = cls.load_tensor("bot_embed")
            slot_embed = cls.load_tensor("slot_embed")
            dial_embed = cls.load_tensor("dial_embed")

            rnn_embed = cls.load_tensor("rnn_embed")
            attn_embed = cls.load_tensor("attn_embed")
            copy_attn_debug = cls.load_tensor("copy_attn_debug")

            all_time_masks = cls.load_tensor("all_time_masks")

            attention_weights = cls.load_tensor("attention_weights")

        encoded_actions_file = os.path.join(
            path, "{}.encoded_all_actions.pkl".format(file_name)
        )

        with open(encoded_actions_file, "rb") as f:
            encoded_all_actions = pickle.load(f)

        return cls(
            featurizer=featurizer,
            priority=meta["priority"],
            encoded_all_actions=encoded_all_actions,
            graph=graph,
            session=sess,
            intent_placeholder=a_in,
            action_placeholder=b_in,
            slots_placeholder=c_in,
            prev_act_placeholder=b_prev_in,
            dialogue_len=dialogue_len,
            similarity_op=sim_op,
            alignment_history=alignment_history,
            user_embed=user_embed,
            bot_embed=bot_embed,
            slot_embed=slot_embed,
            dial_embed=dial_embed,
            rnn_embed=rnn_embed,
            attn_embed=attn_embed,
            copy_attn_debug=copy_attn_debug,
            all_time_masks=all_time_masks,
            attention_weights=attention_weights
        )
