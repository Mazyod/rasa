import numpy as np
import tensorflow as tf
import logging
import scipy.sparse
from typing import Optional, Text, Dict, Any, Union, List
from rasa.core.constants import DIALOGUE
from rasa.nlu.constants import TEXT
from rasa.nlu.tokenizers.tokenizer import Token


from rasa.utils.tensorflow.constants import (
    LABEL,
    HIDDEN_LAYERS_SIZES,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    DENSE_DIMENSION,
    LOSS_TYPE,
    SIMILARITY_TYPE,
    NUM_NEG,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    REGULARIZATION_CONSTANT,
    USE_MAX_NEG_SIM,
    MAX_NEG_SIM,
    MAX_POS_SIM,
    EMBEDDING_DIMENSION,
    DROPRATE_DIALOGUE,
    DROPRATE_LABEL,
    NEG_MARGIN_SCALE,
)


logger = logging.getLogger(__name__)


def normalize(values: np.ndarray, ranking_length: Optional[int] = 0) -> np.ndarray:
    """Normalizes an array of positive numbers over the top `ranking_length` values.
    Other values will be set to 0.
    """

    new_values = values.copy()  # prevent mutation of the input
    if 0 < ranking_length < len(new_values):
        ranked = sorted(new_values, reverse=True)
        new_values[new_values < ranked[ranking_length - 1]] = 0

    if np.sum(new_values) > 0:
        new_values = new_values / np.sum(new_values)

    return new_values


def update_similarity_type(config: Dict[Text, Any]) -> Dict[Text, Any]:
    if config.get(SIMILARITY_TYPE) == "auto":
        if config[LOSS_TYPE] == "softmax":
            config[SIMILARITY_TYPE] = "inner"
        elif config[LOSS_TYPE] == "margin":
            config[SIMILARITY_TYPE] = "cosine"

    return config


def align_tokens(
    tokens_in: List[Text], token_end: int, token_start: int
) -> List[Token]:
    """Align sub-tokens of Language model with tokens return by the WhitespaceTokenizer.

    As a language model might split a single word into multiple tokens, we need to make
    sure that the start and end value of first and last sub-token matches the
    start and end value of the token return by the WhitespaceTokenizer as the
    entities are using those start and end values.
    """

    tokens_out = []

    current_token_offset = token_start

    for index, string in enumerate(tokens_in):
        if index == 0:
            if index == len(tokens_in) - 1:
                s_token_end = token_end
            else:
                s_token_end = current_token_offset + len(string)
            tokens_out.append(Token(string, token_start, end=s_token_end))
        elif index == len(tokens_in) - 1:
            tokens_out.append(Token(string, current_token_offset, end=token_end))
        else:
            tokens_out.append(
                Token(
                    string, current_token_offset, end=current_token_offset + len(string)
                )
            )

        current_token_offset += len(string)

    return tokens_out


def sequence_to_sentence_features(
    features: Union[np.ndarray, scipy.sparse.spmatrix]
) -> Optional[Union[np.ndarray, scipy.sparse.spmatrix]]:
    """Extract the CLS token vector as sentence features.
    Features is a sequence. The last token is the CLS token. The feature vector of
    this token contains the sentence features."""

    if features is None:
        return None

    if isinstance(features, scipy.sparse.spmatrix):
        return scipy.sparse.coo_matrix(features.tocsr()[-1])

    return np.expand_dims(features[-1], axis=0)


def _replace_deprecated_option(
    old_option: Text, new_option: Union[Text, List[Text]], config: Dict[Text, Any]
) -> Dict[Text, Any]:
    if old_option in config:
        if isinstance(new_option, str):
            logger.warning(
                f"Option '{old_option}' got renamed to '{new_option}'. "
                f"Please update your configuration file."
            )
            config[new_option] = config[old_option]
        else:
            logger.warning(
                f"Option '{old_option}' got renamed to "
                f"a dictionary '{new_option[0]}' with a key '{new_option[1]}'. "
                f"Please update your configuration file."
            )
            option_dict = config.get(new_option[0], {})
            option_dict[new_option[1]] = config[old_option]
            config[new_option[0]] = option_dict

    return config


def check_deprecated_options(config: Dict[Text, Any]) -> Dict[Text, Any]:

    config = _replace_deprecated_option(
        "hidden_layers_sizes_pre_dial", [HIDDEN_LAYERS_SIZES, DIALOGUE], config
    )
    config = _replace_deprecated_option(
        "hidden_layers_sizes_bot", [HIDDEN_LAYERS_SIZES, LABEL], config
    )
    config = _replace_deprecated_option("droprate_a", DROPRATE_DIALOGUE, config)
    config = _replace_deprecated_option("droprate_b", DROPRATE_LABEL, config)
    config = _replace_deprecated_option(
        "hidden_layers_sizes_a", [HIDDEN_LAYERS_SIZES, TEXT], config
    )
    config = _replace_deprecated_option(
        "hidden_layers_sizes_b", [HIDDEN_LAYERS_SIZES, LABEL], config
    )
    config = _replace_deprecated_option(
        "num_transformer_layers", NUM_TRANSFORMER_LAYERS, config
    )
    config = _replace_deprecated_option("num_heads", NUM_HEADS, config)
    config = _replace_deprecated_option("dense_dim", DENSE_DIMENSION, config)
    config = _replace_deprecated_option("embed_dim", EMBEDDING_DIMENSION, config)
    config = _replace_deprecated_option("num_neg", NUM_NEG, config)
    config = _replace_deprecated_option("mu_pos", MAX_POS_SIM, config)
    config = _replace_deprecated_option("mu_neg", MAX_NEG_SIM, config)
    config = _replace_deprecated_option("use_max_sim_neg", USE_MAX_NEG_SIM, config)
    config = _replace_deprecated_option("C2", REGULARIZATION_CONSTANT, config)
    config = _replace_deprecated_option("C_emb", NEG_MARGIN_SCALE, config)
    config = _replace_deprecated_option(
        "evaluate_every_num_epochs", EVAL_NUM_EPOCHS, config
    )
    config = _replace_deprecated_option(
        "evaluate_on_num_examples", EVAL_NUM_EXAMPLES, config
    )

    return config
