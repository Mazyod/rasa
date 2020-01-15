from collections import defaultdict
import logging
import scipy.sparse
import typing
from typing import (
    List,
    Optional,
    Text,
    Dict,
    Tuple,
    Union,
    Generator,
    Callable,
    Any,
    NamedTuple,
)
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf

# from tensor2tensor.models.transformer import (
#     transformer_base,
#     transformer_prepare_encoder,
#     transformer_encoder,
# )
# from tensor2tensor.layers.common_attention import large_compatible_negative
from rasa.utils.common import is_logging_disabled


if typing.TYPE_CHECKING:
    from tensor2tensor.utils.hparam import HParams

logger = logging.getLogger(__name__)


# type for all tf session related data
SessionDataType = Dict[Text, List[np.ndarray]]


# namedtuple for training metrics
class TrainingMetrics(NamedTuple):
    loss: Dict[Text, Union[tf.Tensor, float]]
    score: Dict[Text, Union[tf.Tensor, float]]


def load_tf_config(config: Dict[Text, Any]) -> Optional[tf.compat.v1.ConfigProto]:
    """Prepare `tf.compat.v1.ConfigProto` for training"""

    if config.get("tf_config") is not None:
        return tf.compat.v1.ConfigProto(**config.pop("tf_config"))
    else:
        return None


def create_label_ids(label_ids: "np.ndarray") -> "np.ndarray":
    """Convert various size label_ids into single dim array.

    for multi-label y, map each distinct row to a string repr
    using join because str(row) uses an ellipsis if len(row) > 1000.
    Idea taken from sklearn's stratify split.
    """

    if label_ids.ndim == 1:
        return label_ids
    elif label_ids.ndim == 2 and label_ids.shape[-1] == 1:
        return label_ids[:, 0]
    elif label_ids.ndim == 2:
        return np.array([" ".join(row.astype("str")) for row in label_ids])
    elif label_ids.ndim == 3 and label_ids.shape[-1] == 1:
        return np.array([" ".join(row.astype("str")) for row in label_ids[:, :, 0]])
    else:
        raise ValueError("Unsupported label_ids dimensions")


# noinspection PyPep8Naming
def train_val_split(
    session_data: SessionDataType,
    evaluate_on_num_examples: int,
    random_seed: int,
    label_key: Text,
) -> Tuple[SessionDataType, SessionDataType]:
    """Create random hold out validation set using stratified split."""

    if label_key not in session_data or len(session_data[label_key]) > 1:
        raise ValueError(f"Key '{label_key}' not in SessionData.")

    label_ids = create_label_ids(session_data[label_key][0])

    label_counts = dict(zip(*np.unique(label_ids, return_counts=True, axis=0)))

    check_train_test_sizes(evaluate_on_num_examples, label_counts, session_data)

    counts = np.array([label_counts[label] for label in label_ids])

    multi_values = []
    [
        multi_values.append(v[counts > 1])
        for values in session_data.values()
        for v in values
    ]

    solo_values = []
    [
        solo_values.append(v[counts == 1])
        for values in session_data.values()
        for v in values
    ]

    output_values = train_test_split(
        *multi_values,
        test_size=evaluate_on_num_examples,
        random_state=random_seed,
        stratify=label_ids[counts > 1],
    )

    session_data_train, session_data_val = convert_train_test_split(
        output_values, session_data, solo_values
    )

    return session_data_train, session_data_val


def check_train_test_sizes(
    evaluate_on_num_examples: int,
    label_counts: Dict[Any, int],
    session_data: SessionDataType,
):
    num_examples = get_number_of_examples(session_data)

    if evaluate_on_num_examples >= num_examples - len(label_counts):
        raise ValueError(
            f"Validation set of {evaluate_on_num_examples} is too large. Remaining "
            f"train set should be at least equal to number of classes "
            f"{len(label_counts)}."
        )
    elif evaluate_on_num_examples < len(label_counts):
        raise ValueError(
            f"Validation set of {evaluate_on_num_examples} is too small. It should be "
            "at least equal to number of classes {label_counts}."
        )


def convert_train_test_split(
    output_values: List[Any], session_data: SessionDataType, solo_values: List[Any]
):
    session_data_train = defaultdict(list)
    session_data_val = defaultdict(list)

    # output_values = x_train, x_val, y_train, y_val, z_train, z_val, etc.
    # order is kept, e.g. same order as session data keys

    # train datasets have an even index
    index = 0
    for key, values in session_data.items():
        for _ in range(len(values)):
            session_data_train[key].append(
                combine_features(output_values[index * 2], solo_values[index])
            )
            index += 1

    # val datasets have an odd index
    index = 0
    for key, values in session_data.items():
        for _ in range(len(values)):
            session_data_val[key].append(output_values[(index * 2) + 1])
            index += 1

    return session_data_train, session_data_val


def combine_features(
    feature_1: Union[np.ndarray, scipy.sparse.spmatrix],
    feature_2: Union[np.ndarray, scipy.sparse.spmatrix],
) -> Union[np.ndarray, scipy.sparse.spmatrix]:
    """Concatenate features."""

    if isinstance(feature_1, scipy.sparse.spmatrix) and isinstance(
        feature_2, scipy.sparse.spmatrix
    ):
        if feature_2.shape[0] == 0:
            return feature_1
        if feature_1.shape[0] == 0:
            return feature_2
        return scipy.sparse.vstack([feature_1, feature_2])

    return np.concatenate([feature_1, feature_2])


def shuffle_session_data(session_data: SessionDataType) -> SessionDataType:
    """Shuffle session data."""

    data_points = get_number_of_examples(session_data)
    ids = np.random.permutation(data_points)
    return session_data_for_ids(session_data, ids)


def session_data_for_ids(session_data: SessionDataType, ids: np.ndarray):
    """Filter session data by ids."""

    new_session_data = defaultdict(list)
    for k, values in session_data.items():
        for v in values:
            new_session_data[k].append(v[ids])
    return new_session_data


def split_session_data_by_label_ids(
    session_data: SessionDataType,
    label_ids: "np.ndarray",
    unique_label_ids: "np.ndarray",
) -> List[SessionDataType]:
    """Reorganize session data into a list of session data with the same labels."""

    label_data = []
    for label_id in unique_label_ids:
        ids = label_ids == label_id
        label_data.append(session_data_for_ids(session_data, ids))
    return label_data


# noinspection PyPep8Naming
def balance_session_data(
    session_data: SessionDataType, batch_size: int, shuffle: bool, label_key: Text
) -> SessionDataType:
    """Mix session data to account for class imbalance.

    This batching strategy puts rare classes approximately in every other batch,
    by repeating them. Mimics stratified batching, but also takes into account
    that more populated classes should appear more often.
    """

    if label_key not in session_data or len(session_data[label_key]) > 1:
        raise ValueError(f"Key '{label_key}' not in SessionDataType.")

    label_ids = create_label_ids(session_data[label_key][0])

    unique_label_ids, counts_label_ids = np.unique(
        label_ids, return_counts=True, axis=0
    )
    num_label_ids = len(unique_label_ids)

    # need to call every time, so that the data is shuffled inside each class
    label_data = split_session_data_by_label_ids(
        session_data, label_ids, unique_label_ids
    )

    data_idx = [0] * num_label_ids
    num_data_cycles = [0] * num_label_ids
    skipped = [False] * num_label_ids

    new_session_data = defaultdict(list)
    num_examples = get_number_of_examples(session_data)

    while min(num_data_cycles) == 0:
        if shuffle:
            indices_of_labels = np.random.permutation(num_label_ids)
        else:
            indices_of_labels = range(num_label_ids)

        for index in indices_of_labels:
            if num_data_cycles[index] > 0 and not skipped[index]:
                skipped[index] = True
                continue
            else:
                skipped[index] = False

            index_batch_size = (
                int(counts_label_ids[index] / num_examples * batch_size) + 1
            )

            for k, values in label_data[index].items():
                for i, v in enumerate(values):
                    if len(new_session_data[k]) < i + 1:
                        new_session_data[k].append([])
                    new_session_data[k][i].append(
                        v[data_idx[index] : data_idx[index] + index_batch_size]
                    )

            data_idx[index] += index_batch_size
            if data_idx[index] >= counts_label_ids[index]:
                num_data_cycles[index] += 1
                data_idx[index] = 0

            if min(num_data_cycles) > 0:
                break

    final_session_data = defaultdict(list)
    for k, values in new_session_data.items():
        for v in values:
            final_session_data[k].append(np.concatenate(np.array(v)))

    return final_session_data


def get_number_of_examples(session_data: SessionDataType):
    """Obtain number of examples in session data.

    Raise a ValueError if number of examples differ for different data in session data.
    """

    example_lengths = [v.shape[0] for values in session_data.values() for v in values]

    # check if number of examples is the same for all values
    if not all(length == example_lengths[0] for length in example_lengths):
        raise ValueError(
            f"Number of examples differs for keys '{session_data.keys()}'. Number of "
            f"examples should be the same for all data in session data."
        )

    return example_lengths[0]


def gen_batch(
    session_data: SessionDataType,
    batch_size: int,
    label_key: Text,
    batch_strategy: Text = "sequence",
    shuffle: bool = False,
) -> Generator[Tuple, None, None]:
    """Generate batches."""

    if shuffle:
        session_data = shuffle_session_data(session_data)

    if batch_strategy == "balanced":
        session_data = balance_session_data(
            session_data, batch_size, shuffle, label_key
        )

    num_examples = get_number_of_examples(session_data)
    num_batches = num_examples // batch_size + int(num_examples % batch_size > 0)

    for batch_num in range(num_batches):
        start = batch_num * batch_size
        end = start + batch_size

        yield prepare_batch(session_data, start, end)


def prepare_batch(
    session_data: SessionDataType,
    start: Optional[int] = None,
    end: Optional[int] = None,
    tuple_sizes: Dict[Text, int] = None,
) -> Tuple[Optional[np.ndarray]]:
    """Slices session data into batch using given start and end value."""

    batch_data = []

    for key, values in session_data.items():
        # add None for not present values during processing
        if not values:
            if tuple_sizes:
                batch_data += [None] * tuple_sizes[key]
            else:
                batch_data.append(None)
            continue

        for v in values:
            if start is not None and end is not None:
                _data = v[start:end]
            elif start is not None:
                _data = v[start:]
            elif end is not None:
                _data = v[:end]
            else:
                _data = v[:]

            if isinstance(_data[0], scipy.sparse.spmatrix):
                batch_data.extend(scipy_matrix_to_values(_data))
            else:
                batch_data.append(pad_dense_data(_data))

    # len of batch_data is equal to the number of keys in session data
    return tuple(batch_data)


def scipy_matrix_to_values(array_of_sparse: np.ndarray) -> List[np.ndarray]:
    """Convert a scipy matrix into inidces, data, and shape."""

    if not isinstance(array_of_sparse[0], scipy.sparse.coo_matrix):
        array_of_sparse = [x.tocoo() for x in array_of_sparse]

    max_seq_len = max([x.shape[0] for x in array_of_sparse])

    indices = np.hstack(
        [
            np.vstack([i * np.ones_like(x.row), x.row, x.col])
            for i, x in enumerate(array_of_sparse)
        ]
    ).T
    data = np.hstack([x.data for x in array_of_sparse])

    shape = np.array((len(array_of_sparse), max_seq_len, array_of_sparse[0].shape[-1]))

    return [indices.astype(np.int64), data.astype(np.float32), shape.astype(np.int64)]


def pad_dense_data(array_of_dense: np.ndarray) -> np.ndarray:
    """Pad data of different lengths.

    Sequential data is padded with zeros. Zeros are added to the end of data.
    """

    if array_of_dense[0].ndim < 2:
        # data doesn't contain a sequence
        return array_of_dense

    data_size = len(array_of_dense)
    max_seq_len = max([x.shape[0] for x in array_of_dense])

    data_padded = np.zeros(
        [data_size, max_seq_len, array_of_dense[0].shape[-1]],
        dtype=array_of_dense[0].dtype,
    )
    for i in range(data_size):
        data_padded[i, : array_of_dense[i].shape[0], :] = array_of_dense[i]

    return data_padded.astype(np.float32)


def batch_to_session_data(
    batch: Union[Tuple[np.ndarray], Tuple[tf.Tensor]], session_data: SessionDataType
) -> Dict[Text, List[tf.Tensor]]:
    """Convert input batch tensors into batch data format.

    Batch contains any number of batch data. The order is equal to the
    key-value pairs in session data. As sparse data were converted into indices, data,
    shape before, this methods converts them into sparse tensors. Dense data is
    kept.
    """

    batch_data = defaultdict(list)

    idx = 0
    for k, values in session_data.items():
        for v in values:
            if isinstance(v[0], scipy.sparse.spmatrix):
                # explicitly substitute last dimension in shape with known static value
                batch_data[k].append(
                    tf.SparseTensor(
                        batch[idx],
                        batch[idx + 1],
                        [batch[idx + 2][0], batch[idx + 2][1], v[0].shape[-1]],
                    )
                )
                idx += 3
            else:
                batch_data[k].append(batch[idx])
                idx += 1

    return batch_data


def batch_tuple_sizes(
    session_data: SessionDataType
) -> Dict[Text, int]:

    # save the amount of placeholders attributed to session data keys
    tuple_sizes = defaultdict(int)

    idx = 0
    for k, values in session_data.items():
        tuple_sizes[k] = 0
        for v in values:
            if isinstance(v[0], scipy.sparse.spmatrix):
                tuple_sizes[k] += 3
                idx += 3
            else:
                tuple_sizes[k] += 1
                idx += 1

    return tuple_sizes


def create_tf_dataset(
    session_data: SessionDataType,
    batch_size: Union["tf.Tensor", int],
    label_key: Text,
    batch_strategy: Text = "sequence",
    shuffle: bool = False,
) -> "tf.data.Dataset":
    """Create tf dataset."""

    shapes, types = get_shapes_types(session_data)

    return tf.data.Dataset.from_generator(
        lambda batch_size_: gen_batch(
            session_data, batch_size_, label_key, batch_strategy, shuffle
        ),
        output_types=types,
        output_shapes=shapes,
        args=([batch_size]),
    )


def get_shapes_types(session_data: SessionDataType) -> Tuple:
    """Extract shapes and types from session data."""

    types = []
    shapes = []

    def append_shape(v: np.ndarray):
        if isinstance(v[0], scipy.sparse.spmatrix):
            # scipy matrix is converted into indices, data, shape
            shapes.append((None, v[0].ndim + 1))
            shapes.append((None,))
            shapes.append((v[0].ndim + 1))
        elif v[0].ndim == 0:
            shapes.append((None,))
        elif v[0].ndim == 1:
            shapes.append((None, v[0].shape[-1]))
        else:
            shapes.append((None, None, v[0].shape[-1]))

    def append_type(v: np.ndarray):
        if isinstance(v[0], scipy.sparse.spmatrix):
            # scipy matrix is converted into indices, data, shape
            types.append(tf.int64)
            types.append(tf.float32)
            types.append(tf.int64)
        else:
            types.append(tf.float32)

    for values in session_data.values():
        for v in values:
            append_shape(v)
            append_type(v)

    return tuple(shapes), tuple(types)


def _tf_make_flat(x: "tf.Tensor") -> "tf.Tensor":
    """Make tensor 2D."""

    return tf.reshape(x, (-1, x.shape[-1]))


def _tf_sample_neg(
    batch_size: "tf.Tensor", all_bs: "tf.Tensor", neg_ids: "tf.Tensor"
) -> "tf.Tensor":
    """Sample negative examples for given indices"""

    tiled_all_bs = tf.tile(tf.expand_dims(all_bs, 0), (batch_size, 1, 1))

    return tf.gather(tiled_all_bs, neg_ids, batch_dims=-1)


def _tf_get_bad_mask(
    pos_b: "tf.Tensor", all_bs: "tf.Tensor", neg_ids: "tf.Tensor"
) -> "tf.Tensor":
    """Calculate bad mask for given indices.

    Checks that input features are different for positive negative samples.
    """

    pos_b_in_flat = tf.expand_dims(pos_b, -2)
    neg_b_in_flat = _tf_sample_neg(tf.shape(pos_b)[0], all_bs, neg_ids)

    return tf.cast(
        tf.reduce_all(tf.equal(neg_b_in_flat, pos_b_in_flat), axis=-1),
        pos_b_in_flat.dtype,
    )


def _tf_get_negs(
    all_embed: "tf.Tensor", all_raw: "tf.Tensor", raw_pos: "tf.Tensor", num_neg: int
) -> Tuple["tf.Tensor", "tf.Tensor"]:
    """Get negative examples from given tensor."""

    if len(raw_pos.shape) == 3:
        batch_size = tf.shape(raw_pos)[0]
        seq_length = tf.shape(raw_pos)[1]
    else:  # len(raw_pos.shape) == 2
        batch_size = tf.shape(raw_pos)[0]
        seq_length = 1

    raw_flat = _tf_make_flat(raw_pos)

    total_candidates = tf.shape(all_embed)[0]

    all_indices = tf.tile(
        tf.expand_dims(tf.range(0, total_candidates, 1), 0),
        (batch_size * seq_length, 1),
    )
    shuffled_indices = tf.transpose(
        tf.random.shuffle(tf.transpose(all_indices, (1, 0))), (1, 0)
    )
    neg_ids = shuffled_indices[:, :num_neg]

    bad_negs = _tf_get_bad_mask(raw_flat, all_raw, neg_ids)
    if len(raw_pos.shape) == 3:
        bad_negs = tf.reshape(bad_negs, (batch_size, seq_length, -1))

    neg_embed = _tf_sample_neg(batch_size * seq_length, all_embed, neg_ids)
    if len(raw_pos.shape) == 3:
        neg_embed = tf.reshape(
            neg_embed, (batch_size, seq_length, -1, all_embed.shape[-1])
        )

    return neg_embed, bad_negs


def _sample_negatives(
    a_embed: "tf.Tensor",
    b_embed: "tf.Tensor",
    b_raw: "tf.Tensor",
    all_b_embed: "tf.Tensor",
    all_b_raw: "tf.Tensor",
    num_neg: int,
) -> Tuple[
    "tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor"
]:
    """Sample negative examples."""

    neg_dial_embed, dial_bad_negs = _tf_get_negs(
        _tf_make_flat(a_embed), _tf_make_flat(b_raw), b_raw, num_neg
    )

    neg_bot_embed, bot_bad_negs = _tf_get_negs(
        _tf_make_flat(all_b_embed), _tf_make_flat(all_b_raw), b_raw, num_neg
    )
    return (
        tf.expand_dims(a_embed, -2),
        tf.expand_dims(b_embed, -2),
        neg_dial_embed,
        neg_bot_embed,
        dial_bad_negs,
        bot_bad_negs,
    )


def tf_raw_sim(
    a: "tf.Tensor", b: "tf.Tensor", mask: Optional["tf.Tensor"]
) -> "tf.Tensor":
    """Calculate similarity between given tensors."""

    sim = tf.reduce_sum(a * b, -1)
    if mask is not None:
        sim *= tf.expand_dims(mask, 2)

    return sim


def _tf_sim(
    pos_dial_embed: "tf.Tensor",
    pos_bot_embed: "tf.Tensor",
    neg_dial_embed: "tf.Tensor",
    neg_bot_embed: "tf.Tensor",
    dial_bad_negs: "tf.Tensor",
    bot_bad_negs: "tf.Tensor",
    mask: Optional["tf.Tensor"],
) -> Tuple["tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor"]:
    """Define similarity."""

    # calculate similarity with several
    # embedded actions for the loss
    neg_inf = -1e9  # large_compatible_negative(pos_dial_embed.dtype)

    sim_pos = tf_raw_sim(pos_dial_embed, pos_bot_embed, mask)
    sim_neg = tf_raw_sim(pos_dial_embed, neg_bot_embed, mask) + neg_inf * bot_bad_negs
    sim_neg_bot_bot = (
        tf_raw_sim(pos_bot_embed, neg_bot_embed, mask) + neg_inf * bot_bad_negs
    )
    sim_neg_dial_dial = (
        tf_raw_sim(pos_dial_embed, neg_dial_embed, mask) + neg_inf * dial_bad_negs
    )
    sim_neg_bot_dial = (
        tf_raw_sim(pos_bot_embed, neg_dial_embed, mask) + neg_inf * dial_bad_negs
    )

    # output similarities between user input and bot actions
    # and similarities between bot actions and similarities between user inputs
    return sim_pos, sim_neg, sim_neg_bot_bot, sim_neg_dial_dial, sim_neg_bot_dial


def _tf_calc_accuracy(sim_pos: "tf.Tensor", sim_neg: "tf.Tensor") -> "tf.Tensor":
    """Calculate accuracy"""

    max_all_sim = tf.reduce_max(tf.concat([sim_pos, sim_neg], -1), -1)
    return tf.reduce_mean(
        tf.cast(tf.math.equal(max_all_sim, tf.squeeze(sim_pos, -1)), tf.float32)
    )


# noinspection PyPep8Naming
def _tf_loss_margin(
    sim_pos: "tf.Tensor",
    sim_neg: "tf.Tensor",
    sim_neg_bot_bot: "tf.Tensor",
    sim_neg_dial_dial: "tf.Tensor",
    sim_neg_bot_dial: "tf.Tensor",
    mask: Optional["tf.Tensor"],
    mu_pos: float,
    mu_neg: float,
    use_max_sim_neg: bool,
    C_emb: float,
) -> "tf.Tensor":
    """Define max margin loss."""

    # loss for maximizing similarity with correct action
    loss = tf.maximum(0.0, mu_pos - tf.squeeze(sim_pos, -1))

    # loss for minimizing similarity with `num_neg` incorrect actions
    if use_max_sim_neg:
        # minimize only maximum similarity over incorrect actions
        max_sim_neg = tf.reduce_max(sim_neg, -1)
        loss += tf.maximum(0.0, mu_neg + max_sim_neg)
    else:
        # minimize all similarities with incorrect actions
        max_margin = tf.maximum(0.0, mu_neg + sim_neg)
        loss += tf.reduce_sum(max_margin, -1)

    # penalize max similarity between pos bot and neg bot embeddings
    max_sim_neg_bot = tf.maximum(0.0, tf.reduce_max(sim_neg_bot_bot, -1))
    loss += max_sim_neg_bot * C_emb

    # penalize max similarity between pos dial and neg dial embeddings
    max_sim_neg_dial = tf.maximum(0.0, tf.reduce_max(sim_neg_dial_dial, -1))
    loss += max_sim_neg_dial * C_emb

    # penalize max similarity between pos bot and neg dial embeddings
    max_sim_neg_dial = tf.maximum(0.0, tf.reduce_max(sim_neg_bot_dial, -1))
    loss += max_sim_neg_dial * C_emb

    if mask is not None:
        # mask loss for different length sequences
        loss *= mask
        # average the loss over sequence length
        loss = tf.reduce_sum(loss, -1) / tf.reduce_sum(mask, 1)

    # average the loss over the batch
    loss = tf.reduce_mean(loss)

    return loss


def _tf_loss_softmax(
    sim_pos: "tf.Tensor",
    sim_neg: "tf.Tensor",
    sim_neg_bot_bot: "tf.Tensor",
    sim_neg_dial_dial: "tf.Tensor",
    sim_neg_bot_dial: "tf.Tensor",
    mask: Optional["tf.Tensor"],
    scale_loss: bool,
) -> "tf.Tensor":
    """Define softmax loss."""

    logits = tf.concat(
        [sim_pos, sim_neg, sim_neg_bot_bot, sim_neg_dial_dial, sim_neg_bot_dial], -1
    )

    # create label_ids for softmax
    label_ids = tf.zeros_like(logits[..., 0], tf.int32)

    if mask is None:
        mask = 1.0

    if scale_loss:
        # mask loss by prediction confidence
        pos_pred = tf.stop_gradient(tf.nn.softmax(logits)[..., 0])
        scale_mask = mask * tf.pow(tf.minimum(0.5, 1 - pos_pred) / 0.5, 4)
    else:
        scale_mask = mask

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_ids, logits=logits
    )

    # scale loss
    if len(loss.shape) == 2:
        # average over the sequence
        loss = tf.reduce_sum(loss * scale_mask, -1) / tf.reduce_sum(mask, -1)
    else:
        loss *= scale_mask

    # average the loss over all examples
    loss = tf.reduce_mean(loss)

    return loss


# noinspection PyPep8Naming
def _choose_loss(
    sim_pos: "tf.Tensor",
    sim_neg: "tf.Tensor",
    sim_neg_bot_bot: "tf.Tensor",
    sim_neg_dial_dial: "tf.Tensor",
    sim_neg_bot_dial: "tf.Tensor",
    mask: Optional["tf.Tensor"],
    loss_type: Text,
    mu_pos: float,
    mu_neg: float,
    use_max_sim_neg: bool,
    C_emb: float,
    scale_loss: bool,
) -> "tf.Tensor":
    """Use loss depending on given option."""

    if loss_type == "margin":
        return _tf_loss_margin(
            sim_pos,
            sim_neg,
            sim_neg_bot_bot,
            sim_neg_dial_dial,
            sim_neg_bot_dial,
            mask,
            mu_pos,
            mu_neg,
            use_max_sim_neg,
            C_emb,
        )
    elif loss_type == "softmax":
        return _tf_loss_softmax(
            sim_pos,
            sim_neg,
            sim_neg_bot_bot,
            sim_neg_dial_dial,
            sim_neg_bot_dial,
            mask,
            scale_loss,
        )
    else:
        raise ValueError(
            f"Wrong loss type '{loss_type}', " f"should be 'margin' or 'softmax'"
        )


# noinspection PyPep8Naming
def calculate_loss_acc(
    a_embed: "tf.Tensor",
    b_embed: "tf.Tensor",
    b_raw: "tf.Tensor",
    all_b_embed: "tf.Tensor",
    all_b_raw: "tf.Tensor",
    num_neg: int,
    mask: Optional["tf.Tensor"],
    loss_type: Text,
    mu_pos: float,
    mu_neg: float,
    use_max_sim_neg: bool,
    C_emb: float,
    scale_loss: bool,
) -> Tuple["tf.Tensor", "tf.Tensor"]:
    """Calculate loss and accuracy."""

    (
        pos_dial_embed,
        pos_bot_embed,
        neg_dial_embed,
        neg_bot_embed,
        dial_bad_negs,
        bot_bad_negs,
    ) = _sample_negatives(a_embed, b_embed, b_raw, all_b_embed, all_b_raw, num_neg)

    # calculate similarities
    (sim_pos, sim_neg, sim_neg_bot_bot, sim_neg_dial_dial, sim_neg_bot_dial) = _tf_sim(
        pos_dial_embed,
        pos_bot_embed,
        neg_dial_embed,
        neg_bot_embed,
        dial_bad_negs,
        bot_bad_negs,
        mask,
    )

    acc = _tf_calc_accuracy(sim_pos, sim_neg)

    loss = _choose_loss(
        sim_pos,
        sim_neg,
        sim_neg_bot_bot,
        sim_neg_dial_dial,
        sim_neg_bot_dial,
        mask,
        loss_type,
        mu_pos,
        mu_neg,
        use_max_sim_neg,
        C_emb,
        scale_loss,
    )

    return loss, acc


def confidence_from_sim(sim: "tf.Tensor", similarity_type: Text) -> "tf.Tensor":
    if similarity_type == "cosine":
        # clip negative values to zero
        return tf.nn.relu(sim)
    else:
        # normalize result to [0, 1] with softmax
        return tf.nn.softmax(sim)


def linearly_increasing_batch_size(
    epoch: int, batch_size: Union[List[int], int], epochs: int
) -> int:
    """Linearly increase batch size with every epoch.

    The idea comes from https://arxiv.org/abs/1711.00489.
    """

    if not isinstance(batch_size, list):
        return int(batch_size)

    if epochs > 1:
        return int(
            batch_size[0] + epoch * (batch_size[1] - batch_size[0]) / (epochs - 1)
        )
    else:
        return int(batch_size[0])


def output_validation_stat(
    eval_init_op: "tf.Operation",
    metrics: TrainingMetrics,
    session: "tf.Session",
    is_training: "tf.Session",
    batch_size_in: "tf.Tensor",
    ep_batch_size: int,
) -> TrainingMetrics:
    """Output training statistics"""

    session.run(eval_init_op, feed_dict={batch_size_in: ep_batch_size})
    ep_val_metrics = TrainingMetrics(
        loss=defaultdict(lambda: 0.0), score=defaultdict(lambda: 0.0)
    )
    batches_per_epoch = 0
    while True:
        try:
            batch_val_metrics = session.run([metrics], feed_dict={is_training: False})
            batch_val_metrics = batch_val_metrics[0]
            batches_per_epoch += 1
            for name, value in batch_val_metrics.loss.items():
                ep_val_metrics.loss[name] += value
            for name, value in batch_val_metrics.score.items():
                ep_val_metrics.score[name] += value

        except tf.errors.OutOfRangeError:
            break

    for name, value in ep_val_metrics.loss.items():
        ep_val_metrics.loss[name] = value / batches_per_epoch
    for name, value in ep_val_metrics.score.items():
        ep_val_metrics.score[name] = value / batches_per_epoch

    return ep_val_metrics


def _write_training_metrics(
    output_file: Text,
    epoch: int,
    train_metrics: TrainingMetrics,
    val_metrics: TrainingMetrics,
):
    if output_file:
        import datetime

        # output log file
        with open(output_file, "a") as f:
            # make headers on first epoch
            if epoch == 0:
                f.write(f"EPOCH\tTIMESTAMP")
                [f.write(f"\t{key.upper()}") for key in train_metrics.loss.keys()]
                [f.write(f"\t{key.upper()}") for key in train_metrics.score.keys()]
                [f.write(f"\tVAL_{key.upper()}") for key in train_metrics.loss.keys()]
                [f.write(f"\tVAL_{key.upper()}") for key in train_metrics.score.keys()]

            f.write(f"\n{epoch}\t{datetime.datetime.now():%H:%M:%S}")
            [f.write(f"\t{val:.3f}") for val in train_metrics.loss.values()]
            [f.write(f"\t{val:.3f}") for val in train_metrics.score.values()]
            [
                f.write(f"\t{val:.3f}") if val else f.write("\t0.0")
                for val in val_metrics.loss.values()
            ]
            [
                f.write(f"\t{val:.3f}") if val else f.write("\t0.0")
                for val in val_metrics.score.values()
            ]


def extract_attention(attention_weights) -> Optional["tf.Tensor"]:
    """Extract attention probabilities from t2t dict"""

    attention = [
        tf.expand_dims(t, 0)
        for name, t in attention_weights.items()
        # the strings come from t2t library
        if "multihead_attention/dot_product" in name and not name.endswith("/logits")
    ]

    if attention:
        return tf.concat(attention, 0)


def persist_tensor(
    name: Text,
    tensor: Union["tf.Tensor", Tuple["tf.Tensor"], List["tf.Tensor"]],
    graph: "tf.Graph",
) -> None:
    """Add tensor to collection if it is not None"""

    if tensor is not None:
        graph.clear_collection(name)
        if isinstance(tensor, tuple) or isinstance(tensor, list):
            for t in tensor:
                graph.add_to_collection(name, t)
        else:
            graph.add_to_collection(name, tensor)


def load_tensor(name: Text) -> Optional[Union["tf.Tensor", List["tf.Tensor"]]]:
    """Load tensor or set it to None"""

    tensor_list = tf.get_collection(name)

    if not tensor_list:
        return None

    if len(tensor_list) == 1:
        return tensor_list[0]

    return tensor_list
