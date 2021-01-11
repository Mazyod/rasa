import copy
from typing import Union, List

import pytest
import scipy.sparse
import numpy as np

from rasa.utils.tensorflow.model_data import RasaModelData, FeatureArray


@pytest.fixture
async def model_data() -> RasaModelData:
    return RasaModelData(
        label_key="label",
        label_sub_key="ids",
        data={
            "text": {
                "sentence": [
                    FeatureArray(
                        np.array(
                            [
                                np.random.rand(5, 14),
                                np.random.rand(2, 14),
                                np.random.rand(3, 14),
                                np.random.rand(1, 14),
                                np.random.rand(3, 14),
                            ]
                        ),
                        number_of_dimensions=3,
                    ),
                    FeatureArray(
                        np.array(
                            [
                                scipy.sparse.csr_matrix(
                                    np.random.randint(5, size=(5, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(5, size=(2, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(5, size=(3, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(5, size=(1, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(5, size=(3, 10))
                                ),
                            ]
                        ),
                        number_of_dimensions=3,
                    ),
                ]
            },
            "action_text": {
                "sequence": [
                    FeatureArray(
                        np.array(
                            [
                                [
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(5, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(2, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(3, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(1, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(3, 10))
                                    ),
                                ],
                                [
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(5, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(2, 10))
                                    ),
                                ],
                                [
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(5, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(1, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(3, 10))
                                    ),
                                ],
                                [
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(3, 10))
                                    )
                                ],
                                [
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(3, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(1, 10))
                                    ),
                                    scipy.sparse.csr_matrix(
                                        np.random.randint(5, size=(7, 10))
                                    ),
                                ],
                            ]
                        ),
                        number_of_dimensions=4,
                    ),
                    FeatureArray(
                        np.array(
                            [
                                [
                                    np.random.rand(5, 14),
                                    np.random.rand(2, 14),
                                    np.random.rand(3, 14),
                                    np.random.rand(1, 14),
                                    np.random.rand(3, 14),
                                ],
                                [np.random.rand(5, 14), np.random.rand(2, 14)],
                                [
                                    np.random.rand(5, 14),
                                    np.random.rand(1, 14),
                                    np.random.rand(3, 14),
                                ],
                                [np.random.rand(3, 14)],
                                [
                                    np.random.rand(3, 14),
                                    np.random.rand(1, 14),
                                    np.random.rand(7, 14),
                                ],
                            ]
                        ),
                        number_of_dimensions=4,
                    ),
                ]
            },
            "dialogue": {
                "sentence": [
                    FeatureArray(
                        np.array(
                            [
                                np.random.randint(2, size=(5, 10)),
                                np.random.randint(2, size=(2, 10)),
                                np.random.randint(2, size=(3, 10)),
                                np.random.randint(2, size=(1, 10)),
                                np.random.randint(2, size=(3, 10)),
                            ]
                        ),
                        number_of_dimensions=3,
                    )
                ]
            },
            "label": {
                "ids": [FeatureArray(np.array([0, 1, 0, 1, 1]), number_of_dimensions=1)]
            },
            "entities": {
                "tag_ids": [
                    FeatureArray(
                        np.array(
                            [
                                np.array([[0], [1], [1], [0], [2]]),
                                np.array([[2], [0]]),
                                np.array([[0], [1], [1]]),
                                np.array([[0], [1]]),
                                np.array([[0], [0], [0]]),
                            ]
                        ),
                        number_of_dimensions=3,
                    )
                ]
            },
        },
    )


def test_shuffle_session_data(model_data: RasaModelData):
    before = copy.copy(model_data)

    # precondition
    assert np.all(
        np.array(list(before.values())) == np.array(list(model_data.values()))
    )

    data = model_data._shuffled_data(model_data.data)

    # check that original data didn't change
    assert np.all(
        np.array(list(before.values())) == np.array(list(model_data.values()))
    )
    # check that new data is different
    assert np.all(np.array(model_data.values()) != np.array(data.values()))


def test_split_data_by_label(model_data: RasaModelData):
    split_model_data = model_data._split_by_label_ids(
        model_data.data, model_data.get("label", "ids")[0], np.array([0, 1])
    )

    assert len(split_model_data) == 2
    for s in split_model_data:
        assert len(set(s.get("label", "ids")[0])) == 1

    for key, attribute_data in split_model_data[0].items():
        for sub_key, features in attribute_data.items():
            assert len(features) == len(model_data.data[key][sub_key])
            assert len(features[0]) == 2


def test_split_data_by_none_label(model_data: RasaModelData):
    model_data.label_key = None
    model_data.label_sub_key = None

    split_model_data = model_data.split(2, 42)

    assert len(split_model_data) == 2

    train_data = split_model_data[0]
    test_data = split_model_data[1]

    # train data should have 3 examples
    assert len(train_data.get("label", "ids")[0]) == 3
    # test data should have 2 examples
    assert len(test_data.get("label", "ids")[0]) == 2


def test_train_val_split(model_data: RasaModelData):
    train_model_data, test_model_data = model_data.split(2, 42)

    for key, values in model_data.items():
        assert len(values) == len(train_model_data.get(key))
        assert len(values) == len(test_model_data.get(key))
        for sub_key, data in values.items():
            assert len(data) == len(train_model_data.get(key, sub_key))
            assert len(data) == len(test_model_data.get(key, sub_key))
            for i, v in enumerate(data):
                if isinstance(v[0], list):
                    assert (
                        v[0][0].dtype
                        == train_model_data.get(key, sub_key)[i][0][0].dtype
                    )
                else:
                    assert v[0].dtype == train_model_data.get(key, sub_key)[i][0].dtype

    for values in train_model_data.values():
        for data in values.values():
            for v in data:
                assert np.array(v).shape[0] == 3

    for values in test_model_data.values():
        for data in values.values():
            for v in data:
                assert np.array(v).shape[0] == 2


@pytest.mark.parametrize("size", [0, 1, 5])
def test_train_val_split_incorrect_size(model_data: RasaModelData, size: int):
    with pytest.raises(ValueError):
        model_data.split(size, 42)


def test_session_data_for_ids(model_data: RasaModelData):
    filtered_data = model_data._data_for_ids(model_data.data, np.array([0, 1]))

    for values in filtered_data.values():
        for data in values.values():
            for v in data:
                assert np.array(v).shape[0] == 2

    key = model_data.keys()[0]
    sub_key = model_data.keys(key)[0]

    assert np.all(
        np.array(filtered_data[key][sub_key][0][0])
        == np.array(model_data.get(key, sub_key)[0][0])
    )
    assert np.all(
        np.array(filtered_data[key][sub_key][0][1])
        == np.array(model_data.get(key, sub_key)[0][1])
    )


def test_get_number_of_examples(model_data: RasaModelData):
    assert model_data.number_of_examples() == 5


def test_get_number_of_examples_raises_value_error(model_data: RasaModelData):
    model_data.data["dense"] = {}
    model_data.data["dense"]["data"] = [np.random.randint(5, size=(2, 10))]
    with pytest.raises(ValueError):
        model_data.number_of_examples()


def test_gen_batch(model_data: RasaModelData):
    iterator = model_data._gen_batch(2, shuffle=True, batch_strategy="balanced")

    batch = next(iterator)
    assert len(batch) == 11
    assert len(batch[0]) == 2

    batch = next(iterator)
    assert len(batch) == 11
    assert len(batch[0]) == 2

    batch = next(iterator)
    assert len(batch) == 11
    assert len(batch[0]) == 1

    with pytest.raises(StopIteration):
        next(iterator)


def test_is_in_4d_format(model_data: RasaModelData):
    assert model_data.data["action_text"]["sequence"][0].number_of_dimensions == 4
    assert model_data.data["text"]["sentence"][0].number_of_dimensions == 3


def test_balance_model_data(model_data: RasaModelData):
    data = model_data._balanced_data(model_data.data, 2, False)

    assert np.all(np.array(data["label"]["ids"][0]) == np.array([0, 1, 1, 0, 1]))


def test_not_balance_model_data(model_data: RasaModelData):
    test_model_data = RasaModelData(
        label_key="entities", label_sub_key="tag_ids", data=model_data.data
    )

    data = test_model_data._balanced_data(test_model_data.data, 2, False)

    assert np.all(
        data["entities"]["tag_ids"] == test_model_data.get("entities", "tag_ids")
    )


def test_get_num_of_features(model_data: RasaModelData):
    num_features = model_data.number_of_units("text", "sentence")

    assert num_features == 24


@pytest.mark.parametrize(
    "incoming_data, expected_shape",
    [
        (FeatureArray(np.random.rand(7, 12), number_of_dimensions=2), (7, 12)),
        (FeatureArray(np.random.rand(7), number_of_dimensions=1), (7,)),
        (
            FeatureArray(
                np.array(
                    [
                        np.random.rand(1, 10),
                        np.random.rand(3, 10),
                        np.random.rand(7, 10),
                        np.random.rand(1, 10),
                    ]
                ),
                number_of_dimensions=3,
            ),
            (4, 7, 10),
        ),
        (
            FeatureArray(
                np.array(
                    [
                        np.array(
                            [
                                np.random.rand(1, 10),
                                np.random.rand(5, 10),
                                np.random.rand(7, 10),
                            ]
                        ),
                        np.array(
                            [
                                np.random.rand(1, 10),
                                np.random.rand(3, 10),
                                np.random.rand(3, 10),
                                np.random.rand(7, 10),
                            ]
                        ),
                        np.array([np.random.rand(2, 10)]),
                    ]
                ),
                number_of_dimensions=4,
            ),
            (8, 7, 10),
        ),
    ],
)
def test_pad_dense_data(incoming_data: FeatureArray, expected_shape: np.ndarray):
    padded_data = RasaModelData._pad_dense_data(incoming_data)

    assert padded_data.shape == expected_shape


@pytest.mark.parametrize(
    "incoming_data, expected_shape",
    [
        (
            FeatureArray(
                np.array([scipy.sparse.csr_matrix(np.random.randint(5, size=(7, 12)))]),
                number_of_dimensions=3,
            ),
            [1, 7, 12],
        ),
        (
            FeatureArray(
                np.array([scipy.sparse.csr_matrix(np.random.randint(5, size=(7,)))]),
                number_of_dimensions=2,
            ),
            [1, 1, 7],
        ),
        (
            FeatureArray(
                np.array(
                    [
                        scipy.sparse.csr_matrix(np.random.randint(10, size=(1, 10))),
                        scipy.sparse.csr_matrix(np.random.randint(10, size=(3, 10))),
                        scipy.sparse.csr_matrix(np.random.randint(10, size=(7, 10))),
                        scipy.sparse.csr_matrix(np.random.randint(10, size=(1, 10))),
                    ]
                ),
                number_of_dimensions=3,
            ),
            (4, 7, 10),
        ),
        (
            FeatureArray(
                np.array(
                    [
                        np.array(
                            [
                                scipy.sparse.csr_matrix(
                                    np.random.randint(10, size=(1, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(10, size=(5, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(10, size=(7, 10))
                                ),
                            ]
                        ),
                        np.array(
                            [
                                scipy.sparse.csr_matrix(
                                    np.random.randint(10, size=(1, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(10, size=(3, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(10, size=(1, 10))
                                ),
                                scipy.sparse.csr_matrix(
                                    np.random.randint(10, size=(7, 10))
                                ),
                            ]
                        ),
                        np.array(
                            [
                                scipy.sparse.csr_matrix(
                                    np.random.randint(10, size=(2, 10))
                                )
                            ]
                        ),
                    ]
                ),
                number_of_dimensions=4,
            ),
            (8, 7, 10),
        ),
    ],
)
def test_scipy_matrix_to_values(
    incoming_data: FeatureArray, expected_shape: np.ndarray
):
    indices, data, shape = RasaModelData._scipy_matrix_to_values(incoming_data)

    assert np.all(shape == expected_shape)


def test_sort(model_data: RasaModelData):
    assert list(model_data.data.keys()) == [
        "text",
        "action_text",
        "dialogue",
        "label",
        "entities",
    ]

    model_data.sort()

    assert list(model_data.data.keys()) == [
        "action_text",
        "dialogue",
        "entities",
        "label",
        "text",
    ]


def test_update_key(model_data: RasaModelData):
    assert model_data.does_feature_exist("label", "ids")

    model_data.update_key("label", "ids", "intent", "ids")

    assert not model_data.does_feature_exist("label", "ids")
    assert model_data.does_feature_exist("intent", "ids")
    assert "label" not in model_data.data
