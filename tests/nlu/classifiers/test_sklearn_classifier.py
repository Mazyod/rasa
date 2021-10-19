import copy
import logging
from typing import Callable, List, Text, Tuple

import pytest
from _pytest.logging import LogCaptureFixture

from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import (
    SpacyFeaturizerGraphComponent,
)
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizerGraphComponent
from rasa.nlu.utils.spacy_utils import SpacyModel, SpacyNLPGraphComponent
import rasa.shared.nlu.training_data.loading
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.classifiers.sklearn_intent_classifier import (
    SklearnIntentClassifierGraphComponent,
)
from rasa.shared.nlu.training_data.training_data import TrainingData


@pytest.fixture()
def training_data(nlu_data_path: Text) -> TrainingData:
    return rasa.shared.nlu.training_data.loading.load_data(nlu_data_path)


@pytest.fixture()
def default_sklearn_intent_classifier(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext,
):
    return SklearnIntentClassifierGraphComponent.create(
        SklearnIntentClassifierGraphComponent.get_default_config(),
        default_model_storage,
        Resource("sklearn"),
        default_execution_context,
    )


def test_persist_and_load(
    training_data: TrainingData,
    default_sklearn_intent_classifier: SklearnIntentClassifierGraphComponent,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    train_and_preprocess: Callable[..., Tuple[TrainingData, List[GraphComponent]]],
    spacy_nlp_component: SpacyNLPGraphComponent,
    spacy_model: SpacyModel,
):
    training_data = spacy_nlp_component.process_training_data(
        training_data, spacy_model
    )

    training_data, loaded_pipeline = train_and_preprocess(
        pipeline=[
            {"component": SpacyTokenizerGraphComponent},
            {"component": SpacyFeaturizerGraphComponent},
        ],
        training_data=training_data,
    )
    default_sklearn_intent_classifier.train(training_data)

    loaded = SklearnIntentClassifierGraphComponent.load(
        SklearnIntentClassifierGraphComponent.get_default_config(),
        default_model_storage,
        Resource("sklearn"),
        default_execution_context,
    )

    predicted = copy.deepcopy(training_data)
    actual = copy.deepcopy(training_data)
    loaded_messages = loaded.process(predicted.training_examples)
    trained_messages = default_sklearn_intent_classifier.process(
        actual.training_examples
    )

    for m1, m2 in zip(loaded_messages, trained_messages):
        assert m1.get("intent") == m2.get("intent")


def test_loading_from_storage_fail(
    training_data: TrainingData,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.DEBUG):
        loaded = SklearnIntentClassifierGraphComponent.load(
            SklearnIntentClassifierGraphComponent.get_default_config(),
            default_model_storage,
            Resource("test"),
            default_execution_context,
        )
        assert isinstance(loaded, SklearnIntentClassifierGraphComponent)

    assert any(
        "Resource 'test' doesn't exist." in message for message in caplog.messages
    )
