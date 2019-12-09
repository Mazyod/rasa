import numpy as np
import typing
from typing import Any, Optional, Dict, Text

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.featurzier import Featurizer
from rasa.nlu.training_data import Message, TrainingData

if typing.TYPE_CHECKING:
    from spacy.tokens import Doc

from rasa.nlu.constants import (
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_SPACY_FEATURES_NAMES,
    MESSAGE_VECTOR_DENSE_FEATURE_NAMES,
    SPACY_FEATURIZABLE_ATTRIBUTES,
    MESSAGE_TOKENS_NAMES,
    CLS_TOKEN,
)


class SpacyFeaturizer(Featurizer):

    provides = [
        MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute]
        for attribute in SPACY_FEATURIZABLE_ATTRIBUTES
    ]

    requires = [
        MESSAGE_SPACY_FEATURES_NAMES[attribute]
        for attribute in SPACY_FEATURIZABLE_ATTRIBUTES
    ] + [MESSAGE_TOKENS_NAMES[attribute] for attribute in SPACY_FEATURIZABLE_ATTRIBUTES]

    defaults = {
        # if True return a sequence of features (return vector has size
        # token-size x feature-dimension)
        # if False token-size will be equal to 1
        "return_sequence": False
    }

    def __init__(self, component_config: Dict[Text, Any] = None):

        super().__init__(component_config)

        self.return_sequence = self.component_config["return_sequence"]

    def _features_for_doc(self, doc: "Doc") -> np.ndarray:
        """Feature vector for a single document / sentence / tokens."""
        if self.return_sequence:
            return np.array([t.vector for t in doc])
        return np.array([doc.vector])

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig],
        **kwargs: Any,
    ) -> None:

        for example in training_data.intent_examples:
            for attribute in SPACY_FEATURIZABLE_ATTRIBUTES:
                self._set_spacy_features(example, attribute)

    def get_doc(self, message, attribute):

        return message.get(MESSAGE_SPACY_FEATURES_NAMES[attribute])

    def process(self, message: Message, **kwargs: Any) -> None:

        self._set_spacy_features(message)

    def _set_spacy_features(self, message, attribute=MESSAGE_TEXT_ATTRIBUTE):
        """Adds the spacy word vectors to the messages features."""

        message_attribute_doc = self.get_doc(message, attribute)
        tokens = message.get(MESSAGE_TOKENS_NAMES[attribute])
        cls_token_used = tokens[-1].text == CLS_TOKEN if tokens else False

        if message_attribute_doc is not None:
            features = self._features_for_doc(message_attribute_doc)

            if cls_token_used and self.return_sequence:
                # cls token is used, need to append a vector
                cls_token_vec = np.mean(features, axis=0, keepdims=True)
                features = np.concatenate([features, cls_token_vec])

            features = self._combine_with_existing_dense_features(
                message, features, MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute]
            )
            message.set(MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute], features)
