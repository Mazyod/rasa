import numpy as np
import typing
from typing import Any, List, Text, Dict

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.featurzier import Featurizer
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.training_data import Message, TrainingData

if typing.TYPE_CHECKING:
    import mitie

from rasa.nlu.constants import (
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_TOKENS_NAMES,
    MESSAGE_ATTRIBUTES,
    MESSAGE_VECTOR_DENSE_FEATURE_NAMES,
    CLS_TOKEN,
)


class MitieFeaturizer(Featurizer):

    provides = [
        MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute]
        for attribute in MESSAGE_ATTRIBUTES
    ]

    requires = [MESSAGE_TOKENS_NAMES[attribute] for attribute in MESSAGE_ATTRIBUTES] + [
        "mitie_feature_extractor"
    ]

    defaults = {
        # if True return a sequence of features (return vector has size
        # token-size x feature-dimension)
        # if False token-size will be equal to 1
        "return_sequence": False
    }

    def __init__(self, component_config: Dict[Text, Any] = None):

        super().__init__(component_config)

        self.return_sequence = self.component_config["return_sequence"]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["mitie", "numpy"]

    def ndim(self, feature_extractor: "mitie.total_word_feature_extractor"):

        return feature_extractor.num_dimensions

    def get_tokens_by_attribute(self, example, attribute) -> Any:

        return example.get(MESSAGE_TOKENS_NAMES[attribute])

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        mitie_feature_extractor = self._mitie_feature_extractor(**kwargs)
        for example in training_data.intent_examples:

            for attribute in MESSAGE_ATTRIBUTES:

                attribute_tokens = self.get_tokens_by_attribute(example, attribute)
                if attribute_tokens is not None:

                    features = self.features_for_tokens(
                        attribute_tokens, mitie_feature_extractor
                    )
                    example.set(
                        MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute],
                        self._combine_with_existing_dense_features(
                            example,
                            features,
                            MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute],
                        ),
                    )

    def process(self, message: Message, **kwargs: Any) -> None:

        mitie_feature_extractor = self._mitie_feature_extractor(**kwargs)
        features = self.features_for_tokens(
            message.get(MESSAGE_TOKENS_NAMES[MESSAGE_TEXT_ATTRIBUTE]),
            mitie_feature_extractor,
        )
        message.set(
            MESSAGE_VECTOR_DENSE_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE],
            self._combine_with_existing_dense_features(
                message,
                features,
                MESSAGE_VECTOR_DENSE_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE],
            ),
        )

    def _mitie_feature_extractor(self, **kwargs) -> Any:
        mitie_feature_extractor = kwargs.get("mitie_feature_extractor")
        if not mitie_feature_extractor:
            raise Exception(
                "Failed to train 'MitieFeaturizer'. "
                "Missing a proper MITIE feature extractor. "
                "Make sure this component is preceded by "
                "the 'MitieNLP' component in the pipeline "
                "configuration."
            )
        return mitie_feature_extractor

    def features_for_tokens(
        self,
        tokens: List[Token],
        feature_extractor: "mitie.total_word_feature_extractor",
    ) -> np.ndarray:
        cls_token_used = tokens[-1].text == CLS_TOKEN if tokens else False

        tokens_without_cls = tokens
        if cls_token_used:
            tokens_without_cls = tokens[:-1]

        # calculate features
        features = []
        for token in tokens_without_cls:
            features.append(feature_extractor.get_feature_vector(token.text))
        features = np.array(features)

        if cls_token_used and self.return_sequence:
            # cls token is used, need to append a vector
            cls_token_vec = np.mean(features, axis=0, keepdims=True)
            features = np.concatenate([features, cls_token_vec])

        if not self.return_sequence:
            features = np.mean(features, axis=0, keepdims=True)

        return features
