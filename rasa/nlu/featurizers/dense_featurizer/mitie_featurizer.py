import numpy as np
import typing
from typing import Any, List, Text, Optional, Dict

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.training_data import Message, TrainingData

if typing.TYPE_CHECKING:
    import mitie

from rasa.nlu.constants import (
    TEXT,
    TOKENS_NAMES,
    DENSE_FEATURE_NAMES,
    DENSE_FEATURIZABLE_ATTRIBUTES,
)


class MitieFeaturizer(Featurizer):

    provides = [
        DENSE_FEATURE_NAMES[attribute] for attribute in DENSE_FEATURIZABLE_ATTRIBUTES
    ]

    requires = [
        TOKENS_NAMES[attribute] for attribute in DENSE_FEATURIZABLE_ATTRIBUTES
    ] + ["mitie_feature_extractor"]

    defaults = {
        # Specify what pooling operation should be used to calculate the vector of
        # the CLS token. Available options: 'mean' and 'max'
        "pooling": "mean"
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        super().__init__(component_config)

        self.pooling_operation = self.component_config["pooling"]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["mitie", "numpy"]

    def ndim(self, feature_extractor: "mitie.total_word_feature_extractor") -> int:

        return feature_extractor.num_dimensions

    def get_tokens_by_attribute(self, example: Message, attribute: Text) -> Any:
        return example.get(TOKENS_NAMES[attribute])

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:

        mitie_feature_extractor = self._mitie_feature_extractor(**kwargs)
        for example in training_data.intent_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self.process_training_example(
                    example, attribute, mitie_feature_extractor
                )

    def process_training_example(
        self, example: Message, attribute: Text, mitie_feature_extractor: Any
    ):
        attribute_tokens = self.get_tokens_by_attribute(example, attribute)
        if attribute_tokens is not None:
            features = self.features_for_tokens(
                attribute_tokens, mitie_feature_extractor
            )
            example.set(
                DENSE_FEATURE_NAMES[attribute],
                self._combine_with_existing_dense_features(
                    example, features, DENSE_FEATURE_NAMES[attribute]
                ),
            )

    def process(self, message: Message, **kwargs: Any) -> None:

        mitie_feature_extractor = self._mitie_feature_extractor(**kwargs)
        features = self.features_for_tokens(
            message.get(TOKENS_NAMES[TEXT]), mitie_feature_extractor
        )
        message.set(
            DENSE_FEATURE_NAMES[TEXT],
            self._combine_with_existing_dense_features(
                message, features, DENSE_FEATURE_NAMES[TEXT]
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

        # remove CLS token from tokens
        tokens_without_cls = tokens[:-1]

        # calculate features
        features = []
        for token in tokens_without_cls:
            features.append(feature_extractor.get_feature_vector(token.text))
        features = np.array(features)

        cls_token_vec = self._calculate_cls_vector(features, self.pooling_operation)
        features = np.concatenate([features, cls_token_vec])

        return features
