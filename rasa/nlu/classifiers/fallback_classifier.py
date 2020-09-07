import logging
from typing import Any, List, Type, Text, Dict, Union, Tuple, Optional

from rasa.constants import DEFAULT_NLU_FALLBACK_INTENT_NAME
from rasa.core.constants import DEFAULT_NLU_FALLBACK_THRESHOLD
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.components import Component
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import (
    INTENT_RANKING_KEY,
    PREDICTED_CONFIDENCE_KEY,
    INTENT_NAME_KEY,
)
from rasa.shared.nlu.constants import INTENT

THRESHOLD_KEY = "threshold"
AMBIGUITY_THRESHOLD_KEY = "ambiguity_threshold"

logger = logging.getLogger(__name__)


class FallbackClassifier(Component):

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # If all intent confidence scores are beyond this threshold, set the current
        # intent to `FALLBACK_INTENT_NAME`
        THRESHOLD_KEY: DEFAULT_NLU_FALLBACK_THRESHOLD,
        # If the confidence scores for the top two intent predictions are closer than
        # `AMBIGUITY_THRESHOLD_KEY`, then `FALLBACK_INTENT_NAME ` is predicted.
        AMBIGUITY_THRESHOLD_KEY: 0.1,
    }

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [IntentClassifier]

    def process(self, message: Message, **kwargs: Any) -> None:
        """Process an incoming message.

        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`rasa.nlu.components.Component.create`
        of ANY component and
        on any context attributes created by a call to
        :meth:`rasa.nlu.components.Component.process`
        of components previous to this one.

        Args:
            message: The :class:`rasa.shared.nlu.training_data.message.Message` to
            process.

        """

        if not self._should_fallback(message):
            return

        message.data[INTENT] = _fallback_intent()
        message.data[INTENT_RANKING_KEY].insert(0, _fallback_intent())

    def _should_fallback(self, message: Message) -> bool:
        """Check if the fallback intent should be predicted.

        Args:
            message: The current message and its intent predictions.

        Returns:
            `True` if the fallback intent should be predicted.
        """
        intent_name = message.data[INTENT].get(INTENT_NAME_KEY)
        below_threshold, nlu_confidence = self._nlu_confidence_below_threshold(message)

        if below_threshold:
            logger.debug(
                f"NLU confidence {nlu_confidence} for intent '{intent_name}' is lower "
                f"than NLU threshold {self.component_config[THRESHOLD_KEY]:.2f}."
            )
            return True

        ambiguous_prediction, confidence_delta = self._nlu_prediction_ambiguous(message)
        if ambiguous_prediction:
            logger.debug(
                f"The difference in NLU confidences "
                f"for the top two intents ({confidence_delta}) is lower than "
                f"the ambiguity threshold "
                f"{self.component_config[AMBIGUITY_THRESHOLD_KEY]:.2f}. Predicting "
                f"intent '{DEFAULT_NLU_FALLBACK_INTENT_NAME}' instead of "
                f"'{intent_name}'."
            )
            return True

        return False

    def _nlu_confidence_below_threshold(self, message: Message) -> Tuple[bool, float]:
        nlu_confidence = message.data[INTENT].get(PREDICTED_CONFIDENCE_KEY)
        return nlu_confidence < self.component_config[THRESHOLD_KEY], nlu_confidence

    def _nlu_prediction_ambiguous(
        self, message: Message
    ) -> Tuple[bool, Optional[float]]:
        intents = message.data.get(INTENT_RANKING_KEY, [])
        if len(intents) >= 2:
            first_confidence = intents[0].get(PREDICTED_CONFIDENCE_KEY, 1.0)
            second_confidence = intents[1].get(PREDICTED_CONFIDENCE_KEY, 1.0)
            difference = first_confidence - second_confidence
            return (
                difference < self.component_config[AMBIGUITY_THRESHOLD_KEY],
                difference,
            )
        return False, None


def _fallback_intent() -> Dict[Text, Union[Text, float]]:
    return {
        INTENT_NAME_KEY: DEFAULT_NLU_FALLBACK_INTENT_NAME,
        # TODO: Re-consider how we represent the confidence here
        PREDICTED_CONFIDENCE_KEY: 1.0,
    }
