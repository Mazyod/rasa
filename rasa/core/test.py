import logging
from collections import namedtuple
from typing import Any, Dict, List, Optional, Text, Tuple, TYPE_CHECKING

from rasa.shared.core.constants import (
    ACTION_UNLIKELY_INTENT_NAME,
)
from rasa.shared.exceptions import RasaException
from rasa.shared.core.training_data.structures import StoryStep
from rasa.nlu.constants import (
    RESPONSE_SELECTOR_DEFAULT_INTENT,
    RESPONSE_SELECTOR_RETRIEVAL_INTENTS,
)
from rasa.shared.nlu.constants import (
    INTENT,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    EXTRACTOR,
    ENTITY_ATTRIBUTE_TYPE,
    INTENT_RESPONSE_KEY,
    INTENT_NAME_KEY,
    RESPONSE,
    RESPONSE_SELECTOR,
    ENTITY_ATTRIBUTE_TEXT,
)
from rasa.shared.core.events import ActionExecuted, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.formats.readerwriter import TrainingDataWriter
from rasa.shared.utils.io import DEFAULT_ENCODING

if TYPE_CHECKING:
    from rasa.shared.core.events import EntityPrediction

logger = logging.getLogger(__name__)

StoryEvaluation = namedtuple(
    "StoryEvaluation",
    [
        "evaluation_store",
        "failed_stories",
        "successful_stories",
        "stories_with_warnings",
        "action_list",
        "in_training_data_fraction",
    ],
)

PredictionList = List[Optional[Text]]


class WrongPredictionException(RasaException, ValueError):
    """Raised if a wrong prediction is encountered."""


class WarningPredictedAction(ActionExecuted):
    """The model predicted the correct action with warning."""

    type_name = "warning_predicted"

    def __init__(
        self,
        action_name_prediction: Text,
        action_name: Optional[Text] = None,
        policy: Optional[Text] = None,
        confidence: Optional[float] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ):
        """Creates event `action_unlikely_intent` predicted as warning.

        See the docstring of the parent class for more information.
        """
        self.action_name_prediction = action_name_prediction
        super().__init__(action_name, policy, confidence, timestamp, metadata)

    def inline_comment(self, **kwargs: Any) -> Text:
        """A comment attached to this event. Used during dumping."""
        return f"predicted: {self.action_name_prediction}"


class WronglyPredictedAction(ActionExecuted):
    """The model predicted the wrong action.

    Mostly used to mark wrong predictions and be able to
    dump them as stories.
    """

    type_name = "wrong_action"

    def __init__(
        self,
        action_name_target: Text,
        action_text_target: Text,
        action_name_prediction: Text,
        policy: Optional[Text] = None,
        confidence: Optional[float] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict] = None,
        predicted_action_unlikely_intent: bool = False,
    ) -> None:
        """Creates event for a successful event execution.

        See the docstring of the parent class `ActionExecuted` for more information.
        """
        self.action_name_prediction = action_name_prediction
        self.predicted_action_unlikely_intent = predicted_action_unlikely_intent
        super().__init__(
            action_name_target,
            policy,
            confidence,
            timestamp,
            metadata,
            action_text=action_text_target,
        )

    def inline_comment(self, **kwargs: Any) -> Text:
        """A comment attached to this event. Used during dumping."""
        comment = f"predicted: {self.action_name_prediction}"
        if self.predicted_action_unlikely_intent:
            return f"{comment} after {ACTION_UNLIKELY_INTENT_NAME}"
        return comment

    def as_story_string(self) -> Text:
        """Returns the story equivalent representation."""
        return f"{self.action_name}   <!-- {self.inline_comment()} -->"

    def __repr__(self) -> Text:
        """Returns event as string for debugging."""
        return (
            f"WronglyPredictedAction(action_target: {self.action_name}, "
            f"action_prediction: {self.action_name_prediction}, "
            f"policy: {self.policy}, confidence: {self.confidence}, "
            f"metadata: {self.metadata})"
        )


class EvaluationStore:
    """Class storing action, intent and entity predictions and targets."""

    def __init__(
        self,
        action_predictions: Optional[PredictionList] = None,
        action_targets: Optional[PredictionList] = None,
        intent_predictions: Optional[PredictionList] = None,
        intent_targets: Optional[PredictionList] = None,
        entity_predictions: Optional[List["EntityPrediction"]] = None,
        entity_targets: Optional[List["EntityPrediction"]] = None,
    ) -> None:
        """Initialize store attributes."""
        self.action_predictions = action_predictions or []
        self.action_targets = action_targets or []
        self.intent_predictions = intent_predictions or []
        self.intent_targets = intent_targets or []
        self.entity_predictions: List["EntityPrediction"] = entity_predictions or []
        self.entity_targets: List["EntityPrediction"] = entity_targets or []

    def add_to_store(
        self,
        action_predictions: Optional[PredictionList] = None,
        action_targets: Optional[PredictionList] = None,
        intent_predictions: Optional[PredictionList] = None,
        intent_targets: Optional[PredictionList] = None,
        entity_predictions: Optional[List["EntityPrediction"]] = None,
        entity_targets: Optional[List["EntityPrediction"]] = None,
    ) -> None:
        """Add items or lists of items to the store."""
        self.action_predictions.extend(action_predictions or [])
        self.action_targets.extend(action_targets or [])
        self.intent_targets.extend(intent_targets or [])
        self.intent_predictions.extend(intent_predictions or [])
        self.entity_predictions.extend(entity_predictions or [])
        self.entity_targets.extend(entity_targets or [])

    def merge_store(self, other: "EvaluationStore") -> None:
        """Add the contents of other to self."""
        self.add_to_store(
            action_predictions=other.action_predictions,
            action_targets=other.action_targets,
            intent_predictions=other.intent_predictions,
            intent_targets=other.intent_targets,
            entity_predictions=other.entity_predictions,
            entity_targets=other.entity_targets,
        )

    def _check_entity_prediction_target_mismatch(self) -> bool:
        """Checks that same entities were expected and actually extracted.

        Possible duplicates or differences in order should not matter.
        """
        deduplicated_targets = set(
            tuple(entity.items()) for entity in self.entity_targets
        )
        deduplicated_predictions = set(
            tuple(entity.items()) for entity in self.entity_predictions
        )
        return deduplicated_targets != deduplicated_predictions

    def check_prediction_target_mismatch(self) -> bool:
        """Checks if intent, entity or action predictions don't match expected ones."""
        return (
            self.intent_predictions != self.intent_targets
            or self._check_entity_prediction_target_mismatch()
            or self.action_predictions != self.action_targets
        )

    @staticmethod
    def _compare_entities(
        entity_predictions: List["EntityPrediction"],
        entity_targets: List["EntityPrediction"],
        i_pred: int,
        i_target: int,
    ) -> int:
        """Compare the current predicted and target entities and decide which one
        comes first. If the predicted entity comes first it returns -1,
        while it returns 1 if the target entity comes first.
        If target and predicted are aligned it returns 0.
        """
        pred = None
        target = None
        if i_pred < len(entity_predictions):
            pred = entity_predictions[i_pred]
        if i_target < len(entity_targets):
            target = entity_targets[i_target]
        if target and pred:
            # Check which entity has the lower "start" value
            if pred.get(ENTITY_ATTRIBUTE_START) < target.get(ENTITY_ATTRIBUTE_START):
                return -1
            elif target.get(ENTITY_ATTRIBUTE_START) < pred.get(ENTITY_ATTRIBUTE_START):
                return 1
            else:
                # Since both have the same "start" values,
                # check which one has the lower "end" value
                if pred.get(ENTITY_ATTRIBUTE_END) < target.get(ENTITY_ATTRIBUTE_END):
                    return -1
                elif target.get(ENTITY_ATTRIBUTE_END) < pred.get(ENTITY_ATTRIBUTE_END):
                    return 1
                else:
                    # The entities have the same "start" and "end" values
                    return 0
        return 1 if target else -1

    @staticmethod
    def _generate_entity_training_data(entity: Dict[Text, Any]) -> Text:
        return TrainingDataWriter.generate_entity(entity.get("text"), entity)

    def serialise(self) -> Tuple[PredictionList, PredictionList]:
        """Turn targets and predictions to lists of equal size for sklearn."""
        texts = sorted(
            set(
                [str(e.get("text", "")) for e in self.entity_targets]
                + [str(e.get("text", "")) for e in self.entity_predictions]
            )
        )

        aligned_entity_targets: List[Optional[Text]] = []
        aligned_entity_predictions: List[Optional[Text]] = []

        for text in texts:
            # sort the entities of this sentence to compare them directly
            entity_targets = sorted(
                filter(
                    lambda x: x.get(ENTITY_ATTRIBUTE_TEXT) == text, self.entity_targets
                ),
                key=lambda x: x[ENTITY_ATTRIBUTE_START],  # type: ignore[literal-required] # noqa: E501
            )
            entity_predictions = sorted(
                filter(
                    lambda x: x.get(ENTITY_ATTRIBUTE_TEXT) == text,
                    self.entity_predictions,
                ),
                key=lambda x: x[ENTITY_ATTRIBUTE_START],  # type: ignore[literal-required] # noqa: E501
            )

            i_pred, i_target = 0, 0

            while i_pred < len(entity_predictions) or i_target < len(entity_targets):
                cmp = self._compare_entities(
                    entity_predictions, entity_targets, i_pred, i_target
                )
                if cmp == -1:  # predicted comes first
                    aligned_entity_predictions.append(
                        self._generate_entity_training_data(entity_predictions[i_pred])
                    )
                    aligned_entity_targets.append("None")
                    i_pred += 1
                elif cmp == 1:  # target entity comes first
                    aligned_entity_targets.append(
                        self._generate_entity_training_data(entity_targets[i_target])
                    )
                    aligned_entity_predictions.append("None")
                    i_target += 1
                else:  # target and predicted entity are aligned
                    aligned_entity_predictions.append(
                        self._generate_entity_training_data(entity_predictions[i_pred])
                    )
                    aligned_entity_targets.append(
                        self._generate_entity_training_data(entity_targets[i_target])
                    )
                    i_pred += 1
                    i_target += 1

        targets = self.action_targets + self.intent_targets + aligned_entity_targets

        predictions = (
            self.action_predictions
            + self.intent_predictions
            + aligned_entity_predictions
        )
        return targets, predictions


class WronglyClassifiedUserUtterance(UserUttered):
    """The NLU model predicted the wrong user utterance.

    Mostly used to mark wrong predictions and be able to
    dump them as stories.
    """

    type_name = "wrong_utterance"

    def __init__(self, event: UserUttered, eval_store: EvaluationStore) -> None:
        """Set `predicted_intent` and `predicted_entities` attributes."""
        try:
            self.predicted_intent = eval_store.intent_predictions[0]
        except LookupError:
            self.predicted_intent = None

        self.target_entities = eval_store.entity_targets
        self.predicted_entities = eval_store.entity_predictions

        intent = {"name": eval_store.intent_targets[0]}

        super().__init__(
            event.text,
            intent,
            eval_store.entity_targets,
            event.parse_data,
            event.timestamp,
            event.input_channel,
        )

    def inline_comment(self, force_comment_generation: bool = False) -> Optional[Text]:
        """A comment attached to this event. Used during dumping."""
        from rasa.shared.core.events import format_message

        if force_comment_generation or self.predicted_intent != self.intent["name"]:
            predicted_message = format_message(
                self.text, self.predicted_intent, self.predicted_entities
            )

            return f"predicted: {self.predicted_intent}: {predicted_message}"
        else:
            return None

    @staticmethod
    def inline_comment_for_entity(
        predicted: Dict[Text, Any], entity: Dict[Text, Any]
    ) -> Optional[Text]:
        """Returns the predicted entity which is then printed as a comment."""
        if predicted["entity"] != entity["entity"]:
            return "predicted: " + predicted["entity"] + ": " + predicted["value"]
        else:
            return None

    def as_story_string(self, e2e: bool = True) -> Text:
        """Returns text representation of event."""
        from rasa.shared.core.events import format_message

        correct_message = format_message(
            self.text, self.intent.get("name"), self.entities
        )
        return (
            f"{self.intent.get('name')}: {correct_message}   "
            f"<!-- {self.inline_comment()} -->"
        )


def _clean_entity_results(
    text: Text, entity_results: List[Dict[Text, Any]]
) -> List["EntityPrediction"]:
    """Extract only the token variables from an entity dict."""
    cleaned_entities = []

    for r in tuple(entity_results):
        cleaned_entity: EntityPrediction = {ENTITY_ATTRIBUTE_TEXT: text}  # type: ignore[misc]  # noqa E501
        for k in (
            ENTITY_ATTRIBUTE_START,
            ENTITY_ATTRIBUTE_END,
            ENTITY_ATTRIBUTE_TYPE,
            ENTITY_ATTRIBUTE_VALUE,
        ):
            if k in set(r):
                if k == ENTITY_ATTRIBUTE_VALUE and EXTRACTOR in set(r):
                    # convert values to strings for evaluation as
                    # target values are all of type string
                    r[k] = str(r[k])
                cleaned_entity[k] = r[k]  # type: ignore[literal-required]
        cleaned_entities.append(cleaned_entity)

    return cleaned_entities


def _get_full_retrieval_intent(parsed: Dict[Text, Any]) -> Text:
    """Return full retrieval intent, if it's present, or normal intent otherwise.

    Args:
        parsed: Predicted parsed data.

    Returns:
        The extracted intent.
    """
    base_intent = parsed.get(INTENT, {}).get(INTENT_NAME_KEY)
    response_selector = parsed.get(RESPONSE_SELECTOR, {})

    # return normal intent if it's not a retrieval intent
    if base_intent not in response_selector.get(
        RESPONSE_SELECTOR_RETRIEVAL_INTENTS, {}
    ):
        return base_intent

    # extract full retrieval intent
    # if the response selector parameter was not specified in config,
    # the response selector contains a "default" key
    if RESPONSE_SELECTOR_DEFAULT_INTENT in response_selector:
        full_retrieval_intent = (
            response_selector.get(RESPONSE_SELECTOR_DEFAULT_INTENT, {})
            .get(RESPONSE, {})
            .get(INTENT_RESPONSE_KEY)
        )
        return full_retrieval_intent if full_retrieval_intent else base_intent

    # if specified, the response selector contains the base intent as key
    full_retrieval_intent = (
        response_selector.get(base_intent, {})
        .get(RESPONSE, {})
        .get(INTENT_RESPONSE_KEY)
    )
    return full_retrieval_intent if full_retrieval_intent else base_intent


def emulate_loop_rejection(partial_tracker: DialogueStateTracker) -> None:
    """Add `ActionExecutionRejected` event to the tracker.

    During evaluation, we don't run action server, therefore in order to correctly
    test unhappy paths of the loops, we need to emulate loop rejection.

    Args:
        partial_tracker: a :class:`rasa.core.trackers.DialogueStateTracker`
    """
    from rasa.shared.core.events import ActionExecutionRejected

    rejected_action_name = partial_tracker.active_loop_name
    partial_tracker.update(ActionExecutionRejected(rejected_action_name))
