from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, Text, runtime_checkable

import structlog

from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.constants import RASA_DEFAULT_INTENT_PREFIX
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.constants import ENTITY_ATTRIBUTE_TYPE, INTENT_NAME_KEY

import rasa.shared.utils.io
from rasa.utils.llm import (
    DEFAULT_OPENAI_GENERATE_MODEL_NAME,
    DEFAULT_OPENAI_TEMPERATURE,
)

structlogger = structlog.get_logger()

HANDLING_PATTERN_PREFIX = "pattern_"


class UnreachableFlowStepException(RasaException):
    """Raised when a flow step is unreachable."""

    def __init__(self, step: FlowStep, flow: Flow) -> None:
        """Initializes the exception."""
        self.step = step
        self.flow = flow

    def __str__(self) -> Text:
        """Return a string representation of the exception."""
        return (
            f"Step '{self.step.id}' in flow '{self.flow.id}' can not be reached "
            f"from the start step. Please make sure that all steps can be reached "
            f"from the start step, e.g. by "
            f"checking that another step points to this step."
        )


class UnresolvedFlowStepIdException(RasaException):
    """Raised when a flow step is referenced but it's id can not be resolved."""

    def __init__(
        self, step_id: Text, flow: Flow, referenced_from: Optional[FlowStep]
    ) -> None:
        """Initializes the exception."""
        self.step_id = step_id
        self.flow = flow
        self.referenced_from = referenced_from

    def __str__(self) -> Text:
        """Return a string representation of the exception."""
        if self.referenced_from:
            exception_message = (
                f"Step with id '{self.step_id}' could not be resolved. "
                f"'Step '{self.referenced_from.id}' in flow '{self.flow.id}' "
                f"referenced this step but it does not exist. "
            )
        else:
            exception_message = (
                f"Step '{self.step_id}' in flow '{self.flow.id}' can not be resolved. "
            )

        return exception_message + (
            "Please make sure that the step is defined in the same flow."
        )


class UnresolvedFlowException(RasaException):
    """Raised when a flow is referenced but it's id can not be resolved."""

    def __init__(self, flow_id: Text) -> None:
        """Initializes the exception."""
        self.flow_id = flow_id

    def __str__(self) -> Text:
        """Return a string representation of the exception."""
        return (
            f"Flow '{self.flow_id}' can not be resolved. "
            f"Please make sure that the flow is defined."
        )


class FlowsList:
    """Represents the configuration of a list of flow.

    We need this class to be able to fingerprint the flows configuration.
    Fingerprinting is needed to make sure that the model is retrained if the
    flows configuration changes.
    """

    def __init__(self, flows: List[Flow]) -> None:
        """Initializes the configuration of flows.

        Args:
            flows: The flows to be configured.
        """
        self.underlying_flows = flows

    def is_empty(self) -> bool:
        """Returns whether the flows list is empty."""
        return len(self.underlying_flows) == 0

    @classmethod
    def from_json(
        cls, flows_configs: Optional[Dict[Text, Dict[Text, Any]]]
    ) -> FlowsList:
        """Used to read flows from parsed YAML.

        Args:
            flows_configs: The parsed YAML as a dictionary.

        Returns:
            The parsed flows.
        """
        if not flows_configs:
            return cls([])

        return cls(
            [
                Flow.from_json(flow_id, flow_config)
                for flow_id, flow_config in flows_configs.items()
            ]
        )

    def as_json(self) -> List[Dict[Text, Any]]:
        """Returns the flows as a dictionary.

        Returns:
            The flows as a dictionary.
        """
        return [flow.as_json() for flow in self.underlying_flows]

    def fingerprint(self) -> str:
        """Creates a fingerprint of the flows configuration.

        Returns:
            The fingerprint of the flows configuration.
        """
        flow_dicts = [flow.as_json() for flow in self.underlying_flows]
        return rasa.shared.utils.io.get_list_fingerprint(flow_dicts)

    def merge(self, other: FlowsList) -> FlowsList:
        """Merges two lists of flows together."""
        return FlowsList(self.underlying_flows + other.underlying_flows)

    def flow_by_id(self, id: Optional[Text]) -> Optional[Flow]:
        """Return the flow with the given id."""
        if not id:
            return None

        for flow in self.underlying_flows:
            if flow.id == id:
                return flow
        else:
            return None

    def step_by_id(self, step_id: Text, flow_id: Text) -> FlowStep:
        """Return the step with the given id."""
        flow = self.flow_by_id(flow_id)
        if not flow:
            raise UnresolvedFlowException(flow_id)

        step = flow.step_by_id(step_id)
        if not step:
            raise UnresolvedFlowStepIdException(step_id, flow, referenced_from=None)

        return step

    def validate(self) -> None:
        """Validate the flows."""
        for flow in self.underlying_flows:
            flow.validate()


@dataclass
class Flow:
    """Represents the configuration of a flow."""

    id: Text
    """The id of the flow."""
    name: Text
    """The name of the flow."""
    description: Optional[Text]
    """The description of the flow."""
    steps: List[FlowStep]
    """The steps of the flow."""

    @staticmethod
    def from_json(flow_id: Text, flow_config: Dict[Text, Any]) -> Flow:
        """Used to read flows from parsed YAML.

        Args:
            flow_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow.
        """
        return Flow(
            id=flow_id,
            name=flow_config.get("name", ""),
            description=flow_config.get("description"),
            steps=[
                step_from_json(step_config)
                for step_config in flow_config.get("steps", [])
            ],
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow as a dictionary.

        Returns:
            The flow as a dictionary.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [step.as_json() for step in self.steps],
        }

    def validate(self) -> None:
        """Validates the flow configuration.

        This ensures that the flow semantically makes sense. E.g. it
        checks:
            - whether all next links point to existing steps
            - whether all steps can be reached from the start step
        """
        self._validate_all_next_ids_are_availble_steps()
        self._validate_all_steps_can_be_reached()

    def _validate_all_next_ids_are_availble_steps(self) -> None:
        """Validates that all next links point to existing steps."""
        available_steps = {step.id for step in self.steps}
        for step in self.steps:
            for link in step.next.links:
                if link.target not in available_steps:
                    raise UnresolvedFlowStepIdException(link.target, self, step)

    def _validate_all_steps_can_be_reached(self) -> None:
        """Validates that all steps can be reached from the start step."""

        def _reachable_steps(
            step: Optional[FlowStep], reached_steps: Set[Text]
        ) -> Set[Text]:
            """Validates that the given step can be reached from the start step."""
            if step is None or step.id in reached_steps:
                return reached_steps

            reached_steps.add(step.id)
            for link in step.next.links:
                reached_steps = _reachable_steps(
                    self.step_by_id(link.target), reached_steps
                )
            return reached_steps

        reached_steps = _reachable_steps(self.first_step_in_flow(), set())

        for step in self.steps:
            if step.id not in reached_steps:
                raise UnreachableFlowStepException(step, self)

    def step_by_id(self, step_id: Optional[Text]) -> Optional[FlowStep]:
        """Returns the step with the given id."""
        if not step_id:
            return None

        if step_id == START_STEP:
            first_step_in_flow = self.first_step_in_flow()
            return StartFlowStep(first_step_in_flow.id if first_step_in_flow else None)

        if step_id == END_STEP:
            return EndFlowStep()

        for step in self.steps:
            if step.id == step_id:
                return step

        return None

    def first_step_in_flow(self) -> Optional[FlowStep]:
        """Returns the start step of this flow."""
        if len(self.steps) == 0:
            return None
        return self.steps[0]

    def previously_asked_questions(self, step_id: Text) -> List[QuestionFlowStep]:
        """Returns the questions asked before the given step.

        Questions are returned roughly in reverse order, i.e. the first
        question in the list is the one asked last. But due to circles
        in the flow the order is not guaranteed to be exactly reverse.
        """

        def _previously_asked_questions(
            current_step_id: Text, visited_steps: Set[Text]
        ) -> List[QuestionFlowStep]:
            """Returns the questions asked before the given step.

            Keeps track of the steps that have been visited to avoid circles.
            """
            current_step = self.step_by_id(current_step_id)

            questions: List[QuestionFlowStep] = []

            if not current_step:
                return questions

            if isinstance(current_step, QuestionFlowStep):
                questions.append(current_step)

            visited_steps.add(current_step.id)

            for previous_step in self.steps:
                for next_link in previous_step.next.links:
                    if next_link.target != current_step_id:
                        continue
                    if previous_step.id in visited_steps:
                        continue
                    questions.extend(
                        _previously_asked_questions(previous_step.id, visited_steps)
                    )
            return questions

        return _previously_asked_questions(step_id, set())

    def is_handling_pattern(self) -> bool:
        """Returns whether the flow is handling a pattern."""
        return self.id.startswith(HANDLING_PATTERN_PREFIX)

    def get_trigger_intents(self) -> Set[str]:
        """Returns the trigger intents of the flow"""
        results: Set[str] = set()
        if len(self.steps) == 0:
            return results

        first_step = self.steps[0]

        if not isinstance(first_step, UserMessageStep):
            return results

        for condition in first_step.trigger_conditions:
            results.add(condition.intent)

        return results

    def is_user_triggerable(self) -> bool:
        """Test whether a user can trigger the flow with an intent."""
        return len(self.get_trigger_intents()) > 0

    def is_rasa_default_flow(self) -> bool:
        """Test whether something is a rasa default flow."""
        trigger_intents = self.get_trigger_intents()
        any_has_rasa_prefix = any(
            [
                intent.startswith(RASA_DEFAULT_INTENT_PREFIX)
                for intent in trigger_intents
            ]
        )
        return any_has_rasa_prefix

    def get_question_steps(self) -> List[QuestionFlowStep]:
        """Return the question steps of the flow."""
        question_steps = []
        for step in self.steps:
            if isinstance(step, QuestionFlowStep):
                question_steps.append(step)
        return question_steps

    def get_slots(self) -> Set[str]:
        """The slots used in this flow."""
        return {q.question for q in self.get_question_steps()}


def step_from_json(flow_step_config: Dict[Text, Any]) -> FlowStep:
    """Used to read flow steps from parsed YAML.

    Args:
        flow_step_config: The parsed YAML as a dictionary.

    Returns:
        The parsed flow step.
    """
    if "action" in flow_step_config:
        return ActionFlowStep.from_json(flow_step_config)
    if "intent" in flow_step_config:
        return UserMessageStep.from_json(flow_step_config)
    if "question" in flow_step_config:
        return QuestionFlowStep.from_json(flow_step_config)
    if "link" in flow_step_config:
        return LinkFlowStep.from_json(flow_step_config)
    if "set_slots" in flow_step_config:
        return SetSlotsFlowStep.from_json(flow_step_config)
    if "entry_prompt" in flow_step_config:
        return EntryPromptFlowStep.from_json(flow_step_config)
    if "generation_prompt" in flow_step_config:
        return GenerateResponseFlowStep.from_json(flow_step_config)
    else:
        raise ValueError(f"Flow step is missing a type. {flow_step_config}")


@dataclass
class FlowStep:
    """Represents the configuration of a flow step."""

    id: Text
    """The id of the flow step."""
    description: Optional[Text]
    """The description of the flow step."""
    metadata: Dict[Text, Any]
    """Additional, unstructured information about this flow step."""
    next: "FlowLinks"
    """The next steps of the flow step."""

    @classmethod
    def _from_json(cls, flow_step_config: Dict[Text, Any]) -> FlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        return FlowStep(
            id=flow_step_config["id"],
            description=flow_step_config.get("description"),
            metadata=flow_step_config.get("metadata", {}),
            next=FlowLinks.from_json(flow_step_config.get("next", [])),
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = {
            "id": self.id,
            "next": self.next.as_json(),
        }
        if self.description:
            dump["description"] = self.description
        if self.metadata:
            dump["metadata"] = self.metadata
        return dump


START_STEP = "__start__"


@dataclass
class StartFlowStep(FlowStep):
    """Represents the configuration of a start flow step."""

    def __init__(self, start_step_id: Optional[Text]) -> None:
        """Initializes a start flow step.

        Args:
            start_step: The step to start the flow from.
        """
        if start_step_id is not None:
            links: List[FlowLink] = [StaticFlowLink(target=start_step_id)]
        else:
            links = []

        super().__init__(
            id=START_STEP,
            description=None,
            metadata={},
            next=FlowLinks(links=links),
        )

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> ActionFlowStep:
        """Used to read flow steps from parsed JSON.

        Args:
            flow_step_config: The parsed JSON as a dictionary.

        Returns:
            The parsed flow step.
        """
        raise ValueError("A start step cannot be parsed.")

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        raise ValueError("A start step cannot be dumped.")


END_STEP = "__end__"


@dataclass
class EndFlowStep(FlowStep):
    """Represents the configuration of an end to a flow."""

    def __init__(self) -> None:
        """Initializes a start flow step.

        Args:
            start_step: The step to start the flow from.
        """
        super().__init__(
            id=END_STEP,
            description=None,
            metadata={},
            next=FlowLinks(links=[]),
        )

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> ActionFlowStep:
        """Used to read flow steps from parsed JSON.

        Args:
            flow_step_config: The parsed JSON as a dictionary.

        Returns:
            The parsed flow step.
        """
        raise ValueError("An end step cannot be parsed.")

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        raise ValueError("An end step cannot be dumped.")


@dataclass
class ActionFlowStep(FlowStep):
    """Represents the configuration of an action flow step."""

    action: Text
    """The action of the flow step."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> ActionFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        return ActionFlowStep(
            action=flow_step_config.get("action", ""),
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        dump["action"] = self.action
        return dump


@dataclass
class LinkFlowStep(FlowStep):
    """Represents the configuration of a link flow step."""

    link: Text
    """The link of the flow step."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> LinkFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        return LinkFlowStep(
            link=flow_step_config.get("link", ""),
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        dump["link"] = self.link
        return dump


@dataclass
class TriggerCondition:
    """Represents the configuration of a trigger condition."""

    intent: Text
    """The intent to trigger the flow."""
    entities: List[Text]
    """The entities to trigger the flow."""

    def is_triggered(self, intent: Text, entities: List[Text]) -> bool:
        """Check if condition is triggered by the given intent and entities.

        Args:
            intent: The intent to check.
            entities: The entities to check.

        Returns:
            Whether the trigger condition is triggered by the given intent and entities.
        """
        if self.intent != intent:
            return False
        if len(self.entities) == 0:
            return True
        return all(entity in entities for entity in self.entities)


@runtime_checkable
class StepThatCanStartAFlow(Protocol):
    """Represents a step that can start a flow."""

    def is_triggered(self, tracker: DialogueStateTracker) -> bool:
        """Check if a flow should be started for the tracker

        Args:
            tracker: The tracker to check.

        Returns:
            Whether a flow should be started for the tracker.
        """
        ...


@dataclass
class UserMessageStep(FlowStep, StepThatCanStartAFlow):
    """Represents the configuration of an intent flow step."""

    trigger_conditions: List[TriggerCondition]
    """The trigger conditions of the flow step."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> UserMessageStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)

        trigger_conditions = []
        if "intent" in flow_step_config:
            trigger_conditions.append(
                TriggerCondition(
                    intent=flow_step_config["intent"],
                    entities=flow_step_config.get("entities", []),
                )
            )
        elif "or" in flow_step_config:
            for trigger_condition in flow_step_config["or"]:
                trigger_conditions.append(
                    TriggerCondition(
                        intent=trigger_condition.get("intent", ""),
                        entities=trigger_condition.get("entities", []),
                    )
                )

        return UserMessageStep(
            trigger_conditions=trigger_conditions,
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()

        if len(self.trigger_conditions) == 1:
            dump["intent"] = self.trigger_conditions[0].intent
            if self.trigger_conditions[0].entities:
                dump["entities"] = self.trigger_conditions[0].entities
        elif len(self.trigger_conditions) > 1:
            dump["or"] = [
                {
                    "intent": trigger_condition.intent,
                    "entities": trigger_condition.entities,
                }
                for trigger_condition in self.trigger_conditions
            ]

        return dump

    def is_triggered(self, tracker: DialogueStateTracker) -> bool:
        """Returns whether the flow step is triggered by the given intent and entities.

        Args:
            intent: The intent to check.
            entities: The entities to check.

        Returns:
            Whether the flow step is triggered by the given intent and entities.
        """
        if not tracker.latest_message:
            return False

        intent: Text = tracker.latest_message.intent.get(INTENT_NAME_KEY, "")
        entities: List[Text] = [
            e.get(ENTITY_ATTRIBUTE_TYPE, "") for e in tracker.latest_message.entities
        ]
        return any(
            trigger_condition.is_triggered(intent, entities)
            for trigger_condition in self.trigger_conditions
        )


DEFAULT_LLM_CONFIG = {
    "_type": "openai",
    "request_timeout": 5,
    "temperature": DEFAULT_OPENAI_TEMPERATURE,
    "model_name": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
}


@dataclass
class GenerateResponseFlowStep(FlowStep):
    """Represents the configuration of a step prompting an LLM."""

    generation_prompt: Text
    """The prompt template of the flow step."""
    llm_config: Optional[Dict[Text, Any]] = None
    """The LLM configuration of the flow step."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> GenerateResponseFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        return GenerateResponseFlowStep(
            generation_prompt=flow_step_config.get("generation_prompt", ""),
            llm_config=flow_step_config.get("llm", None),
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        dump["generation_prompt"] = self.generation_prompt
        if self.llm_config:
            dump["llm"] = self.llm_config

        return dump

    def generate(self, tracker: DialogueStateTracker) -> Optional[Text]:
        """Generates a response for the given tracker.

        Args:
            tracker: The tracker to generate a response for.

        Returns:
            The generated response.
        """
        from rasa.utils.llm import llm_factory, tracker_as_readable_transcript
        from jinja2 import Template

        context = {
            "history": tracker_as_readable_transcript(tracker, max_turns=5),
            "latest_user_message": tracker.latest_message.text
            if tracker.latest_message
            else "",
        }
        context.update(tracker.current_slot_values())

        llm = llm_factory(self.llm_config, DEFAULT_LLM_CONFIG)
        prompt = Template(self.generation_prompt).render(context)

        try:
            return llm(prompt)
        except Exception as e:
            # unfortunately, langchain does not wrap LLM exceptions which means
            # we have to catch all exceptions here
            structlogger.error(
                "flow.generate_step.llm.error", error=e, step=self.id, prompt=prompt
            )
            return None


@dataclass
class EntryPromptFlowStep(FlowStep, StepThatCanStartAFlow):
    """Represents the configuration of a step prompting an LLM."""

    entry_prompt: Text
    """The prompt template of the flow step."""
    advance_if: Optional[Text]
    """The expected response to start the flow"""
    llm_config: Optional[Dict[Text, Any]] = None
    """The LLM configuration of the flow step."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> EntryPromptFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        return EntryPromptFlowStep(
            entry_prompt=flow_step_config.get("entry_prompt", ""),
            advance_if=flow_step_config.get("advance_if"),
            llm_config=flow_step_config.get("llm", None),
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        dump["entry_prompt"] = self.entry_prompt
        if self.advance_if:
            dump["advance_if"] = self.advance_if

        if self.llm_config:
            dump["llm"] = self.llm_config
        return dump

    def _generate_using_llm(self, prompt: str) -> Optional[str]:
        """Use LLM to generate a response.

        Args:
            prompt: the prompt to send to the LLM

        Returns:
            generated text
        """
        from rasa.utils.llm import llm_factory

        llm = llm_factory(self.llm_config, DEFAULT_LLM_CONFIG)

        try:
            return llm(prompt)
        except Exception as e:
            # unfortunately, langchain does not wrap LLM exceptions which means
            # we have to catch all exceptions here
            structlogger.error(
                "flow.entry_step.llm.error", error=e, step=self.id, prompt=prompt
            )
            return None

    def is_triggered(self, tracker: DialogueStateTracker) -> bool:
        """Returns whether the flow step is triggered by the given intent and entities.

        Args:
            intent: The intent to check.
            entities: The entities to check.

        Returns:
            Whether the flow step is triggered by the given intent and entities.
        """
        from rasa.utils import llm
        from jinja2 import Template

        if not self.entry_prompt:
            return False

        context = {
            "history": llm.tracker_as_readable_transcript(tracker, max_turns=5),
            "latest_user_message": tracker.latest_message.text
            if tracker.latest_message
            else "",
        }
        context.update(tracker.current_slot_values())
        prompt = Template(self.entry_prompt).render(context)

        generated = self._generate_using_llm(prompt)

        expected_response = self.advance_if.lower() if self.advance_if else "yes"
        if generated and generated.lower() == expected_response:
            return True
        else:
            return False


# enumeration of question scopes. scope can either be flow or global
class QuestionScope(str, Enum):
    FLOW = "flow"
    GLOBAL = "global"

    @staticmethod
    def from_str(label: Optional[Text]) -> "QuestionScope":
        """Converts a string to a QuestionScope."""
        if label is None:
            return QuestionScope.FLOW
        elif label.lower() == "flow":
            return QuestionScope.FLOW
        elif label.lower() == "global":
            return QuestionScope.GLOBAL
        else:
            raise NotImplementedError


@dataclass
class QuestionFlowStep(FlowStep):
    """Represents the configuration of a question flow step."""

    question: Text
    """The question of the flow step."""
    skip_if_filled: bool = False
    """Whether to skip the question if the slot is already filled."""
    scope: QuestionScope = QuestionScope.FLOW
    """how the question is scoped, determins when to reset its value."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> QuestionFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        return QuestionFlowStep(
            question=flow_step_config.get("question", ""),
            skip_if_filled=flow_step_config.get("skip_if_filled", False),
            scope=QuestionScope.from_str(flow_step_config.get("scope")),
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        dump["question"] = self.question
        dump["skip_if_filled"] = self.skip_if_filled
        dump["scope"] = self.scope.value

        return dump


@dataclass
class SetSlotsFlowStep(FlowStep):
    """Represents the configuration of a set_slots flow step."""

    slots: List[Dict[str, Any]]
    """Slots to set of the flow step."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> SetSlotsFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        slots = [
            {"key": k, "value": v}
            for slot in flow_step_config.get("set_slots", [])
            for k, v in slot.items()
        ]
        return SetSlotsFlowStep(
            slots=slots,
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        dump["set_slots"] = [{slot["key"]: slot["value"]} for slot in self.slots]
        return dump


@dataclass
class FlowLinks:
    """Represents the configuration of a list of flow links."""

    links: List[FlowLink]

    @staticmethod
    def from_json(flow_links_config: List[Dict[Text, Any]]) -> FlowLinks:
        """Used to read flow links from parsed YAML.

        Args:
            flow_links_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow links.
        """
        if not flow_links_config:
            return FlowLinks(links=[])

        if isinstance(flow_links_config, str):
            return FlowLinks(links=[StaticFlowLink.from_json(flow_links_config)])

        return FlowLinks(
            links=[
                FlowLinks.link_from_json(link_config)
                for link_config in flow_links_config
                if link_config
            ]
        )

    @staticmethod
    def link_from_json(link_config: Dict[Text, Any]) -> FlowLink:
        """Used to read a single flow links from parsed YAML.

        Args:
            link_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow link.
        """
        if "if" in link_config:
            return IfFlowLink.from_json(link_config)
        elif "else" in link_config:
            return ElseFlowLink.from_json(link_config)
        else:
            raise Exception("Invalid flow link")

    def as_json(self) -> Any:
        """Returns the flow links as a dictionary.

        Returns:
            The flow links as a dictionary.
        """
        if not self.links:
            return None

        if len(self.links) == 1 and isinstance(self.links[0], StaticFlowLink):
            return self.links[0].as_json()

        return [link.as_json() for link in self.links]


class FlowLink(Protocol):
    """Represents a flow link."""

    @property
    def target(self) -> Text:
        """Returns the target of the flow link.

        Returns:
            The target of the flow link.
        """
        ...

    def as_json(self) -> Any:
        """Returns the flow link as a dictionary.

        Returns:
            The flow link as a dictionary.
        """
        ...

    @staticmethod
    def from_json(link_config: Any) -> FlowLink:
        """Used to read flow links from parsed YAML.

        Args:
            link_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow link.
        """
        ...


@dataclass
class IfFlowLink:
    """Represents the configuration of an if flow link."""

    target: Text
    """The id of the linked flow."""
    condition: Optional[Text]
    """The condition of the linked flow."""

    @staticmethod
    def from_json(link_config: Dict[Text, Any]) -> IfFlowLink:
        """Used to read flow links from parsed YAML.

        Args:
            link_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow link.
        """
        return IfFlowLink(target=link_config["then"], condition=link_config.get("if"))

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow link as a dictionary.

        Returns:
            The flow link as a dictionary.
        """
        return {
            "if": self.condition,
            "then": self.target,
        }


@dataclass
class ElseFlowLink:
    """Represents the configuration of an else flow link."""

    target: Text
    """The id of the linked flow."""

    @staticmethod
    def from_json(link_config: Dict[Text, Any]) -> ElseFlowLink:
        """Used to read flow links from parsed YAML.

        Args:
            link_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow link.
        """
        return ElseFlowLink(target=link_config["else"])

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow link as a dictionary.

        Returns:
            The flow link as a dictionary.
        """
        return {"else": self.target}


@dataclass
class StaticFlowLink:
    """Represents the configuration of a static flow link."""

    target: Text
    """The id of the linked flow."""

    @staticmethod
    def from_json(link_config: Text) -> StaticFlowLink:
        """Used to read flow links from parsed YAML.

        Args:
            link_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow link.
        """
        return StaticFlowLink(target=link_config)

    def as_json(self) -> Text:
        """Returns the flow link as a dictionary.

        Returns:
            The flow link as a dictionary.
        """
        return self.target
