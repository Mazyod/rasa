import importlib.resources
import re
from typing import Dict, Any, Optional, List, Union

from jinja2 import Template
import structlog
from rasa.dialogue_understanding.stack.utils import top_flow_frame
from rasa.dialogue_understanding.generator import CommandGenerator
from rasa.dialogue_understanding.commands import (
    Command,
    ErrorCommand,
    SetSlotCommand,
    CancelFlowCommand,
    StartFlowCommand,
    HumanHandoffCommand,
    ChitChatAnswerCommand,
    KnowledgeAnswerCommand,
    ClarifyCommand,
)
from rasa.core.policies.flow_policy import DialogueStack
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.flows.flow import FlowStep, FlowsList, CollectInformationFlowStep
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.slots import (
    BooleanSlot,
    CategoricalSlot,
    FloatSlot,
    Slot,
    bool_from_any,
)
from rasa.shared.nlu.constants import (
    TEXT,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils.llm import (
    DEFAULT_OPENAI_CHAT_MODEL_NAME_ADVANCED,
    llm_factory,
    tracker_as_readable_transcript,
    sanitize_message_for_prompt,
)

import os
import json
import tiktoken
from langchain.callbacks import get_openai_callback
from rasa.constants import BENCHMARKS_DIR_PATH

DEFAULT_COMMAND_PROMPT_TEMPLATE = importlib.resources.read_text(
    "rasa.dialogue_understanding.generator", "command_prompt_template.jinja2"
)

structlogger = structlog.get_logger()


DEFAULT_LLM_CONFIG = {
    "_type": "openai",
    "request_timeout": 7,
    "temperature": 0.0,
    "model_name": DEFAULT_OPENAI_CHAT_MODEL_NAME_ADVANCED,
}

LLM_CONFIG_KEY = "llm"


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COMMAND_GENERATOR,
    ],
    is_trainable=True,
)
class LLMCommandGenerator(GraphComponent, CommandGenerator):
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            "prompt": DEFAULT_COMMAND_PROMPT_TEMPLATE,
            LLM_CONFIG_KEY: None,
        }

    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        self.config = {**self.get_default_config(), **config}
        self.prompt_template = self.config["prompt"]
        self._model_storage = model_storage
        self._resource = resource

    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "LLMCommandGenerator":
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    def persist(self) -> None:
        pass

    @classmethod
    def load(
        cls,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> "LLMCommandGenerator":
        """Loads trained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    def train(self, training_data: TrainingData) -> Resource:
        """Train the intent classifier on a data set."""
        self.persist()
        return self._resource

    def _generate_action_list_using_llm(self, prompt: str) -> Optional[str]:
        """Use LLM to generate a response.

        Args:
            prompt: the prompt to send to the LLM

        Returns:
            generated text
        """
        llm = llm_factory(self.config.get(LLM_CONFIG_KEY), DEFAULT_LLM_CONFIG)

        try:

            with get_openai_callback() as cb:
                result = llm(prompt)
                if cb.total_tokens != 0:
                    costs = {
                        'total_tokens': cb.total_tokens,
                        'total_cost': cb.total_cost,

                        'prompt_tokens': cb.prompt_tokens,
                        'prompt_cost': cb.prompt_tokens / cb.total_tokens * cb.total_cost,

                        'completion_tokens': cb.total_tokens - cb.prompt_tokens,
                        'completion_cost': (cb.total_tokens - cb.prompt_tokens) / cb.total_tokens * cb.total_cost,

                        'successful_requests': cb.successful_requests
                    }
                else:
                    costs = {
                        'total_tokens': 0,
                        'total_cost': 0,
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'prompt_cost': 0,
                        'completion_cost': 0,
                        'successful_requests': 0
                    }
                structlogger.info("llm_command_generator.llm.prompt", **costs)
                return result, costs
        except Exception as e:
            # unfortunately, langchain does not wrap LLM exceptions which means
            # we have to catch all exceptions here
            structlogger.error("llm_command_generator.llm.error", error=e)
            return None

    def predict_commands(
        self,
        message: Message,
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
    ) -> List[Command]:
        if tracker is None or flows.is_empty():
            # cannot do anything if there are no flows or no tracker
            return []
        # FINDING: This is where prompts are rendered. Try to extract the size of history, slots, actions, etc.
        flow_prompt, template_details = self.render_template(message, tracker, flows)
        structlogger.info(
            "llm_command_generator.predict_commands.prompt_rendered", prompt=flow_prompt
        )
        action_list, cost = self._generate_action_list_using_llm(flow_prompt)
        structlogger.info(
            "llm_command_generator.predict_commands.actions_generated",
            action_list=action_list,
        )
        commands = self.parse_commands(action_list, tracker)
        structlogger.info(
            "llm_command_generator.predict_commands.finished",
            commands=commands,
        )

        meta = {
            'what': 'llm_command_generator.LLMCommandGenerator.predict_commands',
            'message': message.data['text'],
            'prompt': flow_prompt,
            'completion': action_list,
        }
        meta.update(cost)
        meta.update(template_details)

        # FINDING: Maybe here just open the file and store everything?
        file_name = f"{message.data['metadata']['test_case']['name']}.jsonl"

        with open(os.path.join(BENCHMARKS_DIR_PATH, file_name), 'a') as outfile:
            outfile.write(f"{json.dumps(meta)}\n")

        return commands

    @staticmethod
    def is_none_value(value: str) -> bool:
        return value in {
            "[missing information]",
            "[missing]",
            "None",
            "undefined",
            "null",
        }

    @staticmethod
    def clean_extracted_value(value: str) -> str:
        """Clean up the extracted value from the llm."""
        # replace any combination of single quotes, double quotes, and spaces
        # from the beginning and end of the string
        return re.sub(r"^['\"\s]+|['\"\s]+$", "", value)

    @classmethod
    def coerce_slot_value(
        cls, slot_value: str, slot_name: str, tracker: DialogueStateTracker
    ) -> Union[str, bool, float, None]:
        """Coerce the slot value to the correct type.

        Tries to coerce the slot value to the correct type. If the
        conversion fails, `None` is returned.

        Args:
            value: the value to coerce
            slot_name: the name of the slot
            tracker: the tracker

        Returns:
            the coerced value or `None` if the conversion failed."""
        nullable_value = slot_value if not cls.is_none_value(slot_value) else None
        if slot_name not in tracker.slots:
            return nullable_value

        slot = tracker.slots[slot_name]
        if isinstance(slot, BooleanSlot):
            try:
                return bool_from_any(nullable_value)
            except (ValueError, TypeError):
                return None
        elif isinstance(slot, FloatSlot):
            try:
                return float(nullable_value)
            except (ValueError, TypeError):
                return None
        else:
            return nullable_value

    @classmethod
    def parse_commands(
        cls, actions: Optional[str], tracker: DialogueStateTracker
    ) -> List[Command]:
        """Parse the actions returned by the llm into intent and entities."""
        if not actions:
            return [ErrorCommand()]

        commands: List[Command] = []

        slot_set_re = re.compile(
            r"""SetSlot\(([a-zA-Z_][a-zA-Z0-9_-]*?), ?\"?([^)]*?)\"?\)"""
        )
        start_flow_re = re.compile(r"StartFlow\(([a-zA-Z_][a-zA-Z0-9_-]*?)\)")
        cancel_flow_re = re.compile(r"CancelFlow\(\)")
        chitchat_re = re.compile(r"ChitChat\(\)")
        knowledge_re = re.compile(r"SearchAndReply\(\)")
        humand_handoff_re = re.compile(r"HumandHandoff\(\)")
        clarify_re = re.compile(r"Clarify\(([a-zA-Z0-9_, ]+)\)")

        for action in actions.strip().splitlines():
            if m := slot_set_re.search(action):
                slot_name = m.group(1).strip()
                slot_value = cls.clean_extracted_value(m.group(2))
                # error case where the llm tries to start a flow using a slot set
                if slot_name == "flow_name":
                    commands.append(StartFlowCommand(flow=slot_value))
                else:
                    typed_slot_value = cls.coerce_slot_value(
                        slot_value, slot_name, tracker
                    )
                    commands.append(
                        SetSlotCommand(name=slot_name, value=typed_slot_value)
                    )
            elif m := start_flow_re.search(action):
                commands.append(StartFlowCommand(flow=m.group(1).strip()))
            elif cancel_flow_re.search(action):
                commands.append(CancelFlowCommand())
            elif chitchat_re.search(action):
                commands.append(ChitChatAnswerCommand())
            elif knowledge_re.search(action):
                commands.append(KnowledgeAnswerCommand())
            elif humand_handoff_re.search(action):
                commands.append(HumanHandoffCommand())
            elif m := clarify_re.search(action):
                options = [opt.strip() for opt in m.group(1).split(",")]
                commands.append(ClarifyCommand(options))

        return commands

    @classmethod
    def create_template_inputs(
        cls, flows: FlowsList, tracker: DialogueStateTracker
    ) -> List[Dict[str, Any]]:
        result = []
        for flow in flows.underlying_flows:
            # TODO: check if we should filter more flows; e.g. flows that are
            #  linked to by other flows and that shouldn't be started directly.
            #  we might need a separate flag for that.
            if not flow.is_rasa_default_flow():

                slots_with_info = [
                    {"name": q.collect, "description": q.description}
                    for q in flow.get_collect_steps()
                    if cls.is_extractable(q, tracker)
                ]
                result.append(
                    {
                        "name": flow.id,
                        "description": flow.description,
                        "slots": slots_with_info,
                    }
                )
        return result

    @staticmethod
    def is_extractable(
        q: CollectInformationFlowStep,
        tracker: DialogueStateTracker,
        current_step: Optional[FlowStep] = None,
    ) -> bool:
        """Check if the `collect` can be filled.

        A collect slot can only be filled if the slot exist
        and either the collect has been asked already or the
        slot has been filled already."""
        slot = tracker.slots.get(q.collect)
        if slot is None:
            return False

        return (
            # we can fill because this is a slot that can be filled ahead of time
            not q.ask_before_filling
            # we can fill because the slot has been filled already
            or slot.has_been_set
            # we can fill because the is currently getting asked
            or (
                current_step is not None
                and isinstance(current_step, CollectInformationFlowStep)
                and current_step.collect == q.collect
            )
        )

    def allowed_values_for_slot(self, slot: Slot) -> Optional[str]:
        """Get the allowed values for a slot."""
        if isinstance(slot, BooleanSlot):
            return str([True, False])
        if isinstance(slot, CategoricalSlot):
            return str([v for v in slot.values if v != "__other__"])
        else:
            return None

    @staticmethod
    def slot_value(tracker: DialogueStateTracker, slot_name: str) -> str:
        """Get the slot value from the tracker."""
        slot_value = tracker.get_slot(slot_name)
        if slot_value is None:
            return "undefined"
        else:
            return str(slot_value)

    def render_template(
        self, message: Message, tracker: DialogueStateTracker, flows: FlowsList
    ) -> str:
        flows_without_patterns = FlowsList(
            [f for f in flows.underlying_flows if not f.is_handling_pattern()]
        )
        top_relevant_frame = top_flow_frame(DialogueStack.from_tracker(tracker))
        top_flow = top_relevant_frame.flow(flows) if top_relevant_frame else None
        current_step = top_relevant_frame.step(flows) if top_relevant_frame else None
        if top_flow is not None:
            flow_slots = [
                {
                    "name": q.collect,
                    "value": self.slot_value(tracker, q.collect),
                    "type": tracker.slots[q.collect].type_name,
                    "allowed_values": self.allowed_values_for_slot(
                        tracker.slots[q.collect]
                    ),
                    "description": q.description,
                }
                for q in top_flow.get_collect_steps()
                if self.is_extractable(q, tracker, current_step)
            ]
        else:
            flow_slots = []

        collect, collect_description = (
            (current_step.collect, current_step.description)
            if isinstance(current_step, CollectInformationFlowStep)
            else (None, None)
        )
        current_conversation = tracker_as_readable_transcript(tracker)
        latest_user_message = sanitize_message_for_prompt(message.get(TEXT))
        current_conversation += f"\nUSER: {latest_user_message}"

        inputs = {
            "available_flows": self.create_template_inputs(
                flows_without_patterns, tracker
            ),
            "current_conversation": current_conversation,
            "flow_slots": flow_slots,
            "current_flow": top_flow.id if top_flow is not None else None,
            "collect": collect,
            "collect_description": collect_description,
            "user_message": latest_user_message,
        }

        # FINDING: Here additional info is returned
        encoding = tiktoken.encoding_for_model(self.config["llm"]["model_name"])

        flow_part_template = """
            {% for flow in available_flows %}
            {{ flow.name }}: {{ flow.description }}
                {% for slot in flow.slots -%}
                slot: {{ slot.name }}{% if slot.description %} ({{ slot.description }}){% endif %}
                {% endfor %}
            {%- endfor %}
        """
        active_slots_part_template = """
            {% if current_flow != None %}
            You are currently in the flow "{{ current_flow }}", which {{ current_flow.description }}
            You have just asked the user for the slot "{{ collect }}"{% if collect_description %} ({{ collect_description }}){% endif %}.
            
            {% if flow_slots|length > 0 %}
            Here are the slots of the currently active flow:
            {% for slot in flow_slots -%}
            - name: {{ slot.name }}, value: {{ slot.value }}, type: {{ slot.type }}, description: {{ slot.description}}{% if slot.allowed_values %}, allowed values: {{ slot.allowed_values }}{% endif %}
            {% endfor %}
            {% endif %}
            {% else %}
            You are currently not in any flow and so there are no active slots.
            This means you can only set a slot if you first start a flow that requires that slot.
            {% endif %}
            If you start a flow, first start the flow and then optionally fill that flow's slots with information the user provided in their message.
        """

        flow_part = Template(flow_part_template).render(**inputs)
        flow_part_tokens = len(encoding.encode(flow_part))

        active_slots_part = Template(active_slots_part_template).render(**inputs)
        active_slots_part_tokens = len(encoding.encode(active_slots_part))

        current_conversation_part_tokens =  len(encoding.encode(current_conversation))
        num_conversation_turns = len(list(filter(lambda t: len(t.strip()) > 1, current_conversation.split('\n'))))

        user_message_tokens = len(encoding.encode(latest_user_message))

        meta = {
            "prompt_flow": flow_part,
            "prompt_flow_tokens": flow_part_tokens,
            "num_available_flows": len(inputs["available_flows"]),

            "prompt_active_slots": active_slots_part,
            "prompt_active_slots_tokens": active_slots_part_tokens,
            "num_active_slots": len(inputs["flow_slots"]),

            "prompt_current_conversation": current_conversation,
            "prompt_current_conversation_tokens": current_conversation_part_tokens,
            "num_conversation_turns": num_conversation_turns,

            "user_message_tokens": user_message_tokens,
            "user_message_words": len(latest_user_message.strip().split())
        }
        return Template(self.prompt_template).render(**inputs), meta
