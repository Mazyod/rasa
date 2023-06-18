from __future__ import annotations
from enum import Enum

import rasa.shared.constants as constants


DEFAULT_CATEGORICAL_SLOT_VALUE = "__other__"

USER_INTENT_RESTART = "restart"
USER_INTENT_BACK = "back"
USER_INTENT_OUT_OF_SCOPE = "out_of_scope"
USER_INTENT_SESSION_START = "session_start"
SESSION_START_METADATA_SLOT = "session_started_metadata"

DEFAULT_INTENTS = [
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    USER_INTENT_OUT_OF_SCOPE,
    USER_INTENT_SESSION_START,
    constants.DEFAULT_NLU_FALLBACK_INTENT_NAME,
]

LOOP_NAME = "name"

ACTION_LISTEN_NAME = "action_listen"
ACTION_RESTART_NAME = "action_restart"
ACTION_SEND_TEXT = "action_send_text"
ACTION_SESSION_START_NAME = "action_session_start"
ACTION_DEFAULT_FALLBACK_NAME = "action_default_fallback"
ACTION_DEACTIVATE_LOOP_NAME = "action_deactivate_loop"
ACTION_REVERT_FALLBACK_EVENTS_NAME = "action_revert_fallback_events"
ACTION_DEFAULT_ASK_AFFIRMATION_NAME = "action_default_ask_affirmation"
ACTION_DEFAULT_ASK_REPHRASE_NAME = "action_default_ask_rephrase"
ACTION_BACK_NAME = "action_back"
ACTION_TWO_STAGE_FALLBACK_NAME = "action_two_stage_fallback"
ACTION_UNLIKELY_INTENT_NAME = "action_unlikely_intent"
RULE_SNIPPET_ACTION_NAME = "..."
ACTION_EXTRACT_SLOTS = "action_extract_slots"
ACTION_VALIDATE_SLOT_MAPPINGS = "action_validate_slot_mappings"
ACTION_FLOW_CONTINUE_INERRUPTED_NAME = "action_flow_continue_interrupted"


DEFAULT_ACTION_NAMES = [
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_DEACTIVATE_LOOP_NAME,
    ACTION_REVERT_FALLBACK_EVENTS_NAME,
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
    ACTION_DEFAULT_ASK_REPHRASE_NAME,
    ACTION_TWO_STAGE_FALLBACK_NAME,
    ACTION_UNLIKELY_INTENT_NAME,
    ACTION_BACK_NAME,
    ACTION_SEND_TEXT,
    RULE_SNIPPET_ACTION_NAME,
    ACTION_EXTRACT_SLOTS,
    ACTION_FLOW_CONTINUE_INERRUPTED_NAME,
]

ACTION_SHOULD_SEND_DOMAIN = "send_domain"

# rules allow setting a value of slots or active_loops to None;
# generator substitutes `None`s with this constant to notify rule policy that
# a value should not be set during prediction to activate a rule
SHOULD_NOT_BE_SET = "should_not_be_set"

PREVIOUS_ACTION = "prev_action"
ACTIVE_LOOP = "active_loop"
LOOP_INTERRUPTED = "is_interrupted"
LOOP_REJECTED = "rejected"
TRIGGER_MESSAGE = "trigger_message"
FOLLOWUP_ACTION = "followup_action"

# start of special user message section
EXTERNAL_MESSAGE_PREFIX = "EXTERNAL: "
# Key to access data in the event metadata
# It specifies if an event was caused by an external entity (e.g. a sensor).
IS_EXTERNAL = "is_external"

ACTION_NAME_SENDER_ID_CONNECTOR_STR = "__sender_id:"

REQUESTED_SLOT = "requested_slot"
FLOW_STACK_SLOT = "flow_stack"

# slots for knowledge base
SLOT_LISTED_ITEMS = "knowledge_base_listed_objects"
SLOT_LAST_OBJECT = "knowledge_base_last_object"
SLOT_LAST_OBJECT_TYPE = "knowledge_base_last_object_type"
DEFAULT_KNOWLEDGE_BASE_ACTION = "action_query_knowledge_base"

DEFAULT_SLOT_NAMES = {
    REQUESTED_SLOT,
    FLOW_STACK_SLOT,
    SESSION_START_METADATA_SLOT,
    SLOT_LISTED_ITEMS,
    SLOT_LAST_OBJECT,
    SLOT_LAST_OBJECT_TYPE,
}


SLOT_MAPPINGS = "mappings"
MAPPING_CONDITIONS = "conditions"
MAPPING_TYPE = "type"


class SlotMappingType(Enum):
    """Slot mapping types."""

    FROM_ENTITY = "from_entity"
    FROM_INTENT = "from_intent"
    FROM_TRIGGER_INTENT = "from_trigger_intent"
    FROM_TEXT = "from_text"
    CUSTOM = "custom"

    def __str__(self) -> str:
        """Returns the string representation that should be used in config files."""
        return self.value

    def is_predefined_type(self) -> bool:
        """Returns True iff the mapping type is predefined.

        That is, to evaluate the mapping no custom action execution is needed.
        """
        return self != SlotMappingType.CUSTOM


# the keys for `State` (USER, PREVIOUS_ACTION, SLOTS, ACTIVE_LOOP)
# represent the origin of a `SubState`
USER = "user"
SLOTS = "slots"

USE_TEXT_FOR_FEATURIZATION = "use_text_for_featurization"
ENTITY_LABEL_SEPARATOR = "#"

RULE_ONLY_SLOTS = "rule_only_slots"
RULE_ONLY_LOOPS = "rule_only_loops"

# if you add more policy/classifier names, make sure to add a test as well to ensure
# that the name and the class stay in sync
POLICY_NAME_TWO_STAGE_FALLBACK = "TwoStageFallbackPolicy"
POLICY_NAME_MAPPING = "MappingPolicy"
POLICY_NAME_FALLBACK = "FallbackPolicy"
POLICY_NAME_FORM = "FormPolicy"
POLICY_NAME_RULE = "RulePolicy"

CLASSIFIER_NAME_FALLBACK = "FallbackClassifier"

POLICIES_THAT_EXTRACT_ENTITIES = {"TEDPolicy"}
