import rasa.shared.nlu.constants
from rasa.shared.nlu.constants import ENTITY_ATTRIBUTE_CONFIDENCE

BILOU_ENTITIES = "bilou_entities"
BILOU_ENTITIES_ROLE = "bilou_entities_role"
BILOU_ENTITIES_GROUP = "bilou_entities_group"

ENTITY_ATTRIBUTE_CONFIDENCE_TYPE = (
    f"{ENTITY_ATTRIBUTE_CONFIDENCE}_{rasa.shared.nlu.constants.ENTITY_ATTRIBUTE_TYPE}"
)
ENTITY_ATTRIBUTE_CONFIDENCE_GROUP = (
    f"{ENTITY_ATTRIBUTE_CONFIDENCE}_{rasa.shared.nlu.constants.ENTITY_ATTRIBUTE_GROUP}"
)
ENTITY_ATTRIBUTE_CONFIDENCE_ROLE = (
    f"{ENTITY_ATTRIBUTE_CONFIDENCE}_{rasa.shared.nlu.constants.ENTITY_ATTRIBUTE_ROLE}"
)

EXTRACTOR = "extractor"

PRETRAINED_EXTRACTORS = {"DucklingEntityExtractor"}

NUMBER_OF_SUB_TOKENS = "number_of_sub_tokens"

MESSAGE_ATTRIBUTES = [
    rasa.shared.nlu.constants.TEXT,
    rasa.shared.nlu.constants.INTENT,
    rasa.shared.nlu.constants.RESPONSE,
    rasa.shared.nlu.constants.ACTION_NAME,
    rasa.shared.nlu.constants.ACTION_TEXT,
    rasa.shared.nlu.constants.INTENT_RESPONSE_KEY,
]
# the dense featurizable attributes are essentially text attributes
DENSE_FEATURIZABLE_ATTRIBUTES = [
    rasa.shared.nlu.constants.TEXT,
    rasa.shared.nlu.constants.RESPONSE,
    rasa.shared.nlu.constants.ACTION_TEXT,
]

LANGUAGE_MODEL_DOCS = {
    rasa.shared.nlu.constants.TEXT: "text_language_model_doc",
    rasa.shared.nlu.constants.RESPONSE: "response_language_model_doc",
    rasa.shared.nlu.constants.ACTION_TEXT: "action_text_model_doc",
}

TOKENS_NAMES = {
    rasa.shared.nlu.constants.TEXT: "text_tokens",
    rasa.shared.nlu.constants.INTENT: "intent_tokens",
    rasa.shared.nlu.constants.RESPONSE: "response_tokens",
    rasa.shared.nlu.constants.ACTION_NAME: "action_name_tokens",
    rasa.shared.nlu.constants.ACTION_TEXT: "action_text_tokens",
    rasa.shared.nlu.constants.INTENT_RESPONSE_KEY: "intent_response_key_tokens",
}

SEQUENCE_FEATURES = "sequence_features"
SENTENCE_FEATURES = "sentence_features"

RESPONSE_IDENTIFIER_DELIMITER = "/"

DEFAULT_TRANSFORMER_SIZE = 256

FEATURIZER_CLASS_ALIAS = "alias"

NO_LENGTH_RESTRICTION = -1
