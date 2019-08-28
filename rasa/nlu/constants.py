MESSAGE_TEXT_ATTRIBUTE = "text"

MESSAGE_INTENT_ATTRIBUTE = "intent"

MESSAGE_RESPONSE_ATTRIBUTE = "response"

MESSAGE_ENTITIES_ATTRIBUTE = "entities"

MESSAGE_ATTRIBUTES = [
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_INTENT_ATTRIBUTE,
    MESSAGE_RESPONSE_ATTRIBUTE,
]

MESSAGE_TOKENS_NAMES = {
    MESSAGE_TEXT_ATTRIBUTE: "tokens",
    MESSAGE_INTENT_ATTRIBUTE: "intent_tokens",
    MESSAGE_RESPONSE_ATTRIBUTE: "response_tokens",
}

MESSAGE_VECTOR_FEATURE_NAMES = {
    MESSAGE_TEXT_ATTRIBUTE: "text_features",
    MESSAGE_INTENT_ATTRIBUTE: "intent_features",
    MESSAGE_RESPONSE_ATTRIBUTE: "response_features",
}

MESSAGE_SPACY_FEATURES_NAMES = {
    MESSAGE_TEXT_ATTRIBUTE: "spacy_doc",
    MESSAGE_INTENT_ATTRIBUTE: "intent_spacy_doc",
    MESSAGE_RESPONSE_ATTRIBUTE: "response_spacy_doc",
}

MESSAGE_SELECTOR_PROPERTY_NAME = "selector"
DEFAULT_OPEN_UTTERANCE_TYPE = "default"
OPEN_UTTERANCE_PREDICTION_KEY = "response"
OPEN_UTTERANCE_RANKING_KEY = "ranking"
