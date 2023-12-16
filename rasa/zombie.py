import copy
import json
from json import JSONDecodeError
import logging
import structlog
import re
from re import Match, Pattern
from typing import Dict, Text, List, Any, Optional

from rasa.shared.core.domain import Domain
import rasa.shared.data
import rasa.shared.utils.io
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    INTENT,
    INTENT_NAME_KEY,
    INTENT_RANKING_KEY,
    PREDICTED_CONFIDENCE_KEY,
    TEXT,
    EXTRACTOR,
)
import rasa.shared.utils.validation

from rasa.shared.constants import (
    INTENT_MESSAGE_PREFIX,
    DOCS_URL_STORIES,
)

from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)
structlogger = structlog.get_logger()


def unpack_regex_message(
    message: Message,
    domain: Optional[Domain] = None,
    entity_extractor_name: Optional[Text] = None,
) -> Message:
    """Unpacks the message if `TEXT` contains an encoding of attributes.

    Args:
        message: some message
        domain: the domain
        entity_extractor_name: An extractor name which should be added for the
            entities.

    Returns:
        the given message if that message does not need to be unpacked, and a new
        message with the extracted attributes otherwise
    """
    user_text = message.get(TEXT).strip()

    # If the prefix doesn't match, we don't even need to try to match the pattern.
    if not user_text.startswith(INTENT_MESSAGE_PREFIX):
        return message

    # Try to match the pattern.
    match = _regex_message_pattern().match(user_text)

    # If it doesn't match, then (potentially) something went wrong, because the
    # message text did start with the special prefix -- however, a user might
    # just have decided to start their text this way.
    if not match:
        structlogger.warning(
            "message.parsing.failed", user_text=copy.deepcopy(user_text)
        )
        return message

    # Extract attributes from the match - and validate it via the domain.
    intent_name = _intent_name_from_regex_match(match, domain)
    confidence = _confidences_from_regex_match(match)
    entities = _entities_from_regex_match(
        match, domain, entity_extractor_name
    )

    # The intent name is *not* optional, but during parsing we might find out
    # that the given intent is unknown (and warn). In this case, stop here.
    if intent_name is None:
        return message

    if match.group("rest"):
        rasa.shared.utils.io.raise_warning(
            f"Failed to parse arguments in line '{match.string}'. "
            f"Failed to interpret some parts. "
            f"Make sure your regex string is in the following format:"
            f"{INTENT_MESSAGE_PREFIX}"
            f"<intent_name>@<confidence-value><dictionary of entities> "
            f"Continuing without {match.group('rest')}. "
        )

    # Add the results to the message.
    intent_data = {
        INTENT_NAME_KEY: intent_name,
        PREDICTED_CONFIDENCE_KEY: confidence,
    }
    intent_ranking = [
        {INTENT_NAME_KEY: intent_name, PREDICTED_CONFIDENCE_KEY: confidence}
    ]
    message_data = {}
    message_data[TEXT] = user_text
    message_data[INTENT] = intent_data
    message_data[INTENT_RANKING_KEY] = intent_ranking
    message_data[ENTITIES] = entities
    return Message(message_data, output_properties=set(message_data.keys()))


def _regex_message_pattern() -> Pattern:
    """Builds the pattern that matches `TEXT`s of messages that need to be unpacked.

    Returns:
        pattern with named groups
    """
    return re.compile(
        f"^{INTENT_MESSAGE_PREFIX}"
        f"(?P<{INTENT_NAME_KEY}>[^{{@]+)"  # "{{" is a masked "{" in an f-string
        f"(?P<{PREDICTED_CONFIDENCE_KEY}>@[0-9.]+)?"
        f"(?P<{ENTITIES}>{{.+}})?"  # "{{" is a masked "{" in an f-string
        f"(?P<rest>.*)"
    )


def _intent_name_from_regex_match(match: Match, domain: Domain) -> Optional[Text]:
    intent_name = match.group(INTENT_NAME_KEY).strip()
    if domain and intent_name not in domain.intents:
        rasa.shared.utils.io.raise_warning(
            f"Failed to parse arguments in line '{match.string}'. "
            f"Expected the intent to be one of [{domain.intents}] "
            f"but found {intent_name}."
            f"Continuing with given line as user text.",
            docs=DOCS_URL_STORIES,
        )
        intent_name = None
    return intent_name


def _confidences_from_regex_match(match: Match) -> float:
    """Extracts the optional confidence information from the given pattern match.

    If no confidence is specified, then this method returns the maximum
    confidence `1.0`.
    If a confidence is specified but extraction fails, then this method defaults
    to a confidence of `0.0`.

    Args:
        match: a match produced by `self.pattern`
        domain: the domain

    Returns:
        some confidence value
    """
    confidence_str = match.group(PREDICTED_CONFIDENCE_KEY)
    if confidence_str is None:
        return 1.0
    try:
        confidence_str = confidence_str.strip()[1:]  # remove the "@"
        try:
            confidence = float(confidence_str)
        except ValueError:
            confidence = 0.0
            raise ValueError(
                f"Expected confidence to be a non-negative decimal number but "
                f"found {confidence}. Continuing with 0.0 instead."
            )
        if confidence > 1.0:
            # Due to the pattern we know that this cannot be a negative number.
            original_confidence = confidence
            confidence = min(1.0, confidence)
            raise ValueError(
                f"Expected confidence to be at most 1.0. "
                f"but found {original_confidence}. "
                f"Continuing with {confidence} instead."
            )
        return confidence

    except ValueError as e:
        rasa.shared.utils.io.raise_warning(
            f"Failed to parse arguments in line '{match.string}'. "
            f"Could not extract confidence value from `{confidence_str}'. "
            f"Make sure the intent confidence is an @ followed "
            f"by a decimal number that not negative and at most 1.0. "
            f"Error: {e}",
            docs=DOCS_URL_STORIES,
        )
        return confidence

def _entities_from_regex_match(
    match: Match, domain: Domain, extractor_name: Optional[Text]
) -> List[Dict[Text, Any]]:
    """Extracts the optional entity information from the given pattern match.

    If no entities are specified or if the extraction fails, then an empty list
    is returned.

    Args:
        match: a match produced by `self.pattern`
        domain: the domain
        extractor_name: A extractor name which should be added for the entities

    Returns:
        some list of entities
    """
    entities_str = match.group(ENTITIES)
    if entities_str is None:
        return []

    try:
        parsed_entities = json.loads(entities_str)
        if not isinstance(parsed_entities, dict):
            raise ValueError(
                f"Parsed value isn't a json object "
                f"(instead parser found '{type(parsed_entities)}')"
            )
    except (JSONDecodeError, ValueError) as e:
        rasa.shared.utils.io.raise_warning(
            f"Failed to parse arguments in line '{match.string}'. "
            f"Failed to decode parameters as a json object (dict). "
            f"Make sure the intent is followed by a proper json object (dict). "
            f"Continuing without entities. "
            f"Error: {e}",
            docs=DOCS_URL_STORIES,
        )
        parsed_entities = dict()

    # validate the given entity types
    if domain:
        entity_types = set(parsed_entities.keys())
        unknown_entity_types = entity_types.difference(domain.entities)
        if unknown_entity_types:
            rasa.shared.utils.io.raise_warning(
                f"Failed to parse arguments in line '{match.string}'. "
                f"Expected entities from {domain.entities} "
                f"but found {unknown_entity_types}. "
                f"Continuing without unknown entity types. ",
                docs=DOCS_URL_STORIES,
            )
            parsed_entities = {
                key: value
                for key, value in parsed_entities.items()
                if key not in unknown_entity_types
            }

    # convert them into the list of dictionaries that we expect
    entities: List[Dict[Text, Any]] = []
    default_properties = {}
    if extractor_name:
        default_properties = {EXTRACTOR: extractor_name}

    for entity_type, entity_values in parsed_entities.items():
        if not isinstance(entity_values, list):
            entity_values = [entity_values]

        for entity_value in entity_values:
            entities.append(
                {
                    ENTITY_ATTRIBUTE_TYPE: entity_type,
                    ENTITY_ATTRIBUTE_VALUE: entity_value,
                    ENTITY_ATTRIBUTE_START: match.start(ENTITIES),
                    ENTITY_ATTRIBUTE_END: match.end(ENTITIES),
                    **default_properties,
                }
            )
    return entities
