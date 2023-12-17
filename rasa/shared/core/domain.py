import copy
import collections
import json
import logging
import os
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    NoReturn,
    Optional,
    Set,
    Text,
    Tuple,
    Union,
    TYPE_CHECKING,
    Iterable,
    MutableMapping,
    NamedTuple,
    Callable,
    cast,
)
from dataclasses import dataclass

from ruamel.yaml.scalarstring import DoubleQuotedScalarString

from rasa.shared.constants import (
    DOMAIN_SCHEMA_FILE,
    DOCS_URL_DOMAINS,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
    RESPONSE_CONDITION,
)
import rasa.shared.core.constants
from rasa.shared.exceptions import (
    RasaException,
    YamlException,
    YamlSyntaxException,
)
import rasa.shared.utils.validation
import rasa.shared.utils.io
import rasa.shared.utils.common
from rasa.shared.utils.validation import KEY_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.nlu.constants import (
    RESPONSE_IDENTIFIER_DELIMITER,
    INTENT_NAME_KEY,
    ENTITIES,
)

SESSION_EXPIRATION_TIME_KEY = "session_expiration_time"
USED_ENTITIES_KEY = "used_entities"
USE_ENTITIES_KEY = "use_entities"
IGNORE_ENTITIES_KEY = "ignore_entities"
IS_RETRIEVAL_INTENT_KEY = "is_retrieval_intent"
ENTITY_ROLES_KEY = "roles"
ENTITY_GROUPS_KEY = "groups"
ENTITY_FEATURIZATION_KEY = "influence_conversation"

KEY_INTENTS = "intents"
KEY_ENTITIES = "entities"

ALL_DOMAIN_KEYS = [
    KEY_ENTITIES,
    KEY_INTENTS,
]

PREV_PREFIX = "prev_"

# State is a dictionary with keys (USER, PREVIOUS_ACTION, SLOTS, ACTIVE_LOOP)
# representing the origin of a SubState;
# the values are SubStates, that contain the information needed for featurization
SubStateValue = Union[Text, Tuple[Union[float, Text], ...]]
SubState = MutableMapping[Text, SubStateValue]
State = Dict[Text, SubState]

logger = logging.getLogger(__name__)


class InvalidDomain(RasaException):
    """Exception that can be raised when domain is not valid."""


class ActionNotFoundException(ValueError, RasaException):
    """Raised when an action name could not be found."""


@dataclass
class EntityProperties:
    """Class for keeping track of the properties of entities in the domain."""

    entities: List[Text]
    roles: Dict[Text, List[Text]]
    groups: Dict[Text, List[Text]]
    default_ignored_entities: List[Text]


class Domain:
    """The domain specifies the universe in which the bot's policy acts.

    A Domain subclass provides the actions the bot can take, the intents
    and entities it can recognise.
    """

    @classmethod
    def empty(cls) -> "Domain":
        """Returns empty Domain."""
        return Domain.from_dict({})

    @classmethod
    def load(cls, paths: Union[List[Union[Path, Text]], Text, Path]) -> "Domain":
        """Returns loaded Domain after merging all domain files."""
        if not paths:
            raise InvalidDomain(
                "No domain file was specified. Please specify a path "
                "to a valid domain file."
            )
        elif not isinstance(paths, list) and not isinstance(paths, set):
            paths = [paths]

        domain = Domain.empty()
        for path in paths:
            other = cls.from_path(path)
            domain = domain.merge(other)

        return domain

    @classmethod
    def from_path(cls, path: Union[Text, Path]) -> "Domain":
        """Loads the `Domain` from a path."""
        path = os.path.abspath(path)

        if os.path.isfile(path):
            domain = cls.from_file(path)
        elif os.path.isdir(path):
            domain = cls.from_directory(path)
        else:
            raise InvalidDomain(
                "Failed to load domain specification from '{}'. "
                "File not found!".format(os.path.abspath(path))
            )

        return domain

    @classmethod
    def from_file(cls, path: Text) -> "Domain":
        """Loads the `Domain` from a YAML file."""
        return cls.from_yaml(rasa.shared.utils.io.read_file(path), path)

    @classmethod
    def from_yaml(cls, yaml: Text, original_filename: Text = "") -> "Domain":
        """Loads the `Domain` from YAML text after validating it."""
        try:
            rasa.shared.utils.validation.validate_yaml_schema(yaml, DOMAIN_SCHEMA_FILE)

            data = rasa.shared.utils.io.read_yaml(yaml)
            if not rasa.shared.utils.validation.validate_training_data_format_version(
                data, original_filename
            ):
                return Domain.empty()
            return cls.from_dict(data)
        except YamlException as e:
            e.filename = original_filename
            raise e

    @classmethod
    def from_dict(cls, data: Dict) -> "Domain":
        """Deserializes and creates domain.

        Args:
            data: The serialized domain.

        Returns:
            The instantiated `Domain` object.
        """
        duplicates = data.pop("duplicates", None)
        if duplicates:
            warn_about_duplicates_found_during_domain_merging(duplicates)

        additional_arguments = {
            **data.get("config", {}),
        }
        intents = data.get(KEY_INTENTS, {})

        return cls(
            intents=intents,
            entities=data.get(KEY_ENTITIES, {}),
            data=Domain._cleaned_data(data),
            **additional_arguments,
        )

    @classmethod
    def from_directory(cls, path: Text) -> "Domain":
        """Loads and merges multiple domain files recursively from a directory tree."""
        combined: Dict[Text, Any] = {}
        for root, _, files in os.walk(path, followlinks=True):
            for file in files:
                full_path = os.path.join(root, file)
                if Domain.is_domain_file(full_path):
                    _ = Domain.from_file(full_path)  # does the validation here only
                    other_dict = rasa.shared.utils.io.read_yaml(
                        rasa.shared.utils.io.read_file(full_path)
                    )
                    combined = Domain.merge_domain_dicts(other_dict, combined)

        domain = Domain.from_dict(combined)
        return domain

    def merge(
        self,
        domain: Optional["Domain"],
        override: bool = False,
    ) -> "Domain":
        """Merges this domain dict with another one, combining their attributes.

        This method merges domain dicts, and ensures all attributes (like ``intents``,
        ``entities``, and ``actions``) are known to the Domain when the
        object is created.

        List attributes like ``intents`` and ``actions`` are deduped
        and merged. Single attributes are taken from `domain1` unless
        override is `True`, in which case they are taken from `domain2`.
        """
        if not domain or domain.is_empty():
            return self

        if self.is_empty():
            return domain

        merged_dict = self.__class__.merge_domain_dicts(
            domain.as_dict(), self.as_dict(), override
        )

        return Domain.from_dict(merged_dict)

    @staticmethod
    def merge_domain_dicts(
        domain_dict: Dict,
        combined: Dict,
        override: bool = False,
    ) -> Dict:
        """Combines two domain dictionaries."""
        if not domain_dict:
            return combined

        if not combined:
            return domain_dict

        if override:
            config = domain_dict.get("config", {})
            for key, val in config.items():
                combined["config"][key] = val

        duplicates: Dict[Text, List[Text]] = {}

        merge_func_mappings: Dict[Text, Callable[..., Any]] = {
            KEY_INTENTS: rasa.shared.utils.common.merge_lists_of_dicts,
            KEY_ENTITIES: rasa.shared.utils.common.merge_lists_of_dicts,
        }

        for key, merge_func in merge_func_mappings.items():
            duplicates[key] = rasa.shared.utils.common.extract_duplicates(
                combined.get(key, []), domain_dict.get(key, [])
            )

            default: Union[List[Any], Dict[Text, Any]] = (
                {} if merge_func == rasa.shared.utils.common.merge_dicts else []
            )

            combined[key] = merge_func(
                combined.get(key, default), domain_dict.get(key, default), override
            )

        if duplicates:
            duplicates = rasa.shared.utils.common.clean_duplicates(duplicates)
            combined.update({"duplicates": duplicates})

        return combined

    def _preprocess_domain_dict(
        self,
        data: Dict
    ) -> Dict:
        data = self._add_default_keys_to_domain_dict(
            data
        )
        data = self._sanitize_intents_in_domain_dict(data)

        return data

    @staticmethod
    def _add_default_keys_to_domain_dict(
        data: Dict
    ) -> Dict:
        # add the config, session_config and training data version defaults
        # if not included in the original domain dict
        if "config" not in data:
            data.update(
                {"config": {}}
            )

        if KEY_TRAINING_DATA_FORMAT_VERSION not in data:
            data.update(
                {
                    KEY_TRAINING_DATA_FORMAT_VERSION: DoubleQuotedScalarString(
                        LATEST_TRAINING_DATA_FORMAT_VERSION
                    )
                }
            )

        return data

    @staticmethod
    def _reset_intent_flags(intent: Dict[Text, Any]) -> None:
        for intent_property in intent.values():
            if (
                USE_ENTITIES_KEY in intent_property.keys()
                and not intent_property[USE_ENTITIES_KEY]
            ):
                intent_property[USE_ENTITIES_KEY] = []
            if (
                IGNORE_ENTITIES_KEY in intent_property.keys()
                and not intent_property[IGNORE_ENTITIES_KEY]
            ):
                intent_property[IGNORE_ENTITIES_KEY] = []

    @staticmethod
    def _sanitize_intents_in_domain_dict(data: Dict[Text, Any]) -> Dict[Text, Any]:
        if not data.get(KEY_INTENTS):
            return data

        for intent in data.get(KEY_INTENTS, []):
            if isinstance(intent, dict):
                Domain._reset_intent_flags(intent)

        data[KEY_INTENTS] = Domain._sort_intent_names_alphabetical_order(
            intents=data.get(KEY_INTENTS)
        )

        return data

    @staticmethod
    def _transform_intent_properties_for_internal_use(
        intent: Dict[Text, Any], entity_properties: EntityProperties
    ) -> Dict[Text, Any]:
        """Transforms the intent's parameters in a format suitable for internal use.

        When an intent is retrieved from the `domain.yml` file, it contains two
        parameters, the `use_entities` and the `ignore_entities` parameter.
        With the values of these two parameters the Domain class is updated, a new
        parameter is added to the intent called `used_entities` and the two
        previous parameters are deleted. This happens because internally only the
        parameter `used_entities` is needed to list all the entities that should be
        used for this intent.

        Args:
            intent: The intent as retrieved from the `domain.yml` file thus having two
                parameters, the `use_entities` and the `ignore_entities` parameter.
            entity_properties: Entity properties as provided by the domain file.

        Returns:
            The intent with the new format thus having only one parameter called
            `used_entities` since this is the expected format of the intent
            when used internally.
        """
        name, properties = next(iter(intent.items()))

        if properties:
            properties.setdefault(USE_ENTITIES_KEY, True)
        else:
            raise InvalidDomain(
                f"In the `domain.yml` file, the intent '{name}' cannot have value of"
                f" `{type(properties)}`. If you have placed a ':' character after the"
                f" intent's name without adding any additional parameters to this"
                f" intent then you would need to remove the ':' character. Please see"
                f" {rasa.shared.constants.DOCS_URL_DOMAINS} for more information on how"
                f" to correctly add `intents` in the `domain` and"
                f" {rasa.shared.constants.DOCS_URL_INTENTS} for examples on"
                f" when to use the ':' character after an intent's name."
            )

        properties.setdefault(
            IGNORE_ENTITIES_KEY, entity_properties.default_ignored_entities
        )
        if not properties[USE_ENTITIES_KEY]:  # this covers False, None and []
            properties[USE_ENTITIES_KEY] = []

        # `use_entities` is either a list of explicitly included entities
        # or `True` if all should be included
        # if the listed entities have a role or group label, concatenate the entity
        # label with the corresponding role or group label to make sure roles and
        # groups can also influence the dialogue predictions
        if properties[USE_ENTITIES_KEY] is True:
            included_entities = set(entity_properties.entities) - set(
                entity_properties.default_ignored_entities
            )
            included_entities.update(
                Domain.concatenate_entity_labels(entity_properties.roles)
            )
            included_entities.update(
                Domain.concatenate_entity_labels(entity_properties.groups)
            )
        else:
            included_entities = set(properties[USE_ENTITIES_KEY])
            for entity in list(included_entities):
                included_entities.update(
                    Domain.concatenate_entity_labels(entity_properties.roles, entity)
                )
                included_entities.update(
                    Domain.concatenate_entity_labels(entity_properties.groups, entity)
                )
        excluded_entities = set(properties[IGNORE_ENTITIES_KEY])
        for entity in list(excluded_entities):
            excluded_entities.update(
                Domain.concatenate_entity_labels(entity_properties.roles, entity)
            )
            excluded_entities.update(
                Domain.concatenate_entity_labels(entity_properties.groups, entity)
            )
        used_entities = list(included_entities - excluded_entities)
        used_entities.sort()

        # Only print warning for ambiguous configurations if entities were included
        # explicitly.
        explicitly_included = isinstance(properties[USE_ENTITIES_KEY], list)
        ambiguous_entities = included_entities.intersection(excluded_entities)
        if explicitly_included and ambiguous_entities:
            rasa.shared.utils.io.raise_warning(
                f"Entities: '{ambiguous_entities}' are explicitly included and"
                f" excluded for intent '{name}'."
                f"Excluding takes precedence in this case. "
                f"Please resolve that ambiguity.",
                docs=f"{DOCS_URL_DOMAINS}",
            )

        properties[USED_ENTITIES_KEY] = used_entities
        del properties[USE_ENTITIES_KEY]
        del properties[IGNORE_ENTITIES_KEY]

        return intent

    @classmethod
    def collect_entity_properties(
        cls, domain_entities: List[Union[Text, Dict[Text, Any]]]
    ) -> EntityProperties:
        """Get entity properties for a domain from what is provided by a domain file.

        Args:
            domain_entities: The entities as provided by a domain file.

        Returns:
            An instance of EntityProperties.
        """
        entity_properties = EntityProperties([], {}, {}, [])
        for entity in domain_entities:
            if isinstance(entity, str):
                entity_properties.entities.append(entity)
            elif isinstance(entity, dict):
                for _entity, sub_labels in entity.items():
                    entity_properties.entities.append(_entity)
                    if sub_labels:
                        if ENTITY_ROLES_KEY in sub_labels:
                            entity_properties.roles[_entity] = sub_labels[
                                ENTITY_ROLES_KEY
                            ]
                        if ENTITY_GROUPS_KEY in sub_labels:
                            entity_properties.groups[_entity] = sub_labels[
                                ENTITY_GROUPS_KEY
                            ]
                        if (
                            ENTITY_FEATURIZATION_KEY in sub_labels
                            and sub_labels[ENTITY_FEATURIZATION_KEY] is False
                        ):
                            entity_properties.default_ignored_entities.append(_entity)
                    else:
                        raise InvalidDomain(
                            f"In the `domain.yml` file, the entity '{_entity}' cannot"
                            f" have value of `{type(sub_labels)}`. If you have placed a"
                            f" ':' character after the entity `{_entity}` without"
                            f" adding any additional parameters to this entity then you"
                            f" would need to remove the ':' character. Please see"
                            f" {rasa.shared.constants.DOCS_URL_DOMAINS} for more"
                            f" information on how to correctly add `entities` in the"
                            f" `domain` and {rasa.shared.constants.DOCS_URL_ENTITIES}"
                            f" for examples on when to use the ':' character after an"
                            f" entity's name."
                        )
            else:
                raise InvalidDomain(
                    f"Invalid domain. Entity is invalid, type of entity '{entity}' "
                    f"not supported: '{type(entity).__name__}'"
                )

        return entity_properties

    @classmethod
    def collect_intent_properties(
        cls,
        intents: List[Union[Text, Dict[Text, Any]]],
        entity_properties: EntityProperties,
    ) -> Dict[Text, Dict[Text, Union[bool, List]]]:
        """Get intent properties for a domain from what is provided by a domain file.

        Args:
            intents: The intents as provided by a domain file.
            entity_properties: Entity properties as provided by the domain file.

        Returns:
            The intent properties to be stored in the domain.
        """
        # make a copy to not alter the input argument
        intents = copy.deepcopy(intents)
        intent_properties: Dict[Text, Any] = {}
        duplicates = set()

        for intent in intents:
            intent_name, properties = cls._intent_properties(intent, entity_properties)

            if intent_name in intent_properties.keys():
                duplicates.add(intent_name)

            intent_properties.update(properties)

        if duplicates:
            raise InvalidDomain(
                f"Intents are not unique! Found multiple intents "
                f"with name(s) {sorted(duplicates)}. "
                f"Either rename or remove the duplicate ones."
            )

        cls._add_default_intents(intent_properties, entity_properties)

        return intent_properties

    @classmethod
    def _intent_properties(
        cls, intent: Union[Text, Dict[Text, Any]], entity_properties: EntityProperties
    ) -> Tuple[Text, Dict[Text, Any]]:
        if not isinstance(intent, dict):
            intent_name = intent
            intent = {
                intent_name: {
                    USE_ENTITIES_KEY: True,
                    IGNORE_ENTITIES_KEY: entity_properties.default_ignored_entities,
                }
            }
        else:
            intent_name = next(iter(intent.keys()))

        return (
            intent_name,
            cls._transform_intent_properties_for_internal_use(
                intent, entity_properties
            ),
        )

    @classmethod
    def _add_default_intents(
        cls,
        intent_properties: Dict[Text, Dict[Text, Union[bool, List]]],
        entity_properties: EntityProperties,
    ) -> None:
        for intent_name in rasa.shared.core.constants.DEFAULT_INTENTS:
            if intent_name not in intent_properties:
                _, properties = cls._intent_properties(intent_name, entity_properties)
                intent_properties.update(properties)

    def __init__(
        self,
        intents: Union[Set[Text], List[Text], List[Dict[Text, Any]]],
        entities: List[Union[Text, Dict[Text, Any]]],
        data: Dict,
        **kwargs: Any,
    ) -> None:
        """Create a `Domain`.

        Args:
            intents: Intent labels.
            entities: The names of entities which might be present in user messages.
            data: original domain dict representation.
        """
        self.entity_properties = self.collect_entity_properties(entities)
        self.intent_properties = self.collect_intent_properties(
            intents, self.entity_properties
        )
        self.overridden_default_intents = self._collect_overridden_default_intents(
            intents
        )

        data_copy = copy.deepcopy(data)
        self._data = self._preprocess_domain_dict(
            data_copy
        )

        self._check_domain_sanity()

    def __deepcopy__(self, memo: Optional[Dict[int, Any]]) -> "Domain":
        """Enables making a deep copy of the `Domain` using `copy.deepcopy`.

        See https://docs.python.org/3/library/copy.html#copy.deepcopy
        for more implementation.

        Args:
            memo: Optional dictionary of objects already copied during the current
            copying pass.

        Returns:
            A deep copy of the current domain.
        """
        domain_dict = self.as_dict()
        return self.__class__.from_dict(copy.deepcopy(domain_dict, memo))

    def count_conditional_response_variations(self) -> int:
        """Returns count of conditional response variations."""
        count = 0
        for response_variations in self.responses.values():
            for variation in response_variations:
                if RESPONSE_CONDITION in variation:
                    count += 1

        return count

    @staticmethod
    def _collect_overridden_default_intents(
        intents: Union[Set[Text], List[Text], List[Dict[Text, Any]]]
    ) -> List[Text]:
        """Collects the default intents overridden by the user.

        Args:
            intents: User-provided intents.

        Returns:
            User-defined intents that are default intents.
        """
        intent_names: Set[Text] = {
            next(iter(intent.keys())) if isinstance(intent, dict) else intent
            for intent in intents
        }
        return sorted(
            intent_names.intersection(set(rasa.shared.core.constants.DEFAULT_INTENTS))
        )

    def __hash__(self) -> int:
        """Returns a unique hash for the domain."""
        return int(self.fingerprint(), 16)

    def fingerprint(self) -> Text:
        """Returns a unique hash for the domain which is stable across python runs.

        Returns:
            fingerprint of the domain
        """
        self_as_dict = self.as_dict()
        transformed_intents: List[Text] = []
        for intent in self_as_dict.get(KEY_INTENTS, []):
            if isinstance(intent, dict):
                transformed_intents.append(*intent.keys())
            elif isinstance(intent, str):
                transformed_intents.append(intent)

        self_as_dict[KEY_INTENTS] = sorted(transformed_intents)
        return rasa.shared.utils.io.get_dictionary_fingerprint(self_as_dict)

    @staticmethod
    def _sort_intent_names_alphabetical_order(
        intents: List[Union[Text, Dict]]
    ) -> List[Union[Text, Dict]]:
        def sort(elem: Union[Text, Dict]) -> Union[Text, Dict]:
            if isinstance(elem, dict):
                return next(iter(elem.keys()))
            elif isinstance(elem, str):
                return elem

        sorted_intents = sorted(intents, key=sort)
        return sorted_intents

    @rasa.shared.utils.common.lazy_property
    def user_actions_and_forms(self) -> List[Text]:
        """Returns combination of user actions and forms."""
        return self.user_actions + self.form_names

    @rasa.shared.utils.common.lazy_property
    def num_states(self) -> int:
        """Number of used input states for the action prediction."""
        return len(self.input_states)

    @rasa.shared.utils.common.lazy_property
    def retrieval_intent_responses(self) -> Dict[Text, List[Dict[Text, Any]]]:
        """Return only the responses which are defined for retrieval intents."""
        return dict(
            filter(
                lambda intent_response: self.is_retrieval_intent_response(
                    intent_response
                ),
                self.responses.items(),
            )
        )

    @staticmethod
    def is_retrieval_intent_response(
        response: Tuple[Text, List[Dict[Text, Any]]]
    ) -> bool:
        """Check if the response is for a retrieval intent.

        These responses have a `/` symbol in their name. Use that to filter them from
        the rest.
        """
        return RESPONSE_IDENTIFIER_DELIMITER in response[0]

    # noinspection PyTypeChecker
    @rasa.shared.utils.common.lazy_property
    def entity_states(self) -> List[Text]:
        """Returns all available entity state strings."""
        entity_states = copy.deepcopy(self.entities)
        entity_states.extend(
            Domain.concatenate_entity_labels(self.entity_properties.roles)
        )
        entity_states.extend(
            Domain.concatenate_entity_labels(self.entity_properties.groups)
        )

        return entity_states

    @staticmethod
    def concatenate_entity_labels(
        entity_labels: Dict[Text, List[Text]], entity: Optional[Text] = None
    ) -> List[Text]:
        """Concatenates the given entity labels with their corresponding sub-labels.

        If a specific entity label is given, only this entity label will be
        concatenated with its corresponding sub-labels.

        Args:
            entity_labels: A map of an entity label to its sub-label list.
            entity: If present, only this entity will be considered.

        Returns:
            A list of labels.
        """
        if entity is not None and entity not in entity_labels:
            return []

        if entity:
            return [
                f"{entity}"
                f"{rasa.shared.core.constants.ENTITY_LABEL_SEPARATOR}"
                f"{sub_label}"
                for sub_label in entity_labels[entity]
            ]

        return [
            f"{entity_label}"
            f"{rasa.shared.core.constants.ENTITY_LABEL_SEPARATOR}"
            f"{entity_sub_label}"
            for entity_label, entity_sub_labels in entity_labels.items()
            for entity_sub_label in entity_sub_labels
        ]

    @rasa.shared.utils.common.lazy_property
    def input_state_map(self) -> Dict[Text, int]:
        """Provide a mapping from state names to indices."""
        return {f: i for i, f in enumerate(self.input_states)}

    @rasa.shared.utils.common.lazy_property
    def input_states(self) -> List[Text]:
        """Returns all available states."""
        return (
            self.intents
            + self.entity_states
        )

    def _get_featurized_entities(self, latest_message: Any) -> Set[Text]:
        """Gets the names of all entities that are present and wanted in the message.

        Wherever an entity has a role or group specified as well, an additional role-
        or group-specific entity name is added.
        """
        intent_name = latest_message.intent.get(INTENT_NAME_KEY)
        intent_config = self.intent_config(intent_name)
        entities = latest_message.entities

        # If Entity Roles and Groups is used, we also need to make sure the roles and
        # groups get featurized. We concatenate the entity label with the role/group
        # label using a special separator to make sure that the resulting label is
        # unique (as you can have the same role/group label for different entities).
        entity_names_basic = set(
            entity["entity"] for entity in entities if "entity" in entity.keys()
        )
        entity_names_roles = set(
            f"{entity['entity']}"
            f"{rasa.shared.core.constants.ENTITY_LABEL_SEPARATOR}{entity['role']}"
            for entity in entities
            if "entity" in entity.keys() and "role" in entity.keys()
        )
        entity_names_groups = set(
            f"{entity['entity']}"
            f"{rasa.shared.core.constants.ENTITY_LABEL_SEPARATOR}{entity['group']}"
            for entity in entities
            if "entity" in entity.keys() and "group" in entity.keys()
        )
        entity_names = entity_names_basic.union(entity_names_roles, entity_names_groups)

        # the USED_ENTITIES_KEY of an intent also contains the entity labels and the
        # concatenated entity labels with their corresponding roles and groups labels
        wanted_entities = set(intent_config.get(USED_ENTITIES_KEY, entity_names))

        return entity_names.intersection(wanted_entities)

    def as_dict(self) -> Dict[Text, Any]:
        """Return serialized `Domain`."""
        return self._data

    @staticmethod
    def _cleaned_data(data: Dict[Text, Any]) -> Dict[Text, Any]:
        """Remove empty and redundant keys from merged domain dict.

        Returns:
            A cleaned dictionary version of the domain.
        """
        return {
            key: val
            for key, val in data.items()
            if val != {} and val != [] and val is not None
        }

    def persist(self, filename: Union[Text, Path]) -> None:
        """Write domain to a file."""
        as_yaml = self.as_yaml()
        rasa.shared.utils.io.write_text_file(as_yaml, filename)

    def as_yaml(self) -> Text:
        """Dump the `Domain` object as a YAML string.

        This function preserves the orders of the keys in the domain.

        Returns:
            A string in YAML format representing the domain.
        """
        # setting the `version` key first so that it appears at the top of YAML files
        # thanks to the `should_preserve_key_order` argument
        # of `dump_obj_as_yaml_to_string`
        domain_data: Dict[Text, Any] = {
            KEY_TRAINING_DATA_FORMAT_VERSION: DoubleQuotedScalarString(
                LATEST_TRAINING_DATA_FORMAT_VERSION
            )
        }

        domain_data.update(self.as_dict())

        return rasa.shared.utils.io.dump_obj_as_yaml_to_string(
            domain_data, should_preserve_key_order=True
        )

    def intent_config(self, intent_name: Text) -> Dict[Text, Any]:
        """Return the configuration for an intent."""
        return self.intent_properties.get(intent_name, {})

    @rasa.shared.utils.common.lazy_property
    def intents(self) -> List[Text]:
        """Returns sorted list of intents."""
        return sorted(self.intent_properties.keys())

    @rasa.shared.utils.common.lazy_property
    def entities(self) -> List[Text]:
        """Returns sorted list of entities."""
        return sorted(self.entity_properties.entities)

    @staticmethod
    def _get_symmetric_difference(
        domain_elements: Union[List[Text], Set[Text]],
        training_data_elements: Optional[Union[List[Text], Set[Text]]],
    ) -> Dict[Text, Set[Text]]:
        """Gets the symmetric difference between two sets.

        One set represents domain elements and the other one is a set of training
        data elements.

        Returns a dictionary containing a list of items found in the `domain_elements`
        but not in `training_data_elements` at key `in_domain`, and a list of items
        found in `training_data_elements` but not in `domain_elements` at key
        `in_training_data_set`.
        """
        if training_data_elements is None:
            training_data_elements = set()

        in_domain_diff = set(domain_elements) - set(training_data_elements)
        in_training_data_diff = set(training_data_elements) - set(domain_elements)

        return {"in_domain": in_domain_diff, "in_training_data": in_training_data_diff}

    def _check_domain_sanity(self) -> None:
        """Make sure the domain is properly configured.

        If the domain contains any duplicate slots, intents, actions
        or entities, an InvalidDomain error is raised.  This error
        is also raised when intent-action mappings are incorrectly
        named or a response is missing.
        """

        def get_duplicates(my_items: Iterable[Any]) -> List[Any]:
            """Returns a list of duplicate items in my_items."""
            return [
                item
                for item, count in collections.Counter(my_items).items()
                if count > 1
            ]

        def get_exception_message(
            duplicates: Optional[List[Tuple[List[Text], Text]]] = None,
            mappings: Optional[List[Tuple[Text, Text]]] = None,
        ) -> Text:
            """Return a message given a list of error locations."""
            message = ""
            if duplicates:
                message += get_duplicate_exception_message(duplicates)
            if mappings:
                if message:
                    message += "\n"
                message += get_mapping_exception_message(mappings)
            return message

        def get_mapping_exception_message(mappings: List[Tuple[Text, Text]]) -> Text:
            """Return a message given a list of duplicates."""
            message = ""
            for name, action_name in mappings:
                if message:
                    message += "\n"
                message += (
                    "Intent '{}' is set to trigger action '{}', which is "
                    "not defined in the domain.".format(name, action_name)
                )
            return message

        def get_duplicate_exception_message(
            duplicates: List[Tuple[List[Text], Text]]
        ) -> Text:
            """Return a message given a list of duplicates."""
            message = ""
            for d, name in duplicates:
                if d:
                    if message:
                        message += "\n"
                    message += (
                        f"Duplicate {name} in domain. "
                        f"These {name} occur more than once in "
                        f"the domain: '{', '.join(d)}'."
                    )
            return message

        duplicate_entities = get_duplicates(self.entities)

        if (
            duplicate_entities
        ):
            raise InvalidDomain(
                get_exception_message(
                    [
                        (duplicate_entities, KEY_ENTITIES),
                    ],
                )
            )

    def is_empty(self) -> bool:
        """Check whether the domain is empty."""
        return self.as_dict() == Domain.empty().as_dict()

    @staticmethod
    def is_domain_file(filename: Union[Text, Path]) -> bool:
        """Checks whether the given file path is a Rasa domain file.

        Args:
            filename: Path of the file which should be checked.

        Returns:
            `True` if it's a domain file, otherwise `False`.

        Raises:
            YamlException: if the file seems to be a YAML file (extension) but
                can not be read / parsed.
        """
        from rasa.shared.data import is_likely_yaml_file

        if not is_likely_yaml_file(filename):
            return False

        try:
            content = rasa.shared.utils.io.read_yaml_file(filename)
        except (RasaException, YamlSyntaxException):
            rasa.shared.utils.io.raise_warning(
                message=f"The file {filename} could not be loaded as domain file. "
                + "You can use https://yamlchecker.com/ to validate "
                + "the YAML syntax of your file.",
                category=UserWarning,
            )
            return False

        return any(key in content for key in ALL_DOMAIN_KEYS)

    def __repr__(self) -> Text:
        """Returns text representation of object."""
        return (
            f"{self.__class__.__name__}: "
            f"{len(self.intent_properties)} intents, "
            f"{len(self.entities)} entities"
        )


def warn_about_duplicates_found_during_domain_merging(
    duplicates: Dict[Text, List[Text]]
) -> None:
    """Emits warning about found duplicates while loading multiple domain paths."""
    message = ""
    for key in [
        KEY_INTENTS,
        KEY_ENTITIES,
    ]:
        duplicates_per_key = duplicates.get(key)
        if duplicates_per_key:
            if message:
                message += " \n"

            duplicates_per_key_str = ", ".join(duplicates_per_key)
            message += (
                f"The following duplicated {key} have been found "
                f"across multiple domain files: {duplicates_per_key_str}"
            )

    rasa.shared.utils.io.raise_warning(message, docs=DOCS_URL_DOMAINS)
    return None
