import logging
import os
import re
from typing import Any, Dict, List, Optional, Text

import rasa.utils.io as io_utils
import rasa.nlu.utils.pattern_utils as pattern_utils
from rasa.nlu.model import Metadata
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData
from rasa.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_END,
)
from rasa.nlu.training_data import Message
from rasa.nlu.extractors.extractor import EntityExtractor

logger = logging.getLogger(__name__)


class RegexEntityExtractor(EntityExtractor):
    """Searches for entities in the user's message using the lookup tables and regexes
    defined in the training data."""

    defaults = {
        # lower case the entity value from the lookup file and
        # user message while comparing them
        "lowercase": True
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        patterns: Optional[List[Dict[Text, Text]]] = None,
    ):
        super(RegexEntityExtractor, self).__init__(component_config)

        self.lowercase = self.component_config["lowercase"]
        self.patterns = patterns or []

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        self.patterns = pattern_utils.extract_patterns(training_data)

    def process(self, message: Message, **kwargs: Any) -> None:
        extracted_entities = self._extract_entities(message)
        extracted_entities = self.add_extractor_name(extracted_entities)

        message.set(
            ENTITIES, message.get(ENTITIES, []) + extracted_entities, add_to_output=True
        )

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Extract entities of the given type from the given user message."""
        entities = []

        flags = 0  # default flag
        if self.lowercase:
            flags = re.IGNORECASE

        for pattern in self.patterns:
            matches = re.finditer(pattern["pattern"], message.text, flags=flags)
            matches = list(matches)

            for match in matches:
                start_index = match.start()
                end_index = match.end()
                entities.append(
                    {
                        ENTITY_ATTRIBUTE_TYPE: pattern["name"],
                        ENTITY_ATTRIBUTE_START: start_index,
                        ENTITY_ATTRIBUTE_END: end_index,
                        ENTITY_ATTRIBUTE_VALUE: message.text[start_index:end_index],
                    }
                )

        return entities

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["RegexEntityExtractor"] = None,
        **kwargs: Any,
    ) -> "RegexEntityExtractor":

        file_name = meta.get("file")
        regex_file = os.path.join(model_dir, file_name)

        if os.path.exists(regex_file):
            patterns = io_utils.read_json_file(regex_file)
            return RegexEntityExtractor(meta, patterns=patterns)
        else:
            return RegexEntityExtractor(meta)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        file_name = f"{file_name}.json"
        regex_file = os.path.join(model_dir, file_name)
        io_utils.dump_obj_as_json_to_file(regex_file, self.patterns)

        return {"file": file_name}
