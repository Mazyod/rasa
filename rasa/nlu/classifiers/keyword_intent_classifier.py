import os
import logging
import typing
import re
from typing import Any, Dict, Optional, Text

from rasa.nlu import utils
from rasa.nlu.components import Component
from rasa.nlu.training_data import Message

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa.nlu.config import RasaNLUModelConfig
    from rasa.nlu.training_data import TrainingData
    from rasa.nlu.model import Metadata
    from rasa.nlu.training_data import Message


class KeywordIntentClassifier(Component):
    """Intent classifier using simple keyword matching.


    The classifier takes a list of keywords and associated intents as an input.
    A input sentence is checked for the keywords and the intent is returned.

    """

    provides = ["intent"]

    defaults = {"case_sensitive": True}

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        intent_keyword_map: Optional[Dict] = None,
    ):

        super(KeywordIntentClassifier, self).__init__(component_config)

        self.intent_keyword_map = intent_keyword_map or {}

        self.re_case_flag = (
            0 if self.component_config["case_sensitive"] else re.IGNORECASE
        )

    def train(
        self,
        training_data: "TrainingData",
        cfg: Optional["RasaNLUModelConfig"] = None,
        **kwargs: Any
    ) -> None:

        duplicate_examples = set()
        for ex in training_data.training_examples:
            if (
                ex.text in self.intent_keyword_map.keys()
                and ex.get("intent") != self.intent_keyword_map[ex.text]
            ):
                duplicate_examples.add(ex.text)
                logger.warning(
                    "Keyword '{}' is an example of intent '{}' and of "
                    "intent '{}', it will be removed from the list of "
                    "keyword.\n"
                    "Remove (one of) the duplicates from the training data."
                    "".format(
                        ex.text, self.intent_keyword_map[ex.text], ex.get("intent")
                    )
                )
            else:
                self.intent_keyword_map[ex.text] = ex.get("intent")
        for keyword in duplicate_examples:
            self.intent_keyword_map.pop(keyword)
            logger.debug(
                "Removed '{}' from the list of keywords because it was "
                "a keyword for more than one intent.".format(keyword)
            )

        self._validate_keyword_map()

    def _validate_keyword_map(self):
        re_flag = 0 if self.component_config["case_sensitive"] else re.IGNORECASE

        ambiguous_mappings = []
        for keyword1, intent1 in self.intent_keyword_map.items():
            for keyword2, intent2 in self.intent_keyword_map.items():
                if (
                    re.search(r"\b" + keyword1 + r"\b", keyword2, flags=re_flag)
                    and intent1 != intent2
                ):
                    ambiguous_mappings.append((intent1, keyword1))
                    logger.warning(
                        "Keyword '{}' is an keywordample of intent '{}', "
                        "but also a substring of '{}', which is an "
                        "keywordample of intent '{}."
                        " '{}' will be removed from the list of keywords.\n"
                        "Remove (one of) the conflicting keywordamples for the"
                        " training data."
                        "".format(keyword1, intent1, keyword2, intent2, keyword1)
                    )
        for intent, keyword in ambiguous_mappings:
            self.intent_keyword_map.pop(keyword)
            logger.debug(
                "Removed keyword '{}' from intent '{}' because it matched a "
                "keyword of another intent.".format(keyword, intent)
            )

    def process(self, message: Message, **kwargs: Any) -> None:
        intent_name = self._map_keyword_to_intent(message.text)
        confidence = 0.0 if intent_name is None else 1.0
        intent = {"name": intent_name, "confidence": confidence}
        if message.get("intent") is None or intent is not None:
            message.set("intent", intent, add_to_output=True)

    def _map_keyword_to_intent(self, text: Text) -> Optional[Text]:
        re_flag = 0 if self.component_config["case_sensitive"] else re.IGNORECASE
        for keyword, intent in self.intent_keyword_map.items():
            if re.search(r"\b" + keyword + r"\b", text, flags=re_flag):
                logger.debug(
                    "KeywordClassifier matched keyword '{}' to"
                    " intent '{}'.".format(keyword, intent)
                )
                return intent
        logger.debug("KeywordClassifier did not find any keywords in the message.")
        return None

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """

        file_name = file_name + ".json"
        keyword_file = os.path.join(model_dir, file_name)
        utils.write_json_to_file(keyword_file, self.intent_keyword_map)

        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: "Metadata" = None,
        cached_component: Optional["KeywordIntentClassifier"] = None,
        **kwargs: Any
    ) -> "KeywordIntentClassifier":

        if model_dir and meta.get("file"):
            file_name = meta.get("file")
            keyword_file = os.path.join(model_dir, file_name)
            if os.path.exists(keyword_file):
                intent_keyword_map = utils.read_json_file(keyword_file)
            else:
                logger.warning(
                    "Failed to load IntentKeywordClassifier, maybe "
                    "{} does not exist.".format(keyword_file)
                )
        return cls(meta, intent_keyword_map)
