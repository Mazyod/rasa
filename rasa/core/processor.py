import inspect
import copy
import logging
import structlog
import os
from pathlib import Path
import tarfile
import time
from types import LambdaType
from typing import Any, Dict, List, Optional, Text, Tuple, Union

from rasa.engine import loader
from rasa.engine.constants import PLACEHOLDER_MESSAGE, PLACEHOLDER_TRACKER
from rasa.engine.runner.dask import DaskGraphRunner
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.storage import ModelMetadata
from rasa.model import get_latest_model
from rasa.shared.data import TrainingType
import rasa.shared.utils.io
from rasa.core.channels.channel import (
    UserMessage,
)
import rasa.core.utils
from rasa.engine.runner.interface import GraphRunner
from rasa.exceptions import ModelNotFound
from rasa.shared.core.constants import (
    USER_INTENT_RESTART,
)
from rasa.shared.constants import (
    ASSISTANT_ID_KEY,
    DOCS_URL_DOMAINS,
    DEFAULT_SENDER_ID,
    DOCS_URL_POLICIES,
    UTTER_PREFIX,
)
from rasa.core.lock_store import LockStore
from rasa.utils.common import TempDirectoryPath, get_temp_dir_name
from rasa.shared.nlu.constants import (
    ENTITIES,
    INTENT,
    INTENT_NAME_KEY,
    PREDICTED_CONFIDENCE_KEY,
    TEXT,
)
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)
structlogger = structlog.get_logger()

MAX_NUMBER_OF_PREDICTIONS = int(os.environ.get("MAX_NUMBER_OF_PREDICTIONS", "10"))


class MessageProcessor:
    """The message processor is interface for communicating with a bot model."""

    def __init__(
        self,
        model_path: Union[Text, Path],
        lock_store: LockStore,
        max_number_of_predictions: int = MAX_NUMBER_OF_PREDICTIONS,
        on_circuit_break: Optional[LambdaType] = None,
    ) -> None:
        """Initializes a `MessageProcessor`."""
        self.lock_store = lock_store
        self.max_number_of_predictions = max_number_of_predictions
        self.on_circuit_break = on_circuit_break
        self.model_filename, self.model_metadata, self.graph_runner = self._load_model(
            model_path
        )

        if self.model_metadata.assistant_id is None:
            rasa.shared.utils.io.raise_warning(
                f"The model metadata does not contain a value for the "
                f"'{ASSISTANT_ID_KEY}' attribute. Check that 'config.yml' "
                f"file contains a value for the '{ASSISTANT_ID_KEY}' key "
                f"and re-train the model. Failure to do so will result in "
                f"streaming events without a unique assistant identifier.",
                UserWarning,
            )

        self.model_path = Path(model_path)
        self.domain = self.model_metadata.domain

    @staticmethod
    def _load_model(
        model_path: Union[Text, Path]
    ) -> Tuple[Text, ModelMetadata, GraphRunner]:
        """Unpacks a model from a given path using the graph model loader."""
        try:
            if os.path.isfile(model_path):
                model_tar = model_path
            else:
                model_file_path = get_latest_model(model_path)
                if not model_file_path:
                    raise ModelNotFound(f"No model found at path '{model_path}'.")
                model_tar = model_file_path
        except TypeError:
            raise ModelNotFound(f"Model {model_path} can not be loaded.")

        logger.info(f"Loading model {model_tar}...")
        with TempDirectoryPath(get_temp_dir_name()) as temporary_directory:
            try:
                metadata, runner = loader.load_predict_graph_runner(
                    Path(temporary_directory),
                    Path(model_tar),
                    LocalModelStorage,
                    DaskGraphRunner,
                )
                return os.path.basename(model_tar), metadata, runner
            except tarfile.ReadError:
                raise ModelNotFound(f"Model {model_path} can not be loaded.")

    async def handle_message(
        self, message: UserMessage
    ) -> Optional[List[Dict[Text, Any]]]:
        """Handle a single message with this processor."""
        # preprocess message if necessary
        if self.model_metadata.training_type == TrainingType.NLU:
            rasa.shared.utils.io.raise_warning(
                "No core model. Skipping action prediction and execution.",
                docs=DOCS_URL_POLICIES,
            )
            return None

    def _check_for_unseen_features(self, parse_data: Dict[Text, Any]) -> None:
        """Warns the user if the NLU parse data contains unrecognized features.

        Checks intents and entities picked up by the NLU parsing
        against the domain and warns the user of those that don't match.
        Also considers a list of default intents that are valid but don't
        need to be listed in the domain.

        Args:
            parse_data: Message parse data to check against the domain.
        """
        if not self.domain or self.domain.is_empty():
            return

        intent = parse_data["intent"][INTENT_NAME_KEY]
        if intent and intent not in self.domain.intents:
            rasa.shared.utils.io.raise_warning(
                f"Parsed an intent '{intent}' "
                f"which is not defined in the domain. "
                f"Please make sure all intents are listed in the domain.",
                docs=DOCS_URL_DOMAINS,
            )

        entities = parse_data["entities"] or []
        for element in entities:
            entity = element["entity"]
            if entity and entity not in self.domain.entities:
                rasa.shared.utils.io.raise_warning(
                    f"Parsed an entity '{entity}' "
                    f"which is not defined in the domain. "
                    f"Please make sure all entities are listed in the domain.",
                    docs=DOCS_URL_DOMAINS,
                )

    async def parse_message(
        self,
        message: UserMessage,
        tracker = None,
        only_output_properties: bool = True,
    ) -> Dict[Text, Any]:
        """Interprets the passed message.

        Args:
            message: Message to handle.
            tracker: Tracker to use.
            only_output_properties: If `True`, restrict the output to
                Message.only_output_properties.

        Returns:
            Parsed data extracted from the message.
        """
        from rasa import zombie

        msg = zombie.unpack_regex_message(
            message=Message({TEXT: message.text})
        )
        # Intent is not explicitly present. Pass message to graph.
        if msg.data.get(INTENT) is None:
            parse_data = self._parse_message_with_graph(
                message, tracker, only_output_properties
            )
        else:
            parse_data = {
                TEXT: "",
                INTENT: {INTENT_NAME_KEY: None, PREDICTED_CONFIDENCE_KEY: 0.0},
                ENTITIES: [],
            }
            parse_data.update(
                msg.as_dict(only_output_properties=only_output_properties)
            )

        structlogger.debug(
            "processor.message.parse",
            parse_data_text=copy.deepcopy(parse_data["text"]),
            parse_data_intent=parse_data["intent"],
            parse_data_entities=copy.deepcopy(parse_data["entities"]),
        )

        self._check_for_unseen_features(parse_data)

        return parse_data

    def _parse_message_with_graph(
        self,
        message: UserMessage,
        tracker = None,
        only_output_properties: bool = True,
    ) -> Dict[Text, Any]:
        """Interprets the passed message.

        Arguments:
            message: Message to handle
            tracker: Tracker to use
            only_output_properties: If `True`, restrict the output to
                Message.only_output_properties.

        Returns:
            Parsed data extracted from the message.
        """
        results = self.graph_runner.run(
            inputs={PLACEHOLDER_MESSAGE: [message], PLACEHOLDER_TRACKER: tracker},
            targets=[self.model_metadata.nlu_target],
        )
        parsed_messages = results[self.model_metadata.nlu_target]
        parsed_message = parsed_messages[0]
        parse_data = {
            TEXT: "",
            INTENT: {INTENT_NAME_KEY: None, PREDICTED_CONFIDENCE_KEY: 0.0},
            ENTITIES: [],
        }
        parse_data.update(
            parsed_message.as_dict(only_output_properties=only_output_properties)
        )
        return parse_data
