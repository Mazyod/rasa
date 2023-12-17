from __future__ import annotations
from asyncio import AbstractEventLoop
import functools
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Text, Union
import uuid

from rasa.core.channels.channel import OutputChannel, UserMessage
from rasa.core.exceptions import AgentNotReady
from rasa.shared.constants import DEFAULT_SENDER_ID
from rasa.core.lock_store import InMemoryLockStore, LockStore
from rasa.core.processor import MessageProcessor
from rasa.exceptions import ModelNotFound
from rasa.shared.exceptions import RasaException
import rasa.shared.utils.io
from rasa.utils.common import TempDirectoryPath, get_temp_dir_name

logger = logging.getLogger(__name__)


def _load_and_set_updated_model(
    agent: Agent, model_directory: Text, fingerprint: Text
) -> None:
    """Load the persisted model into memory and set the model on the agent.

    Args:
        agent: Instance of `Agent` to update with the new model.
        model_directory: Rasa model directory.
        fingerprint: Fingerprint of the supplied model at `model_directory`.
    """
    logger.debug(f"Found new model with fingerprint {fingerprint}. Loading...")
    agent.load_model(model_directory, fingerprint)

    logger.debug("Finished updating agent to new model.")


async def load_agent(
    model_path: Optional[Text] = None,
    remote_storage: Optional[Text] = None,
    loop: Optional[AbstractEventLoop] = None,
) -> Agent:
    """Loads agent from server, remote storage or disk.

    Args:
        model_path: Path to the model if it's on disk.
        remote_storage: URL of remote storage for model.
        loop: Optional async loop to pass to broker creation.

    Returns:
        The instantiated `Agent` or `None`.
    """
    agent = Agent(
        remote_storage=remote_storage
    )

    try:
        if remote_storage is not None:
            agent.load_model_from_remote_storage(model_path)

        elif model_path is not None and os.path.exists(model_path):
            try:
                agent.load_model(model_path)
            except ModelNotFound:
                rasa.shared.utils.io.raise_warning(
                    f"No valid model found at {model_path}!"
                )
        else:
            rasa.shared.utils.io.raise_warning(
                "No valid configuration given to load agent. "
                "Agent loaded with no model!"
            )
        return agent

    except Exception as e:
        logger.error(f"Could not load model due to {e}.", exc_info=True)
        return agent


def agent_must_be_ready(f: Callable[..., Any]) -> Callable[..., Any]:
    """Any Agent method decorated with this will raise if the agent is not ready."""

    @functools.wraps(f)
    def decorated(self: Agent, *args: Any, **kwargs: Any) -> Any:
        if not self.is_ready():
            raise AgentNotReady(
                "Agent needs to be prepared before usage. You need to set a "
                "processor and a tracker store."
            )
        return f(self, *args, **kwargs)

    return decorated


class Agent:
    """The Agent class provides an interface for the most important Rasa functionality.

    This includes training, handling messages, loading a dialogue model,
    getting the next action, and handling a channel.
    """

    def __init__(
        self,
        domain = None,
        lock_store: Optional[LockStore] = None,
        fingerprint: Optional[Text] = None,
        remote_storage: Optional[Text] = None,
    ):
        """Initializes an `Agent`."""
        self.domain = domain
        self.processor: Optional[MessageProcessor] = None

        self.lock_store = self._create_lock_store(lock_store)

        self._set_fingerprint(fingerprint)
        self.remote_storage = remote_storage

    @classmethod
    def load(
        cls,
        model_path: Union[Text, Path],
        domain= None,
        lock_store: Optional[LockStore] = None,
        fingerprint: Optional[Text] = None,
        remote_storage: Optional[Text] = None,
    ) -> Agent:
        """Constructs a new agent and loads the processer and model."""
        agent = Agent(
            domain=domain,
            lock_store=lock_store,
            fingerprint=fingerprint,
            remote_storage=remote_storage,
        )
        agent.load_model(model_path=model_path, fingerprint=fingerprint)
        return agent

    def load_model(
        self, model_path: Union[Text, Path], fingerprint: Optional[Text] = None
    ) -> None:
        """Loads the agent's model and processor given a new model path."""
        self.processor = MessageProcessor(
            model_path=model_path,
            lock_store=self.lock_store,
        )
        self.domain = self.processor.domain

        self._set_fingerprint(fingerprint)

    @property
    def model_id(self) -> Optional[Text]:
        """Returns the model_id from processor's model_metadata."""
        return self.processor.model_metadata.model_id if self.processor else None

    @property
    def model_name(self) -> Optional[Text]:
        """Returns the model name from processor's model_path."""
        return self.processor.model_path.name if self.processor else None

    def is_ready(self) -> bool:
        """Check if all necessary components are instantiated to use agent."""
        return self.processor is not None

    @agent_must_be_ready
    async def parse_message(self, message_data: Text) -> Dict[Text, Any]:
        """Handles message text and intent payload input messages.

        The return value of this function is parsed_data.

        Args:
            message_data (Text): Contain the received message in text or\
            intent payload format.

        Returns:
            The parsed message.

        Example:
                {\
                    "text": '/greet{"name":"Rasa"}',\
                    "intent": {"name": "greet", "confidence": 1.0},\
                    "intent_ranking": [{"name": "greet", "confidence": 1.0}],\
                    "entities": [{"entity": "name", "start": 6,\
                                  "end": 21, "value": "Rasa"}],\
                }

        """
        message = UserMessage(message_data)

        return await self.processor.parse_message(message)  # type: ignore[union-attr]

    async def handle_message(
        self, message: UserMessage
    ) -> Optional[List[Dict[Text, Any]]]:
        """Handle a single message."""
        if not self.is_ready():
            logger.info("Ignoring message as there is no agent to handle it.")
            return None

        async with self.lock_store.lock(message.sender_id):
            return await self.processor.handle_message(  # type: ignore[union-attr]
                message
            )

    @agent_must_be_ready
    async def handle_text(
        self,
        text_message: Union[Text, Dict[Text, Any]],
        output_channel: Optional[OutputChannel] = None,
        sender_id: Optional[Text] = DEFAULT_SENDER_ID,
    ) -> Optional[List[Dict[Text, Any]]]:
        """Handle a single message.

        If a message preprocessor is passed, the message will be passed to that
        function first and the return value is then used as the
        input for the dialogue engine.

        The return value of this function depends on the ``output_channel``. If
        the output channel is not set, set to ``None``, or set
        to ``CollectingOutputChannel`` this function will return the messages
        the bot wants to respond.
        """
        if isinstance(text_message, str):
            text_message = {"text": text_message}

        msg = UserMessage(text_message.get("text"), output_channel, sender_id)

        return await self.handle_message(msg)

    def _set_fingerprint(self, fingerprint: Optional[Text] = None) -> None:

        if fingerprint:
            self.fingerprint = fingerprint
        else:
            self.fingerprint = uuid.uuid4().hex

    @staticmethod
    def _create_lock_store(store: Optional[LockStore]) -> LockStore:
        if store is not None:
            return store

        return InMemoryLockStore()

    def load_model_from_remote_storage(self, model_name: Text) -> None:
        """Loads an Agent from remote storage."""
        from rasa.nlu.persistor import get_persistor

        persistor = get_persistor(self.remote_storage)

        if persistor is not None:
            with TempDirectoryPath(get_temp_dir_name()) as temporary_directory:
                persistor.retrieve(model_name, temporary_directory)
                self.load_model(temporary_directory)

        else:
            raise RasaException(
                f"Persistor not found for remote storage: '{self.remote_storage}'."
            )
