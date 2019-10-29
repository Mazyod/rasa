import json
import logging
import typing
import os
from threading import Thread
from typing import Dict, Optional, Text, Union, List, Callable

import time

import rasa.core.brokers.utils as rasa_broker_utils
from rasa.constants import ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES
from rasa.core.brokers.event_channel import EventChannel
from rasa.utils.endpoints import EndpointConfig

if typing.TYPE_CHECKING:
    from pika.adapters.blocking_connection import BlockingChannel
    from pika import SelectConnection, BlockingConnection
    from pika.channel import Channel
    from pika.connection import Parameters, Connection

logger = logging.getLogger(__name__)


def initialise_pika_connection(
    host: Text,
    username: Text,
    password: Text,
    port: Union[Text, int] = 5672,
    connection_attempts: int = 20,
    retry_delay_in_seconds: Union[int, float] = 5,
) -> "BlockingConnection":
    """Create a Pika `BlockingConnection`.

    Args:
        host: Pika host
        username: username for authentication with Pika host
        password: password for authentication with Pika host
        port: port of the Pika host
        connection_attempts: number of channel attempts before giving up
        retry_delay_in_seconds: delay in seconds between channel attempts

    Returns:
        Pika `BlockingConnection` with provided parameters

    """

    import pika

    parameters = _get_pika_parameters(
        host, username, password, port, connection_attempts, retry_delay_in_seconds
    )
    return pika.BlockingConnection(parameters)


def _get_pika_parameters(
    host: Text,
    username: Text,
    password: Text,
    port: Union[Text, int] = 5672,
    connection_attempts: int = 20,
    retry_delay_in_seconds: Union[int, float] = 5,
) -> "Parameters":
    """Create Pika `Parameters`.

    Args:
        host: Pika host
        username: username for authentication with Pika host
        password: password for authentication with Pika host
        port: port of the Pika host
        connection_attempts: number of channel attempts before giving up
        retry_delay_in_seconds: delay in seconds between channel attempts

    Returns:
        Pika `Paramaters` which can be used to create a new connection to a broker.
    """

    import pika

    if host.startswith("amqp"):
        # user supplied a amqp url containing all the info
        parameters = pika.URLParameters(host)
        parameters.connection_attempts = connection_attempts
        parameters.retry_delay = retry_delay_in_seconds
        if username:
            parameters.credentials = pika.PlainCredentials(username, password)
    else:
        # host seems to be just the host, so we use our parameters
        parameters = pika.ConnectionParameters(
            host,
            port=port,
            credentials=pika.PlainCredentials(username, password),
            connection_attempts=connection_attempts,
            # Wait between retries since
            # it can take some time until
            # RabbitMQ comes up.
            retry_delay=retry_delay_in_seconds,
            ssl_options=rasa_broker_utils.create_rabbitmq_ssl_options(host),
        )

    return parameters


def initialise_pika_select_connection(
    parameters: "Parameters",
    on_open_callback: Callable[["SelectConnection"], None],
    on_open_error_callback: Callable[["SelectConnection", Text], None],
) -> "SelectConnection":
    """Create a non-blocking Pika `SelectConnection`.

    Args:
        parameters: Parameters which should be used to connect.
        on_open_callback: Callback which is called when the connection was established.
        on_open_error_callback: Callback which is called when connecting to the broker
            failed.

    Returns:
        An callback based connection to the RabbitMQ event broker.
    """

    import pika

    return pika.SelectConnection(
        parameters,
        on_open_callback=on_open_callback,
        on_open_error_callback=on_open_error_callback,
    )


def initialise_pika_channel(
    host: Text,
    queue: Text,
    username: Text,
    password: Text,
    port: Union[Text, int] = 5672,
    connection_attempts: int = 20,
    retry_delay_in_seconds: Union[int, float] = 5,
) -> "BlockingChannel":
    """Initialise a Pika channel with a durable queue.

    Args:
        host: Pika host
        queue: Pika queue to declare
        username: username for authentication with Pika host
        password: password for authentication with Pika host
        port: port of the Pika host
        connection_attempts: number of channel attempts before giving up
        retry_delay_in_seconds: delay in seconds between channel attempts

    Returns:
        Pika `BlockingChannel` with declared queue

    """

    connection = initialise_pika_connection(
        host, username, password, port, connection_attempts, retry_delay_in_seconds
    )

    return _declare_pika_channel_with_queue(connection, queue)


def _declare_pika_channel_with_queue(
    connection: "BlockingConnection", queue: Text
) -> "BlockingChannel":
    """Declare a durable queue on Pika channel."""

    channel = connection.channel()
    channel.queue_declare(queue, durable=True)

    return channel


def close_pika_channel(channel: "Channel") -> None:
    """Attempt to close Pika channel."""

    from pika.exceptions import AMQPError

    try:
        channel.close()
        logger.debug("Successfully closed Pika channel.")
    except AMQPError:
        logger.exception("Failed to close Pika channel.")


def close_pika_connection(connection: "Connection") -> None:
    """Attempt to close Pika connection."""

    from pika.exceptions import AMQPError

    try:
        connection.close()
        logger.debug("Successfully closed Pika connection with host.")
    except AMQPError:
        logger.exception("Failed to close Pika connection with host.")


class PikaProducer(EventChannel):
    def __init__(
        self,
        host: Text,
        username: Text,
        password: Text,
        port: Union[int, Text] = 5672,
        queue: Text = "rasa_core_events",
        loglevel: Union[Text, int] = os.environ.get(
            ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES
        ),
    ):
        logging.getLogger("pika").setLevel(loglevel)

        self.queue = queue
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.channel: Optional["Channel"] = None

        # List to store unpublished messages which hopefully will be published later
        self._unpublished_messages: List[Text] = []
        self._run_pika()

    def __del__(self) -> None:
        if self.channel:
            close_pika_channel(self.channel)
            close_pika_connection(self.channel.connection)

    @classmethod
    def from_endpoint_config(
        cls, broker_config: Optional["EndpointConfig"]
    ) -> Optional["PikaProducer"]:
        if broker_config is None:
            return None

        return cls(broker_config.url, **broker_config.kwargs)

    def _run_pika(self) -> None:
        parameters = _get_pika_parameters(
            self.host, self.username, self.password, self.port
        )
        self._pika_connection = initialise_pika_select_connection(
            parameters, self._on_open_connection, self._on_open_connection_error
        )
        # Run Pika io loop in extra thread so it's not blocking
        self._run_pika_io_loop_in_thread()

    def _on_open_connection(self, connection: "SelectConnection") -> None:
        logger.debug(f"RabbitMQ connection to '{self.host}' was established.")
        connection.channel(on_open_callback=self._on_channel_open)

    def _on_open_connection_error(self, _, error: Text) -> None:
        logger.warning(
            f"Connecting to '{self.host}' failed with error '{error}'. Trying again."
        )

    def _on_channel_open(self, channel: "Channel") -> None:
        logger.debug("RabbitMQ channel was opened.")
        channel.queue_declare(self.queue, durable=True)

        self.channel = channel

        while self._unpublished_messages:
            # Send unpublished messages
            message = self._unpublished_messages.pop()
            self._publish(message)

    def _run_pika_io_loop_in_thread(self) -> None:
        thread = Thread(target=self._run_pika_io_loop, daemon=True)
        thread.start()

    def _run_pika_io_loop(self) -> None:
        self._pika_connection.ioloop.start()

    def publish(
        self, event: Dict, retries: int = 60, retry_delay_in_seconds: int = 5
    ) -> None:
        """Publish `event` into Pika queue.

        Perform `retries` publish attempts with `retry_delay_in_seconds` between them.
        """

        body = json.dumps(event)

        while retries:
            # noinspection PyBroadException
            try:
                self._publish(body)
                return
            except Exception as e:
                logger.error(
                    "Could not open Pika channel at host '{}'. Failed with error: "
                    "{}".format(self.host, e)
                )
                self.channel = None

            retries -= 1
            time.sleep(retry_delay_in_seconds)

        logger.error(
            "Failed to publish Pika event to queue '{}' on host "
            "'{}':\n{}".format(self.queue, self.host, body)
        )

    def _publish(self, body: Text) -> None:
        if self._pika_connection.is_closed:
            # Try to reset connection
            self._run_pika()
        elif not self.channel:
            logger.warning(
                "RabbitMQ channel has not been assigned. Adding message to "
                "list of unpublished messages and trying to publish them "
                "later."
            )
            self._unpublished_messages.append(body)
        else:
            self.channel.basic_publish("", self.queue, body)

            logger.debug(
                f"Published Pika events to queue '{self.queue}' on host "
                f"'{self.host}':\n{body}"
            )
