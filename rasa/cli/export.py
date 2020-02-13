import logging

import argparse
from typing import List, Text, Optional

import rasa.cli.utils as cli_utils
from rasa.cli.arguments import export as arguments
from rasa.constants import DEFAULT_ENDPOINTS_PATH
from rasa.core.brokers.broker import EventBroker
from rasa.core.brokers.pika import PikaEventBroker
from rasa.core.migrate import Migrator
from rasa.core.tracker_store import TrackerStore
from rasa.core.utils import AvailableEndpoints
from rasa.exceptions import (
    PublishingError,
    RasaException,
)

logger = logging.getLogger(__name__)


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    export_parser_args = {
        "parents": parents,
        "conflict_handler": "resolve",
        "formatter_class": argparse.ArgumentDefaultsHelpFormatter,
        "help": "Export Rasa trackers using an event broker.",
    }

    shell_parser = subparsers.add_parser("export", **export_parser_args)
    shell_parser.set_defaults(func=export_trackers)

    arguments.set_export_arguments(shell_parser)


def _get_tracker_store(endpoints: AvailableEndpoints) -> TrackerStore:
    """Get tracker store from `endpoints`.

    Prints an error and exits if no tracker store could be loaded.

    Args:
        endpoints: `AvailableEndpoints` to initialize the tracker store from.

    Returns:
        Initialized tracker store.

    """
    if not endpoints.tracker_store:
        cli_utils.print_error_and_exit(
            "Could not find a `tracker_store` section in the supplied "
            "endpoints file. Exiting."
        )

    return TrackerStore.create(endpoints.tracker_store)


def _get_event_broker(endpoints: AvailableEndpoints) -> Optional[EventBroker]:
    """Get event broker from `endpoints`.

    Prints an error and exits if no event broker could be loaded.

    Args:
        endpoints: `AvailableEndpoints` to initialize the event broker from.

    Returns:
        Initialized event broker.

    """
    if not endpoints.event_broker:
        cli_utils.print_error_and_exit(
            "Could not find an `event_broker` section in the supplied "
            "endpoints file. Exiting."
        )

    return EventBroker.create(endpoints.event_broker)


def _get_available_endpoints(endpoints_path: Optional[Text]) -> AvailableEndpoints:
    """Get `AvailableEndpoints` object from specified path.

    Args:
        endpoints_path: Path of the endpoints file to be read. If `None` the
            default path for that file is used (`endpoints.yml`).

    Returns:
        `AvailableEndpoints` object read from endpoints file.

    """
    endpoints_config_path = cli_utils.get_validated_path(
        endpoints_path, "endpoints", DEFAULT_ENDPOINTS_PATH, True
    )
    return AvailableEndpoints.read_endpoints(endpoints_config_path)


def _get_requested_conversation_ids(
    conversation_ids_arg: Optional[Text] = None,
) -> Optional[List[Text]]:
    """Get list of conversation IDs requested as a command-line argument.

    Args:
        conversation_ids_arg: Value of `--conversation-ids` command-line argument.
            If provided, this is a string of comma-separated conversation IDs.

    Return:
        List of conversation IDs requested as a command-line argument.
        `None` if that argument was left unspecified.

    """
    if not conversation_ids_arg:
        return None

    return conversation_ids_arg.split(",")


def _validate_timestamp_options(args: argparse.Namespace) -> None:
    """Inspect CLI timestamp parameters.

    Prints an error and exits if a maximum timestamp is provided that is smaller
    than the provided minimum timestamp.

    Args:
        args: Command-line arguments to process.

    """
    min_timestamp = args.minimum_timestamp
    max_timestamp = args.maximum_timestamp

    if (
        min_timestamp is not None
        and max_timestamp is not None
        and max_timestamp < min_timestamp
    ):
        cli_utils.print_error_and_exit(
            f"Maximum timestamp '{max_timestamp}' is smaller than minimum "
            f"timestamp '{min_timestamp}'. Exiting."
        )


def _prepare_event_broker(event_broker: EventBroker) -> None:
    """Sets `should_keep_unpublished_messages` flag to `False` if
    `self.event_broker` is a `PikaEventBroker`.

    If publishing of events fails, the `PikaEventBroker` instance should not keep a
    list of unpublished messages, so we can retry publishing them. This is because
    the instance is launched as part of this short-lived export script, meaning the
    object is destroyed before it might be published.

    In addition, wait until the event broker reports a `ready` state.

    """
    if isinstance(event_broker, PikaEventBroker):
        event_broker.should_keep_unpublished_messages = False
        event_broker.raise_on_failure = True

    if not event_broker.is_ready():
        cli_utils.print_error_and_exit(
            f"Event broker of type '{type(event_broker)}' is not ready. " f"Exiting."
        )


def export_trackers(args: argparse.Namespace) -> None:
    """Export events for a connected tracker store using an event broker.

    Args:
        args: Command-line arguments to process.

    """
    _validate_timestamp_options(args)

    endpoints = _get_available_endpoints(args.endpoints)
    tracker_store = _get_tracker_store(endpoints)
    event_broker = _get_event_broker(endpoints)
    _prepare_event_broker(event_broker)
    requested_conversation_ids = _get_requested_conversation_ids(args.conversation_ids)

    try:
        migrator = Migrator(
            tracker_store,
            event_broker,
            args.endpoints,
            requested_conversation_ids,
            args.minimum_timestamp,
            args.maximum_timestamp,
        )
        published_events = migrator.publish_events()
        cli_utils.print_success(
            f"Done! Successfully published '{published_events}' events 🎉"
        )

    except PublishingError as e:
        # noinspection PyUnboundLocalVariable
        command = _get_continuation_command(migrator, float(e))
        cli_utils.print_error_and_exit(
            f"Encountered error while publishing event with timestamp '{e}'. To "
            f"continue where I left off, run the following command:"
            f"\n\n\t{command}\n\nExiting."
        )

    except RasaException as e:
        cli_utils.print_error_and_exit(str(e))


def _get_continuation_command(migrator: Migrator, timestamp: float) -> Text:
    """Build CLI command to continue 'rasa export' where it was interrupted.

    Called when event publishing stops due to an error.

    Args:
        migrator: Migrator object containing objects relevant for this export.
        timestamp: Timestamp of the last event attempted to be published.

    """
    # build CLI command command based on supplied timestamp and options
    command = f"rasa export"

    if migrator.endpoints_path is not None:
        command += f" --endpoints {migrator.endpoints_path}"

    command += f" --minimum-timestamp {timestamp}"

    if migrator.maximum_timestamp is not None:
        command += f" --maximum-timestamp {migrator.maximum_timestamp}"

    if migrator.requested_conversation_ids:
        command += (
            f" --conversation-ids {','.join(migrator.requested_conversation_ids)}"
        )

    return command
