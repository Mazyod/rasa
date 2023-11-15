import os

import argparse
from typing import Union

from rasa.cli.arguments.default_arguments import add_model_param, add_endpoint_param
from rasa.core import constants


def set_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments for running Rasa directly using `rasa run`."""
    add_model_param(parser)
    add_server_arguments(parser)


def add_interface_argument(
    parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup]
) -> None:
    """Binds the RASA process to a network interface."""
    parser.add_argument(
        "-i",
        "--interface",
        default=constants.DEFAULT_SERVER_INTERFACE,
        type=str,
        help="Network interface to run the server on.",
    )


# noinspection PyProtectedMember
def add_port_argument(
    parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup]
) -> None:
    """Add an argument for port."""
    parser.add_argument(
        "-p",
        "--port",
        default=constants.DEFAULT_SERVER_PORT,
        type=int,
        help="Port to run the server at.",
    )


def add_server_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for running API endpoint."""
    parser.add_argument(
        "--log-file",
        type=str,
        # Rasa should not log to a file by default, otherwise there will be problems
        # when running on OpenShift
        default=None,
        help="Store logs in specified file.",
    )
    parser.add_argument(
        "--use-syslog", action="store_true", help="Add syslog as a log handler"
    )
    parser.add_argument(
        "--syslog-address",
        type=str,
        default=constants.DEFAULT_SYSLOG_HOST,
        help="Address of the syslog server. --use-sylog flag is required",
    )
    parser.add_argument(
        "--syslog-port",
        type=int,
        default=constants.DEFAULT_SYSLOG_PORT,
        help="Port of the syslog server. --use-sylog flag is required",
    )
    parser.add_argument(
        "--syslog-protocol",
        type=str,
        default=constants.DEFAULT_PROTOCOL,
        help="Protocol used with the syslog server. Can be UDP (default) or TCP ",
    )
    add_endpoint_param(
        parser,
        help_text="Configuration file for the model server and the connectors as a "
        "yml file.",
    )
