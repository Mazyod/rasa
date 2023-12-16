import os

import argparse
from typing import Union

from rasa.cli.arguments.default_arguments import add_model_param
from rasa.core import constants


def set_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments for running Rasa directly using `rasa run`."""
    add_model_param(parser)


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
