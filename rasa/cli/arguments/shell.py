import argparse

from rasa.cli.arguments.default_arguments import add_model_param


def set_shell_arguments(parser: argparse.ArgumentParser) -> None:
    add_model_param(parser)


def set_shell_nlu_arguments(parser: argparse.ArgumentParser) -> None:
    add_model_param(parser)
