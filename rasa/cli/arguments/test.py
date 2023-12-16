import argparse
from typing import Union

from rasa.shared.constants import DEFAULT_MODELS_PATH, DEFAULT_RESULTS_PATH

from rasa.cli.arguments.default_arguments import (
    add_model_param,
    add_nlu_data_param,
    add_out_param,
)
from rasa.model import get_latest_model
from rasa.shared.constants import DEFAULT_DOMAIN_PATH


def set_test_arguments(parser: argparse.ArgumentParser) -> None:
    """Sets test arguments for a parser."""
    add_model_param(parser, add_positional_arg=False)

    nlu_arguments = parser.add_argument_group("NLU Test Arguments")
    add_test_nlu_argument_group(nlu_arguments)

    add_no_plot_param(parser)
    add_errors_success_params(parser)
    add_out_param(
        parser,
        default=DEFAULT_RESULTS_PATH,
        help_text="Output path for any files created during the evaluation.",
    )


def set_test_nlu_arguments(parser: argparse.ArgumentParser) -> None:
    add_model_param(parser, add_positional_arg=False)
    add_test_nlu_argument_group(parser)


def add_test_nlu_argument_group(
    parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]
) -> None:
    add_nlu_data_param(parser, help_text="File or folder containing your NLU data.")

    add_out_param(
        parser,
        default=DEFAULT_RESULTS_PATH,
        help_text="Output path for any files created during the evaluation.",
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="+",
        default=None,
        help="Model configuration file. If a single file is passed and cross "
        "validation mode is chosen, cross-validation is performed, if "
        "multiple configs or a folder of configs are passed, models "
        "will be trained and compared directly.",
    )

    parser.add_argument(
        "-d",
        "--domain",
        type=str,
        default=DEFAULT_DOMAIN_PATH,
        help="Domain specification. This can be a single YAML file, or a directory "
        "that contains several files with domain specifications in it. The content "
        "of these files will be read and merged together.",
    )

    cross_validation_arguments = parser.add_argument_group("Cross Validation")
    cross_validation_arguments.add_argument(
        "--cross-validation",
        action="store_true",
        default=False,
        help="Switch on cross validation mode. Any provided model will be ignored.",
    )
    cross_validation_arguments.add_argument(
        "-f",
        "--folds",
        required=False,
        default=5,
        help="Number of cross validation folds (cross validation only).",
    )
    comparison_arguments = parser.add_argument_group("Comparison Mode")
    comparison_arguments.add_argument(
        "-r",
        "--runs",
        required=False,
        default=3,
        type=int,
        help="Number of comparison runs to make.",
    )
    comparison_arguments.add_argument(
        "-p",
        "--percentages",
        required=False,
        nargs="+",
        type=int,
        default=[0, 25, 50, 75],
        help="Percentages of training data to exclude during comparison.",
    )

    add_no_plot_param(parser)
    add_errors_success_params(parser)


def add_no_plot_param(
    parser: argparse.ArgumentParser, default: bool = False, required: bool = False
) -> None:
    parser.add_argument(
        "--no-plot",
        dest="disable_plotting",
        action="store_true",
        default=default,
        help="Don't render evaluation plots.",
        required=required,
    )


def add_errors_success_params(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--successes",
        action="store_true",
        default=False,
        help="If set successful predictions will be written to a file.",
    )
    parser.add_argument(
        "--no-errors",
        action="store_true",
        default=False,
        help="If set incorrect predictions will NOT be written to a file.",
    )
    parser.add_argument(
        "--no-warnings",
        action="store_true",
        default=False,
        help="If set prediction warnings will NOT be written to a file.",
    )
