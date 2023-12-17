import argparse
import logging
from typing import List

import rasa.shared.core.domain
from rasa.cli import SubParsersAction
from rasa.cli.arguments import data as arguments
from rasa.cli.arguments import default_arguments
import rasa.cli.utils
from rasa.shared.constants import (
    DEFAULT_DATA_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOMAIN_PATH,
)
import rasa.shared.data
from rasa.shared.importers.importer import TrainingDataImporter
import rasa.shared.nlu.training_data.loading
import rasa.shared.nlu.training_data.util
import rasa.shared.utils.cli
import rasa.utils.common
import rasa.shared.utils.io

logger = logging.getLogger(__name__)


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all data parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    data_parser = subparsers.add_parser(
        "data",
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Utils for the Rasa training files.",
    )
    data_parser.set_defaults(func=lambda _: data_parser.print_help(None))

    data_subparsers = data_parser.add_subparsers()

    _add_data_split_parsers(data_subparsers, parents)


def _add_data_split_parsers(
    data_subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    split_parser = data_subparsers.add_parser(
        "split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Splits Rasa data into training and test data.",
    )
    split_parser.set_defaults(func=lambda _: split_parser.print_help(None))

    split_subparsers = split_parser.add_subparsers()
    nlu_split_parser = split_subparsers.add_parser(
        "nlu",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Performs a split of your NLU data into training and test data "
        "according to the specified percentages.",
    )
    nlu_split_parser.set_defaults(func=split_nlu_data)

    arguments.set_split_arguments(nlu_split_parser)


def _build_training_data_importer(args: argparse.Namespace) -> "TrainingDataImporter":
    config = rasa.cli.utils.get_validated_path(
        args.config, "config", DEFAULT_CONFIG_PATH, none_is_valid=True
    )

    # Exit the validation if the domain path is invalid
    domain = rasa.cli.utils.get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATH, none_is_valid=False
    )

    return TrainingDataImporter.load_from_config(
        domain_path=domain, training_data_paths=args.data, config_path=config
    )


def split_nlu_data(args: argparse.Namespace) -> None:
    """Load data from a file path and split the NLU data into test and train examples.

    Args:
        args: Commandline arguments
    """
    data_path = rasa.cli.utils.get_validated_path(args.nlu, "nlu", DEFAULT_DATA_PATH)
    data_path = rasa.shared.data.get_nlu_directory(data_path)

    nlu_data = rasa.shared.nlu.training_data.loading.load_data(data_path)
    extension = rasa.shared.nlu.training_data.util.get_file_format_extension(data_path)

    train, test = nlu_data.train_test_split(args.training_fraction, args.random_seed)

    train.persist(args.out, filename=f"training_data{extension}")
    test.persist(args.out, filename=f"test_data{extension}")
