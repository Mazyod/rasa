import argparse
import asyncio
import logging
import os
from typing import List, Optional, Text, Dict, Union, Any

from rasa.cli import SubParsersAction
import rasa.shared.data
from rasa.shared.exceptions import YamlException
import rasa.shared.utils.io
import rasa.shared.utils.cli
from rasa.cli.arguments import test as arguments
from rasa.shared.constants import (
    CONFIG_SCHEMA_FILE,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODELS_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_RESULTS_PATH,
)
import rasa.shared.utils.validation as validation_utils
import rasa.cli.utils
import rasa.utils.common
from rasa.shared.importers.importer import TrainingDataImporter

logger = logging.getLogger(__name__)


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all test parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    test_parser = subparsers.add_parser(
        "test",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Tests Rasa models using your test NLU data.",
    )

    arguments.set_test_arguments(test_parser)

    test_subparsers = test_parser.add_subparsers()

    test_nlu_parser = test_subparsers.add_parser(
        "nlu",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Tests Rasa NLU models using your test NLU data.",
    )
    arguments.set_test_nlu_arguments(test_nlu_parser)

    test_nlu_parser.set_defaults(func=run_nlu_test)


async def run_nlu_test_async(
    config: Optional[Union[Text, List[Text]]],
    data_path: Text,
    models_path: Text,
    output_dir: Text,
    cross_validation: bool,
    percentages: List[int],
    runs: int,
    no_errors: bool,
    domain_path: Text,
    all_args: Dict[Text, Any],
) -> None:
    """Runs NLU tests.

    Args:
        all_args: all arguments gathered in a Dict so we can pass it as one argument
                  to other functions.
        config: it refers to the model configuration file. It can be a single file or
                a list of multiple files or a folder with multiple config files inside.
        data_path: path for the nlu data.
        models_path: path to a trained Rasa model.
        output_dir: output path for any files created during the evaluation.
        cross_validation: indicates if it should test the model using cross validation
                          or not.
        percentages: defines the exclusion percentage of the training data.
        runs: number of comparison runs to make.
        domain_path: path to domain.
        no_errors: indicates if incorrect predictions should be written to a file
                   or not.
    """
    from rasa.model_testing import (
        compare_nlu_models,
        perform_nlu_cross_validation,
        test_nlu,
    )

    data_path = str(
        rasa.cli.utils.get_validated_path(data_path, "nlu", DEFAULT_DATA_PATH)
    )
    test_data_importer = TrainingDataImporter.load_from_dict(
        training_data_paths=[data_path], domain_path=domain_path
    )
    nlu_data = test_data_importer.get_nlu_data()

    output = output_dir or DEFAULT_RESULTS_PATH
    all_args["errors"] = not no_errors
    rasa.shared.utils.io.create_directory(output)

    if config is not None and len(config) == 1:
        config = os.path.abspath(config[0])
        if os.path.isdir(config):
            config = rasa.shared.utils.io.list_files(config)

    if isinstance(config, list):
        logger.info(
            "Multiple configuration files specified, running nlu comparison mode."
        )

        config_files = []
        for file in config:
            try:
                validation_utils.validate_yaml_schema(
                    rasa.shared.utils.io.read_file(file), CONFIG_SCHEMA_FILE
                )
                config_files.append(file)
            except YamlException:
                rasa.shared.utils.io.raise_warning(
                    f"Ignoring file '{file}' as it is not a valid config file."
                )
                continue
        await compare_nlu_models(
            configs=config_files,
            test_data=nlu_data,
            output=output,
            runs=runs,
            exclusion_percentages=percentages,
        )
    elif cross_validation:
        logger.info("Test model using cross validation.")
        # FIXME: supporting Union[Path, Text] down the chain
        # is the proper fix and needs more work
        config = str(
            rasa.cli.utils.get_validated_path(config, "config", DEFAULT_CONFIG_PATH)
        )
        config_importer = TrainingDataImporter.load_from_dict(config_path=config)

        config_dict = config_importer.get_config()
        await perform_nlu_cross_validation(config_dict, nlu_data, output, all_args)
    else:
        model_path = rasa.cli.utils.get_validated_path(
            models_path, "model", DEFAULT_MODELS_PATH
        )

        await test_nlu(model_path, data_path, output, all_args, domain_path=domain_path)


def run_nlu_test(args: argparse.Namespace) -> None:
    """Runs NLU tests.

    Args:
        args: the parsed CLI arguments for 'rasa test nlu'.
    """
    asyncio.run(
        run_nlu_test_async(
            args.config,
            args.nlu,
            args.model,
            args.out,
            args.cross_validation,
            args.percentages,
            args.runs,
            args.no_errors,
            args.domain,
            vars(args),
        )
    )
