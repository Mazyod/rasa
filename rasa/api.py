from typing import Any, Text, Dict, Union, List, Optional, TYPE_CHECKING

import rasa.shared.constants

# WARNING: Be careful about adding any top level imports at this place!
#   These functions are imported in `rasa.__init__` and any top level import
#   added here will get executed as soon as someone runs `import rasa`.
#   Some imports are very slow (e.g. `tensorflow`) and we want them to get
#   imported when running `import rasa`. If you add more imports here,
#   please check that in the chain you are importing, no slow packages
#   are getting imported.

if TYPE_CHECKING:
    from rasa.model_training import TrainingResult


def train(
    domain: "Text",
    config: "Text",
    training_files: "Union[Text, List[Text]]",
    output: "Text" = rasa.shared.constants.DEFAULT_MODELS_PATH,
    dry_run: bool = False,
    force_training: bool = False,
    fixed_model_name: "Optional[Text]" = None,
    persist_nlu_training_data: bool = False,
    nlu_additional_arguments: "Optional[Dict]" = None,
    model_to_finetune: "Optional[Text]" = None,
    finetuning_epoch_fraction: float = 1.0,
) -> "TrainingResult":
    """Runs Rasa Core and NLU training in `async` loop.

    Args:
        domain: Path to the domain file.
        config: Path to the config for Core and NLU.
        training_files: Paths to the training data for Core and NLU.
        output: Output path.
        dry_run: If `True` then no training will be done, and the information about
            whether the training needs to be done will be printed.
        force_training: If `True` retrain model even if data has not changed.
        fixed_model_name: Name of model to be stored.
        persist_nlu_training_data: `True` if the NLU training data should be persisted
            with the model.
        nlu_additional_arguments: Additional training parameters forwarded to training
            method of each NLU component.
        model_to_finetune: Optional path to a model which should be finetuned or
            a directory in case the latest trained model should be used.
        finetuning_epoch_fraction: The fraction currently specified training epochs
            in the model configuration which should be used for finetuning.

    Returns:
        An instance of `TrainingResult`.
    """
    from rasa.model_training import train

    return train(
        domain=domain,
        config=config,
        training_files=training_files,
        output=output,
        dry_run=dry_run,
        force_training=force_training,
        fixed_model_name=fixed_model_name,
        persist_nlu_training_data=persist_nlu_training_data,
        nlu_additional_arguments=nlu_additional_arguments,
        model_to_finetune=model_to_finetune,
        finetuning_epoch_fraction=finetuning_epoch_fraction,
    )


def test(
    model: "Text",
    nlu_data: "Text",
    output: "Text" = rasa.shared.constants.DEFAULT_RESULTS_PATH,
    additional_arguments: "Optional[Dict]" = None,
) -> None:
    """Test a Rasa model against a set of test data.

    Args:
        model: model to test
        nlu_data: path to the NLU test data
        output: path to folder where all output will be stored
        additional_arguments: additional arguments for the test call
    """
    from rasa.model_testing import test_nlu

    if additional_arguments is None:
        additional_arguments = {}

    test_nlu(model, nlu_data, output, additional_arguments)  # type: ignore[unused-coroutine] # noqa: E501
