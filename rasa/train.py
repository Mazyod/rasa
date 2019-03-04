import os
import tempfile
from typing import Text, Optional

from rasa import model
from rasa.cli.utils import create_output_path
from rasa.constants import DEFAULT_MODELS_PATH


def train(domain: Text, config: Text, stories: Text, nlu_data: Text,
          output: Text = DEFAULT_MODELS_PATH) -> Optional[Text]:
    """Trains a Rasa model (Core and NLU).

    Args:
        domain: Path to the domain file.
        config: Path to the config for Core and NLU.
        stories: Path to the Core training data.
        nlu_data: Path to the NLU training data.
        output: Output path.

    Returns:
        Path of the trained model archive.
    """
    from rasa_core.utils import print_success

    train_path = tempfile.mkdtemp()
    old_model = model.get_latest_model(output)
    retrain_core = True
    retrain_nlu = True

    new_fingerprint = model.model_fingerprint(config, domain, nlu_data, stories)
    if old_model:
        unpacked = model.unpack_model(old_model)
        old_core, old_nlu = model.get_model_subdirectories(unpacked)
        last_fingerprint = model.fingerprint_from_path(unpacked)

        if not model.core_fingerprint_changed(last_fingerprint,
                                              new_fingerprint):
            target_path = os.path.join(train_path, "rasa_model", "core")
            retrain_core = not model.merge_model(old_core, target_path)

        if not model.nlu_fingerprint_changed(last_fingerprint, new_fingerprint):
            target_path = os.path.join(train_path, "rasa_model", "nlu")
            retrain_nlu = not model.merge_model(old_nlu, target_path)

    if retrain_core:
        train_core(domain, config, stories, output, train_path)
    else:
        print("Core configuration did not change. No need to retrain "
              "Core model.")

    if retrain_nlu:
        train_nlu(config, nlu_data, output, train_path)
    else:
        print("NLU configuration did not change. No need to retrain NLU model.")

    if retrain_core or retrain_nlu:
        output = create_output_path(output)
        model.create_package_rasa(train_path, "rasa_model", output,
                                  new_fingerprint)

        print("Train path: '{}'.".format(train_path))

        print_success("Your bot is trained and ready to take for a spin!")

        return output
    else:
        print("Nothing changed. You can use the old model: '{}'."
              "".format(old_model))

        return old_model


def train_core(domain: Text, config: Text, stories: Text, output: Text,
               train_path: Optional[Text]) -> Optional[Text]:
    """Trains a Core model.

    Args:
        domain: Path to the domain file.
        config: Path to the config file for Core.
        stories: Path to the Core training data.
        output: Output path.
        train_path: If `None` the model will be trained in a temporary
            directory, otherwise in the provided directory.

    Returns:
        If `train_path` is given it returns the path to the model archive,
        otherwise the path to the directory with the trained model files.

    """
    import rasa_core.train
    from rasa_core.utils import print_success

    # normal (not compare) training
    core_model = rasa_core.train(domain_file=domain, stories_file=stories,
                                 output_path=os.path.join(train_path,
                                                          "rasa_model", "core"),
                                 policy_config=config)

    if not train_path:
        # Only Core was trained.
        output_path = create_output_path(output, prefix="core-")
        new_fingerprint = model.model_fingerprint(config, domain,
                                                  stories=stories)
        model.create_package_rasa(train_path, "rasa_model", output_path,
                                  new_fingerprint)
        print_success("Your Rasa Core model is trained and saved at '{}'."
                      "".format(output_path))

    return core_model


def train_nlu(config: Text, nlu_data: Text, output: Text,
              train_path: Optional[Text]) -> Optional["Interpreter"]:
    """Trains a NLU model.

    Args:
        config: Path to the config file for NLU.
        nlu_data: Path to the NLU training data.
        output: Output path.
        train_path: If `None` the model will be trained in a temporary
            directory, otherwise in the provided directory.

    Returns:
        If `train_path` is given it returns the path to the model archive,
        otherwise the path to the directory with the trained model files.

    """
    import rasa_nlu
    from rasa_core.utils import print_success

    _train_path = train_path or tempfile.mkdtemp()
    _, nlu_model, _ = rasa_nlu.train(config, nlu_data, _train_path,
                                     project="rasa_model",
                                     fixed_model_name="nlu")

    if not train_path:
        output_path = create_output_path(output, prefix="nlu-")
        new_fingerprint = model.model_fingerprint(config, nlu_data=nlu_data)
        model.create_package_rasa(_train_path, "rasa_model", output_path,
                                  new_fingerprint)
        print_success("Your Rasa NLU model is trained and saved at '{}'."
                      "".format(output_path))

    return nlu_model
