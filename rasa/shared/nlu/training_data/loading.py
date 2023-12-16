import json
import logging
import os
import typing
from typing import Optional, Text, Callable, Dict, Any, List

import rasa.shared.utils.io
from rasa.shared.nlu.training_data.training_data import TrainingData

if typing.TYPE_CHECKING:
    from rasa.shared.nlu.training_data.formats.readerwriter import TrainingDataReader

logger = logging.getLogger(__name__)

# Different supported file formats and their identifier
RASA = "rasa_nlu"
RASA_YAML = "rasa_yml"
UNK = "unk"

_json_format_heuristics: Dict[Text, Callable[[Any, Text], bool]] = {
    RASA: lambda js, fn: "rasa_nlu_data" in js,
}


def load_data(resource_name: Text, language: Optional[Text] = "en") -> "TrainingData":
    """Load training data from disk.

    Merges them if loaded from disk and multiple files are found.
    """
    if not os.path.exists(resource_name):
        raise ValueError(f"File '{resource_name}' does not exist.")

    if os.path.isfile(resource_name):
        files = [resource_name]
    else:
        files = rasa.shared.utils.io.list_files(resource_name)

    data_sets = [_load(f, language) for f in files]
    training_data_sets: List[TrainingData] = [ds for ds in data_sets if ds]
    if len(training_data_sets) == 0:
        training_data = TrainingData()
    elif len(training_data_sets) == 1:
        training_data = training_data_sets[0]
    else:
        training_data = training_data_sets[0].merge(*training_data_sets[1:])

    return training_data


def _reader_factory(fformat: Text) -> Optional["TrainingDataReader"]:
    """Generates the appropriate reader class based on the file format."""
    from rasa.shared.nlu.training_data.formats import (
        RasaYAMLReader,
        RasaReader,
    )

    reader: Optional["TrainingDataReader"] = None
    if fformat == RASA:
        reader = RasaReader()
    elif fformat == RASA_YAML:
        reader = RasaYAMLReader()
    return reader


def _load(filename: Text, language: Optional[Text] = "en") -> Optional["TrainingData"]:
    """Loads a single training data file from disk."""
    fformat = guess_format(filename)
    if fformat == UNK:
        raise ValueError(f"Unknown data format for file '{filename}'.")

    reader = _reader_factory(fformat)

    if reader:
        return reader.read(filename, language=language, fformat=fformat)
    else:
        return None


def guess_format(filename: Text) -> Text:
    """Applies heuristics to guess the data format of a file.

    Args:
        filename: file whose type should be guessed

    Returns:
        Guessed file format.
    """
    from rasa.shared.nlu.training_data.formats import RasaYAMLReader

    guess = UNK

    if not os.path.isfile(filename):
        return guess

    try:
        content = rasa.shared.utils.io.read_file(filename)
        js = json.loads(content)
    except ValueError:
        if RasaYAMLReader.is_yaml_nlu_file(filename):
            guess = RASA_YAML
    else:
        for file_format, format_heuristic in _json_format_heuristics.items():
            if format_heuristic(js, filename):
                guess = file_format
                break

    logger.debug(f"Training data format of '{filename}' is '{guess}'.")

    return guess
