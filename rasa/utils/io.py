import asyncio
import errno
import json
import logging
import os
import pickle
import tarfile
import tempfile
import warnings
import zipfile
from asyncio import AbstractEventLoop
from collections import OrderedDict
from io import BytesIO as IOReader, StringIO
from pathlib import Path
from typing import Text, Any, Dict, Union, List, Type, Callable, TYPE_CHECKING

import ruamel.yaml as yaml
from ruamel.yaml import RoundTripRepresenter

from rasa.constants import ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL
from rasa.shared.utils.io import write_text_file, DEFAULT_ENCODING, read_file, read_yaml

if TYPE_CHECKING:
    from prompt_toolkit.validation import Validator

YAML_LINE_MAX_WIDTH = 4096


def configure_colored_logging(loglevel: Text) -> None:
    import coloredlogs

    loglevel = loglevel or os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)

    field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
    field_styles["asctime"] = {}
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
    level_styles["debug"] = {}
    coloredlogs.install(
        level=loglevel,
        use_chroot=False,
        fmt="%(asctime)s %(levelname)-8s %(name)s  - %(message)s",
        level_styles=level_styles,
        field_styles=field_styles,
    )


def enable_async_loop_debugging(
    event_loop: AbstractEventLoop, slow_callback_duration: float = 0.1
) -> AbstractEventLoop:
    logging.info(
        "Enabling coroutine debugging. Loop id {}.".format(id(asyncio.get_event_loop()))
    )

    # Enable debugging
    event_loop.set_debug(True)

    # Make the threshold for "slow" tasks very very small for
    # illustration. The default is 0.1 (= 100 milliseconds).
    event_loop.slow_callback_duration = slow_callback_duration

    # Report all mistakes managing asynchronous resources.
    warnings.simplefilter("always", ResourceWarning)
    return event_loop


def dump_obj_as_json_to_file(filename: Union[Text, Path], obj: Any) -> None:
    """Dump an object as a json string to a file."""

    write_text_file(json.dumps(obj, indent=2), filename)


def pickle_dump(filename: Union[Text, Path], obj: Any) -> None:
    """Saves object to file.

    Args:
        filename: the filename to save the object to
        obj: the object to store
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(filename: Union[Text, Path]) -> Any:
    """Loads an object from a file.

    Args:
        filename: the filename to load the object from

    Returns: the loaded object
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def read_config_file(filename: Text) -> Dict[Text, Any]:
    """Parses a yaml configuration file. Content needs to be a dictionary

    Args:
        filename: The path to the file which should be read.
    """
    content = read_yaml(read_file(filename))

    if content is None:
        return {}
    elif isinstance(content, dict):
        return content
    else:
        raise ValueError(
            "Tried to load invalid config file '{}'. "
            "Expected a key value mapping but found {}"
            ".".format(filename, type(content))
        )


def unarchive(byte_array: bytes, directory: Text) -> Text:
    """Tries to unpack a byte array interpreting it as an archive.

    Tries to use tar first to unpack, if that fails, zip will be used."""

    try:
        tar = tarfile.open(fileobj=IOReader(byte_array))
        tar.extractall(directory)
        tar.close()
        return directory
    except tarfile.TarError:
        zip_ref = zipfile.ZipFile(IOReader(byte_array))
        zip_ref.extractall(directory)
        zip_ref.close()
        return directory


def convert_to_ordered_dict(obj: Any) -> Any:
    """Convert object to an `OrderedDict`.

    Args:
        obj: Object to convert.

    Returns:
        An `OrderedDict` with all nested dictionaries converted if `obj` is a
        dictionary, otherwise the object itself.
    """
    if isinstance(obj, OrderedDict):
        return obj
    # use recursion on lists
    if isinstance(obj, list):
        return [convert_to_ordered_dict(element) for element in obj]

    if isinstance(obj, dict):
        out = OrderedDict()
        # use recursion on dictionaries
        for k, v in obj.items():
            out[k] = convert_to_ordered_dict(v)

        return out

    # return all other objects
    return obj


def _enable_ordered_dict_yaml_dumping() -> None:
    """Ensure that `OrderedDict`s are dumped so that the order of keys is respected."""
    yaml.add_representer(
        OrderedDict,
        RoundTripRepresenter.represent_dict,
        representer=RoundTripRepresenter,
    )


def write_yaml(
    data: Any,
    target: Union[Text, Path, StringIO],
    should_preserve_key_order: bool = False,
) -> None:
    """Writes a yaml to the file or to the stream

    Args:
        data: The data to write.
        target: The path to the file which should be written or a stream object
        should_preserve_key_order: Whether to force preserve key order in `data`.
    """
    _enable_ordered_dict_yaml_dumping()

    if should_preserve_key_order:
        data = convert_to_ordered_dict(data)

    dumper = yaml.YAML()
    # no wrap lines
    dumper.width = YAML_LINE_MAX_WIDTH

    # use `null` to represent `None`
    dumper.representer.add_representer(
        type(None),
        lambda self, _: self.represent_scalar("tag:yaml.org,2002:null", "null"),
    )

    if isinstance(target, StringIO):
        dumper.dump(data, target)
        return

    with Path(target).open("w", encoding=DEFAULT_ENCODING) as outfile:
        dumper.dump(data, outfile)


def is_subdirectory(path: Text, potential_parent_directory: Text) -> bool:
    if path is None or potential_parent_directory is None:
        return False

    path = os.path.abspath(path)
    potential_parent_directory = os.path.abspath(potential_parent_directory)

    return potential_parent_directory in path


def create_temporary_file(data: Any, suffix: Text = "", mode: Text = "w+") -> Text:
    """Creates a tempfile.NamedTemporaryFile object for data.

    mode defines NamedTemporaryFile's  mode parameter in py3."""

    encoding = None if "b" in mode else DEFAULT_ENCODING
    f = tempfile.NamedTemporaryFile(
        mode=mode, suffix=suffix, delete=False, encoding=encoding
    )
    f.write(data)

    f.close()
    return f.name


def create_temporary_directory() -> Text:
    """Creates a tempfile.TemporaryDirectory."""
    f = tempfile.TemporaryDirectory()
    return f.name


def create_path(file_path: Text) -> None:
    """Makes sure all directories in the 'file_path' exists."""

    parent_dir = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)


def create_directory_for_file(file_path: Union[Text, Path]) -> None:
    """Creates any missing parent directories of this file path."""

    create_directory(os.path.dirname(file_path))


def file_type_validator(
    valid_file_types: List[Text], error_message: Text
) -> Type["Validator"]:
    """Creates a `Validator` class which can be used with `questionary` to validate
       file paths.
    """

    def is_valid(path: Text) -> bool:
        return path is not None and any(
            [path.endswith(file_type) for file_type in valid_file_types]
        )

    return create_validator(is_valid, error_message)


def not_empty_validator(error_message: Text) -> Type["Validator"]:
    """Creates a `Validator` class which can be used with `questionary` to validate
    that the user entered something other than whitespace.
    """

    def is_valid(input: Text) -> bool:
        return input is not None and input.strip() != ""

    return create_validator(is_valid, error_message)


def create_validator(
    function: Callable[[Text], bool], error_message: Text
) -> Type["Validator"]:
    """Helper method to create `Validator` classes from callable functions. Should be
    removed when questionary supports `Validator` objects."""

    from prompt_toolkit.validation import Validator, ValidationError
    from prompt_toolkit.document import Document

    class FunctionValidator(Validator):
        @staticmethod
        def validate(document: Document) -> None:
            is_valid = function(document.text)
            if not is_valid:
                raise ValidationError(message=error_message)

    return FunctionValidator


def create_directory(directory_path: Text) -> None:
    """Creates a directory and its super paths.

    Succeeds even if the path already exists."""

    try:
        os.makedirs(directory_path)
    except OSError as e:
        # be happy if someone already created the path
        if e.errno != errno.EEXIST:
            raise


def zip_folder(folder: Text) -> Text:
    """Create an archive from a folder."""
    import shutil

    zipped_path = tempfile.NamedTemporaryFile(delete=False)
    zipped_path.close()

    # WARN: not thread-safe!
    return shutil.make_archive(zipped_path.name, "zip", folder)


def json_unpickle(file_name: Union[Text, Path]) -> Any:
    """Unpickle an object from file using json.

    Args:
        file_name: the file to load the object from

    Returns: the object
    """
    import jsonpickle.ext.numpy as jsonpickle_numpy
    import jsonpickle

    jsonpickle_numpy.register_handlers()

    file_content = read_file(file_name)
    return jsonpickle.loads(file_content)


def json_pickle(file_name: Union[Text, Path], obj: Any) -> None:
    """Pickle an object to a file using json.

    Args:
        file_name: the file to store the object to
        obj: the object to store
    """
    import jsonpickle.ext.numpy as jsonpickle_numpy
    import jsonpickle

    jsonpickle_numpy.register_handlers()

    write_text_file(jsonpickle.dumps(obj), file_name)
