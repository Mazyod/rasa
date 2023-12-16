import json
import logging
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Set, Text, Tuple, Union

import numpy as np

import rasa.shared.utils.io
from rasa.shared.constants import TCP_PROTOCOL

from socket import SOCK_DGRAM, SOCK_STREAM


logger = logging.getLogger(__name__)


def configure_file_logging(
    logger_obj: logging.Logger,
    log_file: Optional[Text],
    use_syslog: Optional[bool],
    syslog_address: Optional[Text] = None,
    syslog_port: Optional[int] = None,
    syslog_protocol: Optional[Text] = None,
) -> None:
    """Configure logging to a file.

    Args:
        logger_obj: Logger object to configure.
        log_file: Path of log file to write to.
        use_syslog: Add syslog as a logger.
        syslog_address: Adress of the syslog server.
        syslog_port: Port of the syslog server.
        syslog_protocol: Protocol with the syslog server
    """
    if use_syslog:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-5.5s] [%(process)d]" " %(message)s"
        )
        socktype = SOCK_STREAM if syslog_protocol == TCP_PROTOCOL else SOCK_DGRAM
        syslog_handler = logging.handlers.SysLogHandler(
            address=(syslog_address, syslog_port), socktype=socktype
        )
        syslog_handler.setLevel(logger_obj.level)
        syslog_handler.setFormatter(formatter)
        logger_obj.addHandler(syslog_handler)
    if log_file:
        formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        file_handler = logging.FileHandler(
            log_file, encoding=rasa.shared.utils.io.DEFAULT_ENCODING
        )
        file_handler.setLevel(logger_obj.level)
        file_handler.setFormatter(formatter)
        logger_obj.addHandler(file_handler)


def one_hot(hot_idx: int, length: int, dtype: Optional[Text] = None) -> np.ndarray:
    """Create a one-hot array.

    Args:
        hot_idx: Index of the hot element.
        length: Length of the array.
        dtype: ``numpy.dtype`` of the array.

    Returns:
        One-hot array.
    """
    if hot_idx >= length:
        raise ValueError(
            "Can't create one hot. Index '{}' is out "
            "of range (length '{}')".format(hot_idx, length)
        )
    r = np.zeros(length, dtype)
    r[hot_idx] = 1
    return r


def dump_obj_as_yaml_to_file(
    filename: Union[Text, Path], obj: Any, should_preserve_key_order: bool = False
) -> None:
    """Writes `obj` to the filename in YAML repr.

    Args:
        filename: Target filename.
        obj: Object to dump.
        should_preserve_key_order: Whether to preserve key order in `obj`.
    """
    rasa.shared.utils.io.write_yaml(
        obj, filename, should_preserve_key_order=should_preserve_key_order
    )


def extract_args(
    kwargs: Dict[Text, Any], keys_to_extract: Set[Text]
) -> Tuple[Dict[Text, Any], Dict[Text, Any]]:
    """Go through the kwargs and filter out the specified keys.

    Return both, the filtered kwargs as well as the remaining kwargs.
    """
    remaining = {}
    extracted = {}
    for k, v in kwargs.items():
        if k in keys_to_extract:
            extracted[k] = v
        else:
            remaining[k] = v

    return extracted, remaining


def is_limit_reached(num_messages: int, limit: Optional[int]) -> bool:
    """Determine whether the number of messages has reached a limit.

    Args:
        num_messages: The number of messages to check.
        limit: Limit on the number of messages.

    Returns:
        `True` if the limit has been reached, otherwise `False`.
    """
    return limit is not None and num_messages >= limit


def file_as_bytes(path: Text) -> bytes:
    """Read in a file as a byte array."""
    with open(path, "rb") as f:
        return f.read()


def replace_floats_with_decimals(obj: Any, round_digits: int = 9) -> Any:
    """Convert all instances in `obj` of `float` to `Decimal`.

    Args:
        obj: Input object.
        round_digits: Rounding precision of `Decimal` values.

    Returns:
        Input `obj` with all `float` types replaced by `Decimal`s rounded to
        `round_digits` decimal places.
    """

    def _float_to_rounded_decimal(s: Text) -> Decimal:
        return Decimal(s).quantize(Decimal(10) ** -round_digits)

    return json.loads(json.dumps(obj), parse_float=_float_to_rounded_decimal)


class DecimalEncoder(json.JSONEncoder):
    """`json.JSONEncoder` that dumps `Decimal`s as `float`s."""

    def default(self, obj: Any) -> Any:
        """Get serializable object for `o`.

        Args:
            obj: Object to serialize.

        Returns:
            `obj` converted to `float` if `o` is a `Decimals`, else the base class
            `default()` method.
        """
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def replace_decimals_with_floats(obj: Any) -> Any:
    """Convert all instances in `obj` of `Decimal` to `float`.

    Args:
        obj: A `List` or `Dict` object.

    Returns:
        Input `obj` with all `Decimal` types replaced by `float`s.
    """
    return json.loads(json.dumps(obj, cls=DecimalEncoder))
