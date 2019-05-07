import logging
import os
from typing import Any, Callable, Dict, List, Text, Optional, Union

import rasa.core.utils
import rasa.utils.io
from rasa.constants import GLOBAL_USER_CONFIG_PATH, DEFAULT_LOG_LEVEL, ENV_LOG_LEVEL


def arguments_of(func: Callable) -> List[Text]:
    """Return the parameters of the function `func` as a list of names."""
    import inspect

    return list(inspect.signature(func).parameters.keys())


def read_global_config() -> Dict[Text, Any]:
    """Read global Rasa configuration."""
    # noinspection PyBroadException
    try:
        return rasa.utils.io.read_yaml_file(GLOBAL_USER_CONFIG_PATH)
    except Exception:
        # if things go south we pretend there is no config
        return {}


def write_global_config_value(name: Text, value: Any) -> None:
    """Read global Rasa configuration."""

    os.makedirs(os.path.dirname(GLOBAL_USER_CONFIG_PATH), exist_ok=True)

    c = read_global_config()
    c[name] = value
    rasa.core.utils.dump_obj_as_yaml_to_file(GLOBAL_USER_CONFIG_PATH, c)


def read_global_config_value(name: Text, unavailable_ok: bool = True) -> Any:
    """Read a value from the global Rasa configuration."""

    def not_found():
        if unavailable_ok:
            return None
        else:
            raise ValueError("Configuration '{}' key not found.".format(name))

    if not os.path.exists(GLOBAL_USER_CONFIG_PATH):
        return not_found()

    c = read_global_config()

    if name in c:
        return c[name]
    else:
        return not_found()


def set_log_level(log_level: Optional[Any] = None):
    import logging

    if not log_level:
        log_level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)
        log_level = logging.getLevelName(log_level)

    logging.getLogger(__name__).setLevel(log_level)

    set_tensorflow_log_level(log_level)
    set_sanic_log_level(log_level)

    os.environ[ENV_LOG_LEVEL] = logging.getLevelName(log_level)


def set_tensorflow_log_level(log_level: logging):
    import tensorflow as tf

    tf_log_level = tf.logging.INFO
    if log_level == logging.DEBUG:
        tf_log_level = tf.logging.DEBUG
    if log_level == logging.WARNING:
        tf_log_level = tf.logging.WARN
    if log_level == logging.ERROR:
        tf_log_level = tf.logging.ERROR

    tf.logging.set_verbosity(tf_log_level)


def set_sanic_log_level(log_level):
    from sanic.log import logger as sanic_logger

    sanic_logger.setLevel(log_level)


def obtain_verbosity():
    log_level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)

    verbosity = 0
    if log_level == "DEBUG":
        verbosity = 2
    if log_level == "INFO":
        verbosity = 1

    return verbosity


def disable_logging():
    log_level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)

    return log_level == "ERROR" or log_level == "WARNING"
