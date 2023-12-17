from abc import ABC, abstractmethod
from functools import reduce
from typing import Text, Optional, List, Dict, Set, Any, Tuple, Type, Union, cast
import logging

import rasa.shared.constants
import rasa.shared.utils.common
import rasa.shared.core.constants
import rasa.shared.utils.io
from rasa.shared.core.domain import (
    Domain
)
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.core.domain import IS_RETRIEVAL_INTENT_KEY

logger = logging.getLogger(__name__)


class TrainingDataImporter(ABC):
    """Common interface for different mechanisms to load training data."""

    @abstractmethod
    def __init__(
        self,
        config_file: Optional[Text] = None,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[Union[List[Text], Text]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the importer."""
        ...

    @abstractmethod
    def get_domain(self) -> Domain:
        """Retrieves the domain of the bot.

        Returns:
            Loaded `Domain`.
        """
        ...

    @abstractmethod
    def get_config(self) -> Dict:
        """Retrieves the configuration that should be used for the training.

        Returns:
            The configuration as dictionary.
        """
        ...

    @abstractmethod
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        ...

    @abstractmethod
    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Retrieves the NLU training data that should be used for training.

        Args:
            language: Can be used to only load training data for a certain language.

        Returns:
            Loaded NLU `TrainingData`.
        """
        ...

    @staticmethod
    def load_from_config(
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        args: Optional[Dict[Text, Any]] = {},
    ) -> "TrainingDataImporter":
        """Loads a `TrainingDataImporter` instance from a configuration file."""
        config = rasa.shared.utils.io.read_config_file(config_path)
        return TrainingDataImporter.load_from_dict(
            config, config_path, domain_path, training_data_paths, args
        )

    @staticmethod
    def load_core_importer_from_config(
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        args: Optional[Dict[Text, Any]] = {},
    ) -> "TrainingDataImporter":
        """Loads core `TrainingDataImporter` instance.

        Instance loaded from configuration file will only read Core training data.
        """
        importer = TrainingDataImporter.load_from_config(
            config_path, domain_path, training_data_paths, args
        )
        return importer

    @staticmethod
    def load_nlu_importer_from_config(
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        args: Optional[Dict[Text, Any]] = {},
    ) -> "TrainingDataImporter":
        """Loads nlu `TrainingDataImporter` instance.

        Instance loaded from configuration file will only read NLU training data.
        """
        importer = TrainingDataImporter.load_from_config(
            config_path, domain_path, training_data_paths, args
        )

        return NluDataImporter(importer)

    @staticmethod
    def load_from_dict(
        config: Optional[Dict] = None,
        config_path: Optional[Text] = None,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        args: Optional[Dict[Text, Any]] = {},
    ) -> "TrainingDataImporter":
        """Loads a `TrainingDataImporter` instance from a dictionary."""
        from rasa.shared.importers.rasa import RasaFileImporter

        config = config or {}
        importers = config.get("importers", [])
        importers = [
            TrainingDataImporter._importer_from_dict(
                importer, config_path, domain_path, training_data_paths, args
            )
            for importer in importers
        ]
        importers = [importer for importer in importers if importer]
        if not importers:
            importers = [
                RasaFileImporter(config_path, domain_path, training_data_paths)
            ]

        return ResponsesSyncImporter(CombinedDataImporter(importers))

    @staticmethod
    def _importer_from_dict(
        importer_config: Dict,
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        args: Optional[Dict[Text, Any]] = {},
    ) -> Optional["TrainingDataImporter"]:
        from rasa.shared.importers.rasa import RasaFileImporter

        module_path = importer_config.pop("name", None)
        if module_path == RasaFileImporter.__name__:
            importer_class: Type[TrainingDataImporter] = RasaFileImporter
        else:
            try:
                importer_class = rasa.shared.utils.common.class_from_module_path(
                    module_path
                )
            except (AttributeError, ImportError):
                logging.warning(f"Importer '{module_path}' not found.")
                return None

        constructor_arguments = rasa.shared.utils.common.minimal_kwargs(
            {**importer_config, **(args or {})}, importer_class
        )

        return importer_class(
            config_path,
            domain_path,
            training_data_paths,
            **constructor_arguments,
        )

    def fingerprint(self) -> Text:
        """Returns a random fingerprint as data shouldn't be cached."""
        return rasa.shared.utils.io.random_string(25)

    def __repr__(self) -> Text:
        """Returns text representation of object."""
        return self.__class__.__name__


class NluDataImporter(TrainingDataImporter):
    """Importer that skips any Core-related file reading."""

    def __init__(self, actual_importer: TrainingDataImporter):
        """Initializes the NLUDataImporter."""
        self._importer = actual_importer

    def get_domain(self) -> Domain:
        """Retrieves model domain (see parent class for full docstring)."""
        return Domain.empty()

    def get_config(self) -> Dict:
        """Retrieves model config (see parent class for full docstring)."""
        return self._importer.get_config()

    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Retrieves NLU training data (see parent class for full docstring)."""
        return self._importer.get_nlu_data(language)

    @rasa.shared.utils.common.cached_method
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        return self._importer.get_config_file_for_auto_config()


class CombinedDataImporter(TrainingDataImporter):
    """A `TrainingDataImporter` that combines multiple importers.

    Uses multiple `TrainingDataImporter` instances
    to load the data as if they were a single instance.
    """

    def __init__(self, importers: List[TrainingDataImporter]):
        self._importers = importers

    @rasa.shared.utils.common.cached_method
    def get_config(self) -> Dict:
        """Retrieves model config (see parent class for full docstring)."""
        configs = [importer.get_config() for importer in self._importers]

        return reduce(lambda merged, other: {**merged, **(other or {})}, configs, {})

    @rasa.shared.utils.common.cached_method
    def get_domain(self) -> Domain:
        """Retrieves model domain (see parent class for full docstring)."""
        domains = [importer.get_domain() for importer in self._importers]

        return reduce(
            lambda merged, other: merged.merge(other),
            domains,
            Domain.empty(),
        )

    @rasa.shared.utils.common.cached_method
    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Retrieves NLU training data (see parent class for full docstring)."""
        nlu_data = [importer.get_nlu_data(language) for importer in self._importers]

        return reduce(
            lambda merged, other: merged.merge(other), nlu_data, TrainingData()
        )

    @rasa.shared.utils.common.cached_method
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        if len(self._importers) != 1:
            rasa.shared.utils.io.raise_warning(
                "Auto-config for multiple importers is not supported; "
                "using config as is."
            )
            return None
        return self._importers[0].get_config_file_for_auto_config()


class ResponsesSyncImporter(TrainingDataImporter):
    """Importer that syncs `responses` between Domain and NLU training data.

    Synchronizes responses between Domain and NLU and
    adds retrieval intent properties from the NLU training data
    back to the Domain.
    """

    def __init__(self, importer: TrainingDataImporter):
        """Initializes the ResponsesSyncImporter."""
        self._importer = importer

    def get_config(self) -> Dict:
        """Retrieves model config (see parent class for full docstring)."""
        return self._importer.get_config()

    @rasa.shared.utils.common.cached_method
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        return self._importer.get_config_file_for_auto_config()

    @rasa.shared.utils.common.cached_method
    def get_domain(self) -> Domain:
        """Merge existing domain with properties of retrieval intents in NLU data."""
        existing_domain = self._importer.get_domain()
        return existing_domain

    @rasa.shared.utils.common.cached_method
    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Updates NLU data with responses for retrieval intents from domain."""
        return self._importer.get_nlu_data(language)
