from functools import reduce
from typing import Text, Optional, Union, List, Dict, Any
import logging
from rasa import data

from rasa.core.domain import Domain, InvalidDomain
from rasa.core.interpreter import RegexInterpreter, NaturalLanguageInterpreter
from rasa.core.training.dsl import StoryFileReader
from rasa.core.training.structures import StoryGraph
from rasa.importers import utils
from rasa.nlu.training_data import TrainingData
import rasa.utils.io as io_utils
import rasa.utils.common as common_utils

logger = logging.getLogger(__name__)


class TrainingFileImporter:
    """Common interface for different mechanisms to load training data."""

    async def get_domain(self) -> Domain:
        """Retrieves the domain which should be used for the training."""
        raise NotImplementedError()

    async def get_story_data(
        self,
        interpreter: "NaturalLanguageInterpreter" = RegexInterpreter(),
        template_variables: Optional[Dict] = None,
        use_e2e: bool = False,
        exclusion_percentage: Optional[int] = None,
    ) -> StoryGraph:
        """Retrieves the story data which should be used for the training."""

        raise NotImplementedError()

    async def get_config(self) -> Dict:
        """Retrieves the configuration which should be used for the training."""
        raise NotImplementedError()

    async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Retrieves the nlu training data which should be used for the training."""
        raise NotImplementedError()

    @staticmethod
    def load_from_config(
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
    ) -> "TrainingFileImporter":
        """Loads a `TrainingFileImporter` instance from a configuration file."""

        config = io_utils.read_config_file(config_path)
        return TrainingFileImporter.load_from_dict(
            config, config_path, domain_path, training_data_paths
        )

    @staticmethod
    def load_from_dict(
        config: Dict,
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
    ) -> "TrainingFileImporter":
        """Loads a `TrainingFileImporter` instance from a dictionary."""

        importers = config.get("importers", [])
        importers = [
            TrainingFileImporter._importer_from_dict(
                importer, config_path, domain_path, training_data_paths
            )
            for importer in importers
        ]
        importers = [importer for importer in importers if importer]

        if not importers:
            importers = [
                SimpleFileImporter(config_path, domain_path, training_data_paths)
            ]

        return CombinedFileImporter(importers)

    @staticmethod
    def _importer_from_dict(
        importer_config: Dict,
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
    ) -> Optional["TrainingFileImporter"]:
        from rasa.importers.skill import SkillSelector

        module_path = importer_config.pop("name", None)
        if module_path == SimpleFileImporter.__name__:
            importer_class = SimpleFileImporter
        elif module_path == SkillSelector.__name__:
            importer_class = SkillSelector
        else:
            try:
                importer_class = common_utils.class_from_module_path(module_path)
            except (AttributeError, ImportError):
                logging.warning("Importer '{}' not found.".format(module_path))
                return None

        import rasa.cli.utils as cli_utils

        constructor_arguments = cli_utils.minimal_kwargs(
            importer_config, importer_class
        )
        return importer_class(
            config_path, domain_path, training_data_paths, **constructor_arguments
        )


class CombinedFileImporter(TrainingFileImporter):
    """`TrainingFileImporter` which supports using multiple `TrainingFileImporter` as
        if it would be a single instance.
    """

    def __init__(self, importers: List[TrainingFileImporter]):
        self._importers = importers

    async def get_config(self) -> Dict:
        configs = [await importer.get_config() for importer in self._importers]
        return reduce(lambda merged, other: {**merged, **(other or {})}, configs, {})

    async def get_domain(self) -> Domain:
        domains = [await importer.get_domain() for importer in self._importers]
        return reduce(
            lambda merged, other: merged.merge(other), domains, Domain.empty()
        )

    async def get_story_data(
        self,
        interpreter: "NaturalLanguageInterpreter" = RegexInterpreter(),
        template_variables: Optional[Dict] = None,
        use_e2e: bool = False,
        exclusion_percentage: Optional[int] = None,
    ) -> StoryGraph:
        story_graphs = [
            await importer.get_story_data(
                interpreter, template_variables, use_e2e, exclusion_percentage
            )
            for importer in self._importers
        ]
        return reduce(
            lambda merged, other: merged.merge(other), story_graphs, StoryGraph([])
        )

    async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        nlu_datas = [
            await importer.get_nlu_data(language) for importer in self._importers
        ]
        return reduce(
            lambda merged, other: merged.merge(other), nlu_datas, TrainingData()
        )


class SimpleFileImporter(TrainingFileImporter):
    """Default `TrainingFileImporter` implementation."""

    def __init__(
        self,
        config_file: Text,
        domain_path: Optional[Text],
        training_data_paths: Optional[Union[List[Text], Text]],
    ):
        self.config = io_utils.read_config_file(config_file)
        self._domain_path = domain_path

        self._story_files, self._nlu_files = data.get_core_nlu_files(
            training_data_paths
        )

        self._domain = None
        self._training_datas = None
        self._story_graph = None

    async def get_config(self, **kwargs: Optional[Dict[Text, Any]]) -> Dict:
        return self.config

    async def get_story_data(
        self,
        interpreter: "NaturalLanguageInterpreter" = RegexInterpreter(),
        template_variables: Optional[Dict] = None,
        use_e2e: bool = False,
        exclusion_percentage: Optional[int] = None,
    ) -> StoryGraph:

        if not self._story_graph:
            story_steps = await StoryFileReader.read_from_files(
                self._story_files,
                await self.get_domain(),
                interpreter,
                template_variables,
                use_e2e,
                exclusion_percentage,
            )
            self._story_graph = StoryGraph(story_steps)

        return self._story_graph

    async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        if not self._training_datas:
            self._training_datas = utils.training_data_from_paths(
                self._nlu_files, language
            )

        return self._training_datas

    async def get_domain(self) -> Domain:
        if not self._domain:
            try:
                self._domain = Domain.load(self._domain_path)
                self._domain.check_missing_templates()
            except InvalidDomain:
                logger.debug(
                    "Loading domain from '{}' failed. Using empty domain.".format(
                        self._domain_path
                    )
                )
                self._domain = Domain.empty()

        return self._domain
