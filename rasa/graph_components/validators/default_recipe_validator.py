from __future__ import annotations
from collections import defaultdict
from typing import Iterable, List, Dict, Text, Any, Set, Type, cast

from rasa.engine.graph import ExecutionContext, GraphComponent, GraphSchema, SchemaNode
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.extractors.regex_entity_extractor import RegexEntityExtractor
from rasa.nlu.extractors.crf_entity_extractor import (
    CRFEntityExtractor,
    CRFEntityExtractorOptions,
)
from rasa.nlu.extractors.entity_synonyms import EntitySynonymMapper
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.constants import (
    DOCS_URL_COMPONENTS,
)
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.training_data import TrainingData
import rasa.shared.utils.io


# TODO: Can we replace this with the registered types from the regitry?
TRAINABLE_EXTRACTORS = [CRFEntityExtractor, DIETClassifier]


def _types_to_str(types: Iterable[Type]) -> Text:
    """Returns a text containing the names of all given types.

    Args:
        types: some types
    Returns:
        text containing all type names
    """
    return ", ".join([type.__name__ for type in types])


class DefaultV1RecipeValidator(GraphComponent):
    """Validates a "DefaultV1" configuration against the training data and domain."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> DefaultV1RecipeValidator:
        """Creates a new `ConfigValidator` (see parent class for full docstring)."""
        return cls(execution_context.graph_schema)

    def __init__(self, graph_schema: GraphSchema) -> None:
        """Instantiates a new `ConfigValidator`.

        Args:
           graph_schema: a graph schema
        """
        self._graph_schema = graph_schema
        self._component_types = set(node.uses for node in graph_schema.nodes.values())

    def validate(self, importer: TrainingDataImporter) -> TrainingDataImporter:
        """Validates the current graph schema against the training data and domain.

        Args:
            importer: the training data importer which can also load the domain
        Raises:
            `InvalidConfigException` or `InvalidDomain` in case there is some mismatch
        """
        nlu_data = importer.get_nlu_data()
        self._validate_nlu(nlu_data)

        return importer

    def _validate_nlu(self, training_data: TrainingData) -> None:
        """Validates whether the configuration matches the training data.

        Args:
           training_data: The training data for the NLU components.
        """
        training_data.validate()

        self._raise_if_more_than_one_tokenizer()
        self._raise_if_featurizers_are_not_compatible()
        self._warn_of_competing_extractors()
        self._warn_of_competition_with_regex_extractor(training_data=training_data)
        self._warn_if_some_training_data_is_unused(training_data=training_data)

    def _warn_if_some_training_data_is_unused(
        self, training_data: TrainingData
    ) -> None:
        """Validates that all training data will be consumed by some component.

        Args:
            training_data: The training data for the NLU components.
        """
        if training_data.entity_examples and self._component_types.isdisjoint(
            TRAINABLE_EXTRACTORS
        ):
            rasa.shared.utils.io.raise_warning(
                f"You have defined training data consisting of entity examples, but "
                f"your NLU configuration does not include an entity extractor "
                f"trained on your training data. "
                f"To extract non-pretrained entities, add one of "
                f"{_types_to_str(TRAINABLE_EXTRACTORS)} to your configuration.",
                docs=DOCS_URL_COMPONENTS,
            )

        if training_data.entity_examples and self._component_types.isdisjoint(
            {DIETClassifier, CRFEntityExtractor}
        ):
            if training_data.entity_roles_groups_used():
                rasa.shared.utils.io.raise_warning(
                    f"You have defined training data with entities that "
                    f"have roles/groups, but your NLU configuration does not "
                    f"include a '{DIETClassifier.__name__}' "
                    f"or a '{CRFEntityExtractor.__name__}'. "
                    f"To train entities that have roles/groups, "
                    f"add either '{DIETClassifier.__name__}' "
                    f"or '{CRFEntityExtractor.__name__}' to your "
                    f"configuration.",
                    docs=DOCS_URL_COMPONENTS,
                )

        if training_data.regex_features and self._component_types.isdisjoint(
            [RegexFeaturizer, RegexEntityExtractor]
        ):
            rasa.shared.utils.io.raise_warning(
                f"You have defined training data with regexes, but "
                f"your NLU configuration does not include a 'RegexFeaturizer' "
                f" or a "
                f"'RegexEntityExtractor'. To use regexes, include either a "
                f"'{RegexFeaturizer.__name__}' or a "
                f"'{RegexEntityExtractor.__name__}' "
                f"in your configuration.",
                docs=DOCS_URL_COMPONENTS,
            )

        if training_data.lookup_tables and self._component_types.isdisjoint(
            [RegexFeaturizer, RegexEntityExtractor]
        ):
            rasa.shared.utils.io.raise_warning(
                f"You have defined training data consisting of lookup tables, but "
                f"your NLU configuration does not include a featurizer "
                f"or an entity extractor using the lookup table."
                f"To use the lookup tables, include either a "
                f"'{RegexFeaturizer.__name__}' "
                f"or a '{RegexEntityExtractor.__name__}' "
                f"in your configuration.",
                docs=DOCS_URL_COMPONENTS,
            )

        if training_data.lookup_tables:

            if self._component_types.isdisjoint([CRFEntityExtractor, DIETClassifier]):
                rasa.shared.utils.io.raise_warning(
                    f"You have defined training data consisting of lookup tables, but "
                    f"your NLU configuration does not include any components "
                    f"that uses the features created from the lookup table. "
                    f"To make use of the features that are created with the "
                    f"help of the lookup tables, "
                    f"add a '{DIETClassifier.__name__}' or a "
                    f"'{CRFEntityExtractor.__name__}' "
                    f"with the 'pattern' feature "
                    f"to your configuration.",
                    docs=DOCS_URL_COMPONENTS,
                )

            elif CRFEntityExtractor in self._component_types:

                crf_schema_nodes = [
                    schema_node
                    for schema_node in self._graph_schema.nodes.values()
                    if schema_node.uses == CRFEntityExtractor
                ]
                has_pattern_feature = any(
                    CRFEntityExtractorOptions.PATTERN in feature_list
                    for crf in crf_schema_nodes
                    for feature_list in crf.config.get("features", [])
                )

                if not has_pattern_feature:
                    rasa.shared.utils.io.raise_warning(
                        f"You have defined training data consisting of "
                        f"lookup tables, but your NLU configuration's "
                        f"'{CRFEntityExtractor.__name__}' "
                        f"does not include the "
                        f"'pattern' feature. To featurize lookup tables, "
                        f"add the 'pattern' feature to the "
                        f"'{CRFEntityExtractor.__name__}' "
                        "in your configuration.",
                        docs=DOCS_URL_COMPONENTS,
                    )

        if (
            training_data.entity_synonyms
            and EntitySynonymMapper not in self._component_types
        ):
            rasa.shared.utils.io.raise_warning(
                f"You have defined synonyms in your training data, but "
                f"your NLU configuration does not include an "
                f"'{EntitySynonymMapper.__name__}'. "
                f"To map synonyms, add an "
                f"'{EntitySynonymMapper.__name__}' to your "
                f"configuration.",
                docs=DOCS_URL_COMPONENTS,
            )

    def _raise_if_more_than_one_tokenizer(self) -> None:
        """Validates that only one tokenizer is present in the configuration.

        Note that the existence of a tokenizer and its position in the graph schema
        will be validated via the validation of required components during
        schema validation.

        Raises:
            `InvalidConfigException` in case there is more than one tokenizer
        """
        types_of_tokenizer_schema_nodes = [
            schema_node.uses
            for schema_node in self._graph_schema.nodes.values()
            if issubclass(schema_node.uses, Tokenizer) and schema_node.fn != "train"
        ]

        allowed_number_of_tokenizers = 1
        if len(types_of_tokenizer_schema_nodes) > allowed_number_of_tokenizers:
            raise InvalidConfigException(
                f"The configuration configuration contains more than one tokenizer, "
                f"which is not possible at this time. You can only use one tokenizer. "
                f"The configuration contains the following tokenizers: "
                f"{_types_to_str(types_of_tokenizer_schema_nodes)}. "
            )

    def _warn_of_competing_extractors(self) -> None:
        """Warns the user when using competing extractors.

        Competing extractors are e.g. `CRFEntityExtractor` and `DIETClassifier`.
        Both of these look for the same entities based on the same training data
        leading to ambiguity in the results.
        """
        extractors_in_configuration: Set[
            Type[GraphComponent]
        ] = self._component_types.intersection(TRAINABLE_EXTRACTORS)
        if len(extractors_in_configuration) > 1:
            rasa.shared.utils.io.raise_warning(
                f"You have defined multiple entity extractors that do the same job "
                f"in your configuration: "
                f"{_types_to_str(extractors_in_configuration)}. "
                f"This can lead to the same entity getting "
                f"extracted multiple times. Please read the documentation section "
                f"on entity extractors to make sure you understand the implications.",
                docs=f"{DOCS_URL_COMPONENTS}#entity-extractors",
            )

    def _warn_of_competition_with_regex_extractor(
        self, training_data: TrainingData
    ) -> None:
        """Warns when regex entity extractor is competing with a general one.

        This might be the case when the following conditions are all met:
        * You are using a general entity extractor and the `RegexEntityExtractor`
        * AND you have regex patterns for entity type A
        * AND you have annotated text examples for entity type A

        Args:
            training_data: The training data for the NLU components.
        """
        present_general_extractors = self._component_types.intersection(
            TRAINABLE_EXTRACTORS
        )
        has_general_extractors = len(present_general_extractors) > 0
        has_regex_extractor = RegexEntityExtractor in self._component_types

        regex_entity_types = {rf["name"] for rf in training_data.regex_features}
        overlap_between_types = training_data.entities.intersection(regex_entity_types)
        has_overlap = len(overlap_between_types) > 0

        if has_general_extractors and has_regex_extractor and has_overlap:
            rasa.shared.utils.io.raise_warning(
                f"You have an overlap between the "
                f"'{RegexEntityExtractor.__name__}' and the "
                f"statistical entity extractors "
                f"{_types_to_str(present_general_extractors)} "
                f"in your configuration. Specifically both types of extractors will "
                f"attempt to extract entities of the types "
                f"{', '.join(overlap_between_types)}. "
                f"This can lead to multiple "
                f"extraction of entities. Please read "
                f"'{RegexEntityExtractor.__name__}''s "
                f"documentation section to make sure you understand the "
                f"implications.",
                docs=f"{DOCS_URL_COMPONENTS}#regexentityextractor",
            )

    def _raise_if_featurizers_are_not_compatible(self) -> None:
        """Raises or warns if there are problems regarding the featurizers.

        Raises:
            `InvalidConfigException` in case the featurizers are not compatible
        """
        featurizers: List[SchemaNode] = [
            node
            for node_name, node in self._graph_schema.nodes.items()
            if issubclass(node.uses, Featurizer)
            # Featurizers are split in `train` and `process_training_data` -
            # we only need to look at the nodes which _add_ features.
            and node.fn == "process_training_data"
            # Tokenizers are re-used in the Core part of the graph when using End-to-End
            and not node_name.startswith("e2e")
        ]

        Featurizer.raise_if_featurizer_configs_are_not_compatible(
            [schema_node.config for schema_node in featurizers]
        )
