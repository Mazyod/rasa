from __future__ import annotations
from abc import ABC, abstractmethod
from collections import ChainMap
from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Type

from rasa.engine.exceptions import GraphComponentException
import rasa.shared.utils.common

logger = logging.getLogger(__name__)


@dataclass
class SchemaNode:
    """Represents one node in the schema."""

    needs: Dict[Text, Text]
    uses: Type[GraphComponent]
    constructor_name: Text
    fn: Text
    config: Dict[Text, Any]
    eager: bool = False
    is_target: bool = False
    is_input: bool = False
    resource_name: Optional[Text] = None


GraphSchema = Dict[Text, SchemaNode]


class GraphComponent(ABC):
    """Interface for any component which will run in a graph."""

    # TODO: This doesn't enforce that it exists in subclasses..
    default_config: Dict[Text, Any]

    @classmethod
    @abstractmethod
    def create(
        cls, config: Dict[Text, Any], execution_context: ExecutionContext
    ) -> GraphComponent:
        """Creates a new `GraphComponent`.

        Args:
            config: This config overrides the `default_config`.
            execution_context: Information about the current graph run.

        Returns: An instantiated `GraphComponent`.
        """
        ...

    @classmethod
    def load(
        cls, config: Dict[Dict, Any], execution_context: ExecutionContext
    ) -> GraphComponent:
        """Creates a component using a persisted version of itself.

        If not overridden this method merely calls `create`.

        Args:
            config: This config overrides the `default_config`.
            execution_context: Information about the current graph run.

        Returns: An instantiated, loaded `GraphComponent`.
        """
        return cls.create(config, execution_context)

    def supported_languages(self) -> Optional[List[Text]]:
        """Determines which languages this component can work with.

        Returns: A list of supported languages, or `None` to signify all are supported.
        """
        return None

    def required_packages(self) -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return []


@dataclass
class ExecutionContext:
    """Holds information about a single graph run."""

    graph_schema: GraphSchema = field(repr=False)
    model_id: Text
    should_add_diagnostic_data: bool = False
    is_finetuning: bool = False


class GraphNode:
    """Instantiates and runs a `GraphComponent` within a graph.

    A `GraphNode` is a wrapper for a `GraphComponent` that allows it to be executed
    in the context of a graph. It is responsible for instantiating the component at the
    correct time, collecting the inputs from the parent nodes, running the run function
    of the component and passing the output onwards.
    """

    def __init__(
        self,
        node_name: Text,
        component_class: Type[GraphComponent],
        constructor_name: Text,
        component_config: Dict[Text, Any],
        fn_name: Text,
        inputs: Dict[Text, Text],
        eager: bool,
        execution_context: ExecutionContext,
    ) -> None:
        """Initializes `GraphNode`.

        Args:
            node_name: The name of the node in the schema.
            component_class: The class to be instantiated and run.
            constructor_name: The method used to instantiate the component.
            component_config: Config to be passed to the component.
            fn_name: The function on the instantiated `GraphComponent` to be run when
                the node executes.
            inputs: A map from input name to parent node name that provides it.
            eager: Determines if the node is instantiated right away, or just before
                being run.
            execution_context: Information about the current graph run.
        """
        self._node_name: Text = node_name
        self._component_class: Type[GraphComponent] = component_class
        self._constructor_name: Text = constructor_name
        self._constructor_fn: Callable = getattr(
            self._component_class, self._constructor_name
        )
        self._component_config: Dict[Text, Any] = {
            **self._component_class.default_config,
            **component_config,
        }
        self._fn_name: Text = fn_name
        self._fn: Callable = getattr(self._component_class, self._fn_name)
        self._inputs: Dict[Text, Text] = inputs
        self._eager: bool = eager
        self._execution_context: ExecutionContext = execution_context

        self._component: Optional[GraphComponent] = None
        if self._eager:
            self._load_component()

    def _load_component(
        self, additional_kwargs: Optional[Dict[Text, Any]] = None
    ) -> None:
        kwargs = additional_kwargs if additional_kwargs else {}

        logger.debug(
            f"Node {self._node_name} loading "
            f"{self._component_class.__name__}.{self._constructor_name} "
            f"with config: {self._component_config}, and kwargs: {kwargs}."
        )

        constructor = getattr(self._component_class, self._constructor_name)
        try:
            self._component: GraphComponent = constructor(  # type: ignore[no-redef]
                config=self._component_config,
                execution_context=self._execution_context,
                **kwargs,
            )
        except Exception as e:
            raise GraphComponentException(
                f"Error initializing graph component for node {self._node_name}."
            ) from e

    def parent_node_names(self) -> List[Text]:
        """The names of the parent nodes of this node."""
        return list(self._inputs.values())

    @staticmethod
    def _collapse_inputs_from_previous_nodes(
        inputs_from_previous_nodes: Tuple[Dict[Text, Any]]
    ) -> Dict[Text, Any]:
        return dict(ChainMap(*inputs_from_previous_nodes))

    def __call__(self, *inputs_from_previous_nodes: Dict[Text, Any]) -> Dict[Text, Any]:
        """Calls the `GraphComponent` run method when the node executes in the graph.

        Args:
            *inputs_from_previous_nodes: The output of all parent nodes. Each is a
                dictionary with a single item mapping the node's name to its output.

        Returns: A dictionary with a single item mapping the node's name to the output.
        """
        received_inputs = self._collapse_inputs_from_previous_nodes(
            inputs_from_previous_nodes
        )
        kwargs = {}
        for input_name, input_node in self._inputs.items():
            kwargs[input_name] = received_inputs[input_node]

        if not self._eager:
            constructor_kwargs = rasa.shared.utils.common.minimal_kwargs(
                kwargs, self._constructor_fn
            )
            self._load_component(constructor_kwargs)

        run_kwargs = rasa.shared.utils.common.minimal_kwargs(kwargs, self._fn)
        logger.debug(
            f"Node {self._node_name} running "
            f"{self._component_class.__name__}.{self._fn_name} "
            f"with kwargs: {run_kwargs}."
        )

        try:
            output = self._fn(self._component, **run_kwargs)
        except Exception as e:
            raise GraphComponentException(
                f"Error running graph component for node {self._node_name}."
            ) from e

        return {self._node_name: output}

    @classmethod
    def from_schema_node(
        cls,
        node_name: Text,
        schema_node: SchemaNode,
        execution_context: ExecutionContext,
    ) -> GraphNode:
        """Creates a `GraphNode` from a `SchemaNode`."""
        return cls(
            node_name=node_name,
            component_class=schema_node.uses,
            constructor_name=schema_node.constructor_name,
            component_config=schema_node.config,
            fn_name=schema_node.fn,
            inputs=schema_node.needs,
            eager=schema_node.eager,
            execution_context=execution_context,
        )
