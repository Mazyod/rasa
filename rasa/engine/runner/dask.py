from __future__ import annotations

from collections import ChainMap
import logging
from typing import Any, Dict, List, Optional, Text

import dask

from rasa.engine.exceptions import GraphRunError
from rasa.engine.graph import (
    ExecutionContext,
    GraphNode,
    GraphSchema,
)
from rasa.engine.runner.interface import GraphRunner

logger = logging.getLogger(__name__)


class DaskGraphRunner(GraphRunner):
    """Dask implementation of a `GraphRunner`."""

    def __init__(
        self, graph_schema: GraphSchema, execution_context: ExecutionContext
    ) -> None:
        """Initializes a `DaskGraphRunner`.

        Args:
            graph_schema: The graph schema that will be run.
            execution_context: Information about the current graph run to be passed to
                each node.
        """
        self._targets: List[Text] = self._targets_from_schema(graph_schema)
        self._instantiated_graph: Dict[Text, GraphNode] = self._instantiate_graph(
            graph_schema, execution_context
        )
        self._execution_context: ExecutionContext = execution_context

    @classmethod
    def create(
        cls, graph_schema: GraphSchema, execution_context: ExecutionContext
    ) -> DaskGraphRunner:
        """Creates the runner (see parent class for full docstring)."""
        return cls(graph_schema, execution_context)

    @staticmethod
    def _targets_from_schema(graph_schema: GraphSchema) -> List[Text]:
        return [
            node_name
            for node_name, schema_node in graph_schema.items()
            if schema_node.is_target
        ]

    def _instantiate_graph(
        self, graph_schema: GraphSchema, execution_context: ExecutionContext
    ) -> Dict[Text, GraphNode]:
        return {
            node_name: GraphNode.from_schema_node(
                node_name, schema_node, execution_context
            )
            for node_name, schema_node in graph_schema.items()
        }

    @staticmethod
    def _instantiated_graph_to_dask_graph(
        instantiated_graph: Dict[Text, GraphNode]
    ) -> Dict[Text, Any]:
        """Builds a dask graph from the instantiated graph.

        For more information about dask graphs
        see: https://docs.dask.org/en/latest/spec.html
        """
        run_graph = {
            node_name: (graph_node, *graph_node.parent_node_names())
            for node_name, graph_node in instantiated_graph.items()
        }
        return run_graph

    def run(
        self,
        inputs: Optional[Dict[Text, Any]] = None,
        targets: Optional[List[Text]] = None,
    ) -> Dict[Text, Any]:
        """Runs the graph (see parent class for full docstring)."""
        run_graph = self._instantiated_graph_to_dask_graph(self._instantiated_graph)

        if inputs:
            self._add_inputs_to_graph(inputs, run_graph)

        run_targets = targets if targets else self._targets

        logger.debug(
            f"Running graph with inputs: {inputs}, targets: {targets} "
            f"and {self._execution_context}."
        )

        try:
            dask_result = dask.get(run_graph, run_targets)
            return dict(ChainMap(*dask_result))
        except RuntimeError as e:
            raise GraphRunError("Error running runner.") from e

    @staticmethod
    def _add_inputs_to_graph(inputs: Optional[Dict[Text, Any]], graph: Any,) -> None:
        for input_name, input_value in inputs.items():
            if input_value in graph.keys():
                raise GraphRunError(
                    f"Input value '{input_value}' clashes with a node name. Make sure "
                    f"that none of the input names passed to the `run` method are the "
                    f"same as node names in the graph schema."
                )
            graph[input_name] = {input_name: input_value}
