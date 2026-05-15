"""
src/graph/dependency_graph.py
=============================
NetworkX-based pipeline dependency graph.

The graph is a DIRECTED graph where edges represent data flow:
    upstream_pipeline → downstream_pipeline

This means: downstream_pipeline DEPENDS ON upstream_pipeline.
If upstream fails, downstream is at risk.

WHY NetworkX?
- Lightweight (pure Python), no infrastructure needed
- Rich set of graph algorithms: centrality, shortest paths, components
- Easy to serialize/load from edge lists
- Perfect for our scale (~15-50 pipelines

In production at larger scale (1000+ pipelines), you'd use a graph DB
(Neo4j, Neptune) — but the features we extract here would be the same.
"""

from typing import Any, Optional

import networkx as nx

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineDependencyGraph:
    """
    Manages the directed dependency graph of data pipelines.

    Nodes = pipeline names
    Edges = dependency relationships (upstream → downstream)

    Usage:
        graph = PipelineDependencyGraph()
        graph.add_dependency("user_etl", "recommendation_pipeline")
        graph.add_dependency("user_etl", "analytics_pipeline")

        features = graph.get_node_features("recommendation_pipeline")
        # → in_degree=1, out_degree=0, is_leaf=True, ...
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        logger.info("Initialised empty pipeline dependency graph")

    def add_dependency(
        self,
        upstream: str,
        downstream: str,
        dependency_type: str = "data",
    ) -> None:
        """
        Add a directed edge: upstream → downstream.

        Args:
            upstream: Pipeline that produces data.
            downstream: Pipeline that consumes data.
            dependency_type: Type of dependency (data, trigger, sensor).
        """
        self._graph.add_edge(
            upstream, downstream,
            dependency_type=dependency_type,
        )

    def add_dependencies_bulk(
        self, edges: list[tuple[str, str, str]]
    ) -> None:
        """
        Add multiple dependencies at once.

        Args:
            edges: List of (upstream, downstream, dependency_type) tuples.
        """
        for upstream, downstream, dep_type in edges:
            self.add_dependency(upstream, downstream, dep_type)
        logger.info(
            "Added %d dependencies. Graph: %d nodes, %d edges",
            len(edges), self._graph.number_of_nodes(), self._graph.number_of_edges(),
        )

    def get_node_features(self, pipeline_name: str) -> dict[str, Any]:
        """
        Extract graph-based features for a single pipeline.

        These features capture the pipeline's POSITION in the dependency
        network — information that tabular features alone can't provide.

        Args:
            pipeline_name: The pipeline to extract features for.

        Returns:
            Dict of graph features (all numeric, ready for ML).
        """
        features: dict[str, Any] = {}

        # If pipeline isn't in the graph, return zeros
        if pipeline_name not in self._graph:
            logger.debug("Pipeline '%s' not found in graph — returning zero features", pipeline_name)
            return self._zero_features()

        # --- Degree features ---
        # In-degree: how many pipelines feed INTO this one
        # High in-degree = many dependencies = more failure points
        features["graph_in_degree"] = self._graph.in_degree(pipeline_name)

        # Out-degree: how many pipelines depend ON this one
        # High out-degree = high blast radius if this pipeline fails
        features["graph_out_degree"] = self._graph.out_degree(pipeline_name)

        # Total degree
        features["graph_total_degree"] = features["graph_in_degree"] + features["graph_out_degree"]

        # --- Position features ---
        # Is this a root node? (no upstream dependencies)
        features["graph_is_root"] = int(features["graph_in_degree"] == 0)

        # Is this a leaf node? (nothing depends on it)
        features["graph_is_leaf"] = int(features["graph_out_degree"] == 0)

        # --- Centrality features ---
        # Betweenness centrality: how often is this node on the shortest
        # path between other nodes? High = critical bottleneck.
        betweenness = nx.betweenness_centrality(self._graph)
        features["graph_betweenness"] = betweenness.get(pipeline_name, 0.0)

        # PageRank: how "important" is this node in the graph?
        # High PageRank = many important pipelines depend on it.
        try:
            pagerank = nx.pagerank(self._graph)
            features["graph_pagerank"] = pagerank.get(pipeline_name, 0.0)
        except nx.PowerIterationFailedConvergence:
            features["graph_pagerank"] = 0.0

        # --- Path features ---
        # Downstream count: total number of pipelines that would be
        # affected if this pipeline fails (transitive closure)
        downstream = nx.descendants(self._graph, pipeline_name)
        features["graph_downstream_count"] = len(downstream)

        # Upstream count: total number of pipelines this one depends on
        upstream = nx.ancestors(self._graph, pipeline_name)
        features["graph_upstream_count"] = len(upstream)

        # Depth: longest path from any root to this node
        # Deep nodes are more likely to fail from cascading issues
        features["graph_depth"] = self._compute_depth(pipeline_name)

        # --- Component features ---
        # Weakly connected component size
        # (how big is the cluster this pipeline belongs to?)
        weak_components = list(nx.weakly_connected_components(self._graph))
        for component in weak_components:
            if pipeline_name in component:
                features["graph_component_size"] = len(component)
                break
        else:
            features["graph_component_size"] = 1

        return features

    def get_upstream_pipelines(self, pipeline_name: str) -> list[str]:
        """Get direct upstream dependencies."""
        if pipeline_name not in self._graph:
            return []
        return list(self._graph.predecessors(pipeline_name))

    def get_downstream_pipelines(self, pipeline_name: str) -> list[str]:
        """Get direct downstream dependents."""
        if pipeline_name not in self._graph:
            return []
        return list(self._graph.successors(pipeline_name))

    def get_all_downstream(self, pipeline_name: str) -> set[str]:
        """Get ALL downstream pipelines (transitive)."""
        if pipeline_name not in self._graph:
            return set()
        return nx.descendants(self._graph, pipeline_name)

    def get_all_upstream(self, pipeline_name: str) -> set[str]:
        """Get ALL upstream pipelines (transitive)."""
        if pipeline_name not in self._graph:
            return set()
        return nx.ancestors(self._graph, pipeline_name)

    def detect_cascade_risk(self, failed_pipeline: str) -> dict[str, Any]:
        """
        Assess the blast radius of a pipeline failure.

        Used by the dashboard to show which pipelines are at risk
        when a failure is detected.

        Returns:
            Dict with at-risk pipelines and severity assessment.
        """
        downstream = self.get_all_downstream(failed_pipeline)
        direct = self.get_downstream_pipelines(failed_pipeline)

        if not downstream:
            severity = "low"
        elif len(downstream) <= 2:
            severity = "medium"
        else:
            severity = "high"

        return {
            "failed_pipeline": failed_pipeline,
            "direct_dependents": direct,
            "total_at_risk": list(downstream),
            "blast_radius": len(downstream),
            "severity": severity,
        }

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    @property
    def nodes(self) -> list[str]:
        return list(self._graph.nodes())

    @property
    def edges(self) -> list[tuple[str, str]]:
        return list(self._graph.edges())

    def _compute_depth(self, pipeline_name: str) -> int:
        """Compute the longest path from any root to this node."""
        if self._graph.in_degree(pipeline_name) == 0:
            return 0

        max_depth = 0
        for root in self._get_roots():
            try:
                for path in nx.all_simple_paths(self._graph, root, pipeline_name):
                    max_depth = max(max_depth, len(path) - 1)
            except nx.NodeNotFound:
                continue
            except nx.NetworkXError:
                continue
        return max_depth

    def _get_roots(self) -> list[str]:
        """Get all root nodes (in-degree = 0)."""
        return [n for n in self._graph.nodes() if self._graph.in_degree(n) == 0]

    @staticmethod
    def _zero_features() -> dict[str, Any]:
        """Return zeroed graph features for unknown pipelines."""
        return {
            "graph_in_degree": 0,
            "graph_out_degree": 0,
            "graph_total_degree": 0,
            "graph_is_root": 0,
            "graph_is_leaf": 1,
            "graph_betweenness": 0.0,
            "graph_pagerank": 0.0,
            "graph_downstream_count": 0,
            "graph_upstream_count": 0,
            "graph_depth": 0,
            "graph_component_size": 1,
        }

    @staticmethod
    def feature_names() -> list[str]:
        """Get ordered list of graph feature names."""
        return [
            "graph_in_degree",
            "graph_out_degree",
            "graph_total_degree",
            "graph_is_root",
            "graph_is_leaf",
            "graph_betweenness",
            "graph_pagerank",
            "graph_downstream_count",
            "graph_upstream_count",
            "graph_depth",
            "graph_component_size",
        ]
