"""
tests/unit/test_phase10_graph.py
================================
Tests for the pipeline dependency graph and graph features.
"""

from typing import Any

import pytest

from src.graph.dependency_graph import PipelineDependencyGraph
from src.graph.seed_dependencies import PIPELINE_DEPENDENCIES, build_default_graph


@pytest.fixture
def simple_graph() -> PipelineDependencyGraph:
    """
    Build a simple test graph:
        A → B → D
        A → C → D
              → E
    """
    graph = PipelineDependencyGraph()
    graph.add_dependency("A", "B", "data")
    graph.add_dependency("A", "C", "data")
    graph.add_dependency("B", "D", "data")
    graph.add_dependency("C", "D", "data")
    graph.add_dependency("C", "E", "data")
    return graph


@pytest.fixture
def default_graph() -> PipelineDependencyGraph:
    """Build the full default graph matching our pipeline registry."""
    return build_default_graph()


# =========================================================================
# Basic Graph Tests
# =========================================================================

class TestPipelineDependencyGraph:
    def test_add_dependency(self) -> None:
        graph = PipelineDependencyGraph()
        graph.add_dependency("A", "B", "data")
        assert graph.node_count == 2
        assert graph.edge_count == 1

    def test_add_dependencies_bulk(self) -> None:
        graph = PipelineDependencyGraph()
        graph.add_dependencies_bulk([
            ("A", "B", "data"),
            ("B", "C", "data"),
            ("C", "D", "trigger"),
        ])
        assert graph.node_count == 4
        assert graph.edge_count == 3

    def test_nodes_and_edges(self, simple_graph: PipelineDependencyGraph) -> None:
        assert set(simple_graph.nodes) == {"A", "B", "C", "D", "E"}
        assert simple_graph.edge_count == 5

    def test_upstream_pipelines(self, simple_graph: PipelineDependencyGraph) -> None:
        assert set(simple_graph.get_upstream_pipelines("D")) == {"B", "C"}
        assert simple_graph.get_upstream_pipelines("A") == []

    def test_downstream_pipelines(self, simple_graph: PipelineDependencyGraph) -> None:
        assert set(simple_graph.get_downstream_pipelines("A")) == {"B", "C"}
        assert simple_graph.get_downstream_pipelines("E") == []

    def test_all_downstream_transitive(self, simple_graph: PipelineDependencyGraph) -> None:
        all_downstream = simple_graph.get_all_downstream("A")
        assert all_downstream == {"B", "C", "D", "E"}

    def test_all_upstream_transitive(self, simple_graph: PipelineDependencyGraph) -> None:
        all_upstream = simple_graph.get_all_upstream("D")
        assert all_upstream == {"A", "B", "C"}

    def test_unknown_pipeline_returns_empty(self) -> None:
        graph = PipelineDependencyGraph()
        assert graph.get_upstream_pipelines("nonexistent") == []
        assert graph.get_downstream_pipelines("nonexistent") == []
        assert graph.get_all_downstream("nonexistent") == set()


# =========================================================================
# Graph Feature Tests
# =========================================================================

class TestGraphFeatures:
    def test_root_node_features(self, simple_graph: PipelineDependencyGraph) -> None:
        features = simple_graph.get_node_features("A")
        assert features["graph_in_degree"] == 0
        assert features["graph_out_degree"] == 2
        assert features["graph_is_root"] == 1
        assert features["graph_is_leaf"] == 0

    def test_leaf_node_features(self, simple_graph: PipelineDependencyGraph) -> None:
        features = simple_graph.get_node_features("E")
        assert features["graph_in_degree"] == 1
        assert features["graph_out_degree"] == 0
        assert features["graph_is_root"] == 0
        assert features["graph_is_leaf"] == 1

    def test_middle_node_features(self, simple_graph: PipelineDependencyGraph) -> None:
        features = simple_graph.get_node_features("C")
        assert features["graph_in_degree"] == 1
        assert features["graph_out_degree"] == 2
        assert features["graph_is_root"] == 0
        assert features["graph_is_leaf"] == 0

    def test_downstream_count(self, simple_graph: PipelineDependencyGraph) -> None:
        features = simple_graph.get_node_features("A")
        assert features["graph_downstream_count"] == 4  # B, C, D, E

    def test_upstream_count(self, simple_graph: PipelineDependencyGraph) -> None:
        features = simple_graph.get_node_features("D")
        assert features["graph_upstream_count"] == 3  # A, B, C

    def test_depth_root(self, simple_graph: PipelineDependencyGraph) -> None:
        features = simple_graph.get_node_features("A")
        assert features["graph_depth"] == 0

    def test_depth_leaf(self, simple_graph: PipelineDependencyGraph) -> None:
        features = simple_graph.get_node_features("D")
        assert features["graph_depth"] == 2  # A→B→D or A→C→D

    def test_betweenness_centrality(self, simple_graph: PipelineDependencyGraph) -> None:
        features = simple_graph.get_node_features("C")
        # C is on the path to E, so it should have some betweenness
        assert features["graph_betweenness"] >= 0.0

    def test_pagerank_positive(self, simple_graph: PipelineDependencyGraph) -> None:
        features = simple_graph.get_node_features("D")
        assert features["graph_pagerank"] > 0.0

    def test_component_size(self, simple_graph: PipelineDependencyGraph) -> None:
        features = simple_graph.get_node_features("A")
        assert features["graph_component_size"] == 5  # All nodes in one component

    def test_unknown_pipeline_returns_zeros(self) -> None:
        graph = PipelineDependencyGraph()
        graph.add_dependency("A", "B", "data")
        features = graph.get_node_features("Z")
        assert features["graph_in_degree"] == 0
        assert features["graph_out_degree"] == 0
        assert features["graph_is_leaf"] == 1

    def test_all_features_numeric(self, simple_graph: PipelineDependencyGraph) -> None:
        features = simple_graph.get_node_features("A")
        for key, value in features.items():
            assert isinstance(value, (int, float)), f"{key} is {type(value)}, expected numeric"

    def test_feature_names_match_features(self, simple_graph: PipelineDependencyGraph) -> None:
        features = simple_graph.get_node_features("A")
        expected_names = PipelineDependencyGraph.feature_names()
        assert set(features.keys()) == set(expected_names)


# =========================================================================
# Cascade Detection Tests
# =========================================================================

class TestCascadeDetection:
    def test_root_failure_cascade(self, simple_graph: PipelineDependencyGraph) -> None:
        risk = simple_graph.detect_cascade_risk("A")
        assert risk["blast_radius"] == 4
        assert risk["severity"] == "high"
        assert set(risk["total_at_risk"]) == {"B", "C", "D", "E"}

    def test_leaf_failure_no_cascade(self, simple_graph: PipelineDependencyGraph) -> None:
        risk = simple_graph.detect_cascade_risk("E")
        assert risk["blast_radius"] == 0
        assert risk["severity"] == "low"

    def test_middle_node_cascade(self, simple_graph: PipelineDependencyGraph) -> None:
        risk = simple_graph.detect_cascade_risk("C")
        assert risk["blast_radius"] == 2  # D and E
        assert set(risk["total_at_risk"]) == {"D", "E"}

    def test_unknown_pipeline_no_cascade(self) -> None:
        graph = PipelineDependencyGraph()
        risk = graph.detect_cascade_risk("nonexistent")
        assert risk["blast_radius"] == 0
        assert risk["severity"] == "low"


# =========================================================================
# Default Graph Tests (our actual pipeline registry)
# =========================================================================

class TestDefaultGraph:
    def test_default_graph_node_count(self, default_graph: PipelineDependencyGraph) -> None:
        # Should have all 15 pipelines from our registry
        assert default_graph.node_count >= 14

    def test_default_graph_edge_count(self, default_graph: PipelineDependencyGraph) -> None:
        assert default_graph.edge_count == len(PIPELINE_DEPENDENCIES)

    def test_user_etl_is_critical(self, default_graph: PipelineDependencyGraph) -> None:
        """user_etl feeds 5 pipelines — should have high out-degree."""
        features = default_graph.get_node_features("user_etl_pipeline")
        assert features["graph_out_degree"] == 5
        assert features["graph_is_root"] == 1
        assert features["graph_downstream_count"] >= 5

    def test_data_quality_is_leaf(self, default_graph: PipelineDependencyGraph) -> None:
        features = default_graph.get_node_features("data_quality_pipeline")
        assert features["graph_is_leaf"] == 1
        assert features["graph_out_degree"] == 0

    def test_user_etl_cascade_is_high(self, default_graph: PipelineDependencyGraph) -> None:
        risk = default_graph.detect_cascade_risk("user_etl_pipeline")
        assert risk["severity"] == "high"
        assert risk["blast_radius"] >= 5
