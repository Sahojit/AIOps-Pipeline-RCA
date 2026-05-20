"""
tests/unit/test_phase11_dashboard.py
=====================================
Tests for the Streamlit monitoring dashboard helper functions.

We test the non-Streamlit logic only:
  - Graph figure builder (valid Plotly Figure, compare mode)
  - Hierarchical layout (correct depth assignment)
  - Scenario presets (well-formed)
  - Constants sanity

We do NOT test the Streamlit page renderers directly — they require a
running Streamlit runtime and are covered by manual / integration testing.
"""

import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so we can import dashboard.app
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from unittest.mock import patch

import plotly.graph_objects as go

from dashboard.app import (
    ALL_ROOT_CAUSES,
    KNOWN_PIPELINES,
    ROOT_CAUSE_COLORS,
    SCENARIO_PRESETS,
    _hierarchical_layout,
    build_graph_figure,
)
from src.graph.dependency_graph import PipelineDependencyGraph


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def simple_graph() -> PipelineDependencyGraph:
    """
    A→B→D
    A→C→D
         →E
    """
    g = PipelineDependencyGraph()
    g.add_dependency("A", "B", "data")
    g.add_dependency("A", "C", "data")
    g.add_dependency("B", "D", "data")
    g.add_dependency("C", "D", "data")
    g.add_dependency("C", "E", "data")
    return g


# =========================================================================
# Hierarchical layout
# =========================================================================

class TestHierarchicalLayout:
    def test_all_nodes_have_positions(self, simple_graph: PipelineDependencyGraph) -> None:
        pos = _hierarchical_layout(simple_graph._graph)
        for node in simple_graph.nodes:
            assert node in pos, f"Node '{node}' missing from layout"

    def test_root_at_depth_zero(self, simple_graph: PipelineDependencyGraph) -> None:
        pos = _hierarchical_layout(simple_graph._graph)
        # A is the root — should be at y=1.0 (depth 0)
        _, y_a = pos["A"]
        assert y_a == pytest.approx(1.0)

    def test_leaf_at_lower_depth(self, simple_graph: PipelineDependencyGraph) -> None:
        pos = _hierarchical_layout(simple_graph._graph)
        _, y_a = pos["A"]
        _, y_d = pos["D"]
        # D is deeper than A, so its y-coordinate should be lower
        assert y_d < y_a

    def test_positions_in_unit_square(self, simple_graph: PipelineDependencyGraph) -> None:
        pos = _hierarchical_layout(simple_graph._graph)
        for node, (x, y) in pos.items():
            assert 0.0 <= x <= 1.0, f"{node} x={x} out of [0,1]"
            assert 0.0 <= y <= 1.0, f"{node} y={y} out of [0,1]"

    def test_empty_graph(self) -> None:
        import networkx as nx
        G = nx.DiGraph()
        pos = _hierarchical_layout(G)
        assert pos == {}


# =========================================================================
# Graph figure builder
# =========================================================================

_DUMMY_FEATURES = {
    "graph_in_degree": 1,
    "graph_out_degree": 1,
    "graph_total_degree": 2,
    "graph_is_root": 0,
    "graph_is_leaf": 0,
    "graph_betweenness": 0.1,
    "graph_pagerank": 0.2,
    "graph_downstream_count": 2,
    "graph_upstream_count": 1,
    "graph_depth": 1,
    "graph_component_size": 5,
}


class TestBuildGraphFigure:
    """
    Tests for build_graph_figure().

    We patch get_node_features() to avoid a scipy dependency in this
    venv — PageRank / centrality computation is already covered by
    test_phase10_graph.py with the full dependency stack.
    """

    def test_returns_plotly_figure(self, simple_graph: PipelineDependencyGraph) -> None:
        with patch.object(simple_graph, "get_node_features", return_value=_DUMMY_FEATURES):
            fig = build_graph_figure(simple_graph)
        assert isinstance(fig, go.Figure)

    def test_figure_has_traces(self, simple_graph: PipelineDependencyGraph) -> None:
        with patch.object(simple_graph, "get_node_features", return_value=_DUMMY_FEATURES):
            fig = build_graph_figure(simple_graph)
        assert len(fig.data) > 0

    def test_figure_with_failed_node(self, simple_graph: PipelineDependencyGraph) -> None:
        """Should not raise even when a failure is highlighted."""
        with patch.object(simple_graph, "get_node_features", return_value=_DUMMY_FEATURES):
            fig = build_graph_figure(simple_graph, failed_pipeline="A")
        assert isinstance(fig, go.Figure)

    def test_figure_with_at_risk_set(self, simple_graph: PipelineDependencyGraph) -> None:
        with patch.object(simple_graph, "get_node_features", return_value=_DUMMY_FEATURES):
            fig = build_graph_figure(
                simple_graph,
                failed_pipeline="A",
                at_risk={"B", "C", "D", "E"},
            )
        assert isinstance(fig, go.Figure)

    def test_figure_compare_mode(self, simple_graph: PipelineDependencyGraph) -> None:
        """Compare mode should add separate node groups without crashing."""
        with patch.object(simple_graph, "get_node_features", return_value=_DUMMY_FEATURES):
            fig = build_graph_figure(
                simple_graph,
                failed_pipeline="A",
                at_risk={"B"},
                compare_pipeline="C",
                compare_at_risk={"D", "E"},
            )
        assert isinstance(fig, go.Figure)
        # Should have more trace groups than a simple graph
        assert len(fig.data) > 0

    def test_figure_with_unknown_failed_node(self, simple_graph: PipelineDependencyGraph) -> None:
        """An unknown failed_pipeline should not crash the function."""
        with patch.object(simple_graph, "get_node_features", return_value=_DUMMY_FEATURES):
            fig = build_graph_figure(simple_graph, failed_pipeline="NONEXISTENT")
        assert isinstance(fig, go.Figure)

    def test_figure_layout_has_no_axes(self, simple_graph: PipelineDependencyGraph) -> None:
        with patch.object(simple_graph, "get_node_features", return_value=_DUMMY_FEATURES):
            fig = build_graph_figure(simple_graph)
        assert fig.layout.xaxis.showticklabels is False
        assert fig.layout.yaxis.showticklabels is False

    def test_annotations_count_matches_edge_count(self, simple_graph: PipelineDependencyGraph) -> None:
        """One annotation (arrow) per edge."""
        with patch.object(simple_graph, "get_node_features", return_value=_DUMMY_FEATURES):
            fig = build_graph_figure(simple_graph)
        assert len(fig.layout.annotations) == simple_graph.edge_count


# =========================================================================
# Constants sanity
# =========================================================================

class TestConstants:
    def test_all_root_causes_have_colours(self) -> None:
        for rc in ALL_ROOT_CAUSES:
            assert rc in ROOT_CAUSE_COLORS, f"'{rc}' missing from ROOT_CAUSE_COLORS"

    def test_colour_values_are_hex(self) -> None:
        for rc, color in ROOT_CAUSE_COLORS.items():
            assert color.startswith("#"), f"Color for '{rc}' is not hex: {color}"

    def test_known_pipelines_nonempty(self) -> None:
        assert len(KNOWN_PIPELINES) > 0

    def test_no_duplicate_pipelines(self) -> None:
        assert len(KNOWN_PIPELINES) == len(set(KNOWN_PIPELINES))


# =========================================================================
# Scenario presets
# =========================================================================

class TestScenarioPresets:
    def test_custom_preset_is_empty(self) -> None:
        assert SCENARIO_PRESETS["(custom)"] == {}

    def test_presets_have_valid_pipeline_names(self) -> None:
        known = set(KNOWN_PIPELINES)
        for name, preset in SCENARIO_PRESETS.items():
            if not preset:
                continue  # skip (custom)
            assert preset["pipeline_name"] in known, (
                f"Preset '{name}' has unknown pipeline: {preset['pipeline_name']}"
            )

    def test_presets_have_required_fields(self) -> None:
        required = {
            "pipeline_name", "task_name", "runtime", "retry_count",
            "rows_processed", "schema_change", "upstream_failed", "error_log",
        }
        for name, preset in SCENARIO_PRESETS.items():
            if not preset:
                continue
            assert required.issubset(preset.keys()), (
                f"Preset '{name}' missing keys: {required - preset.keys()}"
            )

    def test_at_least_three_non_custom_presets(self) -> None:
        non_custom = [k for k in SCENARIO_PRESETS if k != "(custom)"]
        assert len(non_custom) >= 3
