"""
dashboard/app.py
================
Streamlit Monitoring Dashboard for the AIOps Pipeline RCA System.

Tabs:
  Overview       — KPI cards, trend timeline, root cause distribution
  Predictions    — Filterable table with expandable evidence + confidence chart
  Graph Explorer — Interactive DAG with cascade simulation, compare mode, node details
  Live RCA       — Form with scenario presets, instant prediction + history

Run:
    streamlit run dashboard/app.py

With a custom API URL:
    RCA_API_URL=http://my-rca-api:8000 streamlit run dashboard/app.py
"""

import os
import sys
from pathlib import Path

# Add the project root to sys.path so we can import src.*
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src.graph.dependency_graph import PipelineDependencyGraph
from src.graph.seed_dependencies import build_default_graph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = os.getenv("RCA_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 5  # seconds

# Root cause colour palette — consistent across all charts
# Order matches the model's label-encoder class order (alphabetical)
ROOT_CAUSE_COLORS: dict[str, str] = {
    "API Failure": "#3498db",
    "Data Quality Issue": "#1abc9c",
    "Dependency Failure": "#9b59b6",
    "Missing Data": "#f39c12",
    "Resource Exhaustion": "#27ae60",
    "Schema Change": "#e74c3c",
}

ALL_ROOT_CAUSES = list(ROOT_CAUSE_COLORS.keys())

# Pipeline names from the LEMMA-RCA dataset
KNOWN_PIPELINES = [
    "analytics_pipeline",
    "autoscale_pipeline",
    "billing_pipeline",
    "clickstream_pipeline",
    "cloud_infra_pipeline",
    "cluster_ops_pipeline",
    "container_orchestration_pipeline",
    "cost_tracking_pipeline",
    "customer_360_pipeline",
    "data_quality_pipeline",
    "deployment_pipeline",
    "error_tracking_pipeline",
    "fraud_detection_pipeline",
    "inventory_sync",
    "k8s_monitoring_pipeline",
    "latency_monitoring_pipeline",
    "load_balancer_pipeline",
    "marketing_pipeline",
    "microservice_health_pipeline",
    "ml_training_pipeline",
    "network_monitor_pipeline",
    "notification_pipeline",
    "order_etl_pipeline",
    "payment_pipeline",
    "recommendation_pipeline",
    "resource_mgmt_pipeline",
    "search_index_pipeline",
    "security_scan_pipeline",
    "service_mesh_pipeline",
    "user_etl_pipeline",
]

# Scenario presets for the Live RCA form
SCENARIO_PRESETS: dict[str, dict[str, Any]] = {
    "(custom)": {},
    "Schema change in user ETL": {
        "pipeline_name": "user_etl_pipeline",
        "task_name": "extract_users",
        "runtime": 45,
        "retry_count": 3,
        "rows_processed": 0,
        "schema_change": True,
        "upstream_failed": False,
        "error_log": "KeyError: column 'age' not found in dataframe. Schema may have changed upstream.",
    },
    "Upstream dependency failure (analytics)": {
        "pipeline_name": "analytics_pipeline",
        "task_name": "compute_kpis",
        "runtime": 600,
        "retry_count": 3,
        "rows_processed": 0,
        "schema_change": False,
        "upstream_failed": True,
        "error_log": "DependencyError: upstream pipeline 'user_etl_pipeline' has not completed. Sensor timed out after 600s.",
    },
    "API timeout in payment flow": {
        "pipeline_name": "payment_pipeline",
        "task_name": "fetch_payments",
        "runtime": 1800,
        "retry_count": 3,
        "rows_processed": 12400,
        "schema_change": False,
        "upstream_failed": False,
        "error_log": "requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='payments-api.internal', port=443): Read timed out. (read timeout=30)",
    },
    "Resource exhaustion in ML training": {
        "pipeline_name": "ml_training_pipeline",
        "task_name": "train_model",
        "runtime": 3600,
        "retry_count": 1,
        "rows_processed": 2_500_000,
        "schema_change": False,
        "upstream_failed": False,
        "error_log": "MemoryError: Unable to allocate 8.00 GiB for an array with shape (1073741824,) and data type float64. Container OOM killed.",
    },
    "Missing data in clickstream": {
        "pipeline_name": "clickstream_pipeline",
        "task_name": "sessionize",
        "runtime": 120,
        "retry_count": 2,
        "rows_processed": 0,
        "schema_change": False,
        "upstream_failed": False,
        "error_log": "FileNotFoundError: [Errno 2] No such file or directory: 's3://data-lake/clickstream/2026-03-03/events.parquet'. Partition is missing or not yet available.",
    },
    "Schema validation error in order ETL": {
        "pipeline_name": "order_etl_pipeline",
        "task_name": "transform_orders",
        "runtime": 30,
        "retry_count": 0,
        "rows_processed": 50,
        "schema_change": True,
        "upstream_failed": False,
        "error_log": "SchemaValidationError: expected 12 columns but received 11. Missing: [discount_applied]. Schema may have been updated by upstream producer.",
    },
    "K8s pod OOM in cluster ops": {
        "pipeline_name": "cluster_ops_pipeline",
        "task_name": "check_pod_health",
        "runtime": 2400,
        "retry_count": 2,
        "rows_processed": 0,
        "schema_change": False,
        "upstream_failed": False,
        "error_log": "OOMKilled: container 'data-processor' exceeded memory limit 4Gi. Pod was evicted by kubelet.",
    },
    "Data quality failure in fraud detection": {
        "pipeline_name": "fraud_detection_pipeline",
        "task_name": "flag_fraud",
        "runtime": 200,
        "retry_count": 1,
        "rows_processed": 150,
        "schema_change": False,
        "upstream_failed": False,
        "error_log": "DataQualityError: 87% of records failed null checks on field 'transaction_amount'. Data quality thresholds breached.",
    },
}


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Initialise session state keys on first run."""
    defaults = {
        "rca_history": [],       # List of past Live RCA results
        "selected_pipeline": None,  # Cross-tab pipeline focus
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# API Client — all calls go through these helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def fetch_stats(days: int = 30) -> dict[str, Any] | None:
    """Fetch aggregate prediction statistics from /stats."""
    try:
        resp = requests.get(f"{API_URL}/stats", params={"days": days}, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_distribution(days: int = 30) -> list[dict[str, Any]] | None:
    """Fetch root cause distribution from /root-cause-distribution."""
    try:
        resp = requests.get(
            f"{API_URL}/root-cause-distribution",
            params={"days": days},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return None


@st.cache_data(ttl=15, show_spinner=False)
def fetch_predictions(
    limit: int = 100,
    pipeline_name: str | None = None,
    root_cause: str | None = None,
) -> list[dict[str, Any]] | None:
    """Fetch recent predictions from /predictions."""
    params: dict[str, Any] = {"limit": limit}
    if pipeline_name:
        params["pipeline_name"] = pipeline_name
    if root_cause:
        params["root_cause"] = root_cause
    try:
        resp = requests.get(
            f"{API_URL}/predictions", params=params, timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return None


def check_api_health() -> bool:
    """Return True if the API is reachable and healthy."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=REQUEST_TIMEOUT)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def submit_prediction(payload: dict[str, Any]) -> dict[str, Any] | None:
    """POST to /predict-rca and return the result dict."""
    try:
        resp = requests.post(
            f"{API_URL}/predict-rca",
            json=payload,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"API error {resp.status_code}: {resp.text[:200]}")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to RCA API at {API_URL}. Is it running?")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The API took more than 30 seconds.")
    return None


# ---------------------------------------------------------------------------
# Graph visualisation helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def load_dependency_graph() -> PipelineDependencyGraph:
    """Load and cache the pipeline dependency graph (built once per session)."""
    return build_default_graph()


def _hierarchical_layout(G: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """
    Compute a top-down hierarchical layout for a DAG.

    Each node is assigned a depth level via BFS from all root nodes.
    Within each level, nodes are spaced evenly across [0, 1].
    """
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]

    # BFS: assign the *maximum* depth to each node
    depths: dict[str, int] = {r: 0 for r in roots}
    queue = list(roots)
    while queue:
        node = queue.pop(0)
        for successor in G.successors(node):
            new_depth = depths[node] + 1
            if successor not in depths or depths[successor] < new_depth:
                depths[successor] = new_depth
                queue.append(successor)

    # Any disconnected nodes default to depth 0
    for node in G.nodes():
        if node not in depths:
            depths[node] = 0

    # Group nodes by level
    level_nodes: dict[int, list[str]] = {}
    for node, d in depths.items():
        level_nodes.setdefault(d, []).append(node)

    max_depth = max(depths.values()) if depths else 0

    pos: dict[str, tuple[float, float]] = {}
    for level, nodes_at_level in level_nodes.items():
        n = len(nodes_at_level)
        y = 1.0 - (level / max(max_depth, 1))
        for i, node in enumerate(sorted(nodes_at_level)):
            x = (i + 0.5) / n
            pos[node] = (x, y)

    return pos


def build_graph_figure(
    graph: PipelineDependencyGraph,
    failed_pipeline: str | None = None,
    at_risk: set[str] | None = None,
    compare_pipeline: str | None = None,
    compare_at_risk: set[str] | None = None,
) -> go.Figure:
    """
    Build an interactive Plotly figure from the dependency graph.

    Args:
        graph: The PipelineDependencyGraph to visualise.
        failed_pipeline: The node to highlight in red (simulated failure).
        at_risk: Set of downstream nodes to highlight in orange.
        compare_pipeline: Optional second pipeline to highlight in purple.
        compare_at_risk: Optional second set of at-risk nodes (light purple).

    Returns:
        A Plotly Figure with nodes, directed edges, and hover tooltips.
    """
    G = graph._graph  # noqa: SLF001
    pos = _hierarchical_layout(G)

    if at_risk is None:
        at_risk = set()
    if compare_at_risk is None:
        compare_at_risk = set()

    # --- Edge traces ---
    edge_traces: list[go.Scatter] = []
    annotations: list[dict] = []

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        if failed_pipeline and (u == failed_pipeline or u in at_risk):
            color = "rgba(231, 76, 60, 0.6)"
            width = 2.5
        elif compare_pipeline and (u == compare_pipeline or u in compare_at_risk):
            color = "rgba(155, 89, 182, 0.6)"
            width = 2.5
        else:
            color = "rgba(120, 120, 150, 0.3)"
            width = 1.5

        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="none",
                showlegend=False,
            )
        )

        annotations.append(
            dict(
                ax=x0, ay=y0,
                axref="x", ayref="y",
                x=x1, y=y1,
                xref="x", yref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=1.5,
                arrowcolor=color,
            )
        )

    # --- Node grouping ---
    node_groups: dict[str, list[str]] = {
        "failed": [],
        "at_risk": [],
        "compare": [],
        "compare_risk": [],
        "normal": [],
    }
    for node in G.nodes():
        if failed_pipeline and node == failed_pipeline:
            node_groups["failed"].append(node)
        elif compare_pipeline and node == compare_pipeline:
            node_groups["compare"].append(node)
        elif node in at_risk:
            node_groups["at_risk"].append(node)
        elif node in compare_at_risk:
            node_groups["compare_risk"].append(node)
        else:
            node_groups["normal"].append(node)

    group_config = {
        "failed": {"color": "#e74c3c", "size": 24, "label": "Failed"},
        "at_risk": {"color": "#f39c12", "size": 19, "label": "At risk (A)"},
        "compare": {"color": "#9b59b6", "size": 24, "label": "Compare"},
        "compare_risk": {"color": "#d5a6e6", "size": 19, "label": "At risk (B)"},
        "normal": {"color": "#3b82f6", "size": 16, "label": "Pipeline"},
    }

    node_traces: list[go.Scatter] = []
    for group, nodes in node_groups.items():
        if not nodes:
            continue
        cfg = group_config[group]
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]

        hover_texts = []
        for n in nodes:
            feats = graph.get_node_features(n)
            hover_texts.append(
                f"<b>{n}</b><br>"
                f"In-degree: {feats['graph_in_degree']}<br>"
                f"Out-degree: {feats['graph_out_degree']}<br>"
                f"Downstream: {feats['graph_downstream_count']}<br>"
                f"Depth: {feats['graph_depth']}<br>"
                f"PageRank: {feats['graph_pagerank']:.4f}<br>"
                f"Betweenness: {feats['graph_betweenness']:.4f}"
            )

        node_traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(
                    size=cfg["size"],
                    color=cfg["color"],
                    symbol="circle",
                    line=dict(width=2, color="white"),
                ),
                text=[n.replace("_pipeline", "").replace("_", " ") for n in nodes],
                textposition="top center",
                textfont=dict(size=9, color="#334155"),
                hovertext=hover_texts,
                hoverinfo="text",
                name=cfg["label"],
                showlegend=(group != "normal"),
            )
        )

    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        annotations=annotations,
        plot_bgcolor="rgba(248,250,252,1)",
        paper_bgcolor="rgba(248,250,252,1)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=30, b=10),
        height=560,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def render_overview(api_online: bool, days: int) -> None:
    """Tab 1 — KPI cards, trend timeline, root cause distribution."""
    if not api_online:
        st.warning(
            f"The RCA API at **{API_URL}** is not reachable. "
            "Start the stack with `docker compose up` to see live LEMMA data.",
            icon="⚠️",
        )
        return

    stats = fetch_stats(days)
    dist = fetch_distribution(days)

    # --- KPI cards ---
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", f"{stats['total_predictions']:,}")
        with col2:
            conf = stats["avg_confidence"]
            st.metric("Avg Confidence", f"{conf:.1%}")
        with col3:
            most_common = stats.get("most_common_root_cause") or "—"
            st.metric("Top Root Cause", most_common)
        with col4:
            st.metric("Period", f"Last {stats['period_days']}d")

    st.divider()

    # --- Trend timeline (stacked area chart) ---
    st.markdown("#### Failure Trend Over Time")
    df_timeline = None
    all_preds = fetch_predictions(limit=1000)
    if all_preds:
        df_preds = pd.DataFrame(all_preds)
        df_preds["date"] = pd.to_datetime(df_preds["created_at"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        df_preds = df_preds[df_preds["date"] >= cutoff]
        if not df_preds.empty:
            df_timeline = (
                df_preds.groupby(["date", "predicted_root_cause"])
                .size()
                .reset_index(name="count")
                .rename(columns={"predicted_root_cause": "root_cause"})
            )
    if df_timeline is None:
        st.caption("No prediction data in the selected time window.")
        return
    fig_trend = px.area(
        df_timeline,
        x="date",
        y="count",
        color="root_cause",
        color_discrete_map=ROOT_CAUSE_COLORS,
        labels={"count": "Failures", "date": "Date", "root_cause": "Root Cause"},
    )
    fig_trend.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    fig_trend.update_xaxes(showgrid=False)
    fig_trend.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()

    # --- Root cause distribution ---
    if dist:
        df_dist = pd.DataFrame(dist)
        df_dist["color"] = df_dist["root_cause"].map(
            lambda rc: ROOT_CAUSE_COLORS.get(rc, "#95a5a6")
        )

        col_pie, col_bar = st.columns(2)

        with col_pie:
            st.markdown("**Root Cause Distribution**")
            fig_pie = go.Figure(
                go.Pie(
                    labels=df_dist["root_cause"],
                    values=df_dist["count"],
                    marker=dict(colors=df_dist["color"].tolist()),
                    hole=0.45,
                    textinfo="label+percent",
                    textfont_size=11,
                    pull=[0.03] * len(df_dist),
                )
            )
            fig_pie.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=20, b=0),
                height=320,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            st.markdown("**Failure Count by Root Cause**")
            df_sorted = df_dist.sort_values("count", ascending=True)
            fig_bar = go.Figure(
                go.Bar(
                    x=df_sorted["count"],
                    y=df_sorted["root_cause"],
                    orientation="h",
                    marker=dict(color=df_sorted["color"].tolist()),
                    text=df_sorted["count"],
                    textposition="outside",
                )
            )
            fig_bar.update_layout(
                xaxis_title="Count",
                yaxis_title="",
                margin=dict(l=0, r=40, t=20, b=20),
                height=320,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- Pipeline heatmap ---
        st.divider()
        st.markdown("#### Pipeline Failure Heatmap")
        st.caption("Which pipelines fail most often, by root cause")

        rows = fetch_predictions(limit=1000)
        if rows:
            df_heat = pd.DataFrame(rows)
            # Filter to selected days window
            df_heat["date"] = pd.to_datetime(df_heat["created_at"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
            df_heat = df_heat[df_heat["date"] >= cutoff]
            heat_pivot = pd.crosstab(df_heat["pipeline_name"], df_heat["predicted_root_cause"])
            # Ensure all root causes are present as columns
            for rc in ALL_ROOT_CAUSES:
                if rc not in heat_pivot.columns:
                    heat_pivot[rc] = 0
            heat_pivot = heat_pivot[ALL_ROOT_CAUSES]

            fig_heat = go.Figure(
                go.Heatmap(
                    z=heat_pivot.values,
                    x=heat_pivot.columns.tolist(),
                    y=[p.replace("_pipeline", "").replace("_", " ") for p in heat_pivot.index],
                    colorscale="YlOrRd",
                    hovertemplate="<b>%{y}</b><br>%{x}: %{z} failures<extra></extra>",
                )
            )
            fig_heat.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                height=max(250, len(heat_pivot) * 28),
                xaxis=dict(side="top"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)


def render_predictions(api_online: bool) -> None:
    """Tab 2 — Predictions with filters, confidence chart, and expandable detail."""
    # --- Filters row ---
    col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 2, 1])
    with col_f1:
        pipeline_filter = st.selectbox(
            "Pipeline",
            ["All"] + KNOWN_PIPELINES,
            index=0,
            key="pred_pipeline_filter",
        )
    with col_f2:
        cause_filter = st.selectbox(
            "Root cause",
            ["All"] + ALL_ROOT_CAUSES,
            index=0,
            key="pred_cause_filter",
        )
    with col_f3:
        conf_threshold = st.slider(
            "Min confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            format="%.0f%%",
            key="pred_conf_threshold",
        )
    with col_f4:
        limit = st.number_input("Limit", min_value=10, max_value=200, value=50, step=10)

    pipeline_arg = None if pipeline_filter == "All" else pipeline_filter
    cause_arg = None if cause_filter == "All" else cause_filter

    if not api_online:
        st.warning(
            f"The RCA API at **{API_URL}** is not reachable. "
            "Start the stack with `docker compose up` to see live LEMMA data.",
            icon="⚠️",
        )
        return

    rows = fetch_predictions(limit=int(limit), pipeline_name=pipeline_arg, root_cause=cause_arg)

    if not rows:
        st.info("No predictions found for the selected filters.")
        return

    # Apply confidence threshold
    rows = [r for r in rows if r["confidence"] >= conf_threshold]
    rows = rows[:int(limit)]

    if not rows:
        st.info(f"No predictions with confidence >= {conf_threshold:.0%}.")
        return

    df = pd.DataFrame(rows)
    df["created_at_dt"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")

    # --- Confidence distribution chart ---
    st.markdown("#### Confidence Distribution")
    col_chart, col_stats = st.columns([3, 1])

    with col_chart:
        fig_conf = go.Figure()
        for rc in ALL_ROOT_CAUSES:
            rc_data = df[df["predicted_root_cause"] == rc]["confidence"]
            if rc_data.empty:
                continue
            fig_conf.add_trace(
                go.Box(
                    y=rc_data,
                    name=rc.replace(" ", "\n"),
                    marker_color=ROOT_CAUSE_COLORS[rc],
                    boxpoints="all",
                    jitter=0.4,
                    pointpos=-1.5,
                    marker=dict(size=4, opacity=0.5),
                )
            )
        fig_conf.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            height=200,
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Confidence",
        )
        fig_conf.update_yaxes(range=[0, 1.05], gridcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig_conf, use_container_width=True)

    with col_stats:
        st.metric("Shown", f"{len(df)} predictions")
        st.metric("Avg Conf", f"{df['confidence'].mean():.1%}")
        st.metric("Min Conf", f"{df['confidence'].min():.1%}")
        if len(df) > 1:
            high_conf = (df["confidence"] >= 0.8).sum()
            st.metric("High Conf (>80%)", f"{high_conf} ({high_conf/len(df):.0%})")

    st.divider()

    # --- Predictions table ---
    st.markdown("#### Prediction Feed")
    df["confidence_pct"] = df["confidence"].map(lambda c: f"{c:.1%}")
    df["created_at_str"] = df["created_at_dt"].dt.strftime("%Y-%m-%d %H:%M")

    display_cols = {
        "pipeline_name": "Pipeline",
        "predicted_root_cause": "Root Cause",
        "confidence_pct": "Confidence",
        "model_version": "Model",
        "created_at_str": "Timestamp",
    }
    df_display = df[list(display_cols.keys())].rename(columns=display_cols)

    def highlight_cause(val: str) -> str:
        color = ROOT_CAUSE_COLORS.get(val, "#95a5a6")
        return f"background-color: {color}22; color: #1a1a2e; font-weight: 500;"

    st.dataframe(
        df_display.style.map(highlight_cause, subset=["Root Cause"]),
        use_container_width=True,
        hide_index=True,
        height=350,
    )

    # --- Expandable evidence details ---
    st.markdown("#### Evidence Details")
    st.caption("Expand a prediction to see its full SHAP evidence")

    # Show the most recent N as expandable cards
    for idx, row in enumerate(rows[:20]):
        label = (
            f"**{row['pipeline_name']}** — "
            f"{row['predicted_root_cause']} "
            f"({row['confidence']:.1%})"
        )
        with st.expander(label, expanded=False):
            ev_col, meta_col = st.columns([3, 1])
            with ev_col:
                evidence = row.get("evidence", [])
                if evidence:
                    for item in evidence:
                        st.markdown(f"- {item}")
                else:
                    st.caption("No evidence available.")
            with meta_col:
                st.caption(f"**ID:** {row.get('id', '—')}")
                st.caption(f"**Model:** {row.get('model_version', '—')}")
                created = row.get("created_at", "")
                if created:
                    st.caption(f"**Time:** {created[:16]}")


def render_graph(graph: PipelineDependencyGraph) -> None:
    """Tab 3 — Interactive DAG with cascade simulation and compare mode."""

    # --- Controls ---
    col_mode, col_a, col_b = st.columns([1, 2, 2])

    with col_mode:
        compare_mode = st.toggle("Compare two pipelines", value=False, key="compare_toggle")

    pipeline_options = ["(none)"] + sorted(graph.nodes)

    with col_a:
        pipeline_a = st.selectbox(
            "Pipeline A (failure):" if compare_mode else "Simulate a failure in:",
            pipeline_options,
            index=0,
            key="graph_pipeline_a",
        )
    with col_b:
        if compare_mode:
            pipeline_b = st.selectbox(
                "Pipeline B (compare):",
                pipeline_options,
                index=0,
                key="graph_pipeline_b",
            )
        else:
            pipeline_b = "(none)"

    failed_a = None if pipeline_a == "(none)" else pipeline_a
    failed_b = None if pipeline_b == "(none)" else pipeline_b

    risk_a: set[str] = set()
    risk_b: set[str] = set()

    # --- Cascade risk info ---
    if failed_a or failed_b:
        info_cols = st.columns(2 if compare_mode and failed_b else 1)

        if failed_a:
            risk_data_a = graph.detect_cascade_risk(failed_a)
            risk_a = set(risk_data_a["total_at_risk"])
            with info_cols[0]:
                sev_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
                    risk_data_a["severity"], "⚪"
                )
                st.markdown(
                    f"**{failed_a}** &nbsp; {sev_icon} **{risk_data_a['severity'].upper()}** — "
                    f"blast radius: **{risk_data_a['blast_radius']}**"
                )
                if risk_data_a["direct_dependents"]:
                    st.caption(
                        "Direct: " + ", ".join(sorted(risk_data_a["direct_dependents"]))
                    )
                if risk_data_a["total_at_risk"]:
                    st.caption(
                        "All at risk: " + ", ".join(sorted(risk_data_a["total_at_risk"]))
                    )

        if compare_mode and failed_b:
            risk_data_b = graph.detect_cascade_risk(failed_b)
            risk_b = set(risk_data_b["total_at_risk"])
            with info_cols[-1]:
                sev_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
                    risk_data_b["severity"], "⚪"
                )
                st.markdown(
                    f"**{failed_b}** &nbsp; {sev_icon} **{risk_data_b['severity'].upper()}** — "
                    f"blast radius: **{risk_data_b['blast_radius']}**"
                )
                if risk_data_b["total_at_risk"]:
                    st.caption(
                        "All at risk: " + ", ".join(sorted(risk_data_b["total_at_risk"]))
                    )

            # Overlap analysis
            overlap = risk_a & risk_b
            only_a = risk_a - risk_b
            only_b = risk_b - risk_a
            if overlap:
                st.warning(
                    f"**Shared blast zone:** {len(overlap)} pipeline(s) affected by both failures — "
                    + ", ".join(f"`{p}`" for p in sorted(overlap)),
                    icon="⚠️",
                )

    # --- Graph figure ---
    fig = build_graph_figure(
        graph,
        failed_pipeline=failed_a,
        at_risk=risk_a,
        compare_pipeline=failed_b if compare_mode else None,
        compare_at_risk=risk_b if compare_mode else None,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Node detail panel ---
    st.divider()
    detail_pipeline = st.selectbox(
        "Inspect pipeline details:",
        ["(select a pipeline)"] + sorted(graph.nodes),
        index=0,
        key="graph_detail_select",
    )

    if detail_pipeline != "(select a pipeline)":
        feats = graph.get_node_features(detail_pipeline)
        upstream = graph.get_upstream_pipelines(detail_pipeline)
        downstream = graph.get_downstream_pipelines(detail_pipeline)

        col_feat, col_up, col_down = st.columns(3)

        with col_feat:
            st.markdown(f"**Graph features for `{detail_pipeline}`**")
            for k, v in feats.items():
                label = k.replace("graph_", "").replace("_", " ").title()
                display_val = f"{v:.4f}" if isinstance(v, float) else str(v)
                st.text(f"  {label}: {display_val}")

        with col_up:
            st.markdown("**Upstream (depends on)**")
            if upstream:
                for p in sorted(upstream):
                    st.text(f"  {p}")
            else:
                st.caption("  (root node — no upstream)")

        with col_down:
            st.markdown("**Downstream (depended on by)**")
            if downstream:
                for p in sorted(downstream):
                    st.text(f"  {p}")
            else:
                st.caption("  (leaf node — no downstream)")

    # --- Graph stats ---
    with st.expander("Graph summary statistics"):
        s1, s2, s3, s4 = st.columns(4)
        roots = [n for n in graph.nodes if graph.get_upstream_pipelines(n) == []]
        leaves = [n for n in graph.nodes if graph.get_downstream_pipelines(n) == []]
        with s1:
            st.metric("Nodes", graph.node_count)
        with s2:
            st.metric("Edges", graph.edge_count)
        with s3:
            st.metric("Root pipelines", len(roots))
        with s4:
            st.metric("Leaf pipelines", len(leaves))


def render_live_rca(api_online: bool) -> None:
    """Tab 4 — Form with scenario presets, prediction + session history."""

    if not api_online:
        st.warning(
            f"The RCA API at **{API_URL}** is not reachable. "
            "Start it with `uvicorn api.main:app --port 8000`.",
            icon="⚠️",
        )
        return

    # --- Scenario preset selector ---
    st.markdown("#### Quick-fill from a scenario preset")
    preset_name = st.selectbox(
        "Scenario",
        list(SCENARIO_PRESETS.keys()),
        index=0,
        key="scenario_preset",
    )
    preset = SCENARIO_PRESETS[preset_name]

    st.divider()

    # --- Form ---
    with st.form("rca_form"):
        col1, col2 = st.columns(2)
        with col1:
            pipeline_name = st.selectbox(
                "Pipeline name",
                KNOWN_PIPELINES,
                index=KNOWN_PIPELINES.index(preset["pipeline_name"]) if preset.get("pipeline_name") in KNOWN_PIPELINES else 0,
                key="form_pipeline",
            )
            task_name = st.text_input(
                "Task name",
                value=preset.get("task_name", "extract_data"),
                key="form_task",
            )
            runtime = st.number_input(
                "Runtime (seconds)",
                min_value=0,
                max_value=86400,
                value=preset.get("runtime", 300),
            )
            retry_count = st.number_input(
                "Retry count",
                min_value=0,
                max_value=10,
                value=preset.get("retry_count", 1),
            )
        with col2:
            rows_processed = st.number_input(
                "Rows processed",
                min_value=0,
                value=preset.get("rows_processed", 50_000),
                step=1_000,
            )
            schema_change = st.checkbox(
                "Schema change detected?",
                value=preset.get("schema_change", False),
            )
            upstream_failed = st.checkbox(
                "Upstream pipeline failed?",
                value=preset.get("upstream_failed", False),
            )
            error_log = st.text_area(
                "Error log snippet",
                value=preset.get("error_log", ""),
                height=100,
            )

        submitted = st.form_submit_button("Run RCA", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "pipeline_name": pipeline_name,
            "task_name": task_name,
            "runtime": int(runtime),
            "retry_count": int(retry_count),
            "rows_processed": int(rows_processed),
            "schema_change": schema_change,
            "upstream_failed": upstream_failed,
            "error_log": error_log or "Error: pipeline task failed unexpectedly.",
        }

        with st.spinner("Running inference…"):
            result = submit_prediction(payload)

        if result:
            # Store in session history
            st.session_state["rca_history"].append({
                **result,
                "input_pipeline": pipeline_name,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            })

            rc = result.get("predicted_root_cause", "Unknown")
            conf = result.get("confidence", 0.0)
            rc_color = ROOT_CAUSE_COLORS.get(rc, "#95a5a6")

            st.markdown(
                f"""
                <div style="
                    background:{rc_color}22;
                    border-left: 4px solid {rc_color};
                    padding: 1rem 1.25rem;
                    border-radius: 0.5rem;
                    margin: 0.5rem 0 1rem 0;
                ">
                    <div style="font-size:1.4rem; font-weight:700; color:{rc_color};">
                        {rc}
                    </div>
                    <div style="color:#334155; margin-top:0.25rem;">
                        Confidence: <strong>{conf:.1%}</strong> &nbsp;|&nbsp;
                        Model: <strong>{result.get('model_version', 'v1')}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            evidence = result.get("evidence", [])
            if evidence:
                st.markdown("**Evidence:**")
                for item in evidence:
                    st.markdown(f"- {item}")

            # Cascade risk
            graph = load_dependency_graph()
            risk = graph.detect_cascade_risk(pipeline_name)
            if risk["blast_radius"] > 0:
                sev_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
                    risk["severity"], "⚪"
                )
                st.markdown(
                    f"**Cascade Risk:** {sev_icon} {risk['severity'].upper()} — "
                    f"{risk['blast_radius']} pipeline(s) at risk: "
                    + ", ".join(f"`{p}`" for p in sorted(risk["total_at_risk"]))
                )

    # --- Session history ---
    history = st.session_state.get("rca_history", [])
    if history:
        st.divider()
        st.markdown("#### Session History")
        st.caption(f"{len(history)} prediction(s) made this session")

        for i, entry in enumerate(reversed(history)):
            rc = entry.get("predicted_root_cause", "?")
            conf = entry.get("confidence", 0)
            rc_color = ROOT_CAUSE_COLORS.get(rc, "#95a5a6")
            pipe = entry.get("input_pipeline", entry.get("pipeline_name", "?"))
            ts = entry.get("submitted_at", "")[:16]

            st.markdown(
                f"<div style='display:flex;align-items:center;gap:0.75rem;padding:0.4rem 0;'>"
                f"<span style='background:{rc_color}33;color:{rc_color};padding:2px 8px;"
                f"border-radius:4px;font-weight:600;font-size:0.85rem;'>{rc}</span>"
                f"<span style='color:#64748b;'>{pipe}</span>"
                f"<span style='color:#94a3b8;font-size:0.8rem;'>{conf:.1%}</span>"
                f"<span style='color:#cbd5e1;font-size:0.75rem;margin-left:auto;'>{ts}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if st.button("Clear history", key="clear_history"):
            st.session_state["rca_history"] = []
            st.rerun()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="AIOps Pipeline RCA",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session_state()

    # --- Sidebar ---
    with st.sidebar:
        st.title("🔍 AIOps RCA")
        st.caption("Pipeline Root Cause Analysis")
        st.divider()

        api_online = check_api_health()
        if api_online:
            st.success(f"API connected\n`{API_URL}`", icon="✅")
        else:
            st.error(f"API offline\n`{API_URL}`", icon="🔴")

        st.divider()
        days = st.slider(
            "Stats window (days)",
            min_value=1,
            max_value=90,
            value=30,
            step=1,
        )

        st.divider()
        auto_refresh = st.toggle("Auto-refresh", value=False, key="auto_refresh_toggle")
        if auto_refresh:
            refresh_secs = st.slider(
                "Refresh interval (sec)",
                min_value=10,
                max_value=120,
                value=30,
                step=5,
                key="refresh_interval",
            )
            st.caption(f"Refreshing every {refresh_secs}s")

        st.divider()
        if st.button("🔄 Refresh now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption(
            "Built with Streamlit + Plotly + NetworkX\n"
            "Data from the FastAPI RCA service"
        )

    # --- Auto-refresh via JavaScript page reload ---
    if auto_refresh:
        import streamlit.components.v1 as _components
        _components.html(
            f"<script>setTimeout(function(){{window.parent.location.reload();}}, {refresh_secs * 1000});</script>",
            height=0,
        )

    # --- Main content ---
    st.title("AIOps Pipeline RCA — Monitoring Dashboard")

    tab_overview, tab_predictions, tab_graph, tab_live = st.tabs(
        ["📊 Overview", "📋 Predictions", "🕸️ Graph Explorer", "⚡ Live RCA"]
    )

    graph = load_dependency_graph()

    with tab_overview:
        render_overview(api_online, days)

    with tab_predictions:
        render_predictions(api_online)

    with tab_graph:
        render_graph(graph)

    with tab_live:
        render_live_rca(api_online)


if __name__ == "__main__":
    main()
