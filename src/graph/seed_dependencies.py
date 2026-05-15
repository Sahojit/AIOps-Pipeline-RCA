"""
src/graph/seed_dependencies.py
==============================
Seeds the pipeline dependency graph with realistic relationships.

All 30 LEMMA-RCA dataset pipelines are represented across four tiers:
- Infrastructure tier  (cloud_infra, network, k8s, cluster, security)
- Platform tier        (autoscale, load_balancer, container, deployment, resource_mgmt, service_mesh)
- Data / ETL tier      (user_etl, order_etl, clickstream, inventory, payment)
- Analytics / ML tier  (analytics, recommendation, customer_360, fraud, ml_training, search, marketing)
- Observability tier   (latency_monitoring, microservice_health, error_tracking)
- Business tier        (billing, notification, cost_tracking, data_quality)

Usage:
    python -m src.graph.seed_dependencies

This populates:
1. The PipelineDependencyGraph (in-memory, for feature extraction)
2. Optionally, the pipeline_dependencies database table
"""

from src.graph.dependency_graph import PipelineDependencyGraph
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# The dependency registry — ground truth for the pipeline DAG.
# Format: (upstream, downstream, type)
# ---------------------------------------------------------------------------
PIPELINE_DEPENDENCIES: list[tuple[str, str, str]] = [

    # ── Infrastructure tier (no upstream dependencies) ────────────────────
    # k8s monitoring feeds autoscale decisions
    ("k8s_monitoring_pipeline",         "autoscale_pipeline",               "sensor"),
    # network monitor feeds load balancer config
    ("network_monitor_pipeline",        "load_balancer_pipeline",           "sensor"),
    # cluster ops feeds container orchestration
    ("cluster_ops_pipeline",            "container_orchestration_pipeline", "data"),
    # cloud infra feeds resource management
    ("cloud_infra_pipeline",            "resource_mgmt_pipeline",           "data"),
    # security scans gate deployments
    ("security_scan_pipeline",          "deployment_pipeline",              "sensor"),

    # ── Platform tier ────────────────────────────────────────────────────
    # container orchestration + load balancer → service mesh
    ("container_orchestration_pipeline","service_mesh_pipeline",            "data"),
    ("load_balancer_pipeline",          "service_mesh_pipeline",            "data"),
    # deployment needs resource budget
    ("resource_mgmt_pipeline",          "deployment_pipeline",              "sensor"),
    # service mesh feeds latency monitoring
    ("service_mesh_pipeline",           "latency_monitoring_pipeline",      "sensor"),
    # microservice health depends on service mesh
    ("service_mesh_pipeline",           "microservice_health_pipeline",     "sensor"),
    # error tracking depends on deployment and microservice health
    ("deployment_pipeline",             "error_tracking_pipeline",          "trigger"),
    ("microservice_health_pipeline",    "error_tracking_pipeline",          "sensor"),
    # cost tracking depends on resource usage + billing
    ("resource_mgmt_pipeline",          "cost_tracking_pipeline",           "data"),

    # ── Data / ETL tier ───────────────────────────────────────────────────
    # user_etl is the most critical — feeds 5 downstream pipelines
    ("user_etl_pipeline",               "recommendation_pipeline",          "data"),
    ("user_etl_pipeline",               "customer_360_pipeline",            "data"),
    ("user_etl_pipeline",               "analytics_pipeline",               "data"),
    ("user_etl_pipeline",               "fraud_detection_pipeline",         "data"),
    ("user_etl_pipeline",               "notification_pipeline",            "trigger"),

    # order_etl feeds analytics and payment
    ("order_etl_pipeline",              "analytics_pipeline",               "data"),
    ("order_etl_pipeline",              "payment_pipeline",                 "data"),
    ("order_etl_pipeline",              "fraud_detection_pipeline",         "data"),

    # clickstream feeds analytics and recommendations
    ("clickstream_pipeline",            "analytics_pipeline",               "data"),
    ("clickstream_pipeline",            "recommendation_pipeline",          "data"),

    # inventory feeds recommendations and search
    ("inventory_sync",                  "recommendation_pipeline",          "data"),
    ("inventory_sync",                  "search_index_pipeline",            "data"),

    # ── Analytics / ML tier ───────────────────────────────────────────────
    # analytics feeds marketing
    ("analytics_pipeline",              "marketing_pipeline",               "data"),
    # ML training depends on recommendation features
    ("recommendation_pipeline",         "ml_training_pipeline",             "data"),

    # ── Business tier ─────────────────────────────────────────────────────
    # payment → billing → notification chain
    ("payment_pipeline",                "billing_pipeline",                 "data"),
    ("billing_pipeline",                "notification_pipeline",            "trigger"),
    ("billing_pipeline",                "cost_tracking_pipeline",           "data"),

    # ── Quality tier ──────────────────────────────────────────────────────
    ("customer_360_pipeline",           "data_quality_pipeline",            "sensor"),
    ("analytics_pipeline",              "data_quality_pipeline",            "sensor"),
]


def build_default_graph() -> PipelineDependencyGraph:
    """
    Build the default pipeline dependency graph from the registry.

    Returns:
        Populated PipelineDependencyGraph instance.
    """
    graph = PipelineDependencyGraph()
    graph.add_dependencies_bulk(PIPELINE_DEPENDENCIES)
    logger.info(
        "Default dependency graph built: %d nodes, %d edges",
        graph.node_count, graph.edge_count,
    )
    return graph


def seed_database_dependencies() -> None:
    """
    Seed the pipeline_dependencies database table.

    This is for when you want the graph persisted in PostgreSQL
    (useful for the dashboard and API queries).
    """
    from database.crud import create_pipeline_dependency
    from database.session import SessionLocal

    logger.info("Seeding pipeline dependencies in database")
    db = SessionLocal()
    try:
        for upstream, downstream, dep_type in PIPELINE_DEPENDENCIES:
            try:
                create_pipeline_dependency(
                    db=db,
                    upstream_pipeline=upstream,
                    downstream_pipeline=downstream,
                    dependency_type=dep_type,
                )
            except Exception as e:
                db.rollback()
                logger.debug("Dependency %s→%s may already exist: %s", upstream, downstream, e)
                continue
        db.commit()
        logger.info("Seeded %d dependencies in database", len(PIPELINE_DEPENDENCIES))
    finally:
        db.close()


def main() -> None:
    """CLI entry point — build graph and print summary."""
    from src.utils.logger import configure_root_logger
    configure_root_logger("INFO")

    graph = build_default_graph()

    print("\n" + "=" * 60)
    print("PIPELINE DEPENDENCY GRAPH")
    print("=" * 60)
    print(f"Nodes (pipelines): {graph.node_count}")
    print(f"Edges (dependencies): {graph.edge_count}")

    print("\nDependency edges:")
    for u, v in graph.edges:
        print(f"  {u} → {v}")

    print("\nNode features for key pipelines:")
    for pipeline in ["user_etl_pipeline", "k8s_monitoring_pipeline", "service_mesh_pipeline", "data_quality_pipeline"]:
        features = graph.get_node_features(pipeline)
        print(f"\n  {pipeline}:")
        for k, v in features.items():
            print(f"    {k}: {v}")

    print("\nCascade risk for user_etl_pipeline:")
    risk = graph.detect_cascade_risk("user_etl_pipeline")
    print(f"  Blast radius: {risk['blast_radius']} pipelines")
    print(f"  Severity: {risk['severity']}")
    print(f"  At risk: {risk['total_at_risk']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
