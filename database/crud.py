"""
database/crud.py
================
Database CRUD (Create, Read, Update, Delete) operations.

WHY a separate CRUD layer?
- API endpoints should NOT contain raw SQL or ORM queries
- CRUD functions are reusable: API, CLI scripts, and tests all use them
- Easy to unit test — mock the Session, assert the right queries were made
- If you switch from PostgreSQL to another DB, only this file changes

Each function takes a Session and returns ORM objects.
The CALLER is responsible for committing or rolling back.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.models import PipelineDependency, PipelineFailure, RCAPrediction
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =========================================================================
# Pipeline Failures — CREATE + READ
# =========================================================================

def create_pipeline_failure(
    db: Session,
    pipeline_name: str,
    task_name: str,
    runtime: int,
    retry_count: int,
    rows_processed: int,
    schema_change: bool,
    upstream_failed: bool,
    error_log: str,
    event_timestamp: datetime,
) -> PipelineFailure:
    """
    Store a raw pipeline failure event.

    Called by the /predict-rca endpoint BEFORE running inference,
    so we have a record of every event even if inference fails.
    """
    failure = PipelineFailure(
        pipeline_name=pipeline_name,
        task_name=task_name,
        runtime=runtime,
        retry_count=retry_count,
        rows_processed=rows_processed,
        schema_change=schema_change,
        upstream_failed=upstream_failed,
        error_log=error_log,
        event_timestamp=event_timestamp,
    )
    db.add(failure)
    db.flush()  # Assigns the ID without committing — caller commits
    logger.info("Created pipeline failure record: %s", failure.id)
    return failure


def get_recent_failures(
    db: Session,
    limit: int = 50,
    pipeline_name: Optional[str] = None,
) -> list[PipelineFailure]:
    """Get recent failures, optionally filtered by pipeline."""
    query = select(PipelineFailure).order_by(PipelineFailure.created_at.desc())
    if pipeline_name:
        query = query.where(PipelineFailure.pipeline_name == pipeline_name)
    query = query.limit(limit)
    return list(db.scalars(query).all())


# =========================================================================
# RCA Predictions — CREATE + READ + ANALYTICS
# =========================================================================

def create_rca_prediction(
    db: Session,
    failure_id: uuid.UUID,
    pipeline_name: str,
    predicted_root_cause: str,
    confidence: float,
    evidence: list[str],
    model_version: str,
) -> RCAPrediction:
    """
    Store a model prediction linked to a failure event.

    Called by the /predict-rca endpoint AFTER successful inference.
    """
    prediction = RCAPrediction(
        failure_id=failure_id,
        pipeline_name=pipeline_name,
        predicted_root_cause=predicted_root_cause,
        confidence=confidence,
        evidence=evidence,
        model_version=model_version,
    )
    db.add(prediction)
    db.flush()
    logger.info(
        "Created RCA prediction: %s → %s (conf=%.4f)",
        pipeline_name, predicted_root_cause, confidence,
    )
    return prediction


def get_recent_predictions(
    db: Session,
    limit: int = 50,
    pipeline_name: Optional[str] = None,
    root_cause: Optional[str] = None,
) -> list[RCAPrediction]:
    """Get recent predictions with optional filters."""
    query = select(RCAPrediction).order_by(RCAPrediction.created_at.desc())
    if pipeline_name:
        query = query.where(RCAPrediction.pipeline_name == pipeline_name)
    if root_cause:
        query = query.where(RCAPrediction.predicted_root_cause == root_cause)
    query = query.limit(limit)
    return list(db.scalars(query).all())


def get_prediction_by_id(
    db: Session,
    prediction_id: uuid.UUID,
) -> Optional[RCAPrediction]:
    """Get a single prediction by ID."""
    return db.get(RCAPrediction, prediction_id)


def get_root_cause_distribution(
    db: Session,
    days: int = 30,
) -> list[dict[str, Any]]:
    """
    Get root cause distribution for the last N days.
    Used by the monitoring dashboard.

    Returns list of dicts: [{"root_cause": "Schema Change", "count": 42}, ...]
    """
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    query = (
        select(
            RCAPrediction.predicted_root_cause,
            func.count().label("count"),
        )
        .where(RCAPrediction.created_at >= cutoff)
        .group_by(RCAPrediction.predicted_root_cause)
        .order_by(func.count().desc())
    )

    results = db.execute(query).all()
    return [{"root_cause": r[0], "count": r[1]} for r in results]


def get_prediction_stats(
    db: Session,
    days: int = 30,
) -> dict[str, Any]:
    """
    Get aggregate statistics for the dashboard.

    Returns: total predictions, avg confidence, most common root cause, etc.
    """
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Total predictions
    total = db.scalar(
        select(func.count()).select_from(RCAPrediction).where(
            RCAPrediction.created_at >= cutoff
        )
    ) or 0

    # Average confidence
    avg_conf = db.scalar(
        select(func.avg(RCAPrediction.confidence)).where(
            RCAPrediction.created_at >= cutoff
        )
    )

    # Most common root cause
    most_common_query = (
        select(
            RCAPrediction.predicted_root_cause,
            func.count().label("count"),
        )
        .where(RCAPrediction.created_at >= cutoff)
        .group_by(RCAPrediction.predicted_root_cause)
        .order_by(func.count().desc())
        .limit(1)
    )
    most_common_row = db.execute(most_common_query).first()
    most_common = most_common_row[0] if most_common_row else None

    return {
        "total_predictions": total,
        "avg_confidence": round(float(avg_conf), 4) if avg_conf else 0.0,
        "most_common_root_cause": most_common,
        "period_days": days,
    }


# =========================================================================
# Pipeline Dependencies — CREATE + READ (for Phase 10)
# =========================================================================

def create_pipeline_dependency(
    db: Session,
    upstream_pipeline: str,
    downstream_pipeline: str,
    dependency_type: str = "data",
) -> PipelineDependency:
    """Add a dependency edge to the pipeline graph."""
    dep = PipelineDependency(
        upstream_pipeline=upstream_pipeline,
        downstream_pipeline=downstream_pipeline,
        dependency_type=dependency_type,
    )
    db.add(dep)
    db.flush()
    logger.info("Created dependency: %s → %s", upstream_pipeline, downstream_pipeline)
    return dep


def get_all_dependencies(
    db: Session,
    active_only: bool = True,
) -> list[PipelineDependency]:
    """Get all pipeline dependencies (for building the NetworkX graph)."""
    query = select(PipelineDependency)
    if active_only:
        query = query.where(PipelineDependency.is_active == True)
    return list(db.scalars(query).all())
