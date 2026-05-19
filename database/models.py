"""
database/models.py
==================
SQLAlchemy ORM models — define the three database tables.

Tables:
1. pipeline_failures   — Raw failure events (what happened)
2. rca_predictions     — Model predictions with evidence (what we think caused it)
3. pipeline_dependencies — DAG edges for dependency graph (Phase 10)

Design decisions:
- UUIDs for primary keys — avoids sequential ID guessing and works across
  distributed systems. In production, never expose auto-increment IDs in APIs.
- server_default for timestamps — the DATABASE sets the time, not Python.
  This prevents clock skew issues across multiple app servers.
- Indexes on frequently queried columns (pipeline_name, predicted_root_cause,
  created_at) for fast dashboard queries.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.session import Base


class PipelineFailure(Base):
    """
    Raw pipeline failure event — stored when the API receives a prediction request.

    This is the source-of-truth for what actually happened. Even if the model
    is wrong, we always have the original event to re-analyse later.
    """

    __tablename__ = "pipeline_failures"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    pipeline_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    task_name: Mapped[str] = mapped_column(String(200), nullable=False)
    runtime: Mapped[int] = mapped_column(Integer, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False)
    rows_processed: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_change: Mapped[bool] = mapped_column(Boolean, nullable=False)
    upstream_failed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_log: Mapped[str] = mapped_column(Text, nullable=False)
    event_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationship: one failure → one prediction
    prediction: Mapped["RCAPrediction"] = relationship(
        back_populates="failure",
        uselist=False,
    )

    # Indexes for dashboard queries
    __table_args__ = (
        Index("ix_failures_pipeline_created", "pipeline_name", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<PipelineFailure(id={self.id}, pipeline={self.pipeline_name}, "
            f"task={self.task_name})>"
        )


class RCAPrediction(Base):
    """
    Model prediction for a pipeline failure — the core output of our system.

    Stored separately from the failure event so we can:
    1. Re-run predictions without duplicating failure data
    2. Compare predictions across model versions
    3. Track confidence trends over time
    """

    __tablename__ = "rca_predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    failure_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("pipeline_failures.id"),
        nullable=False,
        unique=True,  # One prediction per failure
    )
    pipeline_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    predicted_root_cause: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationship back to the failure
    failure: Mapped["PipelineFailure"] = relationship(back_populates="prediction")

    # Indexes for dashboard queries
    __table_args__ = (
        Index("ix_predictions_rootcause_created", "predicted_root_cause", "created_at"),
        Index("ix_predictions_pipeline_created", "pipeline_name", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<RCAPrediction(id={self.id}, pipeline={self.pipeline_name}, "
            f"root_cause={self.predicted_root_cause}, confidence={self.confidence:.2f})>"
        )


class PipelineDependency(Base):
    """
    Pipeline dependency graph edges — used in Phase 10 for graph features.

    Each row represents: upstream_pipeline → downstream_pipeline
    This lets us compute graph features like in-degree, out-degree,
    and detect cascade patterns.
    """

    __tablename__ = "pipeline_dependencies"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    upstream_pipeline: Mapped[str] = mapped_column(
        String(200), nullable=False, index=True,
    )
    downstream_pipeline: Mapped[str] = mapped_column(
        String(200), nullable=False, index=True,
    )
    dependency_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="data",  # data | trigger | sensor
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_deps_upstream_downstream",
            "upstream_pipeline", "downstream_pipeline",
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<PipelineDependency({self.upstream_pipeline} → "
            f"{self.downstream_pipeline})>"
        )
