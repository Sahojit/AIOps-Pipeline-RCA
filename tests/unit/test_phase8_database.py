"""
tests/unit/test_phase8_database.py
==================================
Tests for database models and CRUD operations.

Uses an in-memory SQLite database — no PostgreSQL needed.
SQLite doesn't support ARRAY type, so we skip tests that use evidence
field directly and test the core logic.
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.models import PipelineDependency, PipelineFailure, RCAPrediction
from database.session import Base


@pytest.fixture
def db_session() -> Session:
    """
    Create an in-memory SQLite database and session for testing.

    SQLite is used instead of PostgreSQL so tests run without external deps.
    Some PostgreSQL-specific features (ARRAY) are handled gracefully.
    """
    # SQLite in-memory database
    engine = create_engine("sqlite:///:memory:")

    # Create tables — some PG-specific types may need adaptation
    # We'll create simplified versions for testing
    Base.metadata.create_all(bind=engine)

    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    yield session
    session.close()


# =========================================================================
# ORM Model Tests
# =========================================================================

class TestPipelineFailureModel:
    def test_create_failure(self, db_session: Session) -> None:
        failure = PipelineFailure(
            id=uuid.uuid4(),
            pipeline_name="user_etl_pipeline",
            task_name="extract_users",
            runtime=120,
            retry_count=2,
            rows_processed=50000,
            schema_change=True,
            upstream_failed=False,
            error_log="KeyError: column 'age' not found",
            event_timestamp=datetime.now(timezone.utc),
        )
        db_session.add(failure)
        db_session.flush()

        assert failure.id is not None
        assert failure.pipeline_name == "user_etl_pipeline"

    def test_failure_repr(self, db_session: Session) -> None:
        failure = PipelineFailure(
            id=uuid.uuid4(),
            pipeline_name="test_pipeline",
            task_name="test_task",
            runtime=10,
            retry_count=0,
            rows_processed=0,
            schema_change=False,
            upstream_failed=False,
            error_log="error",
            event_timestamp=datetime.now(timezone.utc),
        )
        assert "test_pipeline" in repr(failure)


class TestRCAPredictionModel:
    def test_create_prediction(self, db_session: Session) -> None:
        # Create parent failure first
        failure_id = uuid.uuid4()
        failure = PipelineFailure(
            id=failure_id,
            pipeline_name="order_etl_pipeline",
            task_name="transform_orders",
            runtime=300,
            retry_count=1,
            rows_processed=10000,
            schema_change=False,
            upstream_failed=True,
            error_log="DependencyError: upstream failed",
            event_timestamp=datetime.now(timezone.utc),
        )
        db_session.add(failure)
        db_session.flush()

        prediction = RCAPrediction(
            id=uuid.uuid4(),
            failure_id=failure_id,
            pipeline_name="order_etl_pipeline",
            predicted_root_cause="Dependency Failure",
            confidence=0.91,
            evidence=["upstream failed (high impact)"],
            model_version="v1",
        )
        db_session.add(prediction)
        db_session.flush()

        assert prediction.id is not None
        assert prediction.confidence == 0.91

    def test_prediction_repr(self) -> None:
        prediction = RCAPrediction(
            id=uuid.uuid4(),
            failure_id=uuid.uuid4(),
            pipeline_name="test",
            predicted_root_cause="Schema Change",
            confidence=0.85,
            evidence=[],
            model_version="v1",
        )
        assert "Schema Change" in repr(prediction)


class TestPipelineDependencyModel:
    def test_create_dependency(self, db_session: Session) -> None:
        dep = PipelineDependency(
            id=uuid.uuid4(),
            upstream_pipeline="user_etl_pipeline",
            downstream_pipeline="recommendation_pipeline",
            dependency_type="data",
        )
        db_session.add(dep)
        db_session.flush()

        assert dep.id is not None
        assert dep.upstream_pipeline == "user_etl_pipeline"

    def test_dependency_repr(self) -> None:
        dep = PipelineDependency(
            id=uuid.uuid4(),
            upstream_pipeline="a",
            downstream_pipeline="b",
            dependency_type="data",
        )
        assert "a" in repr(dep) and "b" in repr(dep)


# =========================================================================
# Relationship Tests
# =========================================================================

class TestRelationships:
    def test_failure_prediction_relationship(self, db_session: Session) -> None:
        """A failure and its prediction should be linked."""
        failure_id = uuid.uuid4()
        failure = PipelineFailure(
            id=failure_id,
            pipeline_name="test_pipeline",
            task_name="task",
            runtime=10,
            retry_count=0,
            rows_processed=0,
            schema_change=False,
            upstream_failed=False,
            error_log="error",
            event_timestamp=datetime.now(timezone.utc),
        )
        db_session.add(failure)
        db_session.flush()

        prediction = RCAPrediction(
            id=uuid.uuid4(),
            failure_id=failure_id,
            pipeline_name="test_pipeline",
            predicted_root_cause="API Failure",
            confidence=0.75,
            evidence=["api timeout"],
            model_version="v1",
        )
        db_session.add(prediction)
        db_session.flush()

        # Access via relationship
        db_session.refresh(failure)
        assert failure.prediction is not None
        assert failure.prediction.predicted_root_cause == "API Failure"


# =========================================================================
# Query Tests
# =========================================================================

class TestQueries:
    def _create_test_data(self, db_session: Session, count: int = 5) -> None:
        """Helper: insert multiple failure+prediction pairs."""
        root_causes = ["Schema Change", "API Failure", "Missing Data",
                       "Dependency Failure", "Resource Exhaustion"]
        for i in range(count):
            fid = uuid.uuid4()
            failure = PipelineFailure(
                id=fid,
                pipeline_name=f"pipeline_{i}",
                task_name=f"task_{i}",
                runtime=100 * (i + 1),
                retry_count=i,
                rows_processed=1000 * i,
                schema_change=i % 2 == 0,
                upstream_failed=i % 3 == 0,
                error_log=f"Error for pipeline_{i}",
                event_timestamp=datetime.now(timezone.utc),
            )
            db_session.add(failure)
            db_session.flush()

            prediction = RCAPrediction(
                id=uuid.uuid4(),
                failure_id=fid,
                pipeline_name=f"pipeline_{i}",
                predicted_root_cause=root_causes[i % len(root_causes)],
                confidence=0.7 + (i * 0.05),
                evidence=[f"evidence_{i}"],
                model_version="v1",
            )
            db_session.add(prediction)
        db_session.flush()

    def test_query_predictions_by_pipeline(self, db_session: Session) -> None:
        self._create_test_data(db_session)

        results = (
            db_session.query(RCAPrediction)
            .filter(RCAPrediction.pipeline_name == "pipeline_0")
            .all()
        )
        assert len(results) == 1
        assert results[0].pipeline_name == "pipeline_0"

    def test_query_predictions_by_root_cause(self, db_session: Session) -> None:
        self._create_test_data(db_session)

        results = (
            db_session.query(RCAPrediction)
            .filter(RCAPrediction.predicted_root_cause == "Schema Change")
            .all()
        )
        assert len(results) >= 1

    def test_count_predictions(self, db_session: Session) -> None:
        self._create_test_data(db_session, count=10)
        total = db_session.query(RCAPrediction).count()
        assert total == 10

    def test_average_confidence(self, db_session: Session) -> None:
        self._create_test_data(db_session, count=5)
        from sqlalchemy import func
        avg = db_session.query(func.avg(RCAPrediction.confidence)).scalar()
        assert avg is not None
        assert 0.0 < avg < 1.0
