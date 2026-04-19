"""
tests/unit/test_phase2_data.py
==============================
Tests for Pydantic schemas used by the RCA system.
"""

import pytest
from pydantic import ValidationError

from src.ingestion.schemas import (
    PipelineFailureEvent,
    PipelineFailurePredictionRequest,
    RootCause,
)


# ---- Schema Tests ----

class TestRootCauseEnum:
    def test_has_six_classes(self) -> None:
        assert len(RootCause) == 6

    def test_values_are_human_readable(self) -> None:
        """Root cause values should be human-readable, not snake_case codes."""
        for rc in RootCause:
            assert " " in rc.value or rc.value.istitle(), f"Bad value: {rc.value}"


class TestPipelineFailureEvent:
    def test_valid_event_passes(self) -> None:
        event = PipelineFailureEvent(
            pipeline_name="user_etl_pipeline",
            task_name="extract_users",
            runtime=120,
            retry_count=2,
            rows_processed=50000,
            schema_change=False,
            upstream_failed=False,
            error_log="TimeoutError: API call timed out after 30s",
            root_cause=RootCause.API_FAILURE,
            timestamp="2026-03-03T12:00:00",
        )
        assert event.pipeline_name == "user_etl_pipeline"

    def test_negative_runtime_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PipelineFailureEvent(
                pipeline_name="test",
                task_name="test",
                runtime=-1,  # Invalid
                retry_count=0,
                rows_processed=0,
                schema_change=False,
                upstream_failed=False,
                error_log="error",
                root_cause=RootCause.API_FAILURE,
                timestamp="2026-03-03T12:00:00",
            )

    def test_blank_pipeline_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PipelineFailureEvent(
                pipeline_name="   ",  # Whitespace-only
                task_name="test",
                runtime=10,
                retry_count=0,
                rows_processed=0,
                schema_change=False,
                upstream_failed=False,
                error_log="error",
                root_cause=RootCause.API_FAILURE,
                timestamp="2026-03-03T12:00:00",
            )

    def test_empty_error_log_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PipelineFailureEvent(
                pipeline_name="test",
                task_name="test",
                runtime=10,
                retry_count=0,
                rows_processed=0,
                schema_change=False,
                upstream_failed=False,
                error_log="",  # Empty
                root_cause=RootCause.API_FAILURE,
                timestamp="2026-03-03T12:00:00",
            )

    def test_retry_count_max_10(self) -> None:
        with pytest.raises(ValidationError):
            PipelineFailureEvent(
                pipeline_name="test",
                task_name="test",
                runtime=10,
                retry_count=11,  # Over limit
                rows_processed=0,
                schema_change=False,
                upstream_failed=False,
                error_log="error",
                root_cause=RootCause.API_FAILURE,
                timestamp="2026-03-03T12:00:00",
            )


class TestPredictionRequest:
    def test_timestamp_optional(self) -> None:
        req = PipelineFailurePredictionRequest(
            pipeline_name="test_pipeline",
            task_name="extract",
            runtime=60,
            retry_count=1,
            rows_processed=1000,
            schema_change=False,
            upstream_failed=False,
            error_log="some error occurred",
            # timestamp omitted — should be None
        )
        assert req.timestamp is None
