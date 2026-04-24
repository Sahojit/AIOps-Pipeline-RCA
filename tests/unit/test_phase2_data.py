"""
tests/unit/test_phase2_data.py
==============================
Tests for Pydantic schemas used by the RCA system.
"""

import pandas as pd
import pytest
from pydantic import ValidationError

from src.ingestion.lemma_adapter import load_lemma_dataset
from src.ingestion.log_feature_extractor import LogFeatureExtractor
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

    def test_empty_error_log_allowed(self) -> None:
        req = PipelineFailurePredictionRequest(
            pipeline_name="test_pipeline",
            task_name="extract",
            runtime=60,
            retry_count=0,
            rows_processed=0,
            schema_change=False,
            upstream_failed=False,
            error_log="",
        )
        assert req.error_log == ""

    def test_negative_runtime_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PipelineFailurePredictionRequest(
                pipeline_name="p",
                task_name="t",
                runtime=-1,
                retry_count=0,
                rows_processed=0,
                schema_change=False,
                upstream_failed=False,
                error_log="err",
            )


class TestIngestionIntegration:
    """Integration tests: load CSV → parse → validate schema."""

    def test_load_lemma_returns_dataframe(self) -> None:
        df = load_lemma_dataset()
        assert isinstance(df, pd.DataFrame)

    def test_lemma_dataset_has_800_rows(self) -> None:
        df = load_lemma_dataset()
        assert len(df) == 800

    def test_lemma_required_columns_present(self) -> None:
        df = load_lemma_dataset()
        required = {"pipeline_name", "task_name", "runtime", "error_log", "root_cause"}
        assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"

    def test_lemma_root_cause_classes_match_enum(self) -> None:
        df = load_lemma_dataset()
        valid_values = {rc.value for rc in RootCause}
        actual_values = set(df["root_cause"].unique())
        unknown = actual_values - valid_values
        assert not unknown, f"Unknown root cause values in dataset: {unknown}"

    def test_lemma_no_null_pipeline_names(self) -> None:
        df = load_lemma_dataset()
        assert df["pipeline_name"].notna().all()
        assert (df["pipeline_name"].str.strip() != "").all()

    def test_lemma_runtime_non_negative(self) -> None:
        df = load_lemma_dataset()
        assert (df["runtime"] >= 0).all()


class TestIngestionSmoke:
    """Smoke tests: CSV → LogFeatureExtractor → feature matrix."""

    def test_fit_transform_returns_dataframe(self) -> None:
        df = load_lemma_dataset()
        logs = df["error_log"].fillna("").tolist()[:20]
        extractor = LogFeatureExtractor(max_tfidf_features=10)
        result = extractor.fit_transform(logs)
        assert isinstance(result, pd.DataFrame)

    def test_feature_matrix_row_count_matches_input(self) -> None:
        df = load_lemma_dataset()
        logs = df["error_log"].fillna("").tolist()[:20]
        extractor = LogFeatureExtractor(max_tfidf_features=10)
        result = extractor.fit_transform(logs)
        assert len(result) == 20

    def test_feature_names_include_regex_and_tfidf(self) -> None:
        df = load_lemma_dataset()
        logs = df["error_log"].fillna("").tolist()[:20]
        extractor = LogFeatureExtractor(max_tfidf_features=10)
        extractor.fit(logs)
        names = extractor.get_feature_names()
        assert any(n.startswith("tfidf_") for n in names)
        assert "log_length" in names

    def test_no_nan_values_in_feature_matrix(self) -> None:
        df = load_lemma_dataset()
        logs = df["error_log"].fillna("").tolist()[:20]
        extractor = LogFeatureExtractor(max_tfidf_features=10)
        result = extractor.fit_transform(logs)
        assert not result.isnull().any().any(), "Feature matrix contains NaN values"
