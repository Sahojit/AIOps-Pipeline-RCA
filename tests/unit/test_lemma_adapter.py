"""
tests/unit/test_lemma_adapter.py
=================================
Tests for the LEMMA-RCA dataset adapter.
Validates ground truth mapping, log extraction, event generation,
and error log composition.
"""

import csv
import io
import random
import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ingestion.lemma_adapter import (
    FALLBACK_ERROR_TEMPLATES,
    HUGGINGFACE_DATASETS,
    PIPELINE_NAMES,
    SCENARIO_GROUND_TRUTH,
    TASK_NAMES,
    ScenarioGroundTruth,
    _compose_error_log,
    _extract_logs_from_zip,
    _find_error_logs,
    _generate_events_from_scenario,
)
from src.ingestion.schemas import RootCause


# =========================================================================
# Helpers
# =========================================================================

def _create_test_zip(tmp_path: Path, pod_logs: dict[str, list[str]]) -> Path:
    """Create a test zip file with structured log CSVs."""
    zip_path = tmp_path / "test_logs.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for pod_name, messages in pod_logs.items():
            csv_name = f"log_data/pod_removed/{pod_name}_messages_structured.csv"
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=["LineId", "Time", "Content", "EventId", "EventTemplate", "ParameterList"])
            writer.writeheader()
            for i, msg in enumerate(messages):
                writer.writerow({
                    "LineId": i,
                    "Time": f"2024-01-15T{10+i:02d}:00:00Z",
                    "Content": msg,
                    "EventId": str(i),
                    "EventTemplate": f"template_{i}",
                    "ParameterList": "[]",
                })
            z.writestr(csv_name, buf.getvalue())
    return zip_path


def _make_ground_truth(
    date_key: str = "20240115",
    root_cause: RootCause = RootCause.RESOURCE_EXHAUSTION,
    faulty_pod_pattern: str = r"productpage",
    schema_change: bool = False,
    upstream_failed: bool = False,
) -> ScenarioGroundTruth:
    return ScenarioGroundTruth(
        date_key=date_key,
        fault_description="Test fault",
        root_cause=root_cause,
        faulty_pod_pattern=faulty_pod_pattern,
        microservice="test-service",
        schema_change=schema_change,
        upstream_failed=upstream_failed,
    )


# =========================================================================
# TestScenarioGroundTruth
# =========================================================================

class TestScenarioGroundTruth:
    """Verify hardcoded ground truth for LEMMA-RCA scenarios."""

    def test_six_scenarios_defined(self) -> None:
        assert len(SCENARIO_GROUND_TRUTH) == 6

    def test_all_dates_are_valid(self) -> None:
        for gt in SCENARIO_GROUND_TRUTH:
            dt = datetime.strptime(gt.date_key, "%Y%m%d")
            assert dt.year in (2023, 2024)

    def test_all_root_causes_are_valid(self) -> None:
        for gt in SCENARIO_GROUND_TRUTH:
            assert isinstance(gt.root_cause, RootCause)

    def test_resource_exhaustion_is_most_common(self) -> None:
        rc_counts: dict[RootCause, int] = {}
        for gt in SCENARIO_GROUND_TRUTH:
            rc_counts[gt.root_cause] = rc_counts.get(gt.root_cause, 0) + 1
        assert rc_counts[RootCause.RESOURCE_EXHAUSTION] >= 3

    def test_has_api_failure_scenario(self) -> None:
        assert any(gt.root_cause == RootCause.API_FAILURE for gt in SCENARIO_GROUND_TRUTH)

    def test_has_schema_change_scenario(self) -> None:
        assert any(gt.root_cause == RootCause.SCHEMA_CHANGE for gt in SCENARIO_GROUND_TRUTH)

    def test_schema_change_scenario_has_flag_set(self) -> None:
        for gt in SCENARIO_GROUND_TRUTH:
            if gt.root_cause == RootCause.SCHEMA_CHANGE:
                assert gt.schema_change is True

    def test_unique_date_keys(self) -> None:
        dates = [gt.date_key for gt in SCENARIO_GROUND_TRUTH]
        assert len(dates) == len(set(dates))

    def test_faulty_pod_patterns_are_valid_regex(self) -> None:
        import re
        for gt in SCENARIO_GROUND_TRUTH:
            re.compile(gt.faulty_pod_pattern)  # Should not raise


# =========================================================================
# TestExtractLogsFromZip
# =========================================================================

class TestExtractLogsFromZip:
    """Test log extraction from zip archives."""

    def test_extracts_logs_from_matching_pods(self, tmp_path: Path) -> None:
        zip_path = _create_test_zip(tmp_path, {
            "productpage-v1-abc123": ["INFO starting", "ERROR timeout occurred"],
            "ratings-v1-def456": ["INFO healthy"],
        })
        pod_logs = _extract_logs_from_zip(zip_path, r"productpage")
        assert "productpage-v1-abc123" in pod_logs
        assert len(pod_logs["productpage-v1-abc123"]) == 2

    def test_marks_faulty_pod_flag(self, tmp_path: Path) -> None:
        zip_path = _create_test_zip(tmp_path, {
            "productpage-v1-abc": ["error msg"],
            "other-pod-xyz": ["normal msg"],
        })
        pod_logs = _extract_logs_from_zip(zip_path, r"productpage")
        # Faulty pod should have is_faulty_pod=True
        assert pod_logs["productpage-v1-abc"][0]["is_faulty_pod"] is True
        assert pod_logs["other-pod-xyz"][0]["is_faulty_pod"] is False

    def test_extracts_all_pods(self, tmp_path: Path) -> None:
        zip_path = _create_test_zip(tmp_path, {
            "pod-a": ["log1"],
            "pod-b": ["log2"],
            "pod-c": ["log3"],
        })
        pod_logs = _extract_logs_from_zip(zip_path, r"pod-a")
        assert len(pod_logs) == 3

    def test_respects_max_logs_per_pod(self, tmp_path: Path) -> None:
        zip_path = _create_test_zip(tmp_path, {
            "big-pod": [f"log line {i}" for i in range(100)],
        })
        pod_logs = _extract_logs_from_zip(zip_path, r"big-pod", max_logs_per_pod=10)
        assert len(pod_logs["big-pod"]) == 10

    def test_handles_nonexistent_zip(self, tmp_path: Path) -> None:
        pod_logs = _extract_logs_from_zip(tmp_path / "nonexistent.zip", r"pod")
        assert pod_logs == {}

    def test_empty_zip(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w"):
            pass
        pod_logs = _extract_logs_from_zip(zip_path, r"pod")
        assert pod_logs == {}


# =========================================================================
# TestFindErrorLogs
# =========================================================================

class TestFindErrorLogs:
    """Test error log detection."""

    def test_finds_error_keywords(self) -> None:
        logs = [
            {"content": "INFO starting service", "time": "", "template": ""},
            {"content": "ERROR connection timeout occurred", "time": "", "template": ""},
            {"content": "WARN disk space low", "time": "", "template": ""},
            {"content": "INFO request processed", "time": "", "template": ""},
        ]
        errors = _find_error_logs(logs)
        assert len(errors) == 2
        assert "timeout" in errors[0].lower()

    def test_empty_list_returns_empty(self) -> None:
        assert _find_error_logs([]) == []

    def test_no_errors_returns_empty(self) -> None:
        logs = [{"content": "INFO all good", "time": "", "template": ""}]
        assert _find_error_logs(logs) == []

    def test_respects_max_count(self) -> None:
        logs = [{"content": f"ERROR problem {i}", "time": "", "template": ""} for i in range(100)]
        errors = _find_error_logs(logs, max_count=5)
        assert len(errors) == 5


# =========================================================================
# TestComposeErrorLog
# =========================================================================

class TestComposeErrorLog:
    """Test error log composition."""

    def test_not_empty(self) -> None:
        log = _compose_error_log(
            real_logs=["MemoryError: OOM killed", "Process terminated"],
            root_cause=RootCause.RESOURCE_EXHAUSTION,
            faulty_pod="pod-a",
        )
        assert len(log) > 0
        assert isinstance(log, str)

    def test_truncated_to_max_length(self) -> None:
        long_logs = ["x" * 200] * 10
        log = _compose_error_log(long_logs, RootCause.RESOURCE_EXHAUSTION, "pod-a", max_length=500)
        assert len(log) <= 500

    def test_contains_fault_context(self) -> None:
        log = _compose_error_log(
            real_logs=["Some error occurred"],
            root_cause=RootCause.RESOURCE_EXHAUSTION,
            faulty_pod="db-pod",
        )
        assert any(kw in log.lower() for kw in ["memory", "killed", "resource", "cpu", "disk", "db-pod"])

    def test_handles_empty_logs(self) -> None:
        log = _compose_error_log([], RootCause.RESOURCE_EXHAUSTION, "pod-a")
        assert len(log) > 0

    def test_all_root_causes_have_fallback_templates(self) -> None:
        for rc in [RootCause.RESOURCE_EXHAUSTION, RootCause.API_FAILURE,
                    RootCause.DEPENDENCY_FAILURE, RootCause.SCHEMA_CHANGE]:
            assert rc in FALLBACK_ERROR_TEMPLATES
            assert len(FALLBACK_ERROR_TEMPLATES[rc]) >= 3


# =========================================================================
# TestGenerateEventsFromScenario
# =========================================================================

class TestGenerateEventsFromScenario:
    """Test event generation from scenarios."""

    def test_generates_correct_count(self, tmp_path: Path) -> None:
        gt = _make_ground_truth()
        pod_logs = {
            "productpage-v1": [
                {"time": "t1", "content": "ERROR timeout", "template": "t", "is_faulty_pod": True},
            ],
        }
        events = _generate_events_from_scenario(gt, pod_logs, events_per_scenario=25, seed=42)
        assert len(events) == 25

    def test_events_have_correct_schema(self, tmp_path: Path) -> None:
        gt = _make_ground_truth()
        pod_logs = {
            "productpage-v1": [
                {"time": "t1", "content": "ERROR oom killed", "template": "t", "is_faulty_pod": True},
            ],
        }
        events = _generate_events_from_scenario(gt, pod_logs, events_per_scenario=5, seed=42)
        required = {
            "pipeline_name", "task_name", "runtime", "retry_count",
            "rows_processed", "schema_change", "upstream_failed",
            "error_log", "root_cause", "timestamp",
        }
        for event in events:
            assert set(event.keys()) == required

    def test_events_have_correct_root_cause(self) -> None:
        gt = _make_ground_truth(root_cause=RootCause.API_FAILURE)
        pod_logs = {"pod-a": [{"time": "", "content": "error", "template": "", "is_faulty_pod": True}]}
        events = _generate_events_from_scenario(gt, pod_logs, events_per_scenario=3, seed=42)
        for event in events:
            assert event["root_cause"] == "API Failure"

    def test_schema_change_flag_propagated(self) -> None:
        gt = _make_ground_truth(root_cause=RootCause.SCHEMA_CHANGE, schema_change=True)
        pod_logs = {"pod-a": [{"time": "", "content": "config error", "template": "", "is_faulty_pod": True}]}
        events = _generate_events_from_scenario(gt, pod_logs, events_per_scenario=3, seed=42)
        for event in events:
            assert event["schema_change"] is True

    def test_upstream_failed_flag_propagated(self) -> None:
        gt = _make_ground_truth(upstream_failed=True)
        pod_logs = {"pod-a": [{"time": "", "content": "network error", "template": "", "is_faulty_pod": True}]}
        events = _generate_events_from_scenario(gt, pod_logs, events_per_scenario=3, seed=42)
        for event in events:
            assert event["upstream_failed"] is True

    def test_pipeline_names_from_predefined_list(self) -> None:
        gt = _make_ground_truth()
        pod_logs = {"pod-a": [{"time": "", "content": "error", "template": "", "is_faulty_pod": True}]}
        events = _generate_events_from_scenario(gt, pod_logs, events_per_scenario=10, seed=42)
        for event in events:
            assert event["pipeline_name"] in PIPELINE_NAMES

    def test_task_names_from_predefined_list(self) -> None:
        gt = _make_ground_truth()
        pod_logs = {"pod-a": [{"time": "", "content": "error", "template": "", "is_faulty_pod": True}]}
        events = _generate_events_from_scenario(gt, pod_logs, events_per_scenario=10, seed=42)
        for event in events:
            assert event["task_name"] in TASK_NAMES

    def test_runtime_non_negative(self) -> None:
        gt = _make_ground_truth()
        pod_logs = {"pod-a": [{"time": "", "content": "error", "template": "", "is_faulty_pod": True}]}
        events = _generate_events_from_scenario(gt, pod_logs, events_per_scenario=20, seed=42)
        for event in events:
            assert event["runtime"] >= 0

    def test_retry_count_within_range(self) -> None:
        gt = _make_ground_truth()
        pod_logs = {"pod-a": [{"time": "", "content": "error", "template": "", "is_faulty_pod": True}]}
        events = _generate_events_from_scenario(gt, pod_logs, events_per_scenario=20, seed=42)
        for event in events:
            assert 0 <= event["retry_count"] <= 10

    def test_timestamp_is_iso_format(self) -> None:
        gt = _make_ground_truth()
        pod_logs = {"pod-a": [{"time": "", "content": "error", "template": "", "is_faulty_pod": True}]}
        events = _generate_events_from_scenario(gt, pod_logs, events_per_scenario=5, seed=42)
        for event in events:
            dt = datetime.fromisoformat(event["timestamp"])
            assert dt.tzinfo is not None

    def test_handles_empty_pod_logs(self) -> None:
        gt = _make_ground_truth()
        events = _generate_events_from_scenario(gt, {}, events_per_scenario=5, seed=42)
        assert len(events) == 5
        for event in events:
            assert len(event["error_log"]) > 0  # Falls back to template


# =========================================================================
# TestConstants
# =========================================================================

class TestConstants:
    """Test module-level constants."""

    def test_huggingface_datasets_dict(self) -> None:
        assert "cloud" in HUGGINGFACE_DATASETS
        assert "product_review" in HUGGINGFACE_DATASETS
        for key, repo_id in HUGGINGFACE_DATASETS.items():
            assert "Lemma-RCA-NEC" in repo_id

    def test_pipeline_names_not_empty(self) -> None:
        assert len(PIPELINE_NAMES) >= 10

    def test_task_names_not_empty(self) -> None:
        assert len(TASK_NAMES) >= 10

    def test_fallback_templates_cover_four_causes(self) -> None:
        assert RootCause.RESOURCE_EXHAUSTION in FALLBACK_ERROR_TEMPLATES
        assert RootCause.API_FAILURE in FALLBACK_ERROR_TEMPLATES
        assert RootCause.DEPENDENCY_FAILURE in FALLBACK_ERROR_TEMPLATES
        assert RootCause.SCHEMA_CHANGE in FALLBACK_ERROR_TEMPLATES
