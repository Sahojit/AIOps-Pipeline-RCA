"""
tests/unit/test_phase9_airflow.py
=================================
Tests for the Airflow RCA callback.

Does NOT require Airflow to be installed — we mock the context dict
and the HTTP requests. This is the standard pattern for testing
Airflow callbacks in isolation.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from airflow.plugins.rca_callback import (
    RCA_API_URL,
    rca_failure_callback,
    rca_failure_callback_with_xcom,
)


@pytest.fixture
def mock_context() -> dict:
    """
    Simulates the Airflow callback context dict.

    In real Airflow, this is passed automatically when a task fails.
    We mock it here to test the callback in isolation.
    """
    task_instance = MagicMock()
    task_instance.task_id = "extract_users"
    task_instance.try_number = 3  # 3rd try → retry_count = 2
    task_instance.start_date = datetime(2026, 3, 3, 10, 0, 0, tzinfo=timezone.utc)
    task_instance.end_date = datetime(2026, 3, 3, 10, 2, 30, tzinfo=timezone.utc)

    dag = MagicMock()
    dag.dag_id = "user_etl_pipeline"

    return {
        "task_instance": task_instance,
        "dag": dag,
        "exception": KeyError("column 'age' not found in dataframe. Schema may have changed upstream."),
        "execution_date": datetime(2026, 3, 3, 10, 0, 0, tzinfo=timezone.utc),
    }


@pytest.fixture
def mock_api_response() -> MagicMock:
    """Mock a successful RCA API response."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "pipeline_name": "user_etl_pipeline",
        "predicted_root_cause": "Schema Change",
        "confidence": 0.84,
        "evidence": [
            "upstream schema change was detected (very high impact)",
            "error log contains 3 schema-related keywords (high impact)",
        ],
        "model_version": "v1",
        "timestamp": "2026-03-03T10:02:30Z",
    }
    return response


# =========================================================================
# rca_failure_callback Tests
# =========================================================================

class TestRCAFailureCallback:
    @patch("airflow.plugins.rca_callback.requests.post")
    def test_sends_post_request(
        self, mock_post: MagicMock, mock_context: dict, mock_api_response: MagicMock
    ) -> None:
        """Callback must send a POST to /predict-rca."""
        mock_post.return_value = mock_api_response

        rca_failure_callback(mock_context)

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert call_url == f"{RCA_API_URL}/predict-rca"

    @patch("airflow.plugins.rca_callback.requests.post")
    def test_payload_has_correct_pipeline_name(
        self, mock_post: MagicMock, mock_context: dict, mock_api_response: MagicMock
    ) -> None:
        mock_post.return_value = mock_api_response

        rca_failure_callback(mock_context)

        payload = mock_post.call_args[1]["json"]
        assert payload["pipeline_name"] == "user_etl_pipeline"

    @patch("airflow.plugins.rca_callback.requests.post")
    def test_payload_has_correct_task_name(
        self, mock_post: MagicMock, mock_context: dict, mock_api_response: MagicMock
    ) -> None:
        mock_post.return_value = mock_api_response

        rca_failure_callback(mock_context)

        payload = mock_post.call_args[1]["json"]
        assert payload["task_name"] == "extract_users"

    @patch("airflow.plugins.rca_callback.requests.post")
    def test_runtime_calculated_correctly(
        self, mock_post: MagicMock, mock_context: dict, mock_api_response: MagicMock
    ) -> None:
        """Runtime should be ~150 seconds (2min 30sec)."""
        mock_post.return_value = mock_api_response

        rca_failure_callback(mock_context)

        payload = mock_post.call_args[1]["json"]
        assert payload["runtime"] == 150

    @patch("airflow.plugins.rca_callback.requests.post")
    def test_retry_count_derived_from_try_number(
        self, mock_post: MagicMock, mock_context: dict, mock_api_response: MagicMock
    ) -> None:
        """try_number=3 means 2 retries."""
        mock_post.return_value = mock_api_response

        rca_failure_callback(mock_context)

        payload = mock_post.call_args[1]["json"]
        assert payload["retry_count"] == 2

    @patch("airflow.plugins.rca_callback.requests.post")
    def test_schema_change_detected_from_error(
        self, mock_post: MagicMock, mock_context: dict, mock_api_response: MagicMock
    ) -> None:
        """The word 'Schema' in the error should set schema_change=True."""
        mock_post.return_value = mock_api_response

        rca_failure_callback(mock_context)

        payload = mock_post.call_args[1]["json"]
        assert payload["schema_change"] is True

    @patch("airflow.plugins.rca_callback.requests.post")
    def test_error_log_contains_exception_text(
        self, mock_post: MagicMock, mock_context: dict, mock_api_response: MagicMock
    ) -> None:
        mock_post.return_value = mock_api_response

        rca_failure_callback(mock_context)

        payload = mock_post.call_args[1]["json"]
        assert "column" in payload["error_log"]
        assert "age" in payload["error_log"]

    @patch("airflow.plugins.rca_callback.requests.post")
    def test_error_log_truncated(
        self, mock_post: MagicMock, mock_context: dict, mock_api_response: MagicMock
    ) -> None:
        """Very long error logs should be truncated to 5000 chars."""
        mock_context["exception"] = Exception("x" * 10000)
        mock_post.return_value = mock_api_response

        rca_failure_callback(mock_context)

        payload = mock_post.call_args[1]["json"]
        assert len(payload["error_log"]) <= 5000

    @patch("airflow.plugins.rca_callback.requests.post")
    def test_handles_api_failure_gracefully(
        self, mock_post: MagicMock, mock_context: dict
    ) -> None:
        """Callback must NOT crash if the API returns an error."""
        mock_post.return_value = MagicMock(status_code=500, text="Server error")

        # Should not raise
        rca_failure_callback(mock_context)

    @patch("airflow.plugins.rca_callback.requests.post")
    def test_handles_connection_error_gracefully(
        self, mock_post: MagicMock, mock_context: dict
    ) -> None:
        """Callback must NOT crash if the API is unreachable."""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        # Should not raise
        rca_failure_callback(mock_context)

    @patch("airflow.plugins.rca_callback.requests.post")
    def test_handles_timeout_gracefully(
        self, mock_post: MagicMock, mock_context: dict
    ) -> None:
        """Callback must NOT crash if the API times out."""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        # Should not raise
        rca_failure_callback(mock_context)

    @patch("airflow.plugins.rca_callback.requests.post")
    def test_handles_missing_context_fields(
        self, mock_post: MagicMock, mock_api_response: MagicMock
    ) -> None:
        """Callback should handle partial/empty context gracefully."""
        mock_post.return_value = mock_api_response

        # Minimal context — missing most fields
        rca_failure_callback({"task_instance": None, "dag": None, "exception": None})

        # Should still make the API call
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert payload["pipeline_name"] == "unknown_dag"
        assert payload["task_name"] == "unknown_task"

    @patch("airflow.plugins.rca_callback.requests.post")
    def test_timeout_set_to_30_seconds(
        self, mock_post: MagicMock, mock_context: dict, mock_api_response: MagicMock
    ) -> None:
        mock_post.return_value = mock_api_response

        rca_failure_callback(mock_context)

        assert mock_post.call_args[1]["timeout"] == 30


# =========================================================================
# XCom Variant Tests
# =========================================================================

class TestRCACallbackWithXCom:
    @patch("airflow.plugins.rca_callback.requests.post")
    def test_pushes_to_xcom(
        self, mock_post: MagicMock, mock_context: dict, mock_api_response: MagicMock
    ) -> None:
        mock_post.return_value = mock_api_response

        rca_failure_callback_with_xcom(mock_context)

        # Verify xcom_push was called
        task_instance = mock_context["task_instance"]
        task_instance.xcom_push.assert_called_once()
        call_kwargs = task_instance.xcom_push.call_args[1]
        assert call_kwargs["key"] == "rca_result"
        assert call_kwargs["value"]["predicted_root_cause"] == "Schema Change"
