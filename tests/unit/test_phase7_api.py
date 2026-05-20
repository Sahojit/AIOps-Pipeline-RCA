"""
tests/unit/test_phase7_api.py
=============================
Tests for the FastAPI API layer.

Uses a mock inference engine so tests don't need model artifacts on disk.
This is the standard pattern for testing ML APIs — decouple the API logic
from the model loading.
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def mock_engine() -> MagicMock:
    """Create a mock inference engine that returns realistic predictions."""
    engine = MagicMock()
    engine.model_version = "v1-test"
    engine.class_names = [
        "API Failure", "Data Quality Issue", "Dependency Failure",
        "Missing Data", "Resource Exhaustion", "Schema Change",
    ]

    engine.health_check.return_value = {
        "status": "healthy",
        "model_version": "v1-test",
        "n_classes": 6,
        "classes": engine.class_names,
    }

    engine.predict.return_value = {
        "pipeline_name": "user_etl_pipeline",
        "predicted_root_cause": "Schema Change",
        "confidence": 0.8432,
        "evidence": [
            "upstream schema change was detected (very high impact)",
            "error log contains 3 schema-related keywords (high impact)",
        ],
        "model_version": "v1-test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return engine


@pytest.fixture
def client(mock_engine: MagicMock) -> TestClient:
    """TestClient with a mock engine injected into app.state."""
    app.state.engine = mock_engine
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client_no_model() -> TestClient:
    """TestClient with NO engine loaded — simulates missing model."""
    app.state.engine = None
    return TestClient(app, raise_server_exceptions=False)


# Sample valid request payload
VALID_PAYLOAD: dict[str, Any] = {
    "pipeline_name": "user_etl_pipeline",
    "task_name": "extract_users",
    "runtime": 15,
    "retry_count": 1,
    "rows_processed": 100,
    "schema_change": True,
    "upstream_failed": False,
    "error_log": "KeyError: column 'age' not found in dataframe.",
}


# =========================================================================
# Health Endpoint Tests
# =========================================================================

class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_version"] == "v1-test"

    def test_health_returns_503_when_no_model(self, client_no_model: TestClient) -> None:
        response = client_no_model.get("/health")
        assert response.status_code == 503


# =========================================================================
# Model Info Endpoint Tests
# =========================================================================

class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client: TestClient) -> None:
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_version"] == "v1-test"
        assert data["n_classes"] == 6

    def test_model_info_503_when_no_model(self, client_no_model: TestClient) -> None:
        response = client_no_model.get("/model-info")
        assert response.status_code == 503


# =========================================================================
# Predict RCA Endpoint Tests
# =========================================================================

class TestPredictRCAEndpoint:
    def test_valid_request_returns_200(self, client: TestClient) -> None:
        response = client.post("/predict-rca", json=VALID_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_name"] == "user_etl_pipeline"
        assert data["predicted_root_cause"] == "Schema Change"
        assert 0 <= data["confidence"] <= 1
        assert len(data["evidence"]) > 0
        assert "model_version" in data

    def test_response_has_required_fields(self, client: TestClient) -> None:
        response = client.post("/predict-rca", json=VALID_PAYLOAD)
        data = response.json()
        required_fields = {
            "pipeline_name", "predicted_root_cause", "confidence",
            "evidence", "model_version", "timestamp",
        }
        assert required_fields.issubset(set(data.keys()))

    def test_missing_pipeline_name_returns_422(self, client: TestClient) -> None:
        """Pydantic validation should reject missing required fields."""
        bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "pipeline_name"}
        response = client.post("/predict-rca", json=bad_payload)
        assert response.status_code == 422

    def test_empty_error_log_returns_422(self, client: TestClient) -> None:
        bad_payload = {**VALID_PAYLOAD, "error_log": ""}
        response = client.post("/predict-rca", json=bad_payload)
        assert response.status_code == 422

    def test_negative_runtime_returns_422(self, client: TestClient) -> None:
        bad_payload = {**VALID_PAYLOAD, "runtime": -1}
        response = client.post("/predict-rca", json=bad_payload)
        assert response.status_code == 422

    def test_retry_count_over_max_returns_422(self, client: TestClient) -> None:
        bad_payload = {**VALID_PAYLOAD, "retry_count": 11}
        response = client.post("/predict-rca", json=bad_payload)
        assert response.status_code == 422

    def test_blank_pipeline_name_returns_422(self, client: TestClient) -> None:
        bad_payload = {**VALID_PAYLOAD, "pipeline_name": "   "}
        response = client.post("/predict-rca", json=bad_payload)
        assert response.status_code == 422

    def test_optional_timestamp(self, client: TestClient) -> None:
        """Timestamp should be optional — server fills it in."""
        payload_without_ts = {**VALID_PAYLOAD}  # No timestamp field
        response = client.post("/predict-rca", json=payload_without_ts)
        assert response.status_code == 200

    def test_with_explicit_timestamp(self, client: TestClient) -> None:
        payload_with_ts = {
            **VALID_PAYLOAD,
            "timestamp": "2026-03-03T12:00:00Z",
        }
        response = client.post("/predict-rca", json=payload_with_ts)
        assert response.status_code == 200

    def test_503_when_no_model(self, client_no_model: TestClient) -> None:
        response = client_no_model.post("/predict-rca", json=VALID_PAYLOAD)
        assert response.status_code == 503

    def test_engine_predict_called_with_dict(
        self, client: TestClient, mock_engine: MagicMock
    ) -> None:
        """Verify the engine receives a proper dict, not a Pydantic model."""
        client.post("/predict-rca", json=VALID_PAYLOAD)
        mock_engine.predict.assert_called_once()
        call_args = mock_engine.predict.call_args[0][0]
        assert isinstance(call_args, dict)
        assert call_args["pipeline_name"] == "user_etl_pipeline"


# =========================================================================
# Batch Endpoint Tests
# =========================================================================

class TestBatchEndpoint:
    def test_batch_returns_list(self, client: TestClient) -> None:
        response = client.post("/predict-rca/batch", json=[VALID_PAYLOAD, VALID_PAYLOAD])
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_batch_limit_enforced(self, client: TestClient) -> None:
        """Batch of 51 items should be rejected."""
        batch = [VALID_PAYLOAD] * 51
        response = client.post("/predict-rca/batch", json=batch)
        assert response.status_code == 400

    def test_empty_batch_returns_empty(self, client: TestClient) -> None:
        response = client.post("/predict-rca/batch", json=[])
        assert response.status_code == 200
        assert response.json() == []


# =========================================================================
# OpenAPI Schema Test
# =========================================================================

class TestOpenAPISchema:
    def test_openapi_docs_accessible(self, client: TestClient) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "AIOps Pipeline RCA API"
        assert "/predict-rca" in schema["paths"]
