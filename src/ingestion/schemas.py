"""
src/ingestion/schemas.py
========================
Pydantic data contracts for pipeline failure events.

These schemas serve THREE purposes:
1. Validation — reject malformed data at the boundary (API, ingestion)
2. Documentation — schema IS the spec; no separate data dictionary needed
3. Serialisation — automatic JSON conversion for API responses

Every field has a type constraint. Some have value constraints (ge, le).
If upstream sends bad data, it fails HERE with a clear error — not deep
inside the feature engineering layer where it's harder to debug.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class RootCause(str, Enum):
    """
    The six root cause classes our model must classify.
    Using an Enum prevents typos like 'schema_change' vs 'Schema Change'
    from silently corrupting training data.
    """
    SCHEMA_CHANGE = "Schema Change"
    MISSING_DATA = "Missing Data"
    DEPENDENCY_FAILURE = "Dependency Failure"
    API_FAILURE = "API Failure"
    RESOURCE_EXHAUSTION = "Resource Exhaustion"
    DATA_QUALITY = "Data Quality Issue"


class PipelineFailureEvent(BaseModel):
    """
    A single pipeline failure event — the core input to our system.

    This represents one row of training data and also the payload
    that Airflow sends to POST /predict-rca at runtime.
    """
    pipeline_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique identifier for the pipeline (e.g., 'user_etl_pipeline')",
    )
    task_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Specific task within the pipeline that failed",
    )
    runtime: int = Field(
        ...,
        ge=0,
        description="Task execution time in seconds before failure",
    )
    retry_count: int = Field(
        ...,
        ge=0,
        le=10,
        description="Number of automatic retries attempted before final failure",
    )
    rows_processed: int = Field(
        ...,
        ge=0,
        description="Number of rows processed before failure (0 if failed at start)",
    )
    schema_change: bool = Field(
        ...,
        description="Whether an upstream schema change was detected",
    )
    upstream_failed: bool = Field(
        ...,
        description="Whether an upstream dependency pipeline has failed",
    )
    error_log: str = Field(
        default="",
        description="Raw error log text from the failed task",
    )
    root_cause: RootCause = Field(
        ...,
        description="Ground truth root cause label (for training data)",
    )
    timestamp: datetime = Field(
        ...,
        description="When the failure occurred (UTC)",
    )

    @field_validator("pipeline_name", "task_name")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        """Catch whitespace-only strings that pass min_length check."""
        if not v.strip():
            raise ValueError("Field must not be blank or whitespace-only")
        return v.strip()


class PipelineFailurePredictionRequest(BaseModel):
    """
    API request schema — same as PipelineFailureEvent but WITHOUT root_cause.
    At inference time, the model PREDICTS the root cause.
    """
    pipeline_name: str = Field(..., min_length=1, max_length=200)
    task_name: str = Field(..., min_length=1, max_length=200)
    runtime: int = Field(..., ge=0)
    retry_count: int = Field(..., ge=0, le=10)
    rows_processed: int = Field(..., ge=0)
    schema_change: bool
    upstream_failed: bool
    error_log: str = Field(..., min_length=1)
    timestamp: Optional[datetime] = Field(
        default=None,
        description="If not provided, server uses current UTC time",
    )

    @field_validator("pipeline_name", "task_name")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be blank or whitespace-only")
        return v.strip()


class RCAPredictionResponse(BaseModel):
    """
    API response schema — what the caller gets back after POST /predict-rca.
    """
    pipeline_name: str
    predicted_root_cause: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(
        default_factory=list,
        description="Human-readable evidence supporting the prediction",
    )
    model_version: str
    timestamp: datetime
