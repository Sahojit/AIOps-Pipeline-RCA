"""
api/main.py
===========
FastAPI application — the HTTP serving layer for RCA predictions.

Design decisions:
- Model is loaded ONCE at startup via lifespan context manager (not per-request)
- Model lives in app.state so all requests share the same loaded model
- Pydantic schemas handle request/response validation automatically
- Structured logging on every request for observability
- Clean error handling — clients always get structured JSON errors, never stack traces

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Production:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from database.crud import (
    create_pipeline_failure,
    create_rca_prediction,
    get_prediction_stats,
    get_recent_predictions,
    get_root_cause_distribution,
)
from database.session import get_db
from src.config.settings import settings
from src.inference.engine import RCAInferenceEngine
from src.ingestion.schemas import (
    PipelineFailurePredictionRequest,
    RCAPredictionResponse,
)
from src.utils.logger import configure_root_logger, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup, clean up on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI.

    WHY lifespan over @app.on_event?
    - @app.on_event("startup") is deprecated in FastAPI >= 0.109
    - Lifespan gives explicit startup/shutdown phases with proper cleanup
    - The model is loaded into app.state and shared across all requests
    """
    configure_root_logger(settings.log_level)
    logger.info("Starting RCA API server")

    # Create database tables (idempotent — safe to call on every startup)
    try:
        from database.session import create_tables
        create_tables()
        logger.info("Database tables ready")
        app.state.db_available = True
    except Exception as e:
        logger.warning("Database not available — predictions will NOT be stored: %s", e)
        app.state.db_available = False

    # Load the inference engine (model + TF-IDF + SHAP explainer)
    try:
        engine = RCAInferenceEngine.load()
        app.state.engine = engine
        logger.info("Inference engine loaded successfully")
    except FileNotFoundError as e:
        logger.error(
            "Model artifacts not found. Train the model first: "
            "python -m src.training.train_model. Error: %s", e,
        )
        app.state.engine = None

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down RCA API server")
    app.state.engine = None


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AIOps Pipeline RCA API",
    description=(
        "Predicts the root cause of data pipeline failures "
        "using an XGBoost classifier with SHAP explainability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Global exception handler — never leak stack traces to clients
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: BLE001
    """Catch-all handler so clients always get structured JSON errors."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Check server logs.",
        },
    )


# ---------------------------------------------------------------------------
# Dependency — get the engine or raise 503
# ---------------------------------------------------------------------------
def _get_engine(request: Request) -> RCAInferenceEngine:
    """
    Retrieve the inference engine from app.state.
    Raises 503 if the model isn't loaded (e.g., artifacts missing).
    """
    engine: RCAInferenceEngine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first: python -m src.training.train_model",
        )
    return engine


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check(request: Request) -> dict:
    """
    Health check endpoint — used by load balancers, Kubernetes probes, monitoring.

    Returns 200 if the model is loaded, 503 if not.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )
    return engine.health_check()


@app.post("/predict-rca", response_model=RCAPredictionResponse)
async def predict_rca(
    request: Request,
    payload: PipelineFailurePredictionRequest,
    db: Session = Depends(get_db),
) -> RCAPredictionResponse:
    """
    Predict the root cause of a pipeline failure.

    Called by Airflow's on_failure_callback or any monitoring system
    that detects a pipeline failure. Returns the predicted root cause,
    confidence score, and SHAP-derived evidence.
    """
    engine = _get_engine(request)

    logger.info(
        "Received prediction request: pipeline=%s task=%s",
        payload.pipeline_name, payload.task_name,
    )

    event_dict = payload.model_dump()
    if event_dict.get("timestamp"):
        event_dict["timestamp"] = event_dict["timestamp"].isoformat()
    else:
        event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()

    try:
        result = engine.predict(event_dict)
    except Exception as e:
        logger.exception("Inference failed for pipeline %s: %s", payload.pipeline_name, e)
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Store failure + prediction in DB (best-effort — don't block the response)
    db_available = getattr(request.app.state, "db_available", False)
    if db_available:
        try:
            event_timestamp = payload.timestamp or datetime.now(timezone.utc)
            failure = create_pipeline_failure(
                db=db,
                pipeline_name=payload.pipeline_name,
                task_name=payload.task_name,
                runtime=payload.runtime,
                retry_count=payload.retry_count,
                rows_processed=payload.rows_processed,
                schema_change=payload.schema_change,
                upstream_failed=payload.upstream_failed,
                error_log=payload.error_log,
                event_timestamp=event_timestamp,
            )
            create_rca_prediction(
                db=db,
                failure_id=failure.id,
                pipeline_name=result["pipeline_name"],
                predicted_root_cause=result["predicted_root_cause"],
                confidence=result["confidence"],
                evidence=result["evidence"],
                model_version=result["model_version"],
            )
            db.commit()
            logger.info("Prediction stored in database (failure_id=%s)", failure.id)
        except Exception as e:
            db.rollback()
            logger.warning("Failed to store prediction in database: %s", e)

    response = RCAPredictionResponse(
        pipeline_name=result["pipeline_name"],
        predicted_root_cause=result["predicted_root_cause"],
        confidence=result["confidence"],
        evidence=result["evidence"],
        model_version=result["model_version"],
        timestamp=datetime.now(timezone.utc),
    )

    logger.info(
        "Prediction complete: pipeline=%s root_cause=%s confidence=%.4f",
        response.pipeline_name, response.predicted_root_cause, response.confidence,
    )
    return response


@app.post("/predict-rca/batch")
async def predict_rca_batch(
    request: Request,
    payloads: list[PipelineFailurePredictionRequest],
    db: Session = Depends(get_db),
) -> list[RCAPredictionResponse]:
    """
    Batch prediction — diagnose multiple failures at once.
    Limited to 50 items per batch to prevent request timeouts.
    """
    if not payloads:
        return []

    if len(payloads) > 50:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 50 items. Split into multiple requests.",
        )

    engine = _get_engine(request)
    db_available = getattr(request.app.state, "db_available", False)
    logger.info("Batch prediction request: %d items", len(payloads))

    responses: list[RCAPredictionResponse] = []
    for payload in payloads:
        event_dict = payload.model_dump()
        event_timestamp = payload.timestamp or datetime.now(timezone.utc)
        if event_dict.get("timestamp"):
            event_dict["timestamp"] = event_dict["timestamp"].isoformat()
        else:
            event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()

        result = engine.predict(event_dict)

        if db_available:
            try:
                failure = create_pipeline_failure(
                    db=db,
                    pipeline_name=payload.pipeline_name,
                    task_name=payload.task_name,
                    runtime=payload.runtime,
                    retry_count=payload.retry_count,
                    rows_processed=payload.rows_processed,
                    schema_change=payload.schema_change,
                    upstream_failed=payload.upstream_failed,
                    error_log=payload.error_log,
                    event_timestamp=event_timestamp,
                )
                create_rca_prediction(
                    db=db,
                    failure_id=failure.id,
                    pipeline_name=result["pipeline_name"],
                    predicted_root_cause=result["predicted_root_cause"],
                    confidence=result["confidence"],
                    evidence=result["evidence"],
                    model_version=result["model_version"],
                )
            except Exception as e:
                db.rollback()
                logger.warning("Failed to store batch item in database: %s", e)

        responses.append(RCAPredictionResponse(
            pipeline_name=result["pipeline_name"],
            predicted_root_cause=result["predicted_root_cause"],
            confidence=result["confidence"],
            evidence=result["evidence"],
            model_version=result["model_version"],
            timestamp=datetime.now(timezone.utc),
        ))

    if db_available:
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            logger.warning("Failed to commit batch predictions: %s", e)

    logger.info("Batch prediction complete: %d items", len(responses))
    return responses


# ---------------------------------------------------------------------------
# Query Endpoints — used by dashboard and monitoring
# ---------------------------------------------------------------------------

@app.get("/predictions")
async def list_predictions(
    limit: int = 50,
    pipeline_name: str | None = None,
    root_cause: str | None = None,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """List recent predictions with optional filters."""
    predictions = get_recent_predictions(
        db, limit=limit, pipeline_name=pipeline_name, root_cause=root_cause,
    )
    return [
        {
            "id": str(p.id),
            "pipeline_name": p.pipeline_name,
            "predicted_root_cause": p.predicted_root_cause,
            "confidence": p.confidence,
            "evidence": p.evidence,
            "model_version": p.model_version,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in predictions
    ]


@app.get("/stats")
async def prediction_stats(
    days: int = 30,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Aggregate prediction statistics for the dashboard."""
    stats = get_prediction_stats(db, days=days)
    if stats["total_predictions"] == 0:
        return {
            "total_predictions": 0,
            "avg_confidence": 0.0,
            "most_common_root_cause": None,
            "period_days": days,
        }
    return stats


@app.get("/root-cause-distribution")
async def root_cause_distribution(
    days: int = 30,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """Root cause distribution for pie/bar charts in the dashboard."""
    return get_root_cause_distribution(db, days=days)
