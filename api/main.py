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
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.config.settings import settings
from src.inference.engine import RCAInferenceEngine
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
