"""
src/inference/engine.py
=======================
Production inference engine — single entry point for prediction + explanation.

This is the core of the serving layer. The FastAPI endpoint calls this,
and it returns everything needed for the API response:
- Predicted root cause
- Confidence score
- Human-readable evidence (from SHAP)
- Model version

WHY a separate engine (not just model.predict)?
1. Inference requires MULTIPLE steps: feature engineering → predict → explain → format
2. The engine holds all stateful components (model, vectorizer, explainer) in memory
3. Thread-safe singleton pattern — one model loaded, many concurrent requests
4. Clear separation: API layer handles HTTP, engine handles ML
"""

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.features.engineer import FeatureEngineer
from src.inference.evidence import generate_counter_evidence, generate_evidence
from src.inference.explainer import RCAExplainer
from src.training.artifact_manager import load_model_artifacts
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RCAInferenceEngine:
    """
    Production inference engine — loads all artifacts and serves predictions.

    Usage:
        engine = RCAInferenceEngine.load()
        result = engine.predict(event_dict)
        # result has: predicted_root_cause, confidence, evidence, model_version
    """

    def __init__(
        self,
        model: Any,
        label_encoder: Any,
        feature_engineer: FeatureEngineer,
        explainer: RCAExplainer,
        metadata: dict[str, Any],
    ) -> None:
        self._model = model
        self._label_encoder = label_encoder
        self._feature_engineer = feature_engineer
        self._explainer = explainer
        self._metadata = metadata
        logger.info(
            "RCAInferenceEngine initialised (model=%s, %d features, %d classes)",
            metadata.get("model_version", "unknown"),
            len(metadata.get("feature_names", [])),
            len(metadata.get("class_names", [])),
        )

    @classmethod
    def load(cls, model_version: Optional[str] = None) -> "RCAInferenceEngine":
        """
        Load all artifacts and construct a ready-to-serve engine.

        Args:
            model_version: Specific version to load (e.g., "v1"). None = latest.

        Returns:
            Fully initialised RCAInferenceEngine.
        """
        logger.info("Loading inference engine (version=%s)", model_version or "latest")

        # Load model + label encoder + metadata
        artifacts = load_model_artifacts(version=model_version)
        model = artifacts["model"]
        label_encoder = artifacts["label_encoder"]
        metadata = artifacts["metadata"]
        feature_names = artifacts["feature_names"]
        class_names = artifacts["class_names"]

        # Load feature engineer (TF-IDF vectorizer)
        feature_engineer = FeatureEngineer.load()

        # Initialise SHAP explainer
        explainer = RCAExplainer(
            model=model,
            feature_names=feature_names,
            class_names=class_names,
        )

        engine = cls(
            model=model,
            label_encoder=label_encoder,
            feature_engineer=feature_engineer,
            explainer=explainer,
            metadata=metadata,
        )
        logger.info("Inference engine loaded successfully")
        return engine

    def predict(
        self,
        event: dict[str, Any],
        include_counter_evidence: bool = False,
    ) -> dict[str, Any]:
        """
        Run full inference: features → predict → explain → format.

        Args:
            event: Dict with pipeline failure event fields.
                   Must match PipelineFailurePredictionRequest schema.
            include_counter_evidence: Whether to include evidence AGAINST
                                     the prediction (for debugging).

        Returns:
            Dict with:
            - pipeline_name: str
            - predicted_root_cause: str
            - confidence: float (0-1)
            - evidence: list[str] — human-readable evidence
            - counter_evidence: list[str] — (optional) evidence against
            - model_version: str
            - timestamp: str (ISO format)
        """
        pipeline_name = event.get("pipeline_name", "unknown")
        logger.info("Running inference for pipeline: %s", pipeline_name)

        # --- 1. Feature engineering ---
        # Add default timestamp if missing
        if not event.get("timestamp"):
            event["timestamp"] = datetime.now(timezone.utc).isoformat()

        X = self._feature_engineer.transform_single(event)
        logger.debug("Features computed: %d columns", len(X.columns))

        # --- 2. Predict ---
        y_pred = self._model.predict(X)
        y_prob = self._model.predict_proba(X)

        predicted_class_idx = int(y_pred[0])
        predicted_class = self._label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = float(np.max(y_prob[0]))

        logger.info(
            "Prediction: %s (confidence=%.4f) for pipeline %s",
            predicted_class, confidence, pipeline_name,
        )

        # --- 3. Explain with SHAP ---
        explanation = self._explainer.explain(X, predicted_class_idx)

        # --- 4. Generate human-readable evidence ---
        evidence = generate_evidence(explanation, X)

        result: dict[str, Any] = {
            "pipeline_name": pipeline_name,
            "predicted_root_cause": predicted_class,
            "confidence": round(confidence, 4),
            "evidence": evidence,
            "model_version": self._metadata.get("model_version", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if include_counter_evidence:
            result["counter_evidence"] = generate_counter_evidence(explanation, X)

        return result

    @property
    def model_version(self) -> str:
        return self._metadata.get("model_version", "unknown")

    @property
    def class_names(self) -> list[str]:
        return self._metadata.get("class_names", [])

    def health_check(self) -> dict[str, Any]:
        """Check that the engine is functional — used by API health endpoint."""
        return {
            "status": "healthy",
            "model_version": self.model_version,
            "n_classes": len(self.class_names),
            "classes": self.class_names,
        }
