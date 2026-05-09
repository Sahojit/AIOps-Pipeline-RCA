"""
src/training/artifact_manager.py
================================
Model artifact versioning and persistence.

WHY version model artifacts?
- In production, you MUST be able to answer:
    * "What model is serving traffic right now?"
    * "When was it trained? On what data?"
    * "What were its metrics? Should we roll back?"
- Without metadata, a .pkl file is a black box. With metadata,
  it's an auditable artifact.

Artifact structure:
    models/
    ├── rca_model_v1.pkl              # Model binary
    ├── tfidf_vectorizer.pkl          # TF-IDF vectorizer
    ├── label_encoder.pkl             # Label encoder
    └── model_metadata.json           # Human-readable metadata
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_model_artifacts(
    model: XGBClassifier,
    label_encoder: LabelEncoder,
    metrics: dict[str, Any],
    hyperparams: dict[str, Any],
    feature_names: list[str],
    class_names: list[str],
    version: str = "v1",
    training_samples: int = 0,
) -> dict[str, Path]:
    """
    Save all model artifacts with metadata.

    Args:
        model: Trained XGBClassifier.
        label_encoder: Fitted LabelEncoder.
        metrics: Evaluation metrics dict from evaluator.
        hyperparams: Hyperparameters used for training.
        feature_names: Ordered list of feature names.
        class_names: Ordered list of class names.
        version: Version string (e.g., "v1", "v2").
        training_samples: Number of training samples used.

    Returns:
        Dict mapping artifact name to its saved path.
    """
    model_dir = settings.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # --- 1. Save model binary ---
    # protocol=4 avoids pickle protocol mismatch errors when the artifact
    # is saved on Python 3.11 and loaded on 3.10 (or vice versa).
    model_filename = f"rca_model_{version}.pkl"
    model_path = model_dir / model_filename
    joblib.dump(model, model_path, protocol=4)
    paths["model"] = model_path
    logger.info("Model saved to %s", model_path)

    # --- 2. Save label encoder ---
    encoder_path = model_dir / "label_encoder.pkl"
    joblib.dump(label_encoder, encoder_path, protocol=4)
    paths["label_encoder"] = encoder_path
    logger.info("Label encoder saved to %s", encoder_path)

    # --- 3. Save metadata ---
    # This is the most important file — it makes the model auditable.
    metadata = {
        "model_version": version,
        "model_filename": model_filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": training_samples,
        "n_features": len(feature_names),
        "n_classes": len(class_names),
        "class_names": class_names,
        "feature_names": feature_names,
        "hyperparameters": _make_json_serializable(hyperparams),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "avg_confidence": metrics["avg_confidence"],
            "per_class_confidence": metrics["per_class_confidence"],
            "per_class_f1": {
                name: metrics["classification_report"].get(name, {}).get("f1-score", 0)
                for name in class_names
            },
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "library_versions": {
            "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
    }

    metadata_path = settings.model_metadata_path
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    paths["metadata"] = metadata_path
    logger.info("Model metadata saved to %s", metadata_path)

    logger.info(
        "All artifacts saved for model %s (accuracy=%.4f, macro_f1=%.4f)",
        version, metrics["accuracy"], metrics["macro_f1"],
    )
    return paths


def load_model_artifacts(
    version: str | None = None,
) -> dict[str, Any]:
    """
    Load model artifacts from disk.

    Args:
        version: Specific version to load. If None, reads from metadata.

    Returns:
        Dict containing: model, label_encoder, metadata, feature_names, class_names.
    """
    # Read metadata to find model filename
    metadata_path = settings.model_metadata_path
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}")

    metadata = json.loads(metadata_path.read_text())

    if version:
        model_filename = f"rca_model_{version}.pkl"
    else:
        model_filename = metadata["model_filename"]

    model_path = settings.model_dir / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info("Loading model artifacts: %s", model_filename)

    model = joblib.load(model_path)
    label_encoder = joblib.load(settings.model_dir / "label_encoder.pkl")

    return {
        "model": model,
        "label_encoder": label_encoder,
        "metadata": metadata,
        "feature_names": metadata["feature_names"],
        "class_names": metadata["class_names"],
        "version": metadata["model_version"],
    }


def _make_json_serializable(obj: Any) -> Any:
    """Convert numpy types to Python types for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
