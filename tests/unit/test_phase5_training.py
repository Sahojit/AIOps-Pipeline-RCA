"""
tests/unit/test_phase5_training.py
==================================
Tests for model training, evaluation, and artifact management.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.features.engineer import FeatureEngineer
from src.training.artifact_manager import (
    load_model_artifacts,
    save_model_artifacts,
)
from src.training.evaluator import evaluate_model, format_evaluation_summary
from src.training.trainer import RCAModelTrainer


_LEMMA_CSV = Path(__file__).resolve().parents[2] / "data" / "raw" / "pipeline_failures.csv"


@pytest.fixture(scope="module")
def training_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load LEMMA data and generate features — shared across all tests in this module."""
    df = pd.read_csv(_LEMMA_CSV).sample(n=300, random_state=42).reset_index(drop=True)
    engineer = FeatureEngineer(max_tfidf_features=15)
    X, y, feature_names = engineer.fit_transform(df)
    return X, y, feature_names


@pytest.fixture(scope="module")
def trained_result(training_data) -> dict[str, Any]:
    """Train a model — shared across tests to avoid retraining."""
    X, y, _ = training_data
    trainer = RCAModelTrainer(test_size=0.2, random_state=42)
    return trainer.train(X, y)


# =========================================================================
# Trainer Tests
# =========================================================================

class TestRCAModelTrainer:
    def test_train_returns_required_keys(self, trained_result: dict) -> None:
        required_keys = {
            "model", "label_encoder", "X_train", "X_test",
            "y_train", "y_test", "hyperparams", "class_names", "feature_names",
        }
        assert required_keys.issubset(set(trained_result.keys()))

    def test_model_is_xgboost(self, trained_result: dict) -> None:
        assert isinstance(trained_result["model"], XGBClassifier)

    def test_has_six_classes(self, trained_result: dict) -> None:
        assert len(trained_result["class_names"]) == 6

    def test_train_test_split_sizes(self, trained_result: dict) -> None:
        total = len(trained_result["X_train"]) + len(trained_result["X_test"])
        test_ratio = len(trained_result["X_test"]) / total
        assert 0.15 <= test_ratio <= 0.25  # ~20% with some rounding

    def test_model_can_predict(self, trained_result: dict) -> None:
        model = trained_result["model"]
        X_test = trained_result["X_test"]
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_model_predicts_probabilities(self, trained_result: dict) -> None:
        model = trained_result["model"]
        X_test = trained_result["X_test"]
        probs = model.predict_proba(X_test)
        assert probs.shape == (len(X_test), 6)
        # Probabilities should sum to 1 for each sample
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_reproducibility(self, training_data) -> None:
        X, y, _ = training_data
        trainer1 = RCAModelTrainer(random_state=42)
        trainer2 = RCAModelTrainer(random_state=42)
        result1 = trainer1.train(X, y)
        result2 = trainer2.train(X, y)
        # Same seed → same predictions
        pred1 = result1["model"].predict(result1["X_test"])
        pred2 = result2["model"].predict(result2["X_test"])
        np.testing.assert_array_equal(pred1, pred2)


# =========================================================================
# Evaluator Tests
# =========================================================================

class TestEvaluator:
    @pytest.fixture
    def metrics(self, trained_result: dict) -> dict[str, Any]:
        return evaluate_model(
            model=trained_result["model"],
            X_test=trained_result["X_test"],
            y_test=trained_result["y_test"],
            class_names=trained_result["class_names"],
            feature_names=trained_result["feature_names"],
        )

    def test_returns_required_keys(self, metrics: dict) -> None:
        required = {
            "accuracy", "macro_f1", "weighted_f1",
            "classification_report", "confusion_matrix",
            "feature_importances", "avg_confidence",
        }
        assert required.issubset(set(metrics.keys()))

    def test_accuracy_reasonable(self, metrics: dict) -> None:
        # With LEMMA-RCA data, accuracy should be well above random (16.7%)
        assert metrics["accuracy"] > 0.50, f"Accuracy too low: {metrics['accuracy']}"

    def test_macro_f1_reasonable(self, metrics: dict) -> None:
        assert metrics["macro_f1"] > 0.40, f"Macro F1 too low: {metrics['macro_f1']}"

    def test_confidence_in_range(self, metrics: dict) -> None:
        assert 0.0 <= metrics["avg_confidence"] <= 1.0

    def test_confusion_matrix_shape(self, metrics: dict) -> None:
        cm = metrics["confusion_matrix"]
        assert len(cm) == 6
        assert len(cm[0]) == 6

    def test_feature_importances_sum_to_one(self, metrics: dict) -> None:
        total = metrics["feature_importances"]["importance"].sum()
        np.testing.assert_allclose(total, 1.0, atol=0.01)

    def test_per_class_confidence_all_classes(self, metrics: dict) -> None:
        assert len(metrics["per_class_confidence"]) == 6

    def test_format_summary_is_string(self, metrics: dict, trained_result: dict) -> None:
        summary = format_evaluation_summary(metrics, trained_result["class_names"])
        assert isinstance(summary, str)
        assert "Macro F1" in summary
        assert "Accuracy" in summary


# =========================================================================
# Artifact Manager Tests
# =========================================================================

class TestArtifactManager:
    def test_save_and_load_roundtrip(
        self, trained_result: dict, tmp_path: Path
    ) -> None:
        """Save artifacts, load them back, verify predictions match."""
        from unittest.mock import patch

        model = trained_result["model"]
        metrics = evaluate_model(
            model=model,
            X_test=trained_result["X_test"],
            y_test=trained_result["y_test"],
            class_names=trained_result["class_names"],
            feature_names=trained_result["feature_names"],
        )

        # Patch settings to use tmp_path
        with patch("src.training.artifact_manager.settings") as mock_settings:
            mock_settings.model_dir = tmp_path
            mock_settings.model_metadata_path = tmp_path / "model_metadata.json"

            # Save
            paths = save_model_artifacts(
                model=model,
                label_encoder=trained_result["label_encoder"],
                metrics=metrics,
                hyperparams=trained_result["hyperparams"],
                feature_names=trained_result["feature_names"],
                class_names=trained_result["class_names"],
                version="v1",
                training_samples=len(trained_result["X_train"]),
            )

            assert paths["model"].exists()
            assert paths["label_encoder"].exists()
            assert paths["metadata"].exists()

            # Load
            loaded = load_model_artifacts()

        assert loaded["version"] == "v1"
        assert len(loaded["class_names"]) == 6
        assert len(loaded["feature_names"]) == len(trained_result["feature_names"])

        # Verify loaded model produces same predictions
        original_preds = model.predict(trained_result["X_test"])
        loaded_preds = loaded["model"].predict(trained_result["X_test"])
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_metadata_json_valid(self, trained_result: dict, tmp_path: Path) -> None:
        """Metadata file must be valid JSON with required fields."""
        from unittest.mock import patch

        model = trained_result["model"]
        metrics = evaluate_model(
            model=model,
            X_test=trained_result["X_test"],
            y_test=trained_result["y_test"],
            class_names=trained_result["class_names"],
            feature_names=trained_result["feature_names"],
        )

        with patch("src.training.artifact_manager.settings") as mock_settings:
            mock_settings.model_dir = tmp_path
            mock_settings.model_metadata_path = tmp_path / "model_metadata.json"

            save_model_artifacts(
                model=model,
                label_encoder=trained_result["label_encoder"],
                metrics=metrics,
                hyperparams=trained_result["hyperparams"],
                feature_names=trained_result["feature_names"],
                class_names=trained_result["class_names"],
                version="v1",
                training_samples=200,
            )

        metadata = json.loads((tmp_path / "model_metadata.json").read_text())
        assert metadata["model_version"] == "v1"
        assert metadata["training_samples"] == 200
        assert "accuracy" in metadata["metrics"]
        assert "macro_f1" in metadata["metrics"]
        assert "created_at" in metadata


# =========================================================================
# Integration Tests — end-to-end train → evaluate on LEMMA subset
# =========================================================================

class TestTrainingIntegration:
    """
    Lightweight end-to-end integration: raw CSV → features → train → evaluate.

    Uses only 200 rows so the suite stays fast (<10s), but exercises the full
    code path that train_model.py would run in production.
    """

    @pytest.fixture(scope="class")
    def integration_result(self) -> dict[str, Any]:
        df = pd.read_csv(_LEMMA_CSV).sample(n=200, random_state=7).reset_index(drop=True)
        engineer = FeatureEngineer(max_tfidf_features=20)
        X, y, feature_names = engineer.fit_transform(df)
        trainer = RCAModelTrainer(test_size=0.25, random_state=7)
        result = trainer.train(X, y)
        result["feature_names"] = feature_names
        result["metrics"] = evaluate_model(
            model=result["model"],
            X_test=result["X_test"],
            y_test=result["y_test"],
            class_names=result["class_names"],
            feature_names=feature_names,
        )
        return result

    def test_macro_f1_above_threshold(self, integration_result: dict) -> None:
        macro_f1 = integration_result["metrics"]["macro_f1"]
        assert macro_f1 >= 0.50, (
            f"Macro F1 {macro_f1:.3f} below 0.50 — model not learning signal from LEMMA data"
        )

    def test_accuracy_above_chance(self, integration_result: dict) -> None:
        acc = integration_result["metrics"]["accuracy"]
        assert acc > 1 / 6, f"Accuracy {acc:.3f} is at or below random baseline (0.167)"

    def test_all_classes_predicted(self, integration_result: dict) -> None:
        y_pred = integration_result["metrics"]["y_pred"]
        unique_predicted = set(y_pred.tolist())
        assert len(unique_predicted) >= 4, (
            f"Model only predicted {len(unique_predicted)} classes — "
            "may be collapsing to a few classes"
        )

    def test_feature_count_stable(self, integration_result: dict) -> None:
        X_test = integration_result["X_test"]
        assert X_test.shape[1] >= 50, f"Expected ≥50 features, got {X_test.shape[1]}"

    def test_confidence_reasonable(self, integration_result: dict) -> None:
        avg_conf = integration_result["metrics"]["avg_confidence"]
        assert 0.3 <= avg_conf <= 1.0, f"Average confidence {avg_conf:.3f} out of expected range"

    def test_confusion_matrix_diagonal_dominant(self, integration_result: dict) -> None:
        import numpy as np
        cm = np.array(integration_result["metrics"]["confusion_matrix"])
        diagonal_sum = np.trace(cm)
        total = cm.sum()
        diagonal_ratio = diagonal_sum / total if total > 0 else 0
        assert diagonal_ratio >= 0.50, (
            f"Diagonal ratio {diagonal_ratio:.2f} < 0.50 — model is mostly wrong"
        )
