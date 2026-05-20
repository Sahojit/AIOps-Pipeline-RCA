"""
tests/unit/test_phase6_explainability.py
========================================
Tests for SHAP explainability, evidence generation, and inference engine.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import FeatureEngineer
from src.inference.evidence import generate_counter_evidence, generate_evidence
from src.inference.explainer import RCAExplainer
from src.training.trainer import RCAModelTrainer

_LEMMA_CSV = Path(__file__).resolve().parents[2] / "data" / "raw" / "pipeline_failures.csv"


@pytest.fixture(scope="module")
def trained_system() -> dict[str, Any]:
    """
    Train a complete system for testing — shared across all tests.
    Returns model, engineer, feature names, class names, and test data.
    """
    df = pd.read_csv(_LEMMA_CSV).sample(n=300, random_state=42).reset_index(drop=True)
    engineer = FeatureEngineer(max_tfidf_features=15)
    X, y, feature_names = engineer.fit_transform(df)

    trainer = RCAModelTrainer(test_size=0.2, random_state=42)
    result = trainer.train(X, y)

    return {
        "model": result["model"],
        "label_encoder": result["label_encoder"],
        "engineer": engineer,
        "feature_names": feature_names,
        "class_names": result["class_names"],
        "X_test": result["X_test"],
        "y_test": result["y_test"],
        "raw_df": df,
    }


# =========================================================================
# SHAP Explainer Tests
# =========================================================================

class TestRCAExplainer:
    @pytest.fixture
    def explainer(self, trained_system: dict) -> RCAExplainer:
        return RCAExplainer(
            model=trained_system["model"],
            feature_names=trained_system["feature_names"],
            class_names=trained_system["class_names"],
        )

    def test_explain_returns_required_keys(
        self, explainer: RCAExplainer, trained_system: dict
    ) -> None:
        X_single = trained_system["X_test"].iloc[:1]
        pred = trained_system["model"].predict(X_single)[0]
        result = explainer.explain(X_single, int(pred))

        required_keys = {
            "shap_values", "base_value", "feature_contributions",
            "top_positive_features", "top_negative_features", "predicted_class",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_shap_values_length_matches_features(
        self, explainer: RCAExplainer, trained_system: dict
    ) -> None:
        X_single = trained_system["X_test"].iloc[:1]
        pred = trained_system["model"].predict(X_single)[0]
        result = explainer.explain(X_single, int(pred))

        assert len(result["shap_values"]) == len(trained_system["feature_names"])

    def test_feature_contributions_is_dataframe(
        self, explainer: RCAExplainer, trained_system: dict
    ) -> None:
        X_single = trained_system["X_test"].iloc[:1]
        pred = trained_system["model"].predict(X_single)[0]
        result = explainer.explain(X_single, int(pred))

        assert isinstance(result["feature_contributions"], pd.DataFrame)
        assert "feature" in result["feature_contributions"].columns
        assert "shap_value" in result["feature_contributions"].columns

    def test_contributions_sorted_by_absolute_value(
        self, explainer: RCAExplainer, trained_system: dict
    ) -> None:
        X_single = trained_system["X_test"].iloc[:1]
        pred = trained_system["model"].predict(X_single)[0]
        result = explainer.explain(X_single, int(pred))

        abs_values = result["feature_contributions"]["abs_shap"].values
        # Should be sorted descending
        assert all(abs_values[i] >= abs_values[i + 1] for i in range(len(abs_values) - 1))

    def test_predicted_class_is_string(
        self, explainer: RCAExplainer, trained_system: dict
    ) -> None:
        X_single = trained_system["X_test"].iloc[:1]
        pred = trained_system["model"].predict(X_single)[0]
        result = explainer.explain(X_single, int(pred))

        assert isinstance(result["predicted_class"], str)
        assert result["predicted_class"] in trained_system["class_names"]

    def test_explain_all_classes_returns_all(
        self, explainer: RCAExplainer, trained_system: dict
    ) -> None:
        X_single = trained_system["X_test"].iloc[:1]
        all_explanations = explainer.explain_all_classes(X_single)

        assert len(all_explanations) == len(trained_system["class_names"])
        for class_name in trained_system["class_names"]:
            assert class_name in all_explanations

    def test_base_value_is_float(
        self, explainer: RCAExplainer, trained_system: dict
    ) -> None:
        X_single = trained_system["X_test"].iloc[:1]
        pred = trained_system["model"].predict(X_single)[0]
        result = explainer.explain(X_single, int(pred))

        assert isinstance(result["base_value"], float)


# =========================================================================
# Evidence Generation Tests
# =========================================================================

class TestEvidenceGeneration:
    @pytest.fixture
    def sample_explanation(self, trained_system: dict) -> tuple[dict, pd.DataFrame]:
        """Generate a real explanation for testing."""
        explainer = RCAExplainer(
            model=trained_system["model"],
            feature_names=trained_system["feature_names"],
            class_names=trained_system["class_names"],
        )
        X_single = trained_system["X_test"].iloc[:1]
        pred = trained_system["model"].predict(X_single)[0]
        explanation = explainer.explain(X_single, int(pred))
        return explanation, X_single

    def test_evidence_returns_list_of_strings(
        self, sample_explanation: tuple
    ) -> None:
        explanation, X = sample_explanation
        evidence = generate_evidence(explanation, X)

        assert isinstance(evidence, list)
        assert all(isinstance(e, str) for e in evidence)

    def test_evidence_not_empty(self, sample_explanation: tuple) -> None:
        explanation, X = sample_explanation
        evidence = generate_evidence(explanation, X)

        assert len(evidence) > 0

    def test_evidence_max_items_respected(self, sample_explanation: tuple) -> None:
        explanation, X = sample_explanation
        evidence = generate_evidence(explanation, X, max_evidence=3)

        assert len(evidence) <= 3

    def test_evidence_contains_impact_qualifier(self, sample_explanation: tuple) -> None:
        explanation, X = sample_explanation
        evidence = generate_evidence(explanation, X)

        # At least one evidence string should have an impact qualifier
        qualifiers = ["very high impact", "high impact", "moderate impact", "contributing factor"]
        has_qualifier = any(
            any(q in e for q in qualifiers)
            for e in evidence
        )
        assert has_qualifier, f"No impact qualifier found in evidence: {evidence}"

    def test_counter_evidence_returns_list(self, sample_explanation: tuple) -> None:
        explanation, X = sample_explanation
        counter = generate_counter_evidence(explanation, X)

        assert isinstance(counter, list)

    def test_counter_evidence_prefixed_with_against(
        self, sample_explanation: tuple
    ) -> None:
        explanation, X = sample_explanation
        counter = generate_counter_evidence(explanation, X)

        for item in counter:
            assert item.startswith("Against:"), f"Missing 'Against:' prefix: {item}"


# =========================================================================
# End-to-End Inference Test (without loading from disk)
# =========================================================================

class TestInferenceEndToEnd:
    def test_full_prediction_flow(self, trained_system: dict) -> None:
        """Test the complete flow: raw event → features → predict → explain → evidence."""
        model = trained_system["model"]
        engineer = trained_system["engineer"]
        label_encoder = trained_system["label_encoder"]

        # Simulate a raw event
        event = {
            "pipeline_name": "user_etl_pipeline",
            "task_name": "extract_users",
            "runtime": 15,
            "retry_count": 1,
            "rows_processed": 100,
            "schema_change": True,
            "upstream_failed": False,
            "error_log": "KeyError: column 'age' not found in dataframe. Schema may have changed upstream.",
            "timestamp": "2026-03-03T12:00:00+00:00",
        }

        # Feature engineering
        X = engineer.transform_single(event)
        assert X.shape[0] == 1

        # Predict
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        predicted_class = label_encoder.inverse_transform(y_pred)[0]
        confidence = float(np.max(y_prob[0]))

        assert isinstance(predicted_class, str)
        assert 0.0 <= confidence <= 1.0

        # Explain
        explainer = RCAExplainer(
            model=model,
            feature_names=trained_system["feature_names"],
            class_names=trained_system["class_names"],
        )
        explanation = explainer.explain(X, int(y_pred[0]))
        evidence = generate_evidence(explanation, X)

        assert len(evidence) > 0
        assert all(isinstance(e, str) for e in evidence)

    def test_schema_change_event_prediction(self, trained_system: dict) -> None:
        """A clear schema change event should likely predict Schema Change."""
        model = trained_system["model"]
        engineer = trained_system["engineer"]
        label_encoder = trained_system["label_encoder"]

        event = {
            "pipeline_name": "user_etl_pipeline",
            "task_name": "transform_users",
            "runtime": 10,
            "retry_count": 0,
            "rows_processed": 50,
            "schema_change": True,
            "upstream_failed": False,
            "error_log": "SchemaValidationError: expected 12 columns but received 11. Missing: [email_verified]",
            "timestamp": "2026-03-03T10:00:00+00:00",
        }

        X = engineer.transform_single(event)
        y_pred = model.predict(X)
        predicted = label_encoder.inverse_transform(y_pred)[0]

        # With such a clear signal, the model should get this right
        assert predicted == "Schema Change", (
            f"Expected 'Schema Change' but got '{predicted}'"
        )
