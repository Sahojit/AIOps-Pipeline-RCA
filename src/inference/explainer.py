"""
src/inference/explainer.py
==========================
SHAP-based model explainability for RCA predictions.

WHY SHAP over LIME, feature importance, etc.?
1. SHAP gives EXACT Shapley values for tree models (not approximate)
2. TreeExplainer is O(TLD²) — fast enough for real-time API usage
3. Shapley values are the ONLY feature attribution method with a solid
   game-theoretic foundation (consistency, local accuracy, missingness)
4. SHAP values are PER-PREDICTION — not global. Global feature importance
   (from XGBoost) tells you what matters ON AVERAGE. SHAP tells you what
   mattered for THIS specific failure.

Key concepts:
- base_value: The model's average prediction across training data (prior)
- shap_values[i]: How much feature i pushes the prediction AWAY from the base
- Positive SHAP → pushes toward this class
- Negative SHAP → pushes away from this class
"""

from typing import Any

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RCAExplainer:
    """
    Wraps SHAP TreeExplainer for XGBoost multi-class explanations.

    Usage:
        explainer = RCAExplainer(model, feature_names, class_names)
        explanation = explainer.explain(X_single_row)
    """

    def __init__(
        self,
        model: XGBClassifier,
        feature_names: list[str],
        class_names: list[str],
    ) -> None:
        """
        Args:
            model: Trained XGBClassifier.
            feature_names: Ordered list of feature column names.
            class_names: Ordered list of class names (from LabelEncoder).
        """
        self._model = model
        self._feature_names = feature_names
        self._class_names = class_names

        # TreeExplainer is fast for tree-based models — O(TLD²) per sample
        # where T=trees, L=leaves, D=depth. For our model this is ~1ms.
        logger.info("Initialising SHAP TreeExplainer")
        self._explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer ready")

    def explain(
        self,
        X: pd.DataFrame,
        predicted_class_idx: int,
    ) -> dict[str, Any]:
        """
        Generate SHAP explanation for a single prediction.

        Args:
            X: Feature DataFrame (single row or batch).
            predicted_class_idx: Index of the predicted class
                (from model.predict()).

        Returns:
            Dict containing:
            - shap_values: Raw SHAP values for the predicted class
            - top_positive_features: Features pushing TOWARD this class
            - top_negative_features: Features pushing AWAY from this class
            - base_value: Prior probability for this class
            - feature_contributions: All features with their SHAP values
        """
        # Get SHAP values — returns list of arrays, one per class
        shap_values = self._explainer.shap_values(X)

        # For multi-class, shap_values is a list of (n_samples, n_features)
        # Index into the predicted class
        if isinstance(shap_values, list):
            class_shap = shap_values[predicted_class_idx]
        else:
            # Some SHAP versions return (n_samples, n_features, n_classes)
            class_shap = shap_values[:, :, predicted_class_idx]

        # If single sample, squeeze to 1D
        if class_shap.ndim > 1:
            class_shap = class_shap[0]

        # Base value (prior) for the predicted class
        base_values = self._explainer.expected_value
        if isinstance(base_values, (list, np.ndarray)):
            base_value = float(base_values[predicted_class_idx])
        else:
            base_value = float(base_values)

        # Build feature contribution table
        contributions = pd.DataFrame({
            "feature": self._feature_names,
            "shap_value": class_shap,
            "abs_shap": np.abs(class_shap),
        }).sort_values("abs_shap", ascending=False)

        # Top features pushing TOWARD the predicted class (positive SHAP)
        top_positive = (
            contributions[contributions["shap_value"] > 0]
            .head(10)
            .copy()
        )

        # Top features pushing AWAY from the predicted class (negative SHAP)
        top_negative = (
            contributions[contributions["shap_value"] < 0]
            .head(5)
            .copy()
        )

        return {
            "shap_values": class_shap,
            "base_value": base_value,
            "feature_contributions": contributions,
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
            "predicted_class": self._class_names[predicted_class_idx],
        }

    def explain_all_classes(
        self,
        X: pd.DataFrame,
    ) -> dict[str, dict[str, Any]]:
        """
        Get SHAP explanations for ALL classes for one sample.
        Useful for understanding why the model chose one class over another.

        Args:
            X: Single-row feature DataFrame.

        Returns:
            Dict mapping class_name → explanation dict.
        """
        explanations = {}
        for idx, class_name in enumerate(self._class_names):
            explanations[class_name] = self.explain(X, idx)
        return explanations

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names.copy()

    @property
    def class_names(self) -> list[str]:
        return self._class_names.copy()
