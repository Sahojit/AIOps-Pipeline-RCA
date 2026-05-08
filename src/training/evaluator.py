"""
src/training/evaluator.py
=========================
Model evaluation — metrics that matter for production deployment.

In production ML, accuracy alone is MISLEADING for imbalanced classes.
If 22% of failures are Schema Change, a model that always predicts
"Schema Change" gets 22% accuracy — but is completely useless.

We report:
- Per-class precision, recall, F1 — catches weak classes
- Macro-averaged F1 — treats all classes equally (our primary metric)
- Weighted-averaged F1 — accounts for class frequency
- Confusion matrix — shows WHERE the model confuses classes
- Top feature importances — sanity check that the model learned real signals
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier

from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: list[str],
    feature_names: list[str],
    top_k_features: int = 15,
) -> dict[str, Any]:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained XGBClassifier.
        X_test: Test feature matrix.
        y_test: Test labels (integer-encoded).
        class_names: Ordered list of class name strings.
        feature_names: Ordered list of feature column names.
        top_k_features: Number of top features to report.

    Returns:
        Dict containing all metrics, confusion matrix, and feature importances.
    """
    logger.info("Evaluating model on %d test samples", len(X_test))

    # --- Predictions ---
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # --- Core Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    logger.info("Accuracy: %.4f", accuracy)
    logger.info("Macro F1: %.4f", macro_f1)
    logger.info("Weighted F1: %.4f", weighted_f1)

    # --- Per-Class Report ---
    # classification_report gives precision, recall, F1 per class
    report_dict = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    logger.info("Classification Report:\n%s", report_text)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    logger.info("Confusion Matrix:\n%s", cm_df.to_string())

    # --- Feature Importance ---
    # XGBoost's feature_importances_ uses "gain" by default — how much
    # each feature contributes to reducing the loss across all trees.
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    top_features = importance_df.head(top_k_features)
    logger.info("Top %d features:\n%s", top_k_features, top_features.to_string(index=False))

    # --- Average Confidence ---
    # Mean of the max predicted probability across test samples.
    # High confidence (>0.8) = model is decisive. Low (<0.5) = model is guessing.
    avg_confidence = float(np.mean(np.max(y_prob, axis=1)))
    logger.info("Average prediction confidence: %.4f", avg_confidence)

    # --- Per-Class Confidence ---
    per_class_confidence: dict[str, float] = {}
    for i, class_name in enumerate(class_names):
        mask = y_test == i
        if mask.sum() > 0:
            class_probs = y_prob[mask]
            per_class_confidence[class_name] = float(np.mean(np.max(class_probs, axis=1)))
    logger.info("Per-class confidence: %s", per_class_confidence)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_df": cm_df,
        "feature_importances": importance_df,
        "top_features": top_features,
        "avg_confidence": avg_confidence,
        "per_class_confidence": per_class_confidence,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def confusion_matrix_report(metrics: dict[str, Any], class_names: list[str]) -> str:
    """
    Format the confusion matrix as a readable text table.

    Rows = actual class, Columns = predicted class.
    Off-diagonal entries show WHERE the model is confused.
    """
    cm_df = metrics["confusion_matrix_df"]
    lines = [
        "",
        "Confusion Matrix (rows=actual, cols=predicted):",
        "-" * 60,
        cm_df.to_string(),
        "",
    ]

    # Highlight the most common misclassification
    cm = metrics["confusion_matrix"]
    cm_array = np.array(cm)
    np.fill_diagonal(cm_array, 0)
    if cm_array.max() > 0:
        row, col = np.unravel_index(cm_array.argmax(), cm_array.shape)
        lines.append(
            f"Most common error: '{class_names[row]}' predicted as "
            f"'{class_names[col]}' ({cm_array[row, col]} times)"
        )
    return "\n".join(lines)


def format_evaluation_summary(metrics: dict[str, Any], class_names: list[str]) -> str:
    """
    Format evaluation results into a human-readable summary string.
    Used for console output and model metadata.
    """
    lines = [
        "=" * 60,
        "MODEL EVALUATION SUMMARY",
        "=" * 60,
        f"Accuracy:    {metrics['accuracy']:.4f}",
        f"Macro F1:    {metrics['macro_f1']:.4f}  ← primary metric",
        f"Weighted F1: {metrics['weighted_f1']:.4f}",
        f"Avg Confidence: {metrics['avg_confidence']:.4f}",
        "",
        "Per-Class Metrics:",
        "-" * 60,
    ]

    report = metrics["classification_report"]
    for class_name in class_names:
        if class_name in report:
            r = report[class_name]
            conf = metrics["per_class_confidence"].get(class_name, 0)
            lines.append(
                f"  {class_name:25s}  P={r['precision']:.3f}  R={r['recall']:.3f}  "
                f"F1={r['f1-score']:.3f}  Conf={conf:.3f}  n={int(r['support'])}"
            )

    lines.extend([
        "",
        "Top 10 Features:",
        "-" * 60,
    ])
    for _, row in metrics["top_features"].head(10).iterrows():
        lines.append(f"  {row['feature']:40s}  {row['importance']:.4f}")

    lines.append("=" * 60)
    return "\n".join(lines)


def check_go_no_go(metrics: dict[str, Any], min_macro_f1: float = 0.70) -> None:
    """
    Go/No-Go gate: raise if macro F1 is below the minimum threshold.

    This is the single quality gate before saving model artifacts.
    If the model doesn't meet the bar, we refuse to overwrite the
    production artifact — better to keep the old model than deploy a bad one.

    Args:
        metrics: Output of evaluate_model().
        min_macro_f1: Minimum acceptable macro F1 score. Default 0.70.

    Raises:
        ValueError: If macro F1 < min_macro_f1.
    """
    macro_f1 = metrics["macro_f1"]
    if macro_f1 < min_macro_f1:
        raise ValueError(
            f"Go/No-Go FAILED: macro F1 {macro_f1:.4f} < threshold {min_macro_f1:.2f}. "
            "Model artifacts NOT saved. Fix class coverage or review training data."
        )
    logger.info(
        "Go/No-Go PASSED: macro F1 %.4f >= threshold %.2f",
        macro_f1, min_macro_f1,
    )
