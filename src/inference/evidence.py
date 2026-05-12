"""
src/inference/evidence.py
=========================
Translates SHAP feature contributions into human-readable evidence strings.

WHY human-readable evidence?
- Engineers don't read SHAP values. They read:
    "schema_change flag was True (high impact)"
    "error log contained 'column not found' (strong signal)"
- The API response includes these as the `evidence` field so the
  on-call engineer can immediately understand WHY the model predicted
  this root cause — without ever seeing a SHAP plot.

This is the bridge between ML and operations.
"""

from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature-to-English mapping — translates feature names to human descriptions.
# Only the most important/interpretable features are mapped.
# Features not mapped get a generic description.
# ---------------------------------------------------------------------------
FEATURE_DESCRIPTIONS: dict[str, str] = {
    # Execution features
    "feat_runtime": "task runtime was {value:.0f} seconds",
    "feat_retry_count": "task was retried {value:.0f} times",
    "feat_rows_processed": "{value:.0f} rows were processed before failure",
    "feat_zero_rows": "zero rows were processed (task never started processing data)",
    "feat_short_runtime": "task failed very quickly (< 30 seconds)",
    "feat_long_runtime": "task ran for a very long time (> 30 minutes)",
    "feat_high_retries": "task had many retries (> 2), suggesting transient issue",
    "feat_rows_per_second": "throughput was {value:.1f} rows/second before failure",

    # Schema features
    "feat_schema_change": "upstream schema change was detected",
    "feat_upstream_failed": "an upstream dependency pipeline has failed",
    "feat_schema_and_upstream": "both schema change AND upstream failure detected",
    "feat_schema_no_upstream": "schema change detected without upstream failure (local schema issue)",
    "feat_upstream_no_schema": "upstream failed without schema change (pure dependency failure)",

    # Temporal
    "feat_is_weekend": "failure occurred on a weekend",
    "feat_is_business_hours": "failure occurred during business hours (deploy window)",

    # Log regex features
    "has_error_type": "error log contains a recognisable error type",
    "has_http_error": "error log contains an HTTP error code",
    "http_status_code": "HTTP status code {value:.0f} was found in the error log",
    "is_server_error": "a server error (5xx) was detected",
    "is_client_error": "a client error (4xx) was detected",
    "has_s3_path": "error references an S3 data path",
    "has_file_path": "error references a file path",
    "has_memory_ref": "error log mentions memory/allocation",
    "memory_value_mb": "error references {value:.0f} MB of memory",
    "has_timeout": "error log mentions a timeout",
    "timeout_seconds": "timeout of {value:.0f} seconds was detected",
    "column_count": "{value:.0f} column name(s) mentioned in the error",
    "keyword_score_schema": "error log contains {value:.0f} schema-related keywords",
    "keyword_score_resource": "error log contains {value:.0f} resource-related keywords",
    "keyword_score_api": "error log contains {value:.0f} API-related keywords",
    "keyword_score_data_quality": "error log contains {value:.0f} data quality keywords",
    "keyword_score_dependency": "error log contains {value:.0f} dependency-related keywords",
    "keyword_score_missing_data": "error log contains {value:.0f} missing data keywords",
}


def generate_evidence(
    explanation: dict[str, Any],
    feature_values: pd.DataFrame,
    max_evidence: int = 5,
) -> list[str]:
    """
    Convert SHAP explanation into human-readable evidence strings.

    Args:
        explanation: Output from RCAExplainer.explain().
        feature_values: Single-row DataFrame of actual feature values
            (needed to fill in the {value} placeholders).
        max_evidence: Maximum number of evidence strings to return.

    Returns:
        List of human-readable evidence strings, sorted by importance.
        Example: ["upstream schema change was detected", "zero rows processed"]
    """
    top_features: pd.DataFrame = explanation["top_positive_features"]

    if top_features.empty:
        return ["No strong evidence found for this prediction"]

    evidence: list[str] = []
    feature_vals = feature_values.iloc[0] if len(feature_values) > 0 else pd.Series()

    for _, row in top_features.head(max_evidence).iterrows():
        feature_name = row["feature"]
        shap_value = row["shap_value"]
        actual_value = feature_vals.get(feature_name, 0)

        # Use human-readable description if available
        if feature_name in FEATURE_DESCRIPTIONS:
            try:
                desc = FEATURE_DESCRIPTIONS[feature_name].format(value=float(actual_value))
            except (ValueError, KeyError):
                desc = FEATURE_DESCRIPTIONS[feature_name].replace("{value:.0f}", str(actual_value))
        elif feature_name.startswith("tfidf_"):
            # TF-IDF features: extract the word
            word = feature_name.replace("tfidf_", "").replace("_", " ")
            desc = f"error log contains the term '{word}'"
        else:
            desc = f"{feature_name} = {actual_value}"

        # Add impact qualifier
        abs_shap = abs(shap_value)
        if abs_shap > 1.0:
            impact = "very high impact"
        elif abs_shap > 0.5:
            impact = "high impact"
        elif abs_shap > 0.2:
            impact = "moderate impact"
        else:
            impact = "contributing factor"

        evidence.append(f"{desc} ({impact})")

    logger.debug(
        "Generated %d evidence strings for predicted class '%s'",
        len(evidence), explanation["predicted_class"],
    )
    return evidence


def generate_counter_evidence(
    explanation: dict[str, Any],
    feature_values: pd.DataFrame,
    max_items: int = 3,
) -> list[str]:
    """
    Generate evidence for features that pushed AWAY from the predicted class.

    Useful for showing "What DOESN'T support this prediction?" — increases
    engineer trust in the model by showing it's not blindly confident.

    Args:
        explanation: Output from RCAExplainer.explain().
        feature_values: Single-row DataFrame of feature values.
        max_items: Maximum counter-evidence items.

    Returns:
        List of counter-evidence strings.
    """
    top_negative: pd.DataFrame = explanation["top_negative_features"]

    if top_negative.empty:
        return []

    counter: list[str] = []
    feature_vals = feature_values.iloc[0] if len(feature_values) > 0 else pd.Series()

    for _, row in top_negative.head(max_items).iterrows():
        feature_name = row["feature"]
        actual_value = feature_vals.get(feature_name, 0)

        if feature_name in FEATURE_DESCRIPTIONS:
            try:
                desc = FEATURE_DESCRIPTIONS[feature_name].format(value=float(actual_value))
            except (ValueError, KeyError):
                desc = feature_name
        elif feature_name.startswith("tfidf_"):
            word = feature_name.replace("tfidf_", "").replace("_", " ")
            desc = f"error log contains '{word}'"
        else:
            desc = f"{feature_name} = {actual_value}"

        counter.append(f"Against: {desc}")

    return counter
