"""
src/features/feature_definitions.py
====================================
Pure functions that compute individual feature groups from raw event data.

Design principle: each function takes a DataFrame and returns a DataFrame
of NEW columns only. The main FeatureEngineer concatenates them.

WHY separate feature groups?
1. Testability — you can unit-test each group independently
2. Debuggability — when a feature has a bug, you know which group owns it
3. Extensibility — adding a new feature group doesn't touch existing code
4. SHAP interpretation — you can group SHAP values by feature source
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Canonical list of feature groups and their sizes.
# Used by build_features.py to validate the final matrix shape.
FEATURE_GROUPS: dict[str, int] = {
    "execution": 12,
    "schema": 5,
    "temporal": 10,
    "log_signals": 33,
    "tfidf": 50,
}
TOTAL_FEATURES = sum(FEATURE_GROUPS.values())  # 108 before dedup; dedup gives 96


def build_execution_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive features from pipeline execution metrics.

    These capture HOW the pipeline ran before failing:
    - High runtime + high rows = ran a long time → Resource Exhaustion
    - Zero rows + low runtime = never started → Dependency Failure
    - Many retries = transient issue → API Failure, Missing Data

    Input columns required: runtime, retry_count, rows_processed
    """
    logger.info("Building execution features")
    features = pd.DataFrame(index=df.index)

    # --- Direct features (already numeric, pass through) ---
    features["feat_runtime"] = df["runtime"].astype(float)
    features["feat_retry_count"] = df["retry_count"].astype(float)
    features["feat_rows_processed"] = df["rows_processed"].astype(float)

    # --- Derived features ---

    # Log-transform rows_processed: reduces skew from 0 → 5M range.
    # Adding 1 avoids log(0). Tree models handle raw values fine, but
    # log-scale helps with feature importance interpretation.
    features["feat_log_rows_processed"] = np.log1p(df["rows_processed"].astype(float))

    # Log-transform runtime for the same reason
    features["feat_log_runtime"] = np.log1p(df["runtime"].astype(float))

    # Rows per second — throughput before failure.
    # High throughput + sudden failure = different signal than low throughput.
    # Guard against division by zero (runtime=0).
    runtime_safe = df["runtime"].replace(0, 1).astype(float)
    features["feat_rows_per_second"] = df["rows_processed"].astype(float) / runtime_safe

    # Binary: did it process ANY rows?
    # Strong signal — Dependency Failure almost always has 0 rows.
    features["feat_zero_rows"] = (df["rows_processed"] == 0).astype(int)

    # Binary: was runtime very short? (<30s implies early failure)
    features["feat_short_runtime"] = (df["runtime"] < 30).astype(int)

    # Binary: was runtime very long? (>1800s implies resource issues)
    features["feat_long_runtime"] = (df["runtime"] > 1800).astype(int)

    # Binary: many retries? (>2 implies transient/external issue)
    features["feat_high_retries"] = (df["retry_count"] > 2).astype(int)

    # Retry intensity: retries per 100 seconds of runtime.
    # A job that retries 5 times in 10s is very different from 5 retries in 3600s.
    features["feat_retry_rate"] = df["retry_count"].astype(float) / (runtime_safe / 100.0)

    # Rows per retry attempt — efficiency before each failure.
    # High value = processed a lot before needing to retry → transient issue.
    retry_safe = df["retry_count"].replace(0, 1).astype(float)
    features["feat_rows_per_retry"] = df["rows_processed"].astype(float) / retry_safe

    logger.debug("Execution features shape: %s", features.shape)
    return features


def build_schema_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive features from schema change and upstream failure flags.

    These are already strong boolean signals, but we also create
    interaction features that capture COMBINATIONS:
    - schema_change=True AND upstream_failed=True → likely a cascading
      schema migration that broke downstream pipelines.

    Input columns required: schema_change, upstream_failed
    """
    logger.info("Building schema features")
    features = pd.DataFrame(index=df.index)

    # Direct boolean → int
    features["feat_schema_change"] = df["schema_change"].astype(int)
    features["feat_upstream_failed"] = df["upstream_failed"].astype(int)

    # Interaction: both flags set simultaneously
    features["feat_schema_and_upstream"] = (
        df["schema_change"].astype(int) & df["upstream_failed"].astype(int)
    )

    # Interaction: schema change WITHOUT upstream failure
    # (local schema issue vs. cascading from upstream)
    features["feat_schema_no_upstream"] = (
        df["schema_change"].astype(int) & (~df["upstream_failed"]).astype(int)
    )

    # Interaction: upstream failed WITHOUT schema change
    # (pure dependency failure, not schema-related)
    features["feat_upstream_no_schema"] = (
        (~df["schema_change"]).astype(int) & df["upstream_failed"].astype(int)
    )

    logger.debug("Schema features shape: %s", features.shape)
    return features


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive features from the failure timestamp.

    WHY temporal features?
    - Deployments cluster during business hours → Schema Changes peak mid-day
    - Batch jobs run overnight → Resource Exhaustion peaks at night
    - Weekend = fewer deploys → different failure distribution
    - End-of-month = billing pipelines → specific failure patterns

    Input columns required: timestamp (ISO format string or datetime)
    """
    logger.info("Building temporal features")
    features = pd.DataFrame(index=df.index)

    # Parse timestamp if it's a string
    ts = pd.to_datetime(df["timestamp"], utc=True)

    # Hour of day (0-23)
    features["feat_hour"] = ts.dt.hour

    # Day of week (0=Monday, 6=Sunday)
    features["feat_day_of_week"] = ts.dt.dayofweek

    # Is it a weekend? (different failure patterns)
    features["feat_is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Is it during business hours? (9am-6pm)
    features["feat_is_business_hours"] = (
        (ts.dt.hour >= 9) & (ts.dt.hour < 18)
    ).astype(int)

    # Cyclical encoding for hour — so 23:00 and 00:00 are "close" to the model.
    # Without this, a tree model sees hour=23 and hour=0 as maximally different.
    features["feat_hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    features["feat_hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)

    # Cyclical encoding for day of week
    features["feat_dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    features["feat_dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

    # Day of month — end-of-month billing/aggregation pipelines fail differently
    features["feat_day_of_month"] = ts.dt.day

    logger.debug("Temporal features shape: %s", features.shape)
    return features
