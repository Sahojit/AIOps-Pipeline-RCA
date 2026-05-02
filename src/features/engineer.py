"""
src/features/engineer.py
========================
Main feature engineering pipeline — the single entry point for
transforming raw pipeline failure events into ML-ready features.

WHY a single FeatureEngineer class?
- Training and inference MUST use the EXACT same feature pipeline.
  If training adds a feature that inference doesn't, the model breaks.
- This class enforces that contract: fit() during training, transform()
  during inference, same column order guaranteed.
- The LogFeatureExtractor (TF-IDF) is stateful (needs fitting), so
  FeatureEngineer manages its lifecycle.

Pipeline flow:
    Raw DataFrame
        → build_execution_features()
        → build_schema_features()
        → build_temporal_features()
        → LogFeatureExtractor.transform()    ← Phase 3
        → pd.concat(all, axis=1)
        → Final feature matrix (X) + label vector (y)
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.features.feature_definitions import (
    build_execution_features,
    build_schema_features,
    build_temporal_features,
)
from src.ingestion.log_feature_extractor import LogFeatureExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Orchestrates all feature engineering.

    Usage (training):
        engineer = FeatureEngineer()
        X, y, feature_names = engineer.fit_transform(df)
        engineer.save()

    Usage (inference):
        engineer = FeatureEngineer.load()
        X = engineer.transform_single(event_dict)
    """

    # Column that holds the label — used to separate X from y
    LABEL_COLUMN = "root_cause"

    # Raw columns needed from input data
    REQUIRED_COLUMNS = [
        "pipeline_name", "task_name", "runtime", "retry_count",
        "rows_processed", "schema_change", "upstream_failed",
        "error_log", "timestamp",
    ]

    def __init__(self, max_tfidf_features: Optional[int] = None) -> None:
        self._log_extractor = LogFeatureExtractor(max_tfidf_features=max_tfidf_features)
        self._feature_names: list[str] = []
        self._is_fitted = False

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Ensure all required columns are present."""
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Fit stateful components (TF-IDF vectorizer) on training data.
        Must be called before transform().

        Args:
            df: Training DataFrame with all REQUIRED_COLUMNS.

        Returns:
            self
        """
        self._validate_input(df)
        logger.info("Fitting FeatureEngineer on %d rows", len(df))

        # Only the log extractor needs fitting (TF-IDF vocabulary)
        self._log_extractor.fit(df["error_log"].tolist())
        self._is_fitted = True

        # Build feature names by doing a dry-run transform on first row
        # to capture all column names in order
        sample = df.head(1)
        features_df = self._build_all_features(sample)
        self._feature_names = list(features_df.columns)
        logger.info("Feature names captured: %d total features", len(self._feature_names))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw event data into features.

        Args:
            df: DataFrame with REQUIRED_COLUMNS.

        Returns:
            DataFrame of features (no label column).
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer not fitted. Call fit() or load() first.")
        self._validate_input(df)
        logger.info("Transforming %d rows into features", len(df))

        features_df = self._build_all_features(df)

        # Ensure column order matches training — critical for model inference
        features_df = features_df[self._feature_names]
        return features_df

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """
        Fit on data and return (X, y, feature_names).

        This is the primary training-time interface.

        Args:
            df: Training DataFrame with REQUIRED_COLUMNS + LABEL_COLUMN.

        Returns:
            Tuple of:
            - X: Feature DataFrame (n_samples x n_features)
            - y: Label Series (root_cause values)
            - feature_names: List of feature column names
        """
        if self.LABEL_COLUMN not in df.columns:
            raise ValueError(f"Label column '{self.LABEL_COLUMN}' not found in DataFrame")

        self.fit(df)
        X = self.transform(df)
        y = df[self.LABEL_COLUMN].reset_index(drop=True)

        logger.info(
            "Feature engineering complete: X=%s, y=%s, %d features",
            X.shape, y.shape, len(self._feature_names),
        )
        return X, y, self._feature_names

    def transform_single(self, event: dict) -> pd.DataFrame:
        """
        Transform a single event dict into features — used at inference time.

        Args:
            event: Dict matching PipelineFailurePredictionRequest fields.

        Returns:
            Single-row feature DataFrame.
        """
        df = pd.DataFrame([event])
        return self.transform(df)

    def _feature_groups(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Build each feature group independently and return as a named dict.

        Separating group construction from concatenation makes each step
        independently testable and keeps _build_all_features readable.
        """
        log_features = self._log_extractor.transform(df["error_log"].tolist())
        tfidf_count = len([c for c in log_features.columns if c.startswith("tfidf_")])
        logger.debug(
            "Log features: %d regex + %d tfidf",
            len(log_features.columns) - tfidf_count, tfidf_count,
        )
        return {
            "execution": build_execution_features(df),
            "schema": build_schema_features(df),
            "temporal": build_temporal_features(df),
            "log": log_features,
        }

    def _build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal: build all feature groups and concatenate.

        Order matters: features are concatenated in this exact order,
        and the model expects this order at inference time.
        """
        df = df.reset_index(drop=True)
        groups = self._feature_groups(df)

        all_features = pd.concat(list(groups.values()), axis=1)

        # Guard against duplicate column names from feature group collisions
        dupes = all_features.columns[all_features.columns.duplicated()].tolist()
        if dupes:
            logger.warning("Duplicate feature columns detected — dropping: %s", dupes)
            all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        logger.debug(
            "Feature groups: %s → total=%d",
            {k: len(v.columns) for k, v in groups.items()},
            len(all_features.columns),
        )
        return all_features

    @property
    def feature_names(self) -> list[str]:
        """Get the ordered list of feature names."""
        if not self._is_fitted:
            raise RuntimeError("Not fitted.")
        return self._feature_names.copy()

    @property
    def feature_count(self) -> int:
        """Number of features produced by this engineer."""
        return len(self._feature_names) if self._is_fitted else 0

    @property
    def tfidf_feature_names(self) -> list[str]:
        """Get just the TF-IDF feature names (for SHAP grouping)."""
        if not self._is_fitted:
            raise RuntimeError("Not fitted.")
        return [n for n in self._feature_names if n.startswith("tfidf_")]

    def __repr__(self) -> str:
        status = f"fitted, {self.feature_count} features" if self._is_fitted else "not fitted"
        return f"FeatureEngineer({status})"

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save the fitted feature engineer (saves TF-IDF vectorizer).

        Args:
            path: Custom path for TF-IDF artifact.

        Returns:
            Path where the TF-IDF vectorizer was saved.
        """
        return self._log_extractor.save(path)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "FeatureEngineer":
        """
        Load a previously fitted feature engineer.

        Args:
            path: Custom path for TF-IDF artifact.

        Returns:
            FeatureEngineer ready for transform().
        """
        instance = cls()
        instance._log_extractor = LogFeatureExtractor.load(path)
        instance._is_fitted = True

        # Rebuild feature names from a dummy transform
        dummy_df = pd.DataFrame([{
            "pipeline_name": "dummy",
            "task_name": "dummy",
            "runtime": 0,
            "retry_count": 0,
            "rows_processed": 0,
            "schema_change": False,
            "upstream_failed": False,
            "error_log": "dummy error",
            "timestamp": "2026-01-01T00:00:00+00:00",
        }])
        features = instance._build_all_features(dummy_df)
        instance._feature_names = list(features.columns)

        logger.info(
            "FeatureEngineer loaded with %d features", len(instance._feature_names)
        )
        return instance
