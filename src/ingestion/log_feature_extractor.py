"""
src/ingestion/log_feature_extractor.py
======================================
Unified facade combining regex log parsing + TF-IDF vectorization.

WHY a unified extractor?
- Downstream code (feature engineering, inference) should not care
  whether features came from regex or TF-IDF. They call ONE method
  and get a feature DataFrame back.
- This also ensures regex features and TF-IDF features are always
  generated TOGETHER and in the SAME ORDER — critical for inference
  to match training.

Usage:
    # Training time
    extractor = LogFeatureExtractor()
    log_features_df = extractor.fit_transform(error_logs)
    extractor.save()

    # Inference time
    extractor = LogFeatureExtractor.load()
    log_features_df = extractor.transform([single_log])
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.ingestion.log_parser import parse_error_logs_batch
from src.ingestion.text_vectorizer import LogTextVectorizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LogFeatureExtractor:
    """
    Combines regex-based structured features with TF-IDF text features.

    Produces a DataFrame where each row corresponds to one error log,
    with columns for both regex features and TF-IDF features.
    """

    def __init__(self, max_tfidf_features: Optional[int] = None) -> None:
        self._vectorizer = LogTextVectorizer(max_features=max_tfidf_features)
        self._is_fitted = False

    def fit(self, error_logs: list[str]) -> "LogFeatureExtractor":
        """
        Fit the TF-IDF component on training logs.
        Regex features don't need fitting — they're deterministic.

        Args:
            error_logs: Training set error log strings.

        Returns:
            self
        """
        error_logs = self._sanitize(error_logs)
        logger.info("Fitting LogFeatureExtractor on %d logs", len(error_logs))
        self._vectorizer.fit(error_logs)
        self._is_fitted = True
        return self

    @staticmethod
    def _sanitize(logs: list[str]) -> list[str]:
        return [l if l and l.strip() else "" for l in logs]

    def transform(self, error_logs: list[str]) -> pd.DataFrame:
        """
        Extract all log features (regex + TF-IDF) from error logs.

        Args:
            error_logs: List of raw error log strings.

        Returns:
            DataFrame with regex feature columns + TF-IDF feature columns.
            Shape: (n_logs, n_regex_features + n_tfidf_features)
        """
        if not self._is_fitted:
            raise RuntimeError("LogFeatureExtractor not fitted. Call fit() first.")

        # 1. Regex features → list of dicts → DataFrame
        regex_features = parse_error_logs_batch(error_logs)
        regex_df = pd.DataFrame(regex_features)

        # 2. TF-IDF features → numpy array → DataFrame with named columns
        tfidf_matrix = self._vectorizer.transform(error_logs)
        tfidf_names = self._vectorizer.get_feature_names()
        tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_names)

        # 3. Concatenate side by side
        combined = pd.concat([regex_df, tfidf_df], axis=1)

        logger.info(
            "Log features extracted: %d regex + %d tfidf = %d total features",
            len(regex_df.columns), len(tfidf_df.columns), len(combined.columns),
        )
        return combined

    def fit_transform(self, error_logs: list[str]) -> pd.DataFrame:
        """Fit and transform in one call."""
        self.fit(error_logs)
        return self.transform(error_logs)

    def get_feature_names(self) -> list[str]:
        """Get all feature names (regex + tfidf) in order."""
        if not self._is_fitted:
            raise RuntimeError("Not fitted.")
        # Regex feature names are deterministic — use a sample parse to get them
        sample_features = parse_error_logs_batch(["sample error"])[0]
        regex_names = list(sample_features.keys())
        tfidf_names = self._vectorizer.get_feature_names()
        return regex_names + tfidf_names

    def save(self, path: Optional[Path] = None) -> Path:
        """Save the TF-IDF vectorizer (regex parser is stateless)."""
        return self._vectorizer.save(path)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "LogFeatureExtractor":
        """Load a previously saved extractor."""
        instance = cls()
        instance._vectorizer = LogTextVectorizer.load(path)
        instance._is_fitted = True
        return instance
