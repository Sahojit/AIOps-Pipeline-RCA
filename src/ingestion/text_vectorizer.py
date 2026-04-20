"""
src/ingestion/text_vectorizer.py
================================
TF-IDF vectorizer for error log text.

WHY TF-IDF and not embeddings?
- For structured, repetitive error messages, TF-IDF is HIGHLY effective.
- Error logs have a limited vocabulary (~200-500 unique terms). TF-IDF
  captures this perfectly with low dimensionality (50 features).
- Embeddings (BERT, etc.) are overkill for this domain and add latency +
  model size. We'd use them if error logs were free-form natural language.
- TF-IDF is also fully interpretable — SHAP can tell you which WORDS
  drove the prediction.

This module wraps scikit-learn's TfidfVectorizer with:
- Fit/transform interface matching our pipeline
- Persistence (save/load) for production deployment
- Feature name extraction for SHAP explainability
"""

import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _preprocess_log(text: str) -> str:
    """
    Clean a raw error log before TF-IDF tokenisation.

    Steps:
    1. Lowercase
    2. Replace file paths/URLs with a token (reduce vocabulary noise)
    3. Replace numbers with a token (so "503" and "429" don't bloat vocab)
    4. Remove excessive punctuation
    """
    text = text.lower()
    # Replace URLs and paths with placeholder
    text = re.sub(r"https?://\S+", " URL_TOKEN ", text)
    text = re.sub(r"s3://\S+", " S3_PATH_TOKEN ", text)
    text = re.sub(r"/[\w/\-.]+\.\w+", " FILE_PATH_TOKEN ", text)
    # Replace standalone numbers (but keep words like "v2" intact)
    text = re.sub(r"\b\d+(\.\d+)?\s*(gib|gb|mb|kb|bytes|%|s|ms)\b", " NUMERIC_TOKEN ", text)
    text = re.sub(r"\b\d{3,}\b", " NUM_TOKEN ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


class LogTextVectorizer:
    """
    Wraps TfidfVectorizer with fit/transform/save/load lifecycle.

    Usage:
        # Training time
        vectorizer = LogTextVectorizer(max_features=50)
        vectorizer.fit(train_logs)
        tfidf_matrix = vectorizer.transform(train_logs)

        # Save for production
        vectorizer.save()

        # Inference time
        vectorizer = LogTextVectorizer.load()
        tfidf_matrix = vectorizer.transform([new_log])
    """

    def __init__(self, max_features: Optional[int] = None) -> None:
        self.max_features = max_features or settings.tfidf_max_features
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            # Use both unigrams and bigrams — "not found" is more
            # informative than "not" and "found" separately.
            ngram_range=(1, 2),
            # Remove very rare terms (< 2 documents) and very common terms (> 90%)
            min_df=2,
            max_df=0.90,
            # Custom preprocessor
            preprocessor=_preprocess_log,
            # Token pattern: keep words 2+ chars, plus special tokens
            token_pattern=r"(?u)\b\w[\w_]+\b",
            # Strip accents for normalisation
            strip_accents="unicode",
        )
        self._is_fitted = False

    def fit(self, error_logs: list[str]) -> "LogTextVectorizer":
        """
        Fit the TF-IDF vocabulary on training error logs.

        Args:
            error_logs: List of raw error log strings.

        Returns:
            self (for chaining)
        """
        logger.info(
            "Fitting TF-IDF vectorizer on %d logs (max_features=%d)",
            len(error_logs), self.max_features,
        )
        self._vectorizer.fit(error_logs)
        self._is_fitted = True
        vocab_size = len(self._vectorizer.vocabulary_)
        logger.info("TF-IDF vocabulary size: %d terms", vocab_size)
        return self

    def transform(self, error_logs: list[str]) -> np.ndarray:
        """
        Transform error logs into TF-IDF feature matrix.

        Args:
            error_logs: List of raw error log strings.

        Returns:
            Dense numpy array of shape (n_logs, n_features).
            Dense (not sparse) because XGBoost handles dense better
            and 50 features is tiny.
        """
        if not self._is_fitted:
            raise RuntimeError("Vectorizer not fitted. Call fit() first or load() a saved model.")

        matrix = self._vectorizer.transform(error_logs)
        return matrix.toarray()

    def fit_transform(self, error_logs: list[str]) -> np.ndarray:
        """Fit and transform in one call (training convenience)."""
        self.fit(error_logs)
        return self.transform(error_logs)

    def get_feature_names(self) -> list[str]:
        """
        Get TF-IDF feature names — used for SHAP explanations.

        Returns names like: ["tfidf_schema", "tfidf_not_found", "tfidf_timeout"]
        Prefixed with 'tfidf_' to distinguish from regex features.
        """
        if not self._is_fitted:
            raise RuntimeError("Vectorizer not fitted.")
        return [f"tfidf_{name}" for name in self._vectorizer.get_feature_names_out()]

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Persist the fitted vectorizer to disk.

        Args:
            path: Custom save path. Defaults to models/tfidf_vectorizer.pkl.

        Returns:
            Path where the vectorizer was saved.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted vectorizer.")
        save_path = path or (settings.model_dir / "tfidf_vectorizer.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._vectorizer, save_path)
        logger.info("TF-IDF vectorizer saved to %s", save_path)
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "LogTextVectorizer":
        """
        Load a fitted vectorizer from disk.

        Args:
            path: Custom load path. Defaults to models/tfidf_vectorizer.pkl.

        Returns:
            LogTextVectorizer instance ready for transform().
        """
        load_path = path or (settings.model_dir / "tfidf_vectorizer.pkl")
        logger.info("Loading TF-IDF vectorizer from %s", load_path)
        instance = cls()
        instance._vectorizer = joblib.load(load_path)
        instance._is_fitted = True
        instance.max_features = instance._vectorizer.max_features
        return instance
