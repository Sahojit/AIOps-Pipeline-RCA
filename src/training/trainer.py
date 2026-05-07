"""
src/training/trainer.py
=======================
XGBoost classifier trainer for pipeline root cause analysis.

WHY XGBoost over Random Forest, Neural Nets, etc.?
1. Tabular data → gradient-boosted trees are consistently top performers
   (see Kaggle benchmarks, Google's TabNet paper comparisons)
2. Handles mixed feature types (binary, continuous, TF-IDF) natively
3. Built-in handling of class imbalance via sample_weight
4. Fast training on ~2000 rows (seconds, not minutes)
5. SHAP has native XGBoost TreeExplainer — fast exact explanations
6. Small model artifact (~100KB) → easy to serve in production

The trainer handles:
- Label encoding (string → int for XGBoost, with reverse mapping)
- Stratified train/test split (preserves class distribution)
- Class weight computation (handles imbalanced classes)
- Hyperparameter configuration
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Default hyperparameters — tuned for this problem size.
# In production you'd use Optuna or Hyperopt for automated tuning.
DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "n_estimators": 200,        # Number of boosting rounds
    "max_depth": 6,             # Tree depth — 6 is a good default for ~100 features
    "learning_rate": 0.1,       # Step size — lower = more rounds needed but more robust
    "subsample": 0.8,           # Row sampling per tree — reduces overfitting
    "colsample_bytree": 0.8,    # Feature sampling per tree — reduces overfitting
    "min_child_weight": 3,      # Minimum samples per leaf — regularisation
    "gamma": 0.1,               # Minimum loss reduction for split — regularisation
    "reg_alpha": 0.1,           # L1 regularisation
    "reg_lambda": 1.0,          # L2 regularisation
    "objective": "multi:softprob",  # Multi-class with probability output
    "eval_metric": "mlogloss",      # Log loss for multi-class
    "random_state": 42,
    "n_jobs": -1,               # Use all CPU cores
    "verbosity": 0,             # Suppress XGBoost internal logging
}


class RCAModelTrainer:
    """
    Trains and evaluates an XGBoost classifier for root cause analysis.

    Usage:
        trainer = RCAModelTrainer()
        result = trainer.train(X, y)
        # result contains: model, label_encoder, metrics, splits
    """

    def __init__(
        self,
        hyperparams: dict[str, Any] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            hyperparams: XGBoost hyperparameters. Defaults to DEFAULT_HYPERPARAMS.
            test_size: Fraction of data held out for evaluation.
            random_state: Seed for reproducible splits and training.
        """
        self.hyperparams = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        self.test_size = test_size
        self.random_state = random_state
        self._label_encoder = LabelEncoder()

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, Any]:
        """
        Full training pipeline: encode labels → split → train → evaluate.

        Args:
            X: Feature matrix from FeatureEngineer.
            y: Label series (root cause string values).

        Returns:
            Dict containing:
            - model: Trained XGBClassifier
            - label_encoder: Fitted LabelEncoder (for decode at inference)
            - X_train, X_test, y_train, y_test: Split data
            - y_train_encoded, y_test_encoded: Integer-encoded labels
            - hyperparams: The hyperparameters used
            - class_names: Ordered list of class names
        """
        logger.info(
            "Starting model training: %d samples, %d features, %d classes",
            len(X), len(X.columns), y.nunique(),
        )

        # --- 1. Encode labels: string → integer ---
        y_encoded = self._label_encoder.fit_transform(y)
        class_names = list(self._label_encoder.classes_)
        n_classes = len(class_names)
        logger.info("Classes: %s", class_names)

        # Set number of classes in hyperparams
        self.hyperparams["num_class"] = n_classes

        # --- 2. Stratified train/test split ---
        # Stratified ensures each class has proportional representation in both sets.
        # Without this, rare classes (Resource Exhaustion at 10%) might be
        # underrepresented in the test set, giving misleading metrics.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_encoded,
        )
        logger.info(
            "Train/test split: %d train, %d test (%.0f%% holdout)",
            len(X_train), len(X_test), self.test_size * 100,
        )

        # --- 3. Compute sample weights for class imbalance ---
        # This tells XGBoost to pay MORE attention to underrepresented classes.
        # Without this, the model would be biased toward Schema Change (22%)
        # and weak on Resource Exhaustion (10%).
        sample_weights = compute_sample_weight("balanced", y_train)
        logger.info("Sample weights computed for class imbalance handling")

        # --- 4. Train the model ---
        model = XGBClassifier(**self.hyperparams)
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        logger.info("Model training complete: %d boosting rounds", model.n_estimators)

        return {
            "model": model,
            "label_encoder": self._label_encoder,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "hyperparams": self.hyperparams,
            "class_names": class_names,
            "feature_names": list(X.columns),
        }

    @property
    def label_encoder(self) -> LabelEncoder:
        return self._label_encoder

    @property
    def class_names(self) -> list[str]:
        """Ordered class names matching the integer encoding."""
        return list(self._label_encoder.classes_)

    def decode_predictions(self, y_encoded: np.ndarray) -> list[str]:
        """
        Convert integer-encoded predictions back to root cause strings.

        Args:
            y_encoded: Integer array from model.predict().

        Returns:
            List of human-readable root cause labels.
        """
        return list(self._label_encoder.inverse_transform(y_encoded))
