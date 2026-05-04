"""
tests/unit/test_phase4_features.py
==================================
Tests for feature engineering pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.feature_definitions import (
    build_execution_features,
    build_schema_features,
    build_temporal_features,
    low_variance_features,
)
from src.features.engineer import FeatureEngineer

_LEMMA_CSV = Path(__file__).resolve().parents[2] / "data" / "raw" / "pipeline_failures.csv"


def _lemma_sample(n: int, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(_LEMMA_CSV)
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small LEMMA dataset sample for testing."""
    return _lemma_sample(100)


@pytest.fixture
def full_df() -> pd.DataFrame:
    """Larger LEMMA dataset sample for distribution checks."""
    return _lemma_sample(500)


# =========================================================================
# Execution Features Tests
# =========================================================================

class TestExecutionFeatures:
    def test_output_shape(self, sample_df: pd.DataFrame) -> None:
        result = build_execution_features(sample_df)
        assert len(result) == len(sample_df)
        assert len(result.columns) == 12  # 3 direct + 9 derived

    def test_log_transform_positive(self, sample_df: pd.DataFrame) -> None:
        result = build_execution_features(sample_df)
        # log1p(x) >= 0 for x >= 0
        assert (result["feat_log_rows_processed"] >= 0).all()
        assert (result["feat_log_runtime"] >= 0).all()

    def test_zero_rows_flag(self) -> None:
        df = pd.DataFrame({
            "runtime": [100, 200, 0],
            "retry_count": [0, 1, 2],
            "rows_processed": [0, 5000, 0],
        })
        result = build_execution_features(df)
        assert result["feat_zero_rows"].tolist() == [1, 0, 1]

    def test_short_runtime_threshold(self) -> None:
        df = pd.DataFrame({
            "runtime": [5, 29, 30, 100],
            "retry_count": [0, 0, 0, 0],
            "rows_processed": [0, 0, 0, 0],
        })
        result = build_execution_features(df)
        assert result["feat_short_runtime"].tolist() == [1, 1, 0, 0]

    def test_long_runtime_threshold(self) -> None:
        df = pd.DataFrame({
            "runtime": [100, 1800, 1801, 5000],
            "retry_count": [0, 0, 0, 0],
            "rows_processed": [0, 0, 0, 0],
        })
        result = build_execution_features(df)
        assert result["feat_long_runtime"].tolist() == [0, 0, 1, 1]

    def test_rows_per_second_no_div_by_zero(self) -> None:
        df = pd.DataFrame({
            "runtime": [0, 0, 100],
            "retry_count": [0, 0, 0],
            "rows_processed": [0, 100, 1000],
        })
        result = build_execution_features(df)
        # Should not raise and should produce finite values
        assert np.isfinite(result["feat_rows_per_second"]).all()


# =========================================================================
# Schema Features Tests
# =========================================================================

class TestSchemaFeatures:
    def test_output_shape(self, sample_df: pd.DataFrame) -> None:
        result = build_schema_features(sample_df)
        assert len(result) == len(sample_df)
        assert len(result.columns) == 5  # 2 direct + 3 interaction

    def test_interaction_features(self) -> None:
        df = pd.DataFrame({
            "schema_change": [True, True, False, False],
            "upstream_failed": [True, False, True, False],
        })
        result = build_schema_features(df)

        # Both true
        assert result["feat_schema_and_upstream"].tolist() == [1, 0, 0, 0]
        # Schema only
        assert result["feat_schema_no_upstream"].tolist() == [0, 1, 0, 0]
        # Upstream only
        assert result["feat_upstream_no_schema"].tolist() == [0, 0, 1, 0]

    def test_mutually_exclusive_interactions(self, sample_df: pd.DataFrame) -> None:
        """At most one interaction feature should be 1 per row."""
        result = build_schema_features(sample_df)
        interaction_cols = [
            "feat_schema_and_upstream",
            "feat_schema_no_upstream",
            "feat_upstream_no_schema",
        ]
        # Each row: sum of interactions <= 1
        sums = result[interaction_cols].sum(axis=1)
        assert (sums <= 1).all()


# =========================================================================
# Temporal Features Tests
# =========================================================================

class TestTemporalFeatures:
    def test_output_shape(self, sample_df: pd.DataFrame) -> None:
        result = build_temporal_features(sample_df)
        assert len(result) == len(sample_df)
        assert len(result.columns) == 12  # 10 original + is_night + month

    def test_hour_range(self, sample_df: pd.DataFrame) -> None:
        result = build_temporal_features(sample_df)
        assert result["feat_hour"].between(0, 23).all()

    def test_day_of_week_range(self, sample_df: pd.DataFrame) -> None:
        result = build_temporal_features(sample_df)
        assert result["feat_day_of_week"].between(0, 6).all()

    def test_cyclical_encoding_bounds(self, sample_df: pd.DataFrame) -> None:
        """Sin/cos values must be in [-1, 1]."""
        result = build_temporal_features(sample_df)
        for col in ["feat_hour_sin", "feat_hour_cos", "feat_dow_sin", "feat_dow_cos"]:
            assert result[col].between(-1, 1).all(), f"{col} out of bounds"

    def test_weekend_flag(self) -> None:
        df = pd.DataFrame({
            "timestamp": [
                "2026-01-05T12:00:00+00:00",  # Monday
                "2026-01-10T12:00:00+00:00",  # Saturday
                "2026-01-11T12:00:00+00:00",  # Sunday
            ]
        })
        result = build_temporal_features(df)
        assert result["feat_is_weekend"].tolist() == [0, 1, 1]

    def test_business_hours_flag(self) -> None:
        df = pd.DataFrame({
            "timestamp": [
                "2026-01-05T08:00:00+00:00",  # 8am — before business
                "2026-01-05T09:00:00+00:00",  # 9am — business starts
                "2026-01-05T17:00:00+00:00",  # 5pm — still business
                "2026-01-05T18:00:00+00:00",  # 6pm — after business
            ]
        })
        result = build_temporal_features(df)
        assert result["feat_is_business_hours"].tolist() == [0, 1, 1, 0]

    def test_is_night_flag(self) -> None:
        df = pd.DataFrame({
            "timestamp": [
                "2026-01-05T02:00:00+00:00",  # 2am — night
                "2026-01-05T23:00:00+00:00",  # 11pm — night
                "2026-01-05T12:00:00+00:00",  # noon — not night
            ]
        })
        result = build_temporal_features(df)
        assert result["feat_is_night"].tolist() == [1, 1, 0]

    def test_hour_dtype_is_integer(self, sample_df: pd.DataFrame) -> None:
        result = build_temporal_features(sample_df)
        assert result["feat_hour"].dtype == int
        assert result["feat_day_of_week"].dtype == int


# =========================================================================
# FeatureEngineer (end-to-end) Tests
# =========================================================================

class TestFeatureEngineer:
    def test_fit_transform_returns_correct_types(self, sample_df: pd.DataFrame) -> None:
        engineer = FeatureEngineer(max_tfidf_features=10)
        X, y, feature_names = engineer.fit_transform(sample_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(feature_names, list)

    def test_X_and_y_same_length(self, sample_df: pd.DataFrame) -> None:
        engineer = FeatureEngineer(max_tfidf_features=10)
        X, y, _ = engineer.fit_transform(sample_df)
        assert len(X) == len(y) == len(sample_df)

    def test_feature_names_match_columns(self, sample_df: pd.DataFrame) -> None:
        engineer = FeatureEngineer(max_tfidf_features=10)
        X, _, feature_names = engineer.fit_transform(sample_df)
        assert list(X.columns) == feature_names

    def test_no_nans_in_features(self, full_df: pd.DataFrame) -> None:
        """Feature matrix must not contain NaN — XGBoost handles them but we shouldn't produce them."""
        engineer = FeatureEngineer(max_tfidf_features=20)
        X, _, _ = engineer.fit_transform(full_df)
        assert not X.isna().any().any(), f"NaN found in columns: {X.columns[X.isna().any()].tolist()}"

    def test_no_inf_in_features(self, full_df: pd.DataFrame) -> None:
        engineer = FeatureEngineer(max_tfidf_features=20)
        X, _, _ = engineer.fit_transform(full_df)
        numeric_X = X.select_dtypes(include=[np.number])
        assert np.isfinite(numeric_X.values).all(), "Infinite values found in features"

    def test_transform_without_fit_raises(self, sample_df: pd.DataFrame) -> None:
        engineer = FeatureEngineer()
        with pytest.raises(RuntimeError, match="not fitted"):
            engineer.transform(sample_df)

    def test_missing_column_raises(self) -> None:
        engineer = FeatureEngineer()
        bad_df = pd.DataFrame({"pipeline_name": ["test"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            engineer.fit(bad_df)

    def test_transform_single_event(self, sample_df: pd.DataFrame) -> None:
        engineer = FeatureEngineer(max_tfidf_features=10)
        engineer.fit(sample_df)

        single_event = {
            "pipeline_name": "user_etl_pipeline",
            "task_name": "extract_users",
            "runtime": 120,
            "retry_count": 2,
            "rows_processed": 50000,
            "schema_change": False,
            "upstream_failed": False,
            "error_log": "TimeoutError: API call timed out after 30s",
            "timestamp": "2026-03-03T12:00:00+00:00",
        }
        result = engineer.transform_single(single_event)
        assert len(result) == 1
        assert list(result.columns) == engineer.feature_names

    def test_save_and_load(self, sample_df: pd.DataFrame, tmp_path) -> None:
        engineer = FeatureEngineer(max_tfidf_features=10)
        X_original, _, _ = engineer.fit_transform(sample_df)
        save_path = tmp_path / "test_engineer.pkl"
        engineer.save(save_path)

        loaded = FeatureEngineer.load(save_path)
        X_loaded = loaded.transform(sample_df)
        pd.testing.assert_frame_equal(X_original, X_loaded)

    def test_label_column_missing_raises(self) -> None:
        df = pd.DataFrame({
            "pipeline_name": ["test"],
            "task_name": ["task"],
            "runtime": [100],
            "retry_count": [0],
            "rows_processed": [0],
            "schema_change": [False],
            "upstream_failed": [False],
            "error_log": ["error"],
            "timestamp": ["2026-01-01T00:00:00"],
            # No root_cause column
        })
        engineer = FeatureEngineer()
        with pytest.raises(ValueError, match="Label column"):
            engineer.fit_transform(df)


# =========================================================================
# Edge Case Tests
# =========================================================================

def _minimal_df(**overrides) -> pd.DataFrame:
    """Build a minimal single-row DataFrame for edge case tests."""
    row = {
        "pipeline_name": "test_pipeline",
        "task_name": "test_task",
        "runtime": 60,
        "retry_count": 1,
        "rows_processed": 1000,
        "schema_change": False,
        "upstream_failed": False,
        "error_log": "ConnectionError: upstream service timed out",
        "timestamp": "2026-03-01T10:00:00+00:00",
        "root_cause": "API Failure",
    }
    row.update(overrides)
    return pd.DataFrame([row])


class TestEdgeCases:
    def test_empty_error_log_does_not_crash(self) -> None:
        df = _minimal_df(error_log="")
        engineer = FeatureEngineer(max_tfidf_features=10)
        # Fit on a real sample so TF-IDF has a vocabulary
        train_df = _lemma_sample(50)
        engineer.fit(train_df)
        result = engineer.transform(df)
        assert len(result) == 1
        assert not result.isnull().any().any()

    def test_zero_runtime_event(self) -> None:
        df = _minimal_df(runtime=0, rows_processed=0)
        engineer = FeatureEngineer(max_tfidf_features=10)
        train_df = _lemma_sample(50)
        engineer.fit(train_df)
        result = engineer.transform(df)
        assert np.isfinite(result.select_dtypes(include=[np.number]).values).all()

    def test_max_retry_count(self) -> None:
        df = _minimal_df(retry_count=10)
        engineer = FeatureEngineer(max_tfidf_features=10)
        train_df = _lemma_sample(50)
        engineer.fit(train_df)
        result = engineer.transform(df)
        assert result["feat_high_retries"].iloc[0] == 1

    def test_single_row_transform_matches_batch(self) -> None:
        train_df = _lemma_sample(50)
        engineer = FeatureEngineer(max_tfidf_features=10)
        engineer.fit(train_df)

        # Transform a batch of 1 vs transform_single — must produce same result
        single_row = train_df.iloc[[0]]
        event_dict = single_row.drop(columns=["root_cause"]).iloc[0].to_dict()

        batch_result = engineer.transform(single_row)
        single_result = engineer.transform_single(event_dict)
        pd.testing.assert_frame_equal(
            batch_result.reset_index(drop=True),
            single_result.reset_index(drop=True),
        )


# =========================================================================
# Full Dataset Integration Tests
# =========================================================================

class TestFullDatasetIntegration:
    """Run feature engineering over all 800 LEMMA rows."""

    def test_full_dataset_row_count(self) -> None:
        df = pd.read_csv(_LEMMA_CSV)
        engineer = FeatureEngineer(max_tfidf_features=50)
        X, y, _ = engineer.fit_transform(df)
        assert X.shape[0] == len(df)
        assert len(y) == len(df)

    def test_full_dataset_min_feature_count(self) -> None:
        df = pd.read_csv(_LEMMA_CSV)
        engineer = FeatureEngineer(max_tfidf_features=50)
        X, _, feature_names = engineer.fit_transform(df)
        assert X.shape[1] >= 90, f"Expected >=90 features, got {X.shape[1]}"
        assert len(feature_names) == X.shape[1]

    def test_all_six_root_cause_classes_present(self) -> None:
        df = pd.read_csv(_LEMMA_CSV)
        engineer = FeatureEngineer(max_tfidf_features=50)
        _, y, _ = engineer.fit_transform(df)
        assert y.nunique() == 6, f"Expected 6 classes, got: {y.unique()}"

    def test_no_constant_features_in_full_dataset(self) -> None:
        df = pd.read_csv(_LEMMA_CSV)
        engineer = FeatureEngineer(max_tfidf_features=50)
        X, _, _ = engineer.fit_transform(df)
        zero_var = low_variance_features(X, threshold=0.0)
        assert len(zero_var) == 0, f"Constant (zero-variance) features: {zero_var}"

    def test_no_nans_in_full_dataset(self) -> None:
        df = pd.read_csv(_LEMMA_CSV)
        engineer = FeatureEngineer(max_tfidf_features=50)
        X, _, _ = engineer.fit_transform(df)
        nan_cols = X.columns[X.isnull().any()].tolist()
        assert not nan_cols, f"NaN found in columns: {nan_cols}"
