"""
tests/unit/test_phase3_log_parsing.py
=====================================
Tests for the log parsing system: regex parser, TF-IDF vectorizer,
and the unified LogFeatureExtractor.
"""

import numpy as np
import pandas as pd
import pytest

from src.ingestion.log_parser import parse_error_log, parse_error_logs_batch
from src.ingestion.text_vectorizer import LogTextVectorizer, _preprocess_log
from src.ingestion.log_feature_extractor import LogFeatureExtractor


# =========================================================================
# Regex Parser Tests
# =========================================================================

class TestParseErrorLog:
    """Tests for individual regex feature extraction."""

    def test_extracts_error_type(self) -> None:
        features = parse_error_log("KeyError: column 'age' not found")
        assert features["has_error_type"] == 1

    def test_no_error_type_for_plain_text(self) -> None:
        features = parse_error_log("pipeline failed due to upstream issues")
        assert features["has_error_type"] == 0

    def test_detects_http_503(self) -> None:
        features = parse_error_log("HTTPError: 503 Service Unavailable from https://api.stripe.com")
        assert features["has_http_error"] == 1
        assert features["http_status_code"] == 503
        assert features["is_server_error"] == 1
        assert features["is_client_error"] == 0

    def test_detects_http_429(self) -> None:
        features = parse_error_log("HTTPError: 429 Too Many Requests")
        assert features["http_status_code"] == 429
        assert features["is_client_error"] == 1

    def test_no_http_for_non_http_error(self) -> None:
        features = parse_error_log("MemoryError: unable to allocate 8.2 GiB")
        assert features["has_http_error"] == 0
        assert features["http_status_code"] == 0

    def test_detects_s3_path(self) -> None:
        features = parse_error_log("FileNotFoundError: s3://data-lake/users/part-00000.parquet does not exist")
        assert features["has_s3_path"] == 1
        assert features["has_file_path"] == 1

    def test_detects_memory_reference(self) -> None:
        features = parse_error_log("MemoryError: unable to allocate 8.2 GiB. Worker OOM killed.")
        assert features["has_memory_ref"] == 1
        assert features["memory_value_mb"] > 8000  # 8.2 GiB ≈ 8804 MB

    def test_memory_zero_when_absent(self) -> None:
        features = parse_error_log("KeyError: column 'age' not found")
        assert features["memory_value_mb"] == 0.0

    def test_detects_timeout(self) -> None:
        features = parse_error_log("TimeoutError: API call timed out after 30s")
        assert features["has_timeout"] == 1
        assert features["timeout_seconds"] == 30

    def test_counts_column_names(self) -> None:
        features = parse_error_log(
            "expected columns ['user_id', 'email', 'age'] but got ['user_id', 'email']"
        )
        assert features["column_count"] >= 2

    def test_text_statistics(self) -> None:
        log = "This is a test error log message"
        features = parse_error_log(log)
        assert features["log_length"] == len(log)
        assert features["word_count"] == 7

    def test_keyword_score_schema(self) -> None:
        features = parse_error_log("SchemaError: column dtype mismatch in migration")
        assert features["keyword_score_schema"] >= 3

    def test_keyword_score_resource(self) -> None:
        features = parse_error_log("MemoryError: OOM killed, exceeded 16GB memory limit")
        assert features["keyword_score_resource"] >= 3

    def test_keyword_score_api(self) -> None:
        features = parse_error_log("HTTPError: 503 from API endpoint, connection timeout after retries")
        assert features["keyword_score_api"] >= 3

    def test_keyword_score_dependency(self) -> None:
        features = parse_error_log("DependencyError: upstream pipeline DAG failed, sensor waiting")
        assert features["keyword_score_dependency"] >= 3

    def test_returns_all_expected_keys(self) -> None:
        features = parse_error_log("some error")
        expected_keys = {
            "has_error_type", "error_type_hash",
            "has_http_error", "http_status_code", "is_client_error", "is_server_error",
            "has_file_path", "has_s3_path",
            "has_memory_ref", "memory_value_mb",
            "has_timeout", "timeout_seconds",
            "column_count", "numeric_count_in_log",
            "log_length", "word_count",
            "keyword_score_schema", "keyword_score_resource",
            "keyword_score_api", "keyword_score_data_quality",
            "keyword_score_dependency", "keyword_score_missing_data",
        }
        assert set(features.keys()) == expected_keys


class TestBatchParsing:
    def test_batch_returns_correct_count(self) -> None:
        logs = ["error 1", "error 2", "error 3"]
        results = parse_error_logs_batch(logs)
        assert len(results) == 3

    def test_batch_dicts_have_same_keys(self) -> None:
        logs = ["KeyError: col missing", "MemoryError: OOM at 8GB"]
        results = parse_error_logs_batch(logs)
        assert set(results[0].keys()) == set(results[1].keys())


# =========================================================================
# TF-IDF Vectorizer Tests
# =========================================================================

class TestPreprocessLog:
    def test_lowercases(self) -> None:
        assert "error" in _preprocess_log("KeyError: Something")

    def test_replaces_s3_paths(self) -> None:
        result = _preprocess_log("s3://bucket/path/file.parquet")
        assert "s3://" not in result
        assert "S3_PATH_TOKEN" in result.upper()

    def test_replaces_urls(self) -> None:
        result = _preprocess_log("Failed calling https://api.stripe.com/v1/charges")
        assert "https://" not in result


class TestLogTextVectorizer:
    @pytest.fixture
    def sample_logs(self) -> list[str]:
        return [
            "KeyError: column 'age' not found in schema",
            "MemoryError: unable to allocate 8.2 GiB",
            "HTTPError: 503 Service Unavailable from API",
            "DependencyError: upstream pipeline failed",
            "FileNotFoundError: s3://bucket/data.parquet missing",
            "DataQualityError: 45% null values in column email",
            "SchemaError: expected 12 columns but got 11",
            "TimeoutError: API call timed out after 30 seconds",
            "MemoryError: OOM killed worker process",
            "DependencyError: sensor timed out waiting for DAG",
        ]

    def test_fit_transform_returns_numpy_array(self, sample_logs: list[str]) -> None:
        vec = LogTextVectorizer(max_features=20)
        result = vec.fit_transform(sample_logs)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(sample_logs)

    def test_transform_without_fit_raises(self) -> None:
        vec = LogTextVectorizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            vec.transform(["some error"])

    def test_feature_names_prefixed(self, sample_logs: list[str]) -> None:
        vec = LogTextVectorizer(max_features=10)
        vec.fit(sample_logs)
        names = vec.get_feature_names()
        assert all(n.startswith("tfidf_") for n in names)

    def test_max_features_respected(self, sample_logs: list[str]) -> None:
        vec = LogTextVectorizer(max_features=10)
        result = vec.fit_transform(sample_logs)
        assert result.shape[1] <= 10

    def test_save_and_load(self, sample_logs: list[str], tmp_path) -> None:
        vec = LogTextVectorizer(max_features=10)
        vec.fit(sample_logs)
        save_path = tmp_path / "test_tfidf.pkl"
        vec.save(save_path)

        loaded = LogTextVectorizer.load(save_path)
        original = vec.transform(sample_logs)
        reloaded = loaded.transform(sample_logs)
        np.testing.assert_array_equal(original, reloaded)


# =========================================================================
# Unified LogFeatureExtractor Tests
# =========================================================================

class TestLogFeatureExtractor:
    @pytest.fixture
    def sample_logs(self) -> list[str]:
        return [
            "KeyError: column 'age' not found in schema",
            "MemoryError: unable to allocate 8.2 GiB",
            "HTTPError: 503 Service Unavailable from API",
            "DependencyError: upstream pipeline failed",
            "FileNotFoundError: s3://bucket/data.parquet missing",
            "DataQualityError: 45% null values in column email",
            "SchemaError: expected 12 columns but got 11",
            "TimeoutError: API call timed out after 30 seconds",
            "MemoryError: OOM killed worker process",
            "DependencyError: sensor timed out waiting for DAG",
        ]

    def test_fit_transform_returns_dataframe(self, sample_logs: list[str]) -> None:
        ext = LogFeatureExtractor(max_tfidf_features=10)
        result = ext.fit_transform(sample_logs)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_logs)

    def test_contains_both_regex_and_tfidf_columns(self, sample_logs: list[str]) -> None:
        ext = LogFeatureExtractor(max_tfidf_features=10)
        result = ext.fit_transform(sample_logs)
        # Regex columns
        assert "has_error_type" in result.columns
        assert "keyword_score_schema" in result.columns
        # TF-IDF columns
        tfidf_cols = [c for c in result.columns if c.startswith("tfidf_")]
        assert len(tfidf_cols) > 0

    def test_transform_without_fit_raises(self) -> None:
        ext = LogFeatureExtractor()
        with pytest.raises(RuntimeError):
            ext.transform(["some error"])

    def test_get_feature_names(self, sample_logs: list[str]) -> None:
        ext = LogFeatureExtractor(max_tfidf_features=10)
        ext.fit(sample_logs)
        names = ext.get_feature_names()
        assert len(names) > 20  # At least regex features + some tfidf

    def test_save_and_load(self, sample_logs: list[str], tmp_path) -> None:
        ext = LogFeatureExtractor(max_tfidf_features=10)
        ext.fit(sample_logs)
        save_path = tmp_path / "test_extractor.pkl"
        ext.save(save_path)

        loaded = LogFeatureExtractor.load(save_path)
        original = ext.transform(sample_logs)
        reloaded = loaded.transform(sample_logs)
        pd.testing.assert_frame_equal(original, reloaded)
