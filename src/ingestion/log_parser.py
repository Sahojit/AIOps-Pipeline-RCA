"""
src/ingestion/log_parser.py
===========================
Regex-based parser that extracts structured features from raw error logs.
Handles multiline stack traces and normalises log levels across frameworks.

WHY regex and not just TF-IDF?
- TF-IDF captures word-level patterns but misses STRUCTURE:
    * "503" as a word tells TF-IDF nothing — but as an HTTP status code, it's a
      strong signal for API Failure.
    * "8.2 GiB" as tokens are noise — but as a memory value, it signals OOM.
- Regex features capture the SEMANTICS that TF-IDF misses.
- Together they give the model both structured signals AND bag-of-words context.

Each extractor returns a dict of features. All features are numeric (int/float/bool)
so they can go directly into a feature matrix without encoding.
"""

import re
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Compiled regex patterns — compiled once at module load, not per-call.
# ---------------------------------------------------------------------------

# Error type: captures the first CamelCase/Snake word before ":"
# e.g., "KeyError: ..." → "KeyError", "SchemaValidationError: ..." → "SchemaValidationError"
RE_ERROR_TYPE = re.compile(r"^(\w+Error|\w+Exception|HTTPError|Killed\w*)\b", re.IGNORECASE)

# HTTP status codes: 3-digit codes (4xx, 5xx are failures)
RE_HTTP_STATUS = re.compile(r"\b([2-5]\d{2})\b")

# File paths: s3://, /tmp/, /var/, etc.
RE_FILE_PATH = re.compile(r"(s3://\S+|/[\w/\-.]+\.\w+)")

# Memory values: "8.2 GiB", "16GB", "1024 MB"
RE_MEMORY = re.compile(r"(\d+(?:\.\d+)?)\s*(GiB|GB|MB|KB|bytes)", re.IGNORECASE)

# Numeric counts: "12 columns", "1200 rows", "340 duplicate", "45% null"
RE_COUNT = re.compile(r"(\d+(?:\.\d+)?)\s*(%|columns?|rows?|duplicates?|keys?|seconds?|s\b)")

# Timeout values: "30s", "3600s", "7200s wall clock"
RE_TIMEOUT = re.compile(r"(\d+)\s*s(?:econds?|ec)?\b")

# Column/field names in quotes: 'age', "email_verified", [phone_hash]
RE_COLUMN_NAME = re.compile(r"""['"\[](\w+)['"\]]""")


# ---------------------------------------------------------------------------
# Domain keyword sets — presence of these keywords is a binary feature.
# These encode domain knowledge about what words appear in each failure type.
# ---------------------------------------------------------------------------
SCHEMA_KEYWORDS = frozenset([
    "schema", "column", "dtype", "type", "cast", "mismatch",
    "migration", "field", "expected", "received",
])

RESOURCE_KEYWORDS = frozenset([
    "memory", "oom", "killed", "disk", "cpu", "throttle",
    "limit", "exceeded", "gib", "gb", "allocate", "capacity",
])

API_KEYWORDS = frozenset([
    "http", "api", "connection", "timeout", "ssl",
    "retries", "service", "endpoint", "url", "request",
    "503", "429", "500", "502",
])

DATA_KEYWORDS = frozenset([
    "null", "duplicate", "negative", "range", "format",
    "quality", "consistency", "freshness", "stale", "invalid",
])

DEPENDENCY_KEYWORDS = frozenset([
    "upstream", "dependency", "depends", "sensor", "parent",
    "cascade", "waiting", "dag", "failed",
])

MISSING_DATA_KEYWORDS = frozenset([
    "not found", "not exist", "missing", "empty", "unavailable",
    "0 rows", "0 bytes", "null", "not ready", "no partition",
])


def parse_error_log(error_log: str) -> dict[str, Any]:
    """
    Extract structured features from a single error log string.

    Returns a flat dict of numeric features suitable for ML consumption:
    - has_error_type: bool (1/0)
    - error_type_hash: int — simple hash to distinguish different error types
    - has_http_error: bool
    - http_status_code: int (0 if not found)
    - is_client_error: bool (4xx)
    - is_server_error: bool (5xx)
    - has_file_path: bool
    - has_s3_path: bool
    - has_memory_ref: bool
    - memory_value_mb: float (normalised to MB, 0 if not found)
    - has_timeout: bool
    - timeout_seconds: int
    - column_count: int (number of column names mentioned)
    - numeric_count_in_log: int (count of numeric values found)
    - log_length: int (character count — longer logs often = more complex errors)
    - word_count: int
    - keyword_score_schema: int (count of schema-related keywords found)
    - keyword_score_resource: int
    - keyword_score_api: int
    - keyword_score_data_quality: int
    - keyword_score_dependency: int
    - keyword_score_missing_data: int
    """
    log_lower = error_log.lower()
    log_words = set(log_lower.split())

    features: dict[str, Any] = {}

    # --- Error Type ---
    error_match = RE_ERROR_TYPE.search(error_log)
    features["has_error_type"] = int(bool(error_match))
    # Hash the error type string to a numeric feature. Different error types
    # get different values, letting the tree model split on them.
    features["error_type_hash"] = hash(error_match.group(1)) % 1000 if error_match else 0

    # --- HTTP Status ---
    http_matches = RE_HTTP_STATUS.findall(error_log)
    http_code = int(http_matches[0]) if http_matches else 0
    features["has_http_error"] = int(http_code >= 400)
    features["http_status_code"] = http_code
    features["is_client_error"] = int(400 <= http_code < 500)
    features["is_server_error"] = int(http_code >= 500)

    # --- File Paths ---
    file_matches = RE_FILE_PATH.findall(error_log)
    features["has_file_path"] = int(len(file_matches) > 0)
    features["has_s3_path"] = int(any("s3://" in m for m in file_matches))

    # --- Memory ---
    mem_match = RE_MEMORY.search(error_log)
    features["has_memory_ref"] = int(bool(mem_match))
    if mem_match:
        value = float(mem_match.group(1))
        unit = mem_match.group(2).upper()
        # Normalise to MB
        multiplier = {"KB": 0.001, "MB": 1, "GB": 1000, "GIB": 1073.74, "BYTES": 1e-6}
        features["memory_value_mb"] = value * multiplier.get(unit, 1)
    else:
        features["memory_value_mb"] = 0.0

    # --- Timeout ---
    timeout_match = RE_TIMEOUT.search(error_log)
    features["has_timeout"] = int(bool(timeout_match))
    features["timeout_seconds"] = int(timeout_match.group(1)) if timeout_match else 0

    # --- Column Names ---
    col_matches = RE_COLUMN_NAME.findall(error_log)
    features["column_count"] = len(col_matches)

    # --- Numeric Values ---
    count_matches = RE_COUNT.findall(error_log)
    features["numeric_count_in_log"] = len(count_matches)

    # --- Text Statistics ---
    features["log_length"] = len(error_log)
    features["word_count"] = len(error_log.split())

    # --- Domain Keyword Scores ---
    # Count how many keywords from each domain appear in the log.
    # This gives the model a "soft vote" toward each root cause.
    features["keyword_score_schema"] = len(log_words & SCHEMA_KEYWORDS)
    features["keyword_score_resource"] = len(log_words & RESOURCE_KEYWORDS)
    features["keyword_score_api"] = len(log_words & API_KEYWORDS)
    features["keyword_score_data_quality"] = len(log_words & DATA_KEYWORDS)
    features["keyword_score_dependency"] = len(log_words & DEPENDENCY_KEYWORDS)
    features["keyword_score_missing_data"] = _count_phrase_matches(log_lower, MISSING_DATA_KEYWORDS)

    return features


def _count_phrase_matches(text: str, keywords: frozenset[str]) -> int:
    """
    Count keyword matches, supporting multi-word phrases like 'not found'.
    Regular set intersection only works for single words.
    """
    count = 0
    for kw in keywords:
        if kw in text:
            count += 1
    return count


def parse_error_logs_batch(error_logs: list[str]) -> list[dict[str, Any]]:
    """
    Parse a batch of error logs. Used during feature engineering.

    Args:
        error_logs: List of raw error log strings.

    Returns:
        List of feature dicts, one per log entry.
    """
    logger.info("Parsing %d error logs", len(error_logs))
    results = [parse_error_log(log) for log in error_logs]
    logger.info("Log parsing complete. %d features per log.", len(results[0]) if results else 0)
    return results
