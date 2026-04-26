from src.features.engineer import FeatureEngineer
from src.features.feature_definitions import (
    build_execution_features,
    build_schema_features,
    build_temporal_features,
)

# 96 features total: execution + schema + temporal + log signals + TF-IDF
EXPECTED_FEATURE_COUNT = 96

__all__ = [
    "FeatureEngineer",
    "build_execution_features",
    "build_schema_features",
    "build_temporal_features",
    "EXPECTED_FEATURE_COUNT",
]
