"""
src/features/build_features.py
==============================
CLI script to run the full feature engineering pipeline.

Reads raw CSV from data/raw/, transforms into features, saves to data/processed/.

Usage:
    python -m src.features.build_features

This is the kind of script you'd run as an Airflow task BEFORE model training.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.features.engineer import FeatureEngineer
from src.utils.logger import configure_root_logger, get_logger

logger = get_logger(__name__)


def main() -> None:
    configure_root_logger(settings.log_level)

    # --- Load raw data ---
    raw_path: Path = settings.raw_data_dir / "pipeline_failures.csv"
    if not raw_path.exists():
        logger.error(
            "Raw dataset not found at %s. Run 'python -m src.ingestion.load_lemma_dataset' first.",
            raw_path,
        )
        sys.exit(1)

    logger.info("Loading raw data from %s", raw_path)
    df = pd.read_csv(raw_path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # --- Run feature engineering ---
    engineer = FeatureEngineer()
    X, y, feature_names = engineer.fit_transform(df)

    # --- Save processed features ---
    output_dir = settings.processed_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save feature matrix
    features_path = output_dir / "features.csv"
    X.to_csv(features_path, index=False)
    logger.info("Feature matrix saved to %s", features_path)

    # Save labels
    labels_path = output_dir / "labels.csv"
    y.to_csv(labels_path, index=False, header=True)
    logger.info("Labels saved to %s", labels_path)

    # Save feature names for reference
    names_path = output_dir / "feature_names.txt"
    names_path.write_text("\n".join(feature_names))
    logger.info("Feature names saved to %s", names_path)

    # Save per-group feature counts for validation and documentation
    group_counts = {
        "execution": len([f for f in feature_names if f.startswith("feat_")]),
        "log_signals": len([f for f in feature_names if not f.startswith(("feat_", "tfidf_"))]),
        "tfidf": len([f for f in feature_names if f.startswith("tfidf_")]),
        "total": len(feature_names),
    }
    groups_path = output_dir / "feature_groups.json"
    groups_path.write_text(json.dumps(group_counts, indent=2))
    logger.info("Feature group counts saved to %s", groups_path)

    # Save TF-IDF vectorizer for inference
    engineer.save()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"Input:  {len(df)} rows from {raw_path.name}")
    print(f"Output: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"\nFeature groups:")
    exec_feats = [f for f in feature_names if f.startswith("feat_r") or f.startswith("feat_l") or f.startswith("feat_z") or f.startswith("feat_s") or f.startswith("feat_h")]
    schema_feats = [f for f in feature_names if "schema" in f or "upstream" in f]
    temporal_feats = [f for f in feature_names if "hour" in f or "day" in f or "dow" in f or "weekend" in f or "business" in f]
    log_feats = [f for f in feature_names if f.startswith(("has_", "is_", "keyword_", "tfidf_", "http_", "memory_", "timeout_", "column_", "numeric_", "log_", "word_", "error_"))]
    print(f"  Execution:  {len(exec_feats)}")
    print(f"  Schema:     {len(schema_feats)}")
    print(f"  Temporal:   {len(temporal_feats)}")
    print(f"  Log:        {len(log_feats)}")
    print(f"\nLabel distribution:")
    print(y.value_counts().to_string())
    print(f"\nSaved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
