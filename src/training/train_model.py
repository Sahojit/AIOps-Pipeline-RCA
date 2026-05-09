"""
src/training/train_model.py
===========================
End-to-end training script: data → features → train → evaluate → save.

Usage:
    python -m src.training.train_model
    python -m src.training.train_model --version v2

This is the script you'd trigger from an Airflow DAG or CI/CD pipeline
when you want to retrain the model on new data.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.features.engineer import FeatureEngineer
from src.training.artifact_manager import save_model_artifacts
from src.training.evaluator import check_go_no_go, evaluate_model, format_evaluation_summary
from src.training.trainer import RCAModelTrainer
from src.utils.logger import configure_root_logger, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RCA classification model")
    parser.add_argument(
        "--version", type=str, default="v1",
        help="Model version string (e.g., v1, v2)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data for test set (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--data-source", type=str, default="lemma",
        choices=["lemma", "custom"],
        help="Data source: 'lemma' (default, LEMMA-RCA dataset) or 'custom' (provide --data-path)",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Custom CSV path (used with --data-source custom)",
    )
    args = parser.parse_args()

    configure_root_logger(settings.log_level)
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING PIPELINE (data-source=%s)", args.data_source)
    logger.info("=" * 60)

    # --- 1. Load raw data ---
    if args.data_source == "custom":
        if not args.data_path:
            logger.error("--data-path is required when using --data-source custom")
            sys.exit(1)
        raw_path = Path(args.data_path)
    else:
        raw_path = settings.raw_data_dir / "pipeline_failures.csv"

    if not raw_path.exists():
        hint = {
            "lemma": "python -m src.ingestion.load_lemma_dataset",
            "custom": f"file not found: {raw_path}",
        }[args.data_source]
        logger.error("Raw dataset not found at %s. Run: %s", raw_path, hint)
        sys.exit(1)

    df = pd.read_csv(raw_path)
    logger.info("Loaded %d rows from %s", len(df), raw_path)

    # --- 2. Feature engineering ---
    logger.info("Running feature engineering")
    engineer = FeatureEngineer()
    X, y, feature_names = engineer.fit_transform(df)
    logger.info("Feature matrix: %s, Label vector: %s", X.shape, y.shape)

    # Save the TF-IDF vectorizer for inference
    engineer.save()
    logger.info("Feature engineer artifacts saved")

    # --- 3. Train model ---
    logger.info("Training XGBoost classifier")
    trainer = RCAModelTrainer(
        test_size=args.test_size,
        random_state=args.seed,
    )
    train_result = trainer.train(X, y)

    model = train_result["model"]
    label_encoder = train_result["label_encoder"]
    class_names = train_result["class_names"]

    # --- 4. Evaluate model ---
    logger.info("Evaluating model on test set")
    metrics = evaluate_model(
        model=model,
        X_test=train_result["X_test"],
        y_test=train_result["y_test"],
        class_names=class_names,
        feature_names=feature_names,
    )

    # --- 5. Save artifacts ---
    logger.info("Saving model artifacts (version=%s)", args.version)
    save_model_artifacts(
        model=model,
        label_encoder=label_encoder,
        metrics=metrics,
        hyperparams=train_result["hyperparams"],
        feature_names=feature_names,
        class_names=class_names,
        version=args.version,
        training_samples=len(train_result["X_train"]),
    )

    # --- 6. Print summary ---
    summary = format_evaluation_summary(metrics, class_names)
    print("\n" + summary)

    # --- 7. Go/No-Go check ---
    check_go_no_go(metrics, min_macro_f1=0.70)


if __name__ == "__main__":
    main()
