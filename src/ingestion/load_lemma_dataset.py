"""
src/ingestion/load_lemma_dataset.py
====================================
CLI entrypoint to download and transform the LEMMA-RCA dataset.

Usage:
    python -m src.ingestion.load_lemma_dataset
    python -m src.ingestion.load_lemma_dataset --dataset cloud
    python -m src.ingestion.load_lemma_dataset --output data/raw/lemma_failures.csv
"""

import argparse
from pathlib import Path

from src.config.settings import settings
from src.ingestion.lemma_adapter import load_and_transform
from src.utils.logger import configure_root_logger, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and transform LEMMA-RCA dataset for RCA training"
    )
    parser.add_argument(
        "--dataset", type=str, default="cloud",
        choices=["cloud", "product_review"],
        help="Which LEMMA-RCA dataset to use (default: cloud)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: data/raw/pipeline_failures.csv)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    configure_root_logger(settings.log_level)
    logger.info("Starting LEMMA-RCA dataset loading")

    df = load_and_transform(
        dataset_key=args.dataset,
        seed=args.seed,
    )

    output_path = Path(args.output) if args.output else settings.raw_data_dir / "pipeline_failures.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("LEMMA-RCA dataset saved to %s (%d rows)", output_path, len(df))

    # Print summary
    print("\n" + "=" * 60)
    print("LEMMA-RCA DATASET SUMMARY")
    print("=" * 60)
    print(f"Source: LEMMA-RCA ({args.dataset})")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nRoot cause distribution:")
    print(df["root_cause"].value_counts().to_string())
    print(f"\nRuntime statistics:")
    print(df["runtime"].describe().to_string())
    print(f"\nRows processed statistics:")
    print(df["rows_processed"].describe().to_string())
    print(f"\nSchema change rate: {df['schema_change'].mean():.2%}")
    print(f"Upstream failed rate: {df['upstream_failed'].mean():.2%}")
    print(f"\nSample records:")
    print(df.head(3).to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
