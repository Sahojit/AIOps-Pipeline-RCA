"""
src/ingestion/validate_data.py
==============================
Validates a pipeline failure CSV against the Pydantic schema.

This is the kind of script you'd run in a CI/CD pipeline to ensure
data quality BEFORE it enters the ML pipeline.

Usage:
    python -m src.ingestion.validate_data
"""

import sys
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.config.settings import settings
from src.ingestion.schemas import PipelineFailureEvent
from src.utils.logger import configure_root_logger, get_logger

logger = get_logger(__name__)


def validate_dataset(file_path: Path) -> tuple[int, int, list[str]]:
    """
    Validate every row of a CSV against PipelineFailureEvent schema.

    Returns:
        Tuple of (valid_count, invalid_count, list of error messages)
    """
    logger.info("Validating dataset: %s", file_path)
    df = pd.read_csv(file_path)

    valid_count = 0
    invalid_count = 0
    errors: list[str] = []

    for idx, row in df.iterrows():
        try:
            PipelineFailureEvent(**row.to_dict())
            valid_count += 1
        except ValidationError as e:
            invalid_count += 1
            errors.append(f"Row {idx}: {e.error_count()} error(s) — {e.errors()[0]['msg']}")

    logger.info(
        "Validation complete: %d valid, %d invalid out of %d total",
        valid_count, invalid_count, len(df),
    )
    return valid_count, invalid_count, errors


def main() -> None:
    configure_root_logger(settings.log_level)

    file_path = settings.raw_data_dir / "pipeline_failures.csv"
    if not file_path.exists():
        logger.error("Dataset not found at %s. Run 'python -m src.ingestion.load_lemma_dataset' first.", file_path)
        sys.exit(1)

    valid, invalid, errors = validate_dataset(file_path)

    print(f"\nValidation Results: {valid} valid, {invalid} invalid")
    if errors:
        print("\nFirst 10 errors:")
        for err in errors[:10]:
            print(f"  {err}")
        sys.exit(1)
    else:
        print("All rows passed validation.")


if __name__ == "__main__":
    main()
