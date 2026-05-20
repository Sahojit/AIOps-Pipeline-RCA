"""
scripts/seed_db.py
==================
Seeds the PostgreSQL database with LEMMA-RCA derived predictions
by sending pipeline_failures.csv rows to the /predict-rca API endpoint.

Usage:
    python scripts/seed_db.py
    python scripts/seed_db.py --api-url http://localhost:8000 --csv data/raw/pipeline_failures.csv
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_CSV = Path(__file__).parent.parent / "data" / "raw" / "pipeline_failures.csv"
BATCH_SIZE = 50


def seed(api_url: str, csv_path: Path) -> None:
    # Check API health
    try:
        resp = requests.get(f"{api_url}/health", timeout=10)
        resp.raise_for_status()
        print(f"API healthy: {resp.json()['status']} (model {resp.json()['model_version']})")
    except Exception as e:
        print(f"ERROR: Cannot reach API at {api_url}: {e}")
        sys.exit(1)

    # Load LEMMA events
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Root cause distribution:\n{df['root_cause'].value_counts().to_string()}\n")

    # Convert bool columns
    df["schema_change"] = df["schema_change"].astype(str).str.lower() == "true"
    df["upstream_failed"] = df["upstream_failed"].astype(str).str.lower() == "true"

    total = 0
    errors = 0
    batches = [df.iloc[i:i+BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        payload = [
            {
                "pipeline_name": row["pipeline_name"],
                "task_name": row["task_name"],
                "runtime": int(row["runtime"]),
                "retry_count": int(row["retry_count"]),
                "rows_processed": int(row["rows_processed"]),
                "schema_change": bool(row["schema_change"]),
                "upstream_failed": bool(row["upstream_failed"]),
                "error_log": str(row["error_log"]),
                "timestamp": str(row["timestamp"]),
            }
            for _, row in batch.iterrows()
        ]

        try:
            resp = requests.post(
                f"{api_url}/predict-rca/batch",
                json=payload,
                timeout=120,
            )
            if resp.status_code == 200:
                results = resp.json()
                total += len(results)
                print(f"  Batch {batch_idx+1}/{len(batches)}: {len(results)} predictions stored")
            else:
                errors += len(batch)
                print(f"  Batch {batch_idx+1} ERROR {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            errors += len(batch)
            print(f"  Batch {batch_idx+1} EXCEPTION: {e}")

        time.sleep(0.1)  # brief pause between batches

    print(f"\nDone: {total} predictions seeded, {errors} errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    args = parser.parse_args()
    seed(args.api_url, Path(args.csv))
