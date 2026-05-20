"""
airflow/dags/sample_etl_dag.py
==============================
Sample ETL DAG that demonstrates RCA integration.

This DAG simulates a typical data pipeline:
    extract_data → transform_data → load_data

When ANY task fails, the `rca_failure_callback` fires automatically,
sending the failure context to POST /predict-rca for root cause analysis.

HOW TO USE IN YOUR OWN DAGS:
1. Import the callback:
       from plugins.rca_callback import rca_failure_callback
2. Add to default_args:
       default_args = {"on_failure_callback": rca_failure_callback}
3. That's it — every task failure in the DAG gets auto-diagnosed.

DEPLOYMENT:
- Copy this file to your Airflow DAGs folder
- Copy airflow/plugins/rca_callback.py to your Airflow plugins folder
- Set RCA_API_URL environment variable (or it defaults to localhost:8000)
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Import the RCA callback
from plugins.rca_callback import rca_failure_callback


# ---------------------------------------------------------------------------
# DAG default args — apply to ALL tasks in this DAG
# ---------------------------------------------------------------------------
default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": False,       # We use RCA callback instead of email
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "on_failure_callback": rca_failure_callback,  # ← THIS IS THE KEY LINE
}


# ---------------------------------------------------------------------------
# Task functions — simulate real ETL steps
# ---------------------------------------------------------------------------

def extract_data(**kwargs) -> dict:
    """
    Simulate data extraction from a source system.

    In a real pipeline, this would:
    - Query an API or database
    - Download files from S3/GCS
    - Read from a Kafka topic
    """
    import random
    import logging

    logger = logging.getLogger("extract_data")
    logger.info("Starting data extraction")

    # Simulate occasional failures for demonstration
    # Remove this in production — real failures will come naturally
    if random.random() < 0.1:  # 10% chance of simulated failure
        raise ConnectionError(
            "HTTPError: 503 Service Unavailable from https://api.source-system.com/v1/users. "
            "The upstream API may be experiencing downtime."
        )

    rows_extracted = random.randint(10000, 100000)
    logger.info("Extracted %d rows from source system", rows_extracted)

    return {"rows_extracted": rows_extracted}


def transform_data(**kwargs) -> dict:
    """
    Simulate data transformation.

    In a real pipeline, this would:
    - Apply business logic
    - Clean and validate data
    - Join multiple datasets
    """
    import random
    import logging

    logger = logging.getLogger("transform_data")
    logger.info("Starting data transformation")

    # Simulate schema change failure
    if random.random() < 0.1:
        raise KeyError(
            "KeyError: column 'email_verified' not found in dataframe. "
            "Schema may have changed upstream. Expected 12 columns, got 11."
        )

    # Simulate data quality failure
    if random.random() < 0.05:
        raise ValueError(
            "DataQualityError: column 'age' has 45% null values. "
            "Threshold is 5%. 1200 rows have negative 'order_amount'."
        )

    rows_transformed = random.randint(9000, 95000)
    logger.info("Transformed %d rows", rows_transformed)

    return {"rows_transformed": rows_transformed}


def load_data(**kwargs) -> dict:
    """
    Simulate loading data into a target system.

    In a real pipeline, this would:
    - Write to a data warehouse (BigQuery, Redshift, Snowflake)
    - Update a database table
    - Push to a downstream service
    """
    import random
    import logging

    logger = logging.getLogger("load_data")
    logger.info("Starting data load")

    # Simulate resource exhaustion
    if random.random() < 0.05:
        raise MemoryError(
            "MemoryError: unable to allocate 8.2 GiB. "
            "Worker OOM killed by kernel. Increase executor memory."
        )

    rows_loaded = random.randint(8000, 90000)
    logger.info("Loaded %d rows into target", rows_loaded)

    return {"rows_loaded": rows_loaded}


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="sample_etl_pipeline",
    default_args=default_args,
    description="Sample ETL pipeline with RCA integration",
    schedule=timedelta(hours=1),     # Run every hour
    start_date=datetime(2026, 1, 1),
    catchup=False,                   # Don't backfill past runs
    tags=["etl", "rca-enabled"],
) as dag:

    extract = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data,
    )

    transform = PythonOperator(
        task_id="transform_data",
        python_callable=transform_data,
    )

    load = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )

    # Pipeline dependency chain
    extract >> transform >> load
