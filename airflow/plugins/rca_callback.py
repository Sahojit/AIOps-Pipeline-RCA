"""
airflow/plugins/rca_callback.py
===============================
Airflow on_failure_callback that sends failed task context to the RCA API.

HOW IT WORKS:
1. Airflow task fails
2. Airflow calls on_failure_callback with a `context` dict
3. This function extracts pipeline_name, task_name, runtime, error_log, etc.
4. Sends POST /predict-rca to the RCA API
5. Logs the predicted root cause

WHY a callback and not a sensor?
- Callbacks fire IMMEDIATELY when a task fails — zero delay
- Sensors would need to poll for failures — adds latency and complexity
- Callbacks get the full Airflow context (task instance, exception, etc.)

IMPORTANT: This file must be importable by Airflow workers.
It should NOT import heavy ML libraries — it only makes HTTP calls.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import requests

# Use standard logging (not our custom logger) because this runs
# inside Airflow's process, which has its own logging setup.
logger = logging.getLogger("rca_callback")

# RCA API URL — read from environment or default
RCA_API_URL = os.environ.get("RCA_API_URL", "http://localhost:8000")


def rca_failure_callback(context: dict[str, Any]) -> None:
    """
    Airflow on_failure_callback — automatically diagnoses pipeline failures.

    This function is passed directly to `default_args` or individual operators:

        default_args = {
            "on_failure_callback": rca_failure_callback,
        }

    Args:
        context: Airflow's callback context dict containing:
            - task_instance: The failed TaskInstance object
            - dag: The DAG object
            - exception: The exception that caused the failure
            - execution_date: When the DAG run was scheduled
            - And many more fields
    """
    try:
        # --- Extract information from Airflow context ---
        task_instance = context.get("task_instance")
        dag = context.get("dag")
        exception = context.get("exception")

        pipeline_name = dag.dag_id if dag else "unknown_dag"
        task_name = task_instance.task_id if task_instance else "unknown_task"

        # Runtime: difference between start and now (or end)
        runtime = 0
        if task_instance and task_instance.start_date:
            end_time = task_instance.end_date or datetime.now(timezone.utc)
            runtime = int((end_time - task_instance.start_date).total_seconds())

        # Retry count from task instance
        retry_count = 0
        if task_instance:
            retry_count = task_instance.try_number - 1  # try_number is 1-indexed

        # Error log: exception message + any available log content
        error_log = str(exception) if exception else "Unknown error — no exception captured"

        # Detect schema_change and upstream_failed from error text (heuristic)
        error_lower = error_log.lower()
        schema_change = any(
            kw in error_lower
            for kw in ["schema", "column", "dtype", "type mismatch", "field missing"]
        )
        upstream_failed = any(
            kw in error_lower
            for kw in ["upstream", "dependency", "depends on", "sensor timed out"]
        )

        # Check if any upstream tasks in this DAG run also failed
        if task_instance and hasattr(task_instance, "get_dagrun"):
            try:
                dag_run = task_instance.get_dagrun()
                if dag_run:
                    failed_tasks = [
                        ti for ti in dag_run.get_task_instances()
                        if ti.state == "failed" and ti.task_id != task_name
                    ]
                    if failed_tasks:
                        upstream_failed = True
            except Exception:
                pass  # Don't fail the callback if we can't check upstream

        # --- Build RCA API request ---
        payload = {
            "pipeline_name": pipeline_name,
            "task_name": task_name,
            "runtime": max(runtime, 0),
            "retry_count": min(max(retry_count, 0), 10),
            "rows_processed": 0,  # Airflow doesn't track this natively
            "schema_change": schema_change,
            "upstream_failed": upstream_failed,
            "error_log": error_log[:5000],  # Truncate very long logs
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Sending RCA request for %s.%s to %s",
            pipeline_name, task_name, RCA_API_URL,
        )

        # --- Call the RCA API ---
        response = requests.post(
            f"{RCA_API_URL}/predict-rca",
            json=payload,
            timeout=30,  # Don't hang if the API is down
        )

        if response.status_code == 200:
            result = response.json()
            logger.info(
                "RCA PREDICTION for %s.%s:\n"
                "  Root Cause: %s\n"
                "  Confidence: %.2f\n"
                "  Evidence: %s",
                pipeline_name,
                task_name,
                result.get("predicted_root_cause", "Unknown"),
                result.get("confidence", 0),
                json.dumps(result.get("evidence", []), indent=4),
            )
        else:
            logger.error(
                "RCA API returned status %d for %s.%s: %s",
                response.status_code, pipeline_name, task_name, response.text,
            )

    except requests.exceptions.ConnectionError:
        logger.error(
            "Cannot connect to RCA API at %s. Is the service running?", RCA_API_URL,
        )
    except requests.exceptions.Timeout:
        logger.error("RCA API request timed out after 30 seconds")
    except Exception as e:
        # NEVER let the callback crash — it would mask the original failure
        logger.exception("RCA callback failed unexpectedly: %s", e)


def rca_failure_callback_with_xcom(context: dict[str, Any]) -> None:
    """
    Extended callback that also pushes the RCA result to XCom.

    This lets downstream tasks or alert operators read the root cause
    from XCom and include it in notifications (Slack, PagerDuty, etc.).
    """
    try:
        task_instance = context.get("task_instance")
        dag = context.get("dag")
        exception = context.get("exception")

        pipeline_name = dag.dag_id if dag else "unknown_dag"
        task_name = task_instance.task_id if task_instance else "unknown_task"

        runtime = 0
        if task_instance and task_instance.start_date:
            end_time = task_instance.end_date or datetime.now(timezone.utc)
            runtime = int((end_time - task_instance.start_date).total_seconds())

        retry_count = (task_instance.try_number - 1) if task_instance else 0
        error_log = str(exception) if exception else "Unknown error"
        error_lower = error_log.lower()

        payload = {
            "pipeline_name": pipeline_name,
            "task_name": task_name,
            "runtime": max(runtime, 0),
            "retry_count": min(max(retry_count, 0), 10),
            "rows_processed": 0,
            "schema_change": any(kw in error_lower for kw in ["schema", "column", "dtype"]),
            "upstream_failed": any(kw in error_lower for kw in ["upstream", "dependency"]),
            "error_log": error_log[:5000],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        response = requests.post(
            f"{RCA_API_URL}/predict-rca",
            json=payload,
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()

            # Push to XCom so downstream tasks can access the root cause
            if task_instance:
                task_instance.xcom_push(key="rca_result", value=result)
                logger.info("RCA result pushed to XCom for %s.%s", pipeline_name, task_name)
        else:
            logger.error("RCA API returned %d", response.status_code)

    except Exception as e:
        logger.exception("RCA callback (XCom variant) failed: %s", e)
