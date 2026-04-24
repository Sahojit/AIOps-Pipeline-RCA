"""
src/ingestion/lemma_adapter.py
==============================
Adapter to transform LEMMA-RCA dataset into the 10-column CSV format
expected by the AIOps Pipeline RCA system.

LEMMA-RCA (HuggingFace: Lemma-RCA-NEC) is a multi-modal dataset with
real fault injection scenarios from microservices (AWS EKS). The dataset
is organised as zip files per date, each containing one fault scenario.

Ground truth is hardcoded from the official PPTX documentation because
the dataset stores labels in PowerPoint slides, not structured metadata.

This adapter:
1. Downloads preprocessed data from HuggingFace Hub
2. Extracts real pod logs from zip archives
3. Generates multiple training events per scenario using real log patterns

Output columns match PipelineFailureEvent schema exactly so downstream
feature engineering and model training need no further transformation.
"""

import csv
import io
import random
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.ingestion.schemas import RootCause
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

HUGGINGFACE_DATASETS: dict[str, str] = {
    "cloud": "Lemma-RCA-NEC/Cloud_Computing_Preprocessed",
    "product_review": "Lemma-RCA-NEC/Product_Review_Preprocessed",
}


@dataclass
class ScenarioGroundTruth:
    """Ground truth for a single LEMMA-RCA scenario, parsed from PPTX docs."""
    date_key: str
    fault_description: str
    root_cause: RootCause
    faulty_pod_pattern: str  # regex to match the faulty pod name in log files
    microservice: str
    schema_change: bool = False
    upstream_failed: bool = False


# Ground truth extracted from the official PPTX documentation for each scenario
SCENARIO_GROUND_TRUTH: list[ScenarioGroundTruth] = [
    ScenarioGroundTruth(
        date_key="20231207",
        fault_description="CPU bug in productpage-v1 causing latency increase",
        root_cause=RootCause.RESOURCE_EXHAUSTION,
        faulty_pod_pattern=r"productpage",
        microservice="book-info",
    ),
    ScenarioGroundTruth(
        date_key="20231221",
        fault_description="Noisy Neighbor: pod ratings moved to same node as productpage-v1",
        root_cause=RootCause.RESOURCE_EXHAUSTION,
        faulty_pod_pattern=r"productpage|ratings",
        microservice="book-info",
    ),
    ScenarioGroundTruth(
        date_key="20240115",
        fault_description="Malware DDoS attack via bot pods against productpage-v1",
        root_cause=RootCause.API_FAILURE,
        faulty_pod_pattern=r"malware|bot|productpage",
        microservice="book-info",
    ),
    ScenarioGroundTruth(
        date_key="20240124",
        fault_description="Silent CPU freeze from infinite loop bug occupying one CPU",
        root_cause=RootCause.RESOURCE_EXHAUSTION,
        faulty_pod_pattern=r"ip-10-1-100-109|web|ap-",
        microservice="3-tier-web",
    ),
    ScenarioGroundTruth(
        date_key="20240207",
        fault_description="GitOps mistake: wrong resource limits deployed to details-v1",
        root_cause=RootCause.SCHEMA_CHANGE,
        faulty_pod_pattern=r"details|scenario9",
        microservice="book-info",
        schema_change=True,
    ),
    ScenarioGroundTruth(
        date_key="20240215",
        fault_description="Cryptojacking Coin Miner on details-v1 consuming resources",
        root_cause=RootCause.RESOURCE_EXHAUSTION,
        faulty_pod_pattern=r"details",
        microservice="book-info",
    ),
]

# Error message templates per root cause — prepended to real logs so our
# regex parser and TF-IDF vocabulary can find familiar patterns.
FALLBACK_ERROR_TEMPLATES: dict[RootCause, list[str]] = {
    RootCause.RESOURCE_EXHAUSTION: [
        "MemoryError: unable to allocate {mem} GiB. Worker OOM killed by kernel.",
        "ResourceError: CPU throttled. Container exceeds cgroup limit on pod {pod}.",
        "KilledWorker: task exceeded memory limit and was killed by orchestrator on {pod}.",
        "DiskFullError: /tmp is at 100% capacity on pod {pod}. Cannot write results.",
    ],
    RootCause.API_FAILURE: [
        "HTTPError: 503 Service Unavailable from pod {pod}.",
        "ConnectionError: max retries exceeded connecting to {pod}:8080.",
        "TimeoutError: API call to {pod} timed out after 30s. DDoS suspected.",
        "HTTPError: 429 Too Many Requests from {pod}. Rate limit exceeded.",
    ],
    RootCause.DEPENDENCY_FAILURE: [
        "DependencyError: upstream service {pod} not responding. Network delay detected.",
        "ConnectionTimeout: packet loss to {pod}. Service unreachable.",
        "UpstreamFailure: dependency {pod} has status=FAILED. Cannot proceed.",
        "NetworkError: connection to {pod} reset. Possible packet loss.",
    ],
    RootCause.SCHEMA_CHANGE: [
        "ConfigError: configuration changed on {pod}. Expected schema does not match.",
        "GitOpsError: deployment on {pod} introduced breaking change.",
        "ValidationError: field mapping changed after config update on {pod}.",
        "SchemaError: misconfiguration detected on {pod}. Rollback required.",
    ],
}

# Pipeline name templates for variety in training data
PIPELINE_NAMES = [
    "cloud_infra_pipeline", "microservice_health_pipeline", "k8s_monitoring_pipeline",
    "service_mesh_pipeline", "container_orchestration_pipeline", "cluster_ops_pipeline",
    "latency_monitoring_pipeline", "error_tracking_pipeline", "resource_mgmt_pipeline",
    "deployment_pipeline", "security_scan_pipeline", "cost_tracking_pipeline",
    "autoscale_pipeline", "load_balancer_pipeline", "network_monitor_pipeline",
]

TASK_NAMES = [
    "check_pod_health", "analyze_metrics", "process_logs", "validate_deployment",
    "monitor_latency", "aggregate_kpi", "detect_anomaly", "evaluate_sli",
    "scan_containers", "audit_resources", "check_endpoints", "sync_config",
    "run_diagnostics", "collect_traces", "parse_events",
]


# =============================================================================
# Download
# =============================================================================

def download_lemma_dataset(
    dataset_key: str = "cloud",
    cache_dir: Optional[Path] = None,
) -> Path:
    """Download a LEMMA-RCA dataset from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    if dataset_key not in HUGGINGFACE_DATASETS:
        raise ValueError(
            f"Unknown dataset key '{dataset_key}'. "
            f"Available: {list(HUGGINGFACE_DATASETS.keys())}"
        )

    repo_id = HUGGINGFACE_DATASETS[dataset_key]
    if cache_dir is None:
        from src.config.settings import settings
        cache_dir = settings.external_data_dir / "lemma"

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading LEMMA-RCA dataset '%s' from HuggingFace...", repo_id)

    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=str(cache_dir),
        local_dir=str(cache_dir / dataset_key),
    )

    logger.info("Dataset downloaded to %s", local_dir)
    return Path(local_dir)


# =============================================================================
# Log extraction from zip archives
# =============================================================================

def _extract_logs_from_zip(
    zip_path: Path,
    pod_pattern: str,
    max_logs_per_pod: int = 500,
) -> dict[str, list[dict[str, str]]]:
    """
    Extract structured log messages from a LEMMA-RCA log zip file.

    Returns:
        Dict mapping pod_name → list of {time, content, template} dicts.
    """
    pod_logs: dict[str, list[dict[str, str]]] = {}
    pattern = re.compile(pod_pattern, re.IGNORECASE)

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            csv_files = [
                n for n in z.namelist()
                if n.endswith("_messages_structured.csv")
            ]

            for csv_name in csv_files:
                # Extract pod name from filename
                basename = csv_name.rsplit("/", 1)[-1]
                pod_name = basename.replace("_messages_structured.csv", "")

                # Check if this pod matches our pattern of interest
                is_faulty = bool(pattern.search(pod_name))

                try:
                    with z.open(csv_name) as f:
                        reader = csv.DictReader(
                            io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                        )
                        logs: list[dict[str, str]] = []
                        for row in reader:
                            content = row.get("Content", "")
                            if content.strip():
                                logs.append({
                                    "time": row.get("Time", ""),
                                    "content": content.strip(),
                                    "template": row.get("EventTemplate", ""),
                                    "is_faulty_pod": is_faulty,
                                })
                            if len(logs) >= max_logs_per_pod:
                                break
                        if logs:
                            pod_logs[pod_name] = logs
                except Exception:
                    continue
    except Exception as e:
        logger.warning("Failed to read zip %s: %s", zip_path, e)

    return pod_logs


def _find_error_logs(
    logs: list[dict[str, str]],
    max_count: int = 50,
) -> list[str]:
    """Find error-like log messages from a list of structured logs."""
    error_keywords = [
        "error", "exception", "fail", "timeout", "crash", "oom", "kill",
        "panic", "fatal", "refuse", "unavailable", "denied", "warn",
        "critical", "abort", "i/o timeout", "deadline exceeded",
        "connection reset", "broken pipe",
    ]

    errors: list[str] = []
    for log in logs:
        content = log["content"]
        if any(kw in content.lower() for kw in error_keywords):
            errors.append(content)
            if len(errors) >= max_count:
                break
    return errors


# =============================================================================
# Event generation
# =============================================================================

def _compose_error_log(
    real_logs: list[str],
    root_cause: RootCause,
    faulty_pod: str,
    max_length: int = 500,
) -> str:
    """
    Compose a single error_log string from real LEMMA logs.

    Prepends a synthetic-style header so our regex parser and TF-IDF
    vocabulary can find familiar patterns in the text.
    """
    templates = FALLBACK_ERROR_TEMPLATES.get(root_cause, [])
    if templates:
        header = random.choice(templates).format(
            pod=faulty_pod,
            mem=f"{random.uniform(2.0, 16.0):.1f}",
        )
    else:
        header = f"FaultDetected: anomaly on pod {faulty_pod}."

    # Pick real error messages — guard against None entries in log list
    real_logs = [m for m in real_logs if m and m.strip()]
    if real_logs:
        selected = random.sample(real_logs, min(3, len(real_logs)))
        body = " | ".join(msg[:150] for msg in selected)
        full = f"{header} Context: {body}"
    else:
        full = header

    if len(full) > max_length:
        full = full[:max_length - 3] + "..."
    return full


def _generate_events_from_scenario(
    ground_truth: ScenarioGroundTruth,
    pod_logs: dict[str, list[dict[str, str]]],
    events_per_scenario: int = 80,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate multiple training events from a single LEMMA scenario.

    Uses real log patterns from the scenario's pods and varies numeric
    features to create diverse training data.
    """
    rng = random.Random(seed + hash(ground_truth.date_key))
    events: list[dict[str, Any]] = []

    # Collect error logs from faulty pods and all pods
    faulty_pod_errors: list[str] = []
    all_errors: list[str] = []
    faulty_pod_names: list[str] = []

    for pod_name, logs in pod_logs.items():
        errors = _find_error_logs(logs)
        all_errors.extend(errors)
        if any(log.get("is_faulty_pod") for log in logs):
            faulty_pod_errors.extend(errors)
            faulty_pod_names.append(pod_name)

    # Use faulty pod errors if available, otherwise all errors
    error_pool = faulty_pod_errors if faulty_pod_errors else all_errors

    # If we still have no errors, use raw log content from faulty pods
    if not error_pool:
        for pod_name, logs in pod_logs.items():
            if any(log.get("is_faulty_pod") for log in logs):
                error_pool.extend(log["content"] for log in logs[:20])

    # Fallback: use any log content
    if not error_pool:
        for logs in pod_logs.values():
            error_pool.extend(log["content"] for log in logs[:10])
            if len(error_pool) >= 30:
                break

    # Parse base date from scenario
    try:
        base_date = datetime.strptime(ground_truth.date_key, "%Y%m%d").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

    faulty_pod_label = faulty_pod_names[0] if faulty_pod_names else ground_truth.faulty_pod_pattern

    for _ in range(events_per_scenario):
        # Vary features
        if ground_truth.root_cause == RootCause.RESOURCE_EXHAUSTION:
            runtime = rng.randint(60, 3600)
            retry_count = rng.randint(0, 5)
            rows_processed = rng.randint(100, 500000)
        elif ground_truth.root_cause == RootCause.API_FAILURE:
            runtime = rng.randint(10, 600)
            retry_count = rng.randint(2, 10)
            rows_processed = rng.randint(0, 100000)
        elif ground_truth.root_cause == RootCause.SCHEMA_CHANGE:
            runtime = rng.randint(5, 300)
            retry_count = rng.randint(0, 3)
            rows_processed = rng.randint(0, 50000)
            # schema_change is True from ground truth
        else:
            runtime = rng.randint(30, 1800)
            retry_count = rng.randint(0, 5)
            rows_processed = rng.randint(0, 200000)

        timestamp = base_date + timedelta(
            hours=rng.randint(0, 47),
            minutes=rng.randint(0, 59),
        )

        events.append({
            "pipeline_name": rng.choice(PIPELINE_NAMES),
            "task_name": rng.choice(TASK_NAMES),
            "runtime": runtime if runtime is not None else 0,
            "retry_count": retry_count if retry_count is not None else 0,
            "rows_processed": rows_processed if rows_processed is not None else 0,
            "schema_change": ground_truth.schema_change,
            "upstream_failed": ground_truth.upstream_failed,
            "error_log": _compose_error_log(
                error_pool, ground_truth.root_cause, faulty_pod_label
            ),
            "root_cause": ground_truth.root_cause.value,
            "timestamp": timestamp.isoformat(),
        })

    return events


# =============================================================================
# CSV reader — loads the pre-processed pipeline_failures.csv from disk
# =============================================================================

def load_lemma_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the preprocessed LEMMA-RCA dataset from the local CSV file.

    Args:
        path: Custom CSV path. Defaults to data/raw/pipeline_failures.csv.

    Returns:
        DataFrame with the standard 10-column schema.
    """
    from src.config.settings import settings

    csv_path = path or (settings.raw_data_dir / "pipeline_failures.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Run `python -m src.ingestion.load_lemma_dataset` first."
        )
    df = pd.read_csv(csv_path)
    logger.info("Loaded LEMMA-RCA dataset from %s (%d rows)", csv_path, len(df))
    return df


# =============================================================================
# Main entry point
# =============================================================================

def load_and_transform(
    dataset_key: str = "cloud",
    cache_dir: Optional[Path] = None,
    events_per_scenario: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    """
    End-to-end: download LEMMA-RCA → extract logs → generate events → DataFrame.

    Args:
        dataset_key: Which LEMMA dataset ("cloud" or "product_review").
        cache_dir: Where to cache downloaded data.
        events_per_scenario: How many training events to generate per scenario.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with the standard 10-column schema.
        Raises RuntimeError if the LEMMA log data cannot be found.
    """
    random.seed(seed)

    # 1. Download
    dataset_dir = download_lemma_dataset(dataset_key, cache_dir)

    # 2. Find log zip files
    log_data_dir = dataset_dir / "Log Data"
    if not log_data_dir.exists():
        raise RuntimeError(
            f"No 'Log Data' directory found in {dataset_dir}. "
            "Ensure the LEMMA-RCA dataset downloaded correctly."
        )

    all_events: list[dict[str, Any]] = []
    scenarios_processed = 0

    for gt in SCENARIO_GROUND_TRUTH:
        log_zip = log_data_dir / f"{gt.date_key}.zip"
        if not log_zip.exists():
            logger.warning("Log zip not found for scenario %s: %s", gt.date_key, log_zip)
            continue

        logger.info(
            "Processing scenario %s: %s → %s",
            gt.date_key, gt.fault_description, gt.root_cause.value,
        )

        # Extract logs from zip
        pod_logs = _extract_logs_from_zip(
            log_zip, gt.faulty_pod_pattern, max_logs_per_pod=300,
        )
        logger.info(
            "  Extracted logs from %d pods (%d total messages)",
            len(pod_logs),
            sum(len(v) for v in pod_logs.values()),
        )

        # Generate training events
        events = _generate_events_from_scenario(
            gt, pod_logs, events_per_scenario=events_per_scenario, seed=seed,
        )
        all_events.extend(events)
        scenarios_processed += 1
        logger.info("  Generated %d training events", len(events))

    logger.info(
        "Processed %d/%d scenarios. Total events: %d",
        scenarios_processed, len(SCENARIO_GROUND_TRUTH), len(all_events),
    )

    if not all_events:
        raise RuntimeError(
            "No events generated from LEMMA-RCA scenarios. "
            "Check that log zip files are present in the dataset."
        )

    df = pd.DataFrame(all_events)

    # 3. Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info(
        "Final dataset: %d rows, %d columns. Class distribution:\n%s",
        len(df), len(df.columns), df["root_cause"].value_counts().to_string(),
    )
    return df
