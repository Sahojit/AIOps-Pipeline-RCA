"""
src/utils/logger.py
===================
Centralised structured logger for the entire RCA system.

Why a shared logger factory?
- Consistent log format across ALL modules (timestamp, level, module)
- Single place to change log format, handlers, or add file rotation
- Each module gets its own named logger so you can trace exactly
  which module emitted each log line

Usage (in any module):
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Feature engineering started for pipeline: %s", pipeline_name)

Output format:
    2026-03-03 12:34:56,789 | INFO     | src.features.engineer | Building features
"""

import logging
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Log format — mirrors what you'd ship to a log aggregator (Datadog, Loki)
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Returns a named logger configured with the project's standard format.

    Args:
        name: Usually __name__ from the calling module.
              Produces hierarchical names like 'src.features.engineer'.
        level: Override log level for this specific logger.
               Falls back to the root logger level if not provided.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when the same logger is retrieved
    # multiple times (common in long-running processes like Uvicorn).
    if logger.handlers:
        return logger

    # Stream to stdout so Docker / Kubernetes log collectors pick it up
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)

    if level:
        logger.setLevel(level.upper())

    # Prevent log records from bubbling up to the root logger's handlers,
    # which would cause duplicate output in some setups.
    logger.propagate = False

    return logger


def configure_root_logger(level: str = "INFO") -> None:
    """
    Configure the root logger once at application startup.

    Call this in:
    - main entrypoints (api/main.py, training scripts)
    - Never in library modules (they use get_logger() only)

    Args:
        level: Log level string — DEBUG | INFO | WARNING | ERROR | CRITICAL
    """
    logging.basicConfig(
        level=level.upper(),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stdout,
    )

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
