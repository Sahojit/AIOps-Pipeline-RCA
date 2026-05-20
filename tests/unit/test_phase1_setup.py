"""
tests/unit/test_phase1_setup.py
================================
Smoke tests for Phase 1 — project setup validation.
These verify that config loads correctly and logger works.
"""

import logging
from pathlib import Path


def test_project_root_exists() -> None:
    """Verify that PROJECT_ROOT resolves to a real directory."""
    from src.config.settings import PROJECT_ROOT
    assert PROJECT_ROOT.is_dir(), f"PROJECT_ROOT does not exist: {PROJECT_ROOT}"


def test_settings_load() -> None:
    """Settings must load without raising exceptions."""
    from src.config.settings import settings
    assert settings.app_name == "AIOps Pipeline RCA System"
    assert settings.api_port == 8000
    assert "rca_db" in settings.database_url


def test_model_path_property() -> None:
    """model_path property must return a Path combining model_dir + filename."""
    from src.config.settings import settings
    expected = settings.model_dir / settings.model_filename
    assert settings.model_path == expected


def test_logger_returns_logger_instance() -> None:
    """get_logger must return a standard logging.Logger."""
    from src.utils.logger import get_logger
    logger = get_logger("tests.unit.test_phase1")
    assert isinstance(logger, logging.Logger)


def test_logger_no_duplicate_handlers() -> None:
    """Calling get_logger twice on the same name must not add duplicate handlers."""
    from src.utils.logger import get_logger
    logger_a = get_logger("tests.dedup_check")
    logger_b = get_logger("tests.dedup_check")
    assert len(logger_b.handlers) == 1, "Duplicate handlers detected"


def test_logger_emits_messages(capfd) -> None:
    """Logger must write to stdout."""
    from src.utils.logger import get_logger
    logger = get_logger("tests.emit_check")
    logger.info("Phase 1 smoke test message")
    captured = capfd.readouterr()
    assert "Phase 1 smoke test message" in captured.out


def test_data_dirs_exist() -> None:
    """data/raw, data/processed, data/logs must exist."""
    from src.config.settings import settings
    for d in [settings.raw_data_dir, settings.processed_data_dir, settings.logs_data_dir]:
        assert d.is_dir(), f"Expected directory does not exist: {d}"
