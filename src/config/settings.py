"""
src/config/settings.py
======================
Centralised configuration for the entire RCA system.

Design decisions:
- pydantic-settings BaseSettings automatically reads from:
    1. Environment variables (highest priority)
    2. .env file (fallback)
    3. Field defaults (lowest priority)
- This means the same code runs locally (using .env) and in
  production containers (using real env vars injected by the
  orchestration layer — Kubernetes, ECS, etc.)
- Never import raw `os.getenv()` elsewhere in the codebase.
  Always import from here.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# Project root — one level above src/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """
    All runtime configuration lives here.
    Add new config values here — never use os.getenv() elsewhere.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,       # DATABASE_URL and database_url both work
        extra="ignore",             # Ignore unknown env vars (safe default)
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    app_name: str = Field(default="AIOps Pipeline RCA System")
    environment: str = Field(default="development")  # development | staging | production
    log_level: str = Field(default="INFO")           # DEBUG | INFO | WARNING | ERROR

    # -------------------------------------------------------------------------
    # API Server
    # -------------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=1)              # Increase in production

    # -------------------------------------------------------------------------
    # Database (PostgreSQL — external)
    # -------------------------------------------------------------------------
    database_url: str = Field(
        default="postgresql://rca_user:rca_pass@localhost:5432/rca_db"
    )
    db_pool_size: int = Field(default=5)
    db_max_overflow: int = Field(default=10)
    db_echo: bool = Field(default=False)             # Set True to log SQL queries

    # -------------------------------------------------------------------------
    # Model Artifacts
    # -------------------------------------------------------------------------
    model_dir: Path = Field(default=PROJECT_ROOT / "models")
    model_filename: str = Field(default="rca_model_v1.pkl")
    model_metadata_file: str = Field(default="model_metadata.json")

    @property
    def model_path(self) -> Path:
        """Full path to the active model artifact."""
        return self.model_dir / self.model_filename

    @property
    def model_metadata_path(self) -> Path:
        return self.model_dir / self.model_metadata_file

    # -------------------------------------------------------------------------
    # Data Paths
    # -------------------------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    raw_data_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    logs_data_dir: Path = Field(default=PROJECT_ROOT / "data" / "logs")
    external_data_dir: Path = Field(default=PROJECT_ROOT / "data" / "external")

    # -------------------------------------------------------------------------
    # Airflow Integration
    # -------------------------------------------------------------------------
    airflow_host: str = Field(default="http://localhost:8080")
    rca_api_url: str = Field(default="http://localhost:8000")   # Used by Airflow callbacks

    # -------------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------------
    tfidf_max_features: int = Field(default=50)      # Max TF-IDF vocabulary size


# ---------------------------------------------------------------------------
# Module-level singleton — import `settings` everywhere.
# This is instantiated once at import time; env vars are read at that point.
# ---------------------------------------------------------------------------
settings = Settings()
