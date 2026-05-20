"""
tests/unit/test_phase12_docker.py
=================================
Validates Docker deployment artifacts:
  - Dockerfiles (syntax, required directives)
  - docker-compose.yml (structure, services, healthchecks)
  - .dockerignore (exists, excludes sensitive paths)
  - Requirements files (production deps present, no test/dev deps)

These tests do NOT build or run containers — they validate the
configuration files are well-formed before the first `docker compose build`.
"""

import sys
from pathlib import Path

import yaml
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent


# =========================================================================
# Helpers
# =========================================================================

def _read(relpath: str) -> str:
    """Read a project file relative to the project root."""
    return (_PROJECT_ROOT / relpath).read_text()


def _lines(relpath: str) -> list[str]:
    """Read non-empty, non-comment lines from a file."""
    return [
        line.strip()
        for line in _read(relpath).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


# =========================================================================
# .dockerignore
# =========================================================================

class TestDockerignore:
    def test_exists(self) -> None:
        assert (_PROJECT_ROOT / ".dockerignore").is_file()

    @pytest.mark.parametrize(
        "pattern",
        [".venv/", "__pycache__/", ".git/", ".env", "airflow/", "notebooks/"],
    )
    def test_excludes_pattern(self, pattern: str) -> None:
        content = _read(".dockerignore")
        assert pattern in content, f".dockerignore should exclude '{pattern}'"


# =========================================================================
# Dockerfile.api
# =========================================================================

class TestDockerfileApi:
    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.content = _read("docker/Dockerfile.api")
        self.lines = self.content.splitlines()

    def test_exists(self) -> None:
        assert (_PROJECT_ROOT / "docker/Dockerfile.api").is_file()

    def test_has_from_directive(self) -> None:
        assert any(line.startswith("FROM") for line in self.lines)

    def test_base_image_is_python_slim(self) -> None:
        from_lines = [l for l in self.lines if l.startswith("FROM")]
        assert any("python:3.11-slim" in l for l in from_lines)

    def test_has_workdir(self) -> None:
        assert any("WORKDIR" in line for line in self.lines)

    def test_exposes_port_8000(self) -> None:
        assert any("EXPOSE 8000" in line for line in self.lines)

    def test_has_healthcheck(self) -> None:
        assert "HEALTHCHECK" in self.content

    def test_has_cmd(self) -> None:
        assert any(line.startswith("CMD") for line in self.lines)

    def test_cmd_uses_uvicorn(self) -> None:
        cmd_lines = [l for l in self.lines if l.startswith("CMD")]
        assert any("uvicorn" in l for l in cmd_lines)

    def test_copies_src(self) -> None:
        assert "COPY src/" in self.content

    def test_copies_api(self) -> None:
        assert "COPY api/" in self.content

    def test_copies_database(self) -> None:
        assert "COPY database/" in self.content

    def test_installs_requirements(self) -> None:
        assert "pip install" in self.content
        assert "requirements" in self.content


# =========================================================================
# Dockerfile.dashboard
# =========================================================================

class TestDockerfileDashboard:
    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.content = _read("docker/Dockerfile.dashboard")
        self.lines = self.content.splitlines()

    def test_exists(self) -> None:
        assert (_PROJECT_ROOT / "docker/Dockerfile.dashboard").is_file()

    def test_has_from_directive(self) -> None:
        assert any(line.startswith("FROM") for line in self.lines)

    def test_base_image_is_python_slim(self) -> None:
        from_lines = [l for l in self.lines if l.startswith("FROM")]
        assert any("python:3.11-slim" in l for l in from_lines)

    def test_exposes_port_8501(self) -> None:
        assert any("EXPOSE 8501" in line for line in self.lines)

    def test_has_healthcheck(self) -> None:
        assert "HEALTHCHECK" in self.content

    def test_has_cmd(self) -> None:
        assert any(line.startswith("CMD") for line in self.lines)

    def test_cmd_uses_streamlit(self) -> None:
        cmd_lines = [l for l in self.lines if l.startswith("CMD")]
        assert any("streamlit" in l for l in cmd_lines)

    def test_copies_dashboard(self) -> None:
        assert "COPY dashboard/" in self.content

    def test_copies_src_graph(self) -> None:
        assert "src/graph/" in self.content

    def test_copies_src_config(self) -> None:
        assert "src/config/" in self.content


# =========================================================================
# docker-compose.yml
# =========================================================================

class TestDockerCompose:
    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.path = _PROJECT_ROOT / "docker/docker-compose.yml"
        self.raw = self.path.read_text()
        self.config = yaml.safe_load(self.raw)

    def test_exists(self) -> None:
        assert self.path.is_file()

    def test_parses_as_valid_yaml(self) -> None:
        assert isinstance(self.config, dict)

    def test_has_services_key(self) -> None:
        assert "services" in self.config

    def test_has_three_services(self) -> None:
        services = self.config["services"]
        assert "postgres" in services
        assert "rca-api" in services
        assert "rca-dashboard" in services

    def test_postgres_image(self) -> None:
        pg = self.config["services"]["postgres"]
        assert "postgres" in pg["image"]

    def test_postgres_has_healthcheck(self) -> None:
        pg = self.config["services"]["postgres"]
        assert "healthcheck" in pg

    def test_postgres_environment(self) -> None:
        env = self.config["services"]["postgres"]["environment"]
        assert "POSTGRES_USER" in env
        assert "POSTGRES_PASSWORD" in env
        assert "POSTGRES_DB" in env

    def test_api_depends_on_postgres(self) -> None:
        api = self.config["services"]["rca-api"]
        assert "postgres" in api["depends_on"]

    def test_api_exposes_port_8000(self) -> None:
        api = self.config["services"]["rca-api"]
        ports = api["ports"]
        assert any("8000" in str(p) for p in ports)

    def test_api_has_database_url(self) -> None:
        env = self.config["services"]["rca-api"]["environment"]
        assert "DATABASE_URL" in env
        assert "postgres" in env["DATABASE_URL"]  # points to postgres service

    def test_api_has_healthcheck(self) -> None:
        api = self.config["services"]["rca-api"]
        assert "healthcheck" in api

    def test_api_image_tag(self) -> None:
        api = self.config["services"]["rca-api"]
        assert api["image"] == "sahojit/rca-api:latest"

    def test_dashboard_depends_on_api(self) -> None:
        dash = self.config["services"]["rca-dashboard"]
        assert "rca-api" in dash["depends_on"]

    def test_dashboard_exposes_port_8501(self) -> None:
        dash = self.config["services"]["rca-dashboard"]
        ports = dash["ports"]
        assert any("8501" in str(p) for p in ports)

    def test_dashboard_has_api_url(self) -> None:
        env = self.config["services"]["rca-dashboard"]["environment"]
        assert "RCA_API_URL" in env
        assert "rca-api" in env["RCA_API_URL"]

    def test_dashboard_image_tag(self) -> None:
        dash = self.config["services"]["rca-dashboard"]
        assert dash["image"] == "sahojit/rca-dashboard:latest"

    def test_volumes_defined(self) -> None:
        assert "volumes" in self.config
        assert "pgdata" in self.config["volumes"]


# =========================================================================
# Requirements files
# =========================================================================

class TestRequirementsApi:
    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.lines = _lines("docker/requirements-api.txt")

    def test_exists(self) -> None:
        assert (_PROJECT_ROOT / "docker/requirements-api.txt").is_file()

    @pytest.mark.parametrize(
        "pkg",
        ["fastapi", "uvicorn", "pydantic", "sqlalchemy", "psycopg2-binary",
         "xgboost", "shap", "pandas", "networkx", "scikit-learn"],
    )
    def test_has_production_dep(self, pkg: str) -> None:
        joined = " ".join(self.lines).lower()
        assert pkg in joined, f"requirements-api.txt should include '{pkg}'"

    @pytest.mark.parametrize("pkg", ["pytest", "black", "ruff", "mypy", "streamlit", "apache-airflow"])
    def test_no_dev_dep(self, pkg: str) -> None:
        joined = " ".join(self.lines).lower()
        assert pkg not in joined, f"requirements-api.txt should NOT include '{pkg}'"


class TestRequirementsDashboard:
    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.lines = _lines("docker/requirements-dashboard.txt")

    def test_exists(self) -> None:
        assert (_PROJECT_ROOT / "docker/requirements-dashboard.txt").is_file()

    @pytest.mark.parametrize(
        "pkg",
        ["streamlit", "plotly", "networkx", "requests", "pandas", "pydantic"],
    )
    def test_has_dashboard_dep(self, pkg: str) -> None:
        joined = " ".join(self.lines).lower()
        assert pkg in joined, f"requirements-dashboard.txt should include '{pkg}'"

    @pytest.mark.parametrize("pkg", ["pytest", "xgboost", "shap", "apache-airflow", "fastapi"])
    def test_no_non_dashboard_dep(self, pkg: str) -> None:
        joined = " ".join(self.lines).lower()
        assert pkg not in joined, f"requirements-dashboard.txt should NOT include '{pkg}'"
