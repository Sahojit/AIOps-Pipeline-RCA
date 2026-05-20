#!/bin/bash
# =============================================================================
# airflow/airflow_setup.sh
# =============================================================================
# Quick setup script for running Airflow locally with the RCA DAGs.
#
# Prerequisites:
# - Docker installed and running
# - Python 3.11+ with pip
#
# This script:
# 1. Installs Airflow in standalone mode (for local development)
# 2. Configures the DAGs folder to point to our airflow/dags/
# 3. Sets the RCA_API_URL environment variable
# 4. Initialises the Airflow database
# 5. Creates an admin user
#
# For production, use the official Airflow Docker Compose or Helm chart.
# =============================================================================

set -e

echo "=========================================="
echo "  Airflow Setup for RCA Integration"
echo "=========================================="

# Get the project root (parent of airflow/ directory)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AIRFLOW_HOME="${PROJECT_ROOT}/airflow"

export AIRFLOW_HOME
export AIRFLOW__CORE__DAGS_FOLDER="${AIRFLOW_HOME}/dags"
export AIRFLOW__CORE__PLUGINS_FOLDER="${AIRFLOW_HOME}/plugins"
export AIRFLOW__CORE__LOAD_EXAMPLES="false"
export RCA_API_URL="http://localhost:8000"

echo ""
echo "Project root:    ${PROJECT_ROOT}"
echo "AIRFLOW_HOME:    ${AIRFLOW_HOME}"
echo "DAGs folder:     ${AIRFLOW__CORE__DAGS_FOLDER}"
echo "Plugins folder:  ${AIRFLOW__CORE__PLUGINS_FOLDER}"
echo "RCA API URL:     ${RCA_API_URL}"
echo ""

# --- Step 1: Install Airflow ---
echo "[1/5] Checking Airflow installation..."
if ! command -v airflow &> /dev/null; then
    echo "Installing Apache Airflow..."
    pip install "apache-airflow==2.9.1" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.1/constraints-3.11.txt"
else
    echo "Airflow already installed: $(airflow version)"
fi

# --- Step 2: Initialise Airflow database ---
echo ""
echo "[2/5] Initialising Airflow database..."
airflow db migrate

# --- Step 3: Create admin user ---
echo ""
echo "[3/5] Creating admin user..."
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    2>/dev/null || echo "Admin user already exists"

# --- Step 4: Verify DAGs ---
echo ""
echo "[4/5] Listing available DAGs..."
airflow dags list

# --- Step 5: Print start instructions ---
echo ""
echo "[5/5] Setup complete!"
echo ""
echo "=========================================="
echo "  HOW TO START AIRFLOW"
echo "=========================================="
echo ""
echo "Option A: Standalone mode (webserver + scheduler in one command):"
echo ""
echo "  export AIRFLOW_HOME=${AIRFLOW_HOME}"
echo "  export RCA_API_URL=http://localhost:8000"
echo "  airflow standalone"
echo ""
echo "Option B: Separate processes (for production-like setup):"
echo ""
echo "  # Terminal 1: Webserver"
echo "  export AIRFLOW_HOME=${AIRFLOW_HOME}"
echo "  airflow webserver --port 8080"
echo ""
echo "  # Terminal 2: Scheduler"
echo "  export AIRFLOW_HOME=${AIRFLOW_HOME}"
echo "  export RCA_API_URL=http://localhost:8000"
echo "  airflow scheduler"
echo ""
echo "Airflow UI: http://localhost:8080"
echo "Login: admin / admin"
echo ""
echo "IMPORTANT: Make sure the RCA API is running before testing:"
echo "  uvicorn api.main:app --port 8000"
echo "=========================================="
