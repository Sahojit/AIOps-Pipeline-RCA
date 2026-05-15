# AIOps Pipeline RCA System

> **Automatic Root Cause Analysis for data pipeline failures using XGBoost + SHAP + FastAPI + Streamlit — fully containerised with Docker.**

When a pipeline fails, this system automatically classifies the root cause in real time, explains *why* using SHAP feature attributions, and displays a live monitoring dashboard — all triggered via an Airflow callback hook.

**Status:** Active development — building out the ML pipeline, API, and dashboard from scratch.

**Dataset:** [LEMMA-RCA Cloud](https://huggingface.co/datasets/Lemma-RCA-NEC/Cloud_Computing_Preprocessed) — 800 real fault-injection events across 6 root cause classes.

---

## Demo

| Component | URL |
|-----------|-----|
| Streamlit Dashboard | `http://localhost:8501` |
| FastAPI Docs (Swagger) | `http://localhost:8000/docs` |
| API Health | `http://localhost:8000/health` |

---

## Architecture

```
 Apache Airflow DAGs
        │  on_failure_callback
        ▼
┌──────────────────┐     POST /predict-rca     ┌─────────────────────┐
│  rca_callback.py │ ─────────────────────────► │   FastAPI (port 8000)│
│  (Airflow plugin)│                            │   src/inference/     │
└──────────────────┘                            │   XGBoost + SHAP     │
                                                └────────┬────────────┘
                                                         │ writes prediction
                                                         ▼
                                                ┌─────────────────────┐
                                                │  PostgreSQL (5432)   │
                                                │  predictions table   │
                                                └────────┬────────────┘
                                                         │ reads
                                                         ▼
                                                ┌─────────────────────┐
                                                │ Streamlit Dashboard  │
                                                │    (port 8501)       │
                                                └─────────────────────┘
```

### ML Pipeline (offline training)

```
data/raw/pipeline_failures.csv
        │
        ▼
src/ingestion/   →  log parsing, TF-IDF vectorisation, schema validation
        │
        ▼
src/features/    →  96 engineered features (execution + temporal + log + graph)
        │
        ▼
src/training/    →  XGBClassifier, stratified split, evaluation, artifact save
        │
        ▼
models/          →  rca_model_v1.pkl  label_encoder.pkl  model_metadata.json
```

---

## Root Cause Classes

| Class | Description |
|-------|-------------|
| API Failure | External API timeout, 5xx errors, DDoS |
| Data Quality Issue | Null values, type mismatches, failed validation |
| Dependency Failure | Upstream pipeline failed, network partition |
| Missing Data | Source partition unavailable, empty dataset |
| Resource Exhaustion | OOM kill, CPU throttle, disk full |
| Schema Change | Column renamed/removed, config drift, GitOps mistake |

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **99.4%** |
| Macro F1 | **0.991** |
| Weighted F1 | **0.994** |
| Avg Confidence | **0.989** |
| Algorithm | XGBoost (multi:softprob) |
| Features | 96 (execution + log + temporal + TF-IDF) |
| Training data | 800 events (LEMMA-RCA Cloud dataset) |

---

## Model Card

### Classes and Per-Class Performance

| Root Cause Class | Precision | Recall | F1 | Support |
|-----------------|-----------|--------|----|---------|
| API Failure | 1.000 | 0.938 | 0.968 | 16 |
| Data Quality Issue | 1.000 | 1.000 | 1.000 | 23 |
| Dependency Failure | 1.000 | 1.000 | 1.000 | 21 |
| Missing Data | 1.000 | 1.000 | 1.000 | 20 |
| Resource Exhaustion | 0.985 | 1.000 | 0.992 | 64 |
| Schema Change | 1.000 | 1.000 | 1.000 | 16 |

> Evaluated on a held-out 20% test split (160 samples). `API Failure` recall of 0.938 reflects one misclassified sample — the model predicted `Resource Exhaustion` for a CPU-throttled API pod, which is arguable.

### Feature Groups

| Group | Count | Description |
|-------|-------|-------------|
| Execution | 12 | Duration, retries, row counts, error rate, retry rate |
| Schema | 5 | Column count, null fraction, schema version |
| Temporal | 12 | Hour, day-of-week, is_weekend, is_night, month, lag features |
| Log signals | 23 | Regex-extracted counts: OOM, timeout, connection error, etc. |
| TF-IDF | 50 | Unigram/bigram log message tokens (fitted on training set) |
| **Total** | **96** | |

### Training Configuration

```
Algorithm:    XGBoost  multi:softprob
n_estimators: 300
max_depth:    6
learning_rate: 0.1
subsample:    0.8
colsample_bytree: 0.8
eval_metric:  mlogloss
early_stopping_rounds: 20
```

### Artifact Files

| File | Purpose |
|------|---------|
| `models/rca_model_v1.pkl` | Trained XGBClassifier (~800KB) |
| `models/label_encoder.pkl` | LabelEncoder for 6 classes |
| `models/tfidf_vectorizer.pkl` | TF-IDF vocab (50 features, training-fitted) |
| `models/model_metadata.json` | Version, metrics, hyperparams, library versions |

---

## Dataset

Built on the **[LEMMA-RCA](https://huggingface.co/datasets/Lemma-RCA-NEC/Cloud_Computing_Preprocessed)** dataset — real fault injection scenarios from a microservices deployment on AWS EKS (BookInfo + 3-tier-web apps). Six fault scenarios were used, extracting structured pod logs from zip archives and generating training events with varied numeric features.

---

## Project Structure

```
AIOps-Pipeline-RCA/
│
├── api/
│   └── main.py                  # FastAPI app — POST /predict-rca, GET /stats, etc.
│
├── dashboard/
│   └── app.py                   # Streamlit 4-tab monitoring UI
│
├── database/
│   ├── models.py                # SQLAlchemy ORM — PipelineFailure table
│   ├── crud.py                  # DB queries: stats, distribution, recent predictions
│   └── session.py               # Engine + session factory
│
├── src/
│   ├── config/
│   │   └── settings.py          # pydantic-settings — all config in one place
│   │
│   ├── ingestion/
│   │   ├── schemas.py           # Pydantic v2 data contracts + RootCause enum
│   │   ├── lemma_adapter.py     # LEMMA-RCA download, log extraction, event generation
│   │   ├── log_parser.py        # Regex-based structured log feature extraction
│   │   ├── log_feature_extractor.py  # 30+ log signal features
│   │   ├── text_vectorizer.py   # TF-IDF vectoriser (50 features)
│   │   └── validate_data.py     # Dataset schema validation
│   │
│   ├── features/
│   │   ├── feature_definitions.py  # Canonical 96-feature definition
│   │   ├── engineer.py          # Execution + temporal feature engineering
│   │   └── build_features.py    # End-to-end feature matrix builder
│   │
│   ├── training/
│   │   ├── train_model.py       # CLI entry point: python -m src.training.train_model
│   │   ├── trainer.py           # XGBoost training with cross-validation
│   │   ├── evaluator.py         # Metrics: accuracy, macro F1, confusion matrix, SHAP
│   │   └── artifact_manager.py  # save/load model artifacts + metadata JSON
│   │
│   ├── inference/
│   │   ├── engine.py            # RCAEngine: predict + explain in one call
│   │   ├── explainer.py         # SHAP TreeExplainer — top feature attributions
│   │   └── evidence.py          # Human-readable evidence string builder
│   │
│   └── graph/
│       ├── dependency_graph.py  # NetworkX DAG — propagation depth, upstream failures
│       └── seed_dependencies.py # 30 LEMMA pipelines, 32 dependency edges, 5-tier arch
│
├── airflow/
│   ├── dags/
│   │   └── sample_etl_dag.py    # ETL DAG with rca_failure_callback wired in
│   └── plugins/
│       └── rca_callback.py      # on_failure_callback → POST /predict-rca
│
├── docker/
│   ├── docker-compose.yml       # 3-service stack: postgres + api + dashboard
│   ├── Dockerfile.api           # FastAPI image
│   ├── Dockerfile.dashboard     # Streamlit image
│   ├── requirements-api.txt     # Pinned production deps for API
│   └── requirements-dashboard.txt
│
├── models/                      # Trained artifacts (git-ignored in production)
│   ├── rca_model_v1.pkl
│   ├── label_encoder.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_metadata.json
│
├── data/
│   ├── raw/pipeline_failures.csv   # 800 LEMMA-derived training events
│   └── processed/                  # Feature matrix + labels (generated)
│
├── scripts/
│   └── seed_db.py               # Seeds 800 predictions into PostgreSQL
│
└── tests/
    └── unit/                    # 12 test modules — one per build phase
```

---

## Quick Start — Docker (Recommended)

```bash
# 1. Clone the repo
git clone https://github.com/your-username/aiops-pipeline-rca.git
cd aiops-pipeline-rca

# 2. Start the full stack
cd docker
docker compose up -d

# 3. Open the dashboard
open http://localhost:8501

# 4. Test the API
curl http://localhost:8000/health
```

All three containers start automatically and restart on reboot.

---

## Quick Start — Local Development

### Prerequisites
- Python 3.11+
- Docker Desktop (for PostgreSQL)

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env             # edit DATABASE_URL if needed

# 4. Start only PostgreSQL
cd docker && docker compose up -d postgres && cd ..

# 5. Train the model
python -m src.training.train_model

# 6. Start API
uvicorn api.main:app --reload --port 8000

# 7. Start dashboard (new terminal)
streamlit run dashboard/app.py
```

---

## API Reference

### `POST /predict-rca`

Predict root cause for a pipeline failure event.

**Request:**
```json
{
  "pipeline_name": "user_etl_pipeline",
  "task_name":     "load_users",
  "runtime":       3600,
  "retry_count":   3,
  "rows_processed": 0,
  "schema_change": false,
  "upstream_failed": true,
  "error_log": "ConnectionError: upstream service not responding",
  "timestamp": "2026-03-05T10:00:00Z"
}
```

**Response:**
```json
{
  "predicted_root_cause": "Dependency Failure",
  "confidence": 0.97,
  "all_probabilities": {
    "API Failure": 0.01,
    "Data Quality Issue": 0.01,
    "Dependency Failure": 0.97,
    "Missing Data": 0.00,
    "Resource Exhaustion": 0.00,
    "Schema Change": 0.01
  },
  "evidence": [
    "upstream_failed flag is True",
    "High SHAP: upstream_failed (+0.82)",
    "High SHAP: tfidf_upstream (+0.41)"
  ],
  "model_version": "v1"
}
```

### Other Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + model version |
| GET | `/stats?days=30` | Prediction counts and accuracy stats |
| GET | `/root-cause-distribution?days=30` | Breakdown by root cause |
| GET | `/recent-predictions?limit=50` | Latest prediction records |
| GET | `/pipeline-failures?pipeline_name=X` | Failures for a specific pipeline |
| POST | `/predict-rca/batch` | Batch predictions |

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Overview** | KPI cards (total predictions, top root cause, avg confidence), trend chart, root cause distribution pie |
| **Predictions** | Filterable table of all predictions with confidence scores |
| **Live RCA** | Manual failure submission form — get instant root cause prediction |
| **Graph Explorer** | Interactive NetworkX dependency graph — 30 pipelines, 5-tier architecture |

---

## Airflow Integration

Add automatic RCA to any DAG by importing the callback:

```python
from plugins.rca_callback import rca_failure_callback

default_args = {
    "on_failure_callback": rca_failure_callback,
}
```

When a task fails, Airflow immediately calls `POST /predict-rca` with the task context (pipeline name, runtime, retry count, exception message) and logs the predicted root cause.

---

## Feature Engineering (96 Features)

Features are built by `src/features/engineer.py` via five independent feature groups, then concatenated into a single matrix. Each group is unit-tested independently.

| Group | Count | What it captures |
|-------|-------|-----------------|
| **Execution** | 12 | `runtime`, `retry_count`, `rows_processed` + derived: `log_rows`, `rows_per_second`, `retry_rate`, `zero_rows` flag, `short/long_runtime` flags |
| **Schema** | 5 | `schema_change`, `upstream_failed` + 3 interaction terms (both, schema-only, upstream-only) |
| **Temporal** | 12 | `hour`, `day_of_week`, `is_weekend`, `is_business_hours`, `is_night`, `month`, cyclical sin/cos encodings |
| **Lag** | 2 | `lag_seconds` (time since last failure on same pipeline), `is_burst` flag (<5 min gap) |
| **Log signals** | 23 | HTTP status codes, memory values, timeout values, error type hash, keyword scores per root-cause domain |
| **TF-IDF** | 50 | Top 50 tokens from error log corpus (unigrams + bigrams, after preprocessing) |

The TF-IDF vectorizer vocabulary is fitted on the training set and saved as `models/tfidf_vectorizer.pkl`. At inference time the same vocabulary is loaded to guarantee feature alignment.

---

## Docker Images

Pre-built images are available on Docker Hub:

```bash
docker pull sahojit/rca-api:latest
docker pull sahojit/rca-dashboard:latest
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost 3.2 (multi:softprob) |
| Explainability | SHAP 0.50 (TreeExplainer) |
| Feature engineering | Pandas 2.3, scikit-learn 1.8 |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 16 + SQLAlchemy 2 |
| Dashboard | Streamlit 1.4x |
| Graph analysis | NetworkX |
| Containerisation | Docker + Docker Compose |
| Config management | pydantic-settings v2 |
| Workflow orchestration | Apache Airflow 2.x (external) |
| Data source | LEMMA-RCA (HuggingFace) |

---

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Project setup + config | ✅ |
| 2 | LEMMA-RCA data ingestion | ✅ |
| 3 | Log parsing system | ✅ |
| 4 | Feature engineering (96 features) | ✅ |
| 5 | XGBoost model training | ✅ |
| 6 | SHAP explainability | ✅ |
| 7 | FastAPI prediction service | ✅ |
| 8 | PostgreSQL integration | ✅ |
| 9 | Airflow callback integration | ✅ |
| 10 | NetworkX dependency graph | ✅ |
| 11 | Streamlit monitoring dashboard | ✅ |
| 12 | Docker containerisation + Docker Hub | ✅ |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT
