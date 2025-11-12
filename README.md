# Sentiment Analysis – Project 7

[![Deploy to Cloud Run](https://github.com/Laeti-dev/P7-sentiments_analysis/actions/workflows/deploy.yml/badge.svg)](https://github.com/Laeti-dev/P7-sentiments_analysis/actions/workflows/deploy.yml)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)
![MLflow Tracking](https://img.shields.io/badge/MLflow-enabled-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-live-FF4B4B?logo=streamlit&logoColor=white)

> Production-ready sentiment analysis pipeline built end-to-end for the Air Paradis client: from data prep to deployment.

**TL;DR**
- Real-time tweet classification powered by Bidirectional GRU and backup transformer models.
- FastAPI backend, Streamlit frontend, and an MLflow feedback loop for continuous learning.
- Containerized with Docker, automated with GitHub Actions, deployable to Google Cloud Run.

Production-ready sentiment analysis pipeline built for the Air Paradis client. The repository covers the full lifecycle: exploratory notebooks, experiment tracking, model packaging, FastAPI backend, Streamlit interface, automated testing, and cloud deployment assets.

## Table of Contents
- [Project Highlights](#project-highlights)
- [Repository Layout](#repository-layout)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Testing and Quality Gates](#testing-and-quality-gates)
- [Experiment Tracking](#experiment-tracking)
- [Modeling Pipeline](#modeling-pipeline)
- [Deployment](#deployment)
- [Contributing & Maintenance](#contributing--maintenance)

## Project Highlights
- **Business objective**: anticipate negative buzz on social media by classifying tweets as positive or negative.
- **Modeling**: classical ML baselines, RNN/LSTM/GRU architectures, and fine-tuned BERT; tracked with MLflow.
- **Serving**: FastAPI service exposing `/predict` and feedback endpoints; Streamlit UI for business users.
- **MLOps**: pytest suite, GitHub Actions CI/CD, Docker images, Cloud Run deployment scripts, monitoring hooks.
- **Documentation**: detailed notebooks, comparison artifacts, and guides for CI/CD, testing, and deployment.

## Repository Layout
```txt
P7-Sentiments_analysis/
├── backend/                  # FastAPI application and tests
│   ├── app/
│   │   ├── main.py           # API with prediction + feedback endpoints
│   │   ├── model/            # Deployed model + tokenizer assets
│   │   └── feedback/         # Persistent feedback CSV
│   ├── tests/                # Pytest suite for the API
│   └── Dockerfile
├── frontend/                 # Streamlit client for manual testing and demos
│   ├── main.py
│   └── tests/
├── notebooks/                # Data exploration, modeling, benchmarking
├── models/                   # Trained models ready for reuse
├── checkpoints/              # Intermediate training checkpoints
├── artifacts/                # Plots, reports, and metrics exported from notebooks
├── mlruns/                   # MLflow tracking data
├── documentation/            # Supplemental guides (CI/CD, deployment, testing)
├── docker-compose.yml        # Local multi-service orchestration
├── requirements.txt          # Base Python dependencies
├── run_tests.sh              # Convenience helper for CI
└── README.md
```

## Getting Started

### 1. Prerequisites
- Python 3.10 (required for `transformers` compatibility on macOS/Apple Silicon)
- [`uv`](https://docs.astral.sh/uv/getting-started/) package manager
- Docker (optional, for containerized workflows)

### 2. Environment Setup
```bash
# Install uv if needed
pip install uv

# Create and activate the virtual environment
uv venv --python 3.10
source .uv-venv/bin/activate  # macOS/Linux
# On Windows: .\.uv-venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

Copy `.env_sample` to `.env` and populate:
```bash
cp .env_sample .env
```
Key variables include:
- `MLFLOW_TRACKING_URI` (points to the local or remote MLflow tracking server)
- `MLFLOW_GCS_BUCKET`, `GCP_PROJECT_ID`, `RUN_ID` / `MLFLOW_MODEL_URI` for loading the production model
- `APPINSIGHTS_KEY` for optional telemetry

### 3. Run the Services Locally

**API**
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
- Swagger UI: http://localhost:8000/docs
- Feedback log stored in `backend/app/feedback/feedback_log.csv`

**Streamlit UI**
```bash
cd frontend
streamlit run main.py
```
Set `API_URL` environment variable if the backend runs elsewhere (defaults to `http://localhost:8000/predict`).

**Docker Compose**
```bash
docker compose up --build
```
Starts FastAPI, Streamlit, and supporting services with a single command.
To run MLflow tracking alongside the app:
```bash
docker compose up mlflow db
```
The MLflow service publishes on `http://localhost:5500` and persists artifacts in the configured GCS bucket.

### 4. Key Dependencies
- `pandas`, `numpy`, `matplotlib`, `seaborn` for data manipulation and visualization
- `nltk`, `gensim`, `wordcloud`, `pyspellchecker` for NLP preprocessing and embeddings
- `tensorflow-macos`, `tensorflow-metal`, `tensorflow-hub`, `torch`, `transformers` for deep learning models
- `scikit-learn`, `mlflow`, `ipywidgets`, `ipykernel` for classical ML and experiment tracking
- `fastapi`, `uvicorn`, `pydantic`, `requests` for the API layer
- `streamlit`, `plotly` for the frontend interface
- `pytest` for automated testing

### Documentation
- [CI/CD setup guide](documentation/CI:CD_setup.md)
- [Deploy to Google Cloud Run](documentation/deploy_to_gcp.md)
- [Run tests locally and in CI](documentation/run_tests.md)
- [Configure MLflow with Google Cloud Storage](documentation/mlflow_setup.md)

### Testing and Quality Gates
```bash
pytest                 # run unit tests (backend + frontend)
./run_tests.sh         # wrapper used in CI
```
Key tests cover endpoint responses, schema validation, and feedback persistence.

### Experiment Tracking
- Local fallback: `mlflow ui --backend-store-uri mlruns`
- Remote setup: `docker compose up mlflow db` then open `http://localhost:5000`
- Artifacts are automatically uploaded to the bucket `gs://${MLFLOW_GCS_BUCKET}`
- Comparison figures live in `artifacts/`, including confusion matrices and ROC curves for all models.

### CLI Training Script
- Before launching, ensure your environment exposes the GCP service-account key so MLflow can push artifacts to the GCS bucket:
  ```bash
  export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/gcp-key.json"
  export GOOGLE_CLOUD_PROJECT=analyse-de-sentiments-475921   # adjust to your project
  ```
- Train the production Bidirectional GRU (or any supported variant) from the command line:
  ```bash
  python training/train_neural_model.py \
    --data-path data/processed_sample_tweets.csv \
    --text-column advanced_processed_text_lem \
    --target-column target \
    --experiment-name P7-Sentiments_Analysis_neural_network \
    --run-name Bidirectional_GRU_glove_training \
    --mlflow-uri http://localhost:5500
  ```
- Pass `--register-model-name ""` if the tracking server does not expose the Model Registry endpoint; the script will fall back automatically if the registry is unavailable.
- The script mirrors the notebook pipeline, logs metrics and artifacts to MLflow, and exports the model/tokenizer into `backend/app/model/` for the API.

## Modeling Pipeline
- **Dataset**: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) (1.6M tweets).
- **Preprocessing**: normalization, mention/url handling, tokenization, sequence padding (`MAX_SEQUENCE_LENGTH=100` for RNN family).
- **Benchmarks**: logistic regression, SVM, gradient boosting (baseline notebooks).
- **Deep Learning**: Bidirectional LSTM/GRU with GloVe, Word2Vec, and FastText embeddings.
- **Transformer**: `bert-base-uncased` fine-tuned for binary sentiment classification.
- **Selection**: best-performing Bidirectional GRU (GloVe) exported to `backend/app/model/`.
- **Feedback Loop**: Streamlit UI records user corrections; data goes into `feedback_log.csv` for retraining.

Notebooks of interest:
- `notebooks/P7_eda.ipynb`: exploratory data analysis.
- `notebooks/P7_basic_ML_models.ipynb`: classical baselines.
- `notebooks/P7_neural_network.ipynb`: RNN/LSTM experiments.
- `notebooks/P7_BERT.ipynb`: transformer fine-tuning and evaluation.

## Deployment
- **CI/CD**: GitHub Actions pipeline builds, tests, and deploys the backend image to Google Cloud Run.
- **Docker**: `backend/Dockerfile` and `frontend/Dockerfile` for reproducible builds.
- **GCP Scripts**: `deploy-gcp.sh` and documentation in `documentation/deploy_to_gcp.md`.
- **Monitoring**: Azure Application Insights integration (optional via `APPINSIGHTS_KEY`).

## Contributing & Maintenance
- Follow the coding standards enforced by `pytest` and linting (see `documentation/run_tests.md`).
- Update MLflow experiment IDs and artifact paths when introducing new models.
- Document any changes to feedback schema or API contract in `backend/app/main.py`.

---
Feel free to open an issue if you discover bugs, have questions about the pipeline, or want to propose enhancements.
