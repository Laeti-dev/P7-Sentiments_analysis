## MLflow Remote Tracking with Google Cloud Storage

### 1. Google Cloud prerequisites
- Create a dedicated GCS bucket (e.g. `gs://p7-mlflow-artifacts`) in the same project as the rest of your infrastructure.
- Enable the IAM API if needed and create a dedicated service account (e.g. `mlflow-tracking`).
- Grant the service account at least the `roles/storage.objectAdmin` role on the bucket.
- Download the service account JSON key and store it locally as `gcp-key.json` (or upload it to your CI/CD secret manager).

### 2. Environment variables
Add the following keys to your `.env` (or secret provider):

```
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_GCS_BUCKET=p7-mlflow-artifacts
GCP_PROJECT_ID=<your-gcp-project>
RUN_ID=<optional: MLflow run to load in the API>
MODEL_ARTIFACT_PATH=model
MODEL_FILENAME=Bidirectional_GRU_glove_model.h5
TOKENIZER_FILENAME=tokenizer.pkl
```

> Replace `RUN_ID` with the run ID you want to promote to production. If you leverage the Model Registry you can set `MLFLOW_MODEL_URI=models:/sentiment-gru/Production` instead and omit `RUN_ID`.

### 3. Start the MLflow server
- Make sure `gcp-key.json` points to the correct service account key.
- Run `docker compose up mlflow db` to start the tracking server connected to Postgres and GCS.
- Open `http://localhost:5000` to verify the MLflow UI and confirm that runs log their artifacts to the bucket.

### 4. Run your trainings
- In notebooks or scripts call `mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))`.
- Execute your training runs; ensure that artifacts (model, tokenizer, etc.) are logged under `artifact_path="model"` so the API can retrieve them.
- Optional: register model versions in the Model Registry (`mlflow.register_model(...)`).
- For headless jobs, use the CLI helper: `python training/train_neural_model.py --run-name Bidirectional_GRU_glove_training`.
- Export the Google credentials before launching a run so GCS uploads succeed:
  ```
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-key.json
  export GOOGLE_CLOUD_PROJECT=<your-gcp-project>
  ```
  (Add the same entries to your `.env` to make them persistent.)

### 5. API and model loading
- The FastAPI service automatically downloads artifacts based on the configured `RUN_ID` or `MLFLOW_MODEL_URI`.
- On startup the container places the files under `backend/app/model/` if they are missing.
- The service keeps the existing endpoints (`/predict`, `/feedback`, `/feedback/stats`) while avoiding bundling the weights into the Docker image.

### 6. CI/CD and secrets
- In GitHub Actions (or any pipeline) store `MLFLOW_TRACKING_URI`, `MLFLOW_GCS_BUCKET`, `GCP_PROJECT_ID`, and the service account key in the secrets store.
- Mount the key at runtime (e.g. `actions/upload-artifact`, `gcloud auth activate-service-account`, or direct file write).
- Training or promotion jobs can then run in the cloud, log results, and promote models.

### 7. Good practices
- Version your training scripts (`training/train_*.py`) so each run can be replayed.
- Periodically clean up stale artifacts or configure lifecycle policies on the bucket.
- Monitor GCS and Postgres quotas and set up backups if required.
