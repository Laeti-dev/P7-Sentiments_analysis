# 🚀 Deploy FastAPI \& Streamlit (Docker) on Google Cloud Platform (GCP)


***

## 1. **Pre-requisites**

- macOS (M1/M2/M3), Intel, Linux, or Windows
- Google Cloud SDK (`gcloud`)
- Docker (with Buildx enabled)
- Google Cloud account (free tier available)

***

## 2. **Project Structure Example**

```
project-root/
├── backend/
│   ├── app/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
└── docker-compose.yml
```


***

## 3. **Create a New Google Cloud Project**

> *Skip this section if you already have a GCP project and want to use it!*

```bash
# Login to Google Cloud
gcloud auth login

# Create a new project (choose a unique PROJECT_ID, only lowercase/numbers/- allowed)
export PROJECT_ID="your-unique-project-id"
gcloud projects create $PROJECT_ID --name="My Sentiment Analysis App"

# Set your new project as the active one
gcloud config set project $PROJECT_ID

# Link the project to a billing account (replace with your BILLING_ACCOUNT_ID)
gcloud billing accounts list
gcloud billing projects link $PROJECT_ID --billing-account=YOUR-BILLING-ACCOUNT-ID
```


***

## 4. **Enable Required GCP Services**

```bash
gcloud services enable artifactregistry.googleapis.com run.googleapis.com cloudbuild.googleapis.com
```


***

## 5. **Create Artifact Registry Docker Repository**

```bash
# Choose your region (change as needed)
export REGION="europe-west1"
export REPO_NAME="sentiments-analysis-repo"

gcloud artifacts repositories create $REPO_NAME \
  --repository-format=docker \
  --location=$REGION \
  --description="Sentiment Analysis Docker Repo"

# Configure Docker to authenticate with GCP
gcloud auth configure-docker $REGION-docker.pkg.dev
```


***

## 6. **Build and Push Docker Images (Apple Silicon: Use linux/amd64!)**

```bash
# Backend
docker buildx build --platform linux/amd64 \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/backend:latest \
  --push ./backend

# Frontend
docker buildx build --platform linux/amd64 \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/frontend:latest \
  --push ./frontend
```


***

## 7. **Deploy to Cloud Run**

```bash
# Deploy backend
gcloud run deploy sentiment-backend \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/backend:latest \
  --platform=managed \
  --region=$REGION \
  --allow-unauthenticated \
  --port=8000 \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --min-instances=0 \
  --max-instances=10

# Get backend URL for frontend config
export BACKEND_URL=$(gcloud run services describe sentiment-backend \
  --region=$REGION --format='value(status.url)')

# Deploy frontend (set API_URL to backend)
gcloud run deploy sentiment-frontend \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/frontend:latest \
  --platform=managed \
  --region=$REGION \
  --allow-unauthenticated \
  --port=8501 \
  --memory=1Gi \
  --cpu=1 \
  --timeout=300 \
  --min-instances=0 \
  --max-instances=5 \
  --set-env-vars="API_URL=${BACKEND_URL}/predict"
```


***

## 8. **Access \& Test Your App**

- **Find frontend URL:**

```bash
gcloud run services describe sentiment-frontend --region=$REGION --format="value(status.url)"
```

- **Backend URL:**

```bash
gcloud run services describe sentiment-backend --region=$REGION --format="value(status.url)"
```

- **Open frontend in browser.** Start using your app!

***

## 9. **Monitor \& Check Logs**

```bash
gcloud run logs read sentiment-backend --region=$REGION --limit=50
gcloud run logs read sentiment-frontend --region=$REGION --limit=50
```


***

## 10. **Cost \& Cleanup**

- **Scale-to-zero by default (free for minimal use).**
- To delete:

```bash
gcloud run services delete sentiment-backend --region=$REGION
gcloud run services delete sentiment-frontend --region=$REGION
gcloud artifacts repositories delete $REPO_NAME --location=$REGION
```

- **Delete the project** (optional, if you want to remove everything!)

```bash
gcloud projects delete $PROJECT_ID
```


***

## **Tips**

- Always use `--platform linux/amd64` for Cloud Run compatibility!
- Tag images for production (avoid always using `:latest`).
- Set up billing alerts in GCP Billing Console.
- Update: Rebuild + push new image + redeploy.
- Save your service URLs for reference.

***

