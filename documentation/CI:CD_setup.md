# 🔄 CI/CD with GitHub Actions for Cloud Run Deployment

## **What You'll Achieve**

- Auto-build (`linux/amd64`) and push Docker images on every push to `main`
- Deploy both backend (FastAPI) and frontend (Streamlit) to Cloud Run
- Secure service with GCP service account and GitHub Secrets

***

## **1. Create a GCP Service Account \& Permissions**

```bash
# Set vars
export PROJECT_ID="your-gcp-project-id"
export SERVICE_ACCOUNT_NAME="github-actions-deployer"

# Create the service account
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
  --display-name="GitHub Actions Deployer" \
  --project=$PROJECT_ID

# Grant permissions
export SA_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" --role="roles/storage.admin"

# Generate a new service account key, save as gcp-key.json
gcloud iam service-accounts keys create gcp-key.json \
  --iam-account=$SA_EMAIL
```

Copy the content of `gcp-key.json` (you'll add to GitHub Secrets next).

***

## **2. Add GitHub Actions Secrets**

Go to **GitHub** ➔ Your repository ➔ **Settings** ➔ **Secrets and variables** ➔ **Actions**:

Add new secrets:


| Name | Value (example) |
| :-- | :-- |
| GCP_PROJECT_ID | your-gcp-project-id |
| GCP_REGION | europe-west1 |
| GCP_REPO_NAME | sentiments-analysis-repo |
| GCP_SA_KEY | *Paste entire gcp-key.json content* |


***

## **3. Add Workflow File to Your Repo**

Create:

```
.github/
└── workflows/
    └── deploy.yml
```

Paste the following into `.github/workflows/deploy.yml`:

```yaml
name: Build and Deploy to Cloud Run

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: ${{ secrets.GCP_REGION }}
  REPO_NAME: ${{ secrets.GCP_REPO_NAME }}

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Google Cloud SDK
        uses: google-github-actions/setup-gcloud@v2
        with:
          service_account_key: ${{ secrets.GCP_SA_KEY }}
          project_id: ${{ secrets.GCP_PROJECT_ID }}

      - name: Configure Docker for Artifact Registry
        run: gcloud auth configure-docker ${{ env.REGION }}-docker.pkg.dev

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build and Push Backend Docker Image
        uses: docker/build-push-action@v5
        with:
          context: ./backend
          platforms: linux/amd64
          push: true
          tags: |
            ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.REPO_NAME }}/backend:latest

      - name: Build and Push Frontend Docker Image
        uses: docker/build-push-action@v5
        with:
          context: ./frontend
          platforms: linux/amd64
          push: true
          tags: |
            ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.REPO_NAME }}/frontend:latest

      - name: Deploy Backend to Cloud Run
        id: deploy-backend
        run: |
          gcloud run deploy sentiment-backend \
            --image=${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.REPO_NAME }}/backend:latest \
            --platform=managed \
            --region=${{ env.REGION }} \
            --allow-unauthenticated \
            --port=8000 \
            --memory=2Gi \
            --cpu=2 \
            --timeout=300 \
            --min-instances=0 \
            --max-instances=10 \
            --quiet

      - name: Get Backend URL
        id: backend-url
        run: |
          BACKEND_URL=$(gcloud run services describe sentiment-backend \
            --region=${{ env.REGION }} --format='value(status.url)')
          echo "url=$BACKEND_URL" >> $GITHUB_OUTPUT

      - name: Deploy Frontend to Cloud Run
        run: |
          gcloud run deploy sentiment-frontend \
            --image=${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.REPO_NAME }}/frontend:latest \
            --platform=managed \
            --region=${{ env.REGION }} \
            --allow-unauthenticated \
            --port=8501 \
            --memory=1Gi \
            --cpu=1 \
            --timeout=300 \
            --min-instances=0 \
            --max-instances=5 \
            --set-env-vars="API_URL=${{ steps.backend-url.outputs.url }}/predict" \
            --quiet

      - name: Get Frontend URL
        run: |
          FRONTEND_URL=$(gcloud run services describe sentiment-frontend \
            --region=${{ env.REGION }} --format='value(status.url)')
          echo "Frontend: $FRONTEND_URL"
```


***

## **4. Commit \& Test Your Pipeline**

```bash
git add .github/workflows/deploy.yml
git commit -m "Add CI/CD pipeline for GCP Cloud Run"
git push
```

This will start your first CI/CD build and deploy.

***

## **5. Optional: Add a Status Badge to README**

Add this at the top of your `README.md`:

```markdown
[![Deploy to Cloud Run](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/deploy.yml/badge.svg)](https://github.com/YOUR-USERNAME/YOUR-REPO/actions/workflows/deploy.yml)
```


***

## **6. (Optional) Separate Workflows for Backend/Frontend**

If you want to deploy them individually when only one changes, use the alternate workflows provided in the detailed guide above.

***

## **🎉 You now have professional CI/CD!**

- Your Docker images are always built (`linux/amd64`) and uploaded automatically
- Your Cloud Run backend/frontend are always up to date with your latest `main` branch
- Creds are secret, and your repo shows deployment status with a badge!
