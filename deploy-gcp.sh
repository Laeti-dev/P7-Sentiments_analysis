#!/bin/bash

# Exit on error
set -e

# Configuration
export PROJECT_ID=$(gcloud config get-value project)
export REGION=europe-west1
export REPO_NAME=sentiment-repo

echo "📋 Project: $PROJECT_ID"
echo "🌍 Region: $REGION"
echo "📦 Repository: $REPO_NAME"
echo ""

# Build images
echo "🏗️  Building backend image..."
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/backend:latest ./backend

echo "🏗️  Building frontend image..."
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/frontend:latest ./frontend

# Push images
echo "⬆️  Pushing backend image..."
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/backend:latest

echo "⬆️  Pushing frontend image..."
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/frontend:latest

# Deploy backend
echo "🚀 Deploying backend..."
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
    --max-instances=10 \
    --quiet

# Get backend URL
export BACKEND_URL=$(gcloud run services describe sentiment-backend \
    --region=$REGION \
    --format='value(status.url)')

# Deploy frontend
echo "🚀 Deploying frontend..."
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
    --set-env-vars="API_URL=${BACKEND_URL}/predict" \
    --quiet

# Get frontend URL
export FRONTEND_URL=$(gcloud run services describe sentiment-frontend \
    --region=$REGION \
    --format='value(status.url)')

echo ""
echo "✅ Deployment complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 Backend URL:  $BACKEND_URL"
echo "📱 Frontend URL: $FRONTEND_URL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Open your app: $FRONTEND_URL"
