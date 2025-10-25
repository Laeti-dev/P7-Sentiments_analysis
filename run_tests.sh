#!/bin/bash

# Test runner script for UV environment
set -e

echo "🚀 Starting Test Suite with UV Environment"
echo "==========================================="

# Ensure we're in project root
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Activate UV environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating UV environment..."
    uv venv
fi

echo "🔧 Activating UV environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing backend dependencies..."
uv pip install -r backend/requirements.txt
uv pip install -r backend/requirements-dev.txt

echo "📥 Installing frontend dependencies..."
uv pip install -r frontend/requirements.txt
uv pip install -r frontend/requirements-dev.txt

# Run backend tests
echo ""
echo "🧪 Running Backend Tests..."
echo "============================"
cd backend
pytest tests/ -v --cov=app --cov-report=term-missing

# Run frontend tests
echo ""
echo "🧪 Running Frontend Tests..."
echo "============================="
cd ../frontend
pytest tests/ -v

# Back to root
cd ..

echo ""
echo "✅ All tests completed successfully! 🎉"
echo "📊 Check htmlcov/index.html for coverage report"
