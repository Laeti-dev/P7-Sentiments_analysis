import sys
import os
import pytest
from fastapi.testclient import TestClient

# Ensure the app directory is in sys.path for import
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../app")
    )
)

from api import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Sentiment Analysis API"}

def test_model_and_tokenizer_loaded():
    # Import model and tokenizer from the api module
    from api import model, tokenizer
    assert model is not None, "Model should be loaded"
    assert tokenizer is not None, "Tokenizer should be loaded"

def test_predict_positive_sentiment():
    response = client.post("/predict", json={"text": "I love this product, it is fantastic!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert 0.0 <= data["confidence"] <= 1.0
    assert "positive" in data["probabilities"]
    assert "negative" in data["probabilities"]

def test_predict_negative_sentiment():
    response = client.post("/predict", json={"text": "This is terrible, I hate it."})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "negative"
    assert 0.0 <= data["confidence"] <= 1.0
    assert "positive" in data["probabilities"]
    assert "negative" in data["probabilities"]

def test_predict_missing_text():
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Unprocessable Entity
    data = response.json()
    assert "detail" in data
    # FastAPI returns a list of validation errors for missing required fields
    assert isinstance(data["detail"], list)
    assert any(
        err.get("loc") == ["body", "text"] and err.get("msg") == "Field required"
        for err in data["detail"]
    )

def test_predict_empty_text():
    # Send a request with an empty 'text' field
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "A text should be provided for sentiment analysis."

def test_predict_model_or_tokenizer_missing(monkeypatch):
    # Simulate missing model and tokenizer
    from api import model, tokenizer

    monkeypatch.setattr("api.model", None)
    monkeypatch.setattr("api.tokenizer", None)

    response = client.post("/predict", json={"text": "Any text"})
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert data["detail"] == "Model or tokenizer not loaded"

def test_predict_internal_error(monkeypatch):
    # Simulate an error during model prediction
    from api import model

    def mock_predict(*args, **kwargs):
        raise RuntimeError("Simulated prediction error")

    monkeypatch.setattr(model, "predict", mock_predict)

    response = client.post("/predict", json={"text": "Any text"})
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "Prediction error: Simulated prediction error" in data["detail"]
