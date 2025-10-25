import sys
import os
import pytest

# Patch sys.path to import front.py
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../app")
    )
)

# Change import front to use the correct path for test discovery
import importlib.util

front_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../front.py")
)
spec = importlib.util.spec_from_file_location("front", front_path)
front = importlib.util.module_from_spec(spec)
spec.loader.exec_module(front)

# This file contains unit tests for the Streamlit front-end logic (specifically, get_sentiment_prediction).
# These are unit tests, not integration tests, because they mock the requests and Streamlit API.

def test_get_sentiment_prediction_success(monkeypatch):
    class MockResponse:
        status_code = 200
        def json(self):
            return {
                "sentiment": "positive",
                "confidence": 0.95,
                "probabilities": {"positive": 0.95, "negative": 0.05}
            }
    def mock_post(*args, **kwargs):
        return MockResponse()
    monkeypatch.setattr(front.requests, "post", mock_post)
    result = front.get_sentiment_prediction("Great product!")
    assert result["sentiment"] == "positive"
    assert result["confidence"] == 0.95
    assert "positive" in result["probabilities"]

def test_get_sentiment_prediction_api_error(monkeypatch):
    class MockResponse:
        status_code = 422
        def json(self):
            return {"detail": "A text should be provided for sentiment analysis."}
        text = "A text should be provided for sentiment analysis."
    def mock_post(*args, **kwargs):
        return MockResponse()
    errors = []
    monkeypatch.setattr(front.requests, "post", mock_post)
    monkeypatch.setattr(front.st, "error", lambda msg: errors.append(msg))
    result = front.get_sentiment_prediction("")
    assert result is None
    assert any("A text should be provided for sentiment analysis." in e for e in errors)

def test_get_sentiment_prediction_connection_error(monkeypatch):
    def mock_post(*args, **kwargs):
        raise front.requests.exceptions.RequestException("Connection failed")
    errors = []
    monkeypatch.setattr(front.requests, "post", mock_post)
    monkeypatch.setattr(front.st, "error", lambda msg: errors.append(msg))
    result = front.get_sentiment_prediction("Test")
    assert result is None
    assert any("Error connecting to API" in e for e in errors)
