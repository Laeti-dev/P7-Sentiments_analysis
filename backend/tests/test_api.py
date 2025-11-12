import csv
import uuid

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def override_feedback_storage(monkeypatch, tmp_path):
    """Use a temporary feedback storage for each test run."""
    feedback_dir = tmp_path / "feedback"
    feedback_dir.mkdir()
    feedback_file = feedback_dir / "feedback_log.csv"

    monkeypatch.setattr(app_main, "FEEDBACK_DIR", feedback_dir)
    monkeypatch.setattr(app_main, "FEEDBACK_FILE", feedback_file)

    with feedback_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "prediction_id",
                "timestamp",
                "text",
                "predicted_sentiment",
                "confidence",
                "is_correct",
                "actual_sentiment",
                "user_comments",
            ]
        )

    yield


class TestHealthEndpoint:
    """Test health check and root endpoint"""

    def test_read_root(self):
        """Test root endpoint returns welcome message"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert "Welcome" in response.json()["message"]


class TestPredictEndpoint:
    """Test sentiment prediction endpoint"""

    def test_predict_positive_sentiment(self):
        """Test prediction with positive text"""
        response = client.post(
            "/predict",
            json={"text": "I love this movie! It was amazing and fantastic!"}
        )
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "sentiment" in data
        assert "confidence" in data
        assert "probabilities" in data

        # Check data types
        assert isinstance(data["sentiment"], str)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["probabilities"], dict)

        # Check probabilities
        assert "positive" in data["probabilities"]
        assert "negative" in data["probabilities"]
        assert 0 <= data["confidence"] <= 1

    def test_predict_negative_sentiment(self):
        """Test prediction with negative text"""
        response = client.post(
            "/predict",
            json={"text": "This is terrible and awful. I hate it!"}
        )
        assert response.status_code == 200
        data = response.json()

        assert "sentiment" in data
        assert data["sentiment"] in ["positive", "negative"]
        assert 0 <= data["confidence"] <= 1

    def test_predict_empty_text(self):
        """Test prediction with empty text"""
        response = client.post(
            "/predict",
            json={"text": ""}
        )
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_predict_missing_text(self):
        """Test prediction without text field"""
        response = client.post(
            "/predict",
            json={}
        )
        assert response.status_code == 422

    def test_predict_whitespace_only(self):
        """Test prediction with whitespace only"""
        response = client.post(
            "/predict",
            json={"text": "   "}
        )
        assert response.status_code == 422

    def test_predict_special_characters(self):
        """Test prediction with special characters and URLs"""
        response = client.post(
            "/predict",
            json={"text": "Check out this link http://example.com @user #hashtag"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data

    def test_predict_long_text(self):
        """Test prediction with very long text"""
        long_text = "This is great! " * 100
        response = client.post(
            "/predict",
            json={"text": long_text}
        )
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data


class TestModelValidation:
    """Test model loading, feedback and statistics endpoints"""

    def test_model_loaded(self):
        """Verify model is loaded on startup"""
        assert app_main.model is not None, "Model should be loaded"
        assert app_main.tokenizer is not None, "Tokenizer should be loaded"

    def test_predict_returns_500_when_model_missing(self, monkeypatch):
        """Ensure API fails gracefully when model is not available"""
        original_model = app_main.model
        original_tokenizer = app_main.tokenizer
        monkeypatch.setattr(app_main, "model", None)
        monkeypatch.setattr(app_main, "tokenizer", original_tokenizer)

        response = client.post(
            "/predict",
            json={"text": "Any input should fail when model is missing"},
        )

        assert response.status_code == 500
        assert "Model or tokenizer not loaded" in response.json()["detail"]

        monkeypatch.setattr(app_main, "model", original_model)

    def test_submit_feedback_success(self):
        """Feedback endpoint should append a row and return metadata"""
        payload = {
            "prediction_id": str(uuid.uuid4()),
            "is_correct": False,
            "actual_sentiment": "negative",
            "comments": "Model predicted positive on a clearly negative tweet.",
        }

        response = client.post("/feedback", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["feedback_id"] == payload["prediction_id"]

        with app_main.FEEDBACK_FILE.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert any(row["prediction_id"] == payload["prediction_id"] for row in rows)

    def test_submit_feedback_validation_error(self):
        """Missing required fields should return 422"""
        response = client.post(
            "/feedback",
            json={"comments": "Missing mandatory fields"},
        )

        assert response.status_code == 422

    def test_feedback_stats_empty(self):
        """Stats endpoint should handle empty feedback gracefully"""
        response = client.get("/feedback/stats")
        payload = response.json()

        assert response.status_code == 200
        assert payload["total_feedback"] == 0
        assert payload["accuracy"] == 0.0

    def test_feedback_stats_counts(self):
        """Stats endpoint should aggregate feedback entries"""
        first = {
            "prediction_id": str(uuid.uuid4()),
            "is_correct": True,
            "actual_sentiment": "positive",
        }
        second = {
            "prediction_id": str(uuid.uuid4()),
            "is_correct": False,
            "actual_sentiment": "negative",
            "comments": "Wrong sentiment",
        }

        client.post("/feedback", json=first)
        client.post("/feedback", json=second)

        response = client.get("/feedback/stats")
        payload = response.json()

        assert payload["total_feedback"] == 2
        assert payload["correct_predictions"] == 1
        assert payload["incorrect_predictions"] == 1
        assert payload["accuracy"] == 50.0
