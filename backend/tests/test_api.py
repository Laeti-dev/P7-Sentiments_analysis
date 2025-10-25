import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


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
    """Test model loading and validation"""

    def test_model_loaded(self):
        """Verify model is loaded on startup"""
        from app.main import model, tokenizer
        assert model is not None, "Model should be loaded"
        assert tokenizer is not None, "Tokenizer should be loaded"
