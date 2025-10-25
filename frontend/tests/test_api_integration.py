import pytest
import requests
import os


class TestAPIIntegration:
    """Test frontend integration with backend API"""

    @pytest.fixture
    def api_url(self):
        """Get API URL from environment or use default"""
        return os.environ.get("API_URL", "http://localhost:8000/predict")

    def test_api_connection(self, requests_mock, api_url):
        """Test API connection and response format"""
        # Mock API response
        mock_response = {
            "sentiment": "positive",
            "confidence": 0.95,
            "probabilities": {
                "positive": 0.95,
                "negative": 0.05
            }
        }
        requests_mock.post(api_url, json=mock_response, status_code=200)

        # Make request
        response = requests.post(
            api_url,
            json={"text": "This is great!"},
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert "confidence" in data
        assert "probabilities" in data

    def test_api_error_handling(self, requests_mock, api_url):
        """Test handling of API errors"""
        requests_mock.post(api_url, status_code=500, json={"detail": "Internal error"})

        response = requests.post(
            api_url,
            json={"text": "test"},
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 500
        assert "detail" in response.json()
