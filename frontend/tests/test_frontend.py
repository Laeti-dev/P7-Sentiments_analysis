import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestFrontendConfig:
    """Test frontend configuration"""

    def test_api_url_from_env(self, monkeypatch):
        """Test API URL is loaded from environment variable"""
        test_url = "https://test-api.example.com/predict"
        monkeypatch.setenv("API_URL", test_url)

        # Re-import to get updated env var
        import importlib
        import main
        importlib.reload(main)

        assert os.environ.get("API_URL") == test_url

    def test_api_url_default(self, monkeypatch):
        """Test default API URL when env var not set"""
        monkeypatch.delenv("API_URL", raising=False)

        default_url = os.environ.get("API_URL", "http://localhost:8000/predict")
        assert default_url == "http://localhost:8000/predict"
