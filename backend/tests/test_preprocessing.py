import pytest
from app.main import preprocess_text


class TestPreprocessing:
    """Test text preprocessing function"""

    def test_lowercase_conversion(self):
        """Test text is converted to lowercase"""
        text = "HELLO WORLD"
        result = preprocess_text(text)
        assert result == "hello world"

    def test_url_replacement(self):
        """Test URLs are replaced with URL token"""
        text = "Check this out http://example.com"
        result = preprocess_text(text)
        assert "URL" in result
        assert "http://" not in result

    def test_mention_replacement(self):
        """Test @mentions are replaced with USER token"""
        text = "Hey @john how are you?"
        result = preprocess_text(text)
        assert "USER" in result
        assert "@john" not in result

    def test_hashtag_preservation(self):
        """Test hashtags are preserved"""
        text = "#python is awesome"
        result = preprocess_text(text)
        assert "#python" in result

    def test_whitespace_normalization(self):
        """Test multiple spaces are normalized"""
        text = "Hello    world   test"
        result = preprocess_text(text)
        assert "  " not in result

    def test_strip_whitespace(self):
        """Test leading/trailing whitespace is removed"""
        text = "  hello world  "
        result = preprocess_text(text)
        assert result == "hello world"

    def test_non_string_input(self):
        """Test handling of non-string input"""
        result = preprocess_text(None)
        assert result == ""

        result = preprocess_text(123)
        assert result == ""
