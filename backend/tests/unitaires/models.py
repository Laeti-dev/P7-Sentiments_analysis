import pytest
from pydantic import ValidationError
from app.backend.main import SentimentRequest, SentimentResponse, FeedbackRequest

def test_sentiment_request_valid():
    req = SentimentRequest(text="Hello world")
    assert req.text == "Hello world"

def test_sentiment_request_invalid():
    with pytest.raises(ValidationError):
        SentimentRequest()

def test_sentiment_response_valid():
    resp = SentimentResponse(
        sentiment="positive",
        confidence=0.9,
        probabilities={"positive": 0.9, "negative": 0.1}
    )
    assert resp.sentiment == "positive"
    assert resp.confidence == 0.9
    assert resp.probabilities["positive"] == 0.9

def test_feedback_request_valid():
    fb = FeedbackRequest(
        text_text="A test",
        prediction="positive",
        confidence=0.8,
        is_correct=True,
        corrected_sentiment="positive",
        comments="Looks good"
    )
    assert fb.text_text == "A test"
    assert fb.prediction == "positive"
    assert fb.confidence == 0.8
    assert fb.is_correct is True
    assert fb.corrected_sentiment == "positive"
    assert fb.comments == "Looks good"

def test_feedback_request_extra_field():
    with pytest.raises(ValidationError):
        FeedbackRequest(
            text_text="A test",
            prediction="positive",
            confidence=0.8,
            is_correct=True,
            extra_field="not allowed"
        )
