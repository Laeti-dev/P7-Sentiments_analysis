from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import pickle
import os
import re
from datetime import datetime
import csv
from pathlib import Path
import numpy as np
from typing import Dict, Optional
import tensorflow as tf
from contextlib import asynccontextmanager
import uuid
from dotenv import load_dotenv
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

load_dotenv()

# Configuration de MLflow
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
run_id = os.getenv("RUN_ID")
APPINSIGHTS_KEY = os.getenv("APPINSIGHTS_KEY")

# Set up logger (do this once, e.g., in your main or startup)
logger = logging.getLogger("feedback_logger")
if APPINSIGHTS_KEY:
    logger.addHandler(AzureLogHandler(connection_string=f'InstrumentationKey={APPINSIGHTS_KEY}'))
logger.setLevel(logging.INFO)

# Paramètres par défaut si non spécifiés dans le modèle
MAX_SEQUENCE_LENGTH = 100

# Répertoire local pour sauvegarder les artefacts du modèle
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Feedback storage path
FEEDBACK_DIR = BASE_DIR / "feedback"
FEEDBACK_DIR.mkdir(exist_ok=True)
FEEDBACK_FILE = FEEDBACK_DIR / "feedback_log.csv"

# Initialize feedback CSV if it doesn't exist
if not FEEDBACK_FILE.exists():
    with open(FEEDBACK_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'prediction_id',
            'timestamp',
            'text',
            'predicted_sentiment',
            'confidence',
            'is_correct',
            'actual_sentiment',
            'user_comments'
        ])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    global model, tokenizer
    if model is None or tokenizer is None:
        load_model_and_tokenizer()
    yield
    # (Optional) Shutdown code

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment from text using a pre-trained model",
    version="1.0.0",
    lifespan=lifespan
)

# Define request model
class SentimentRequest(BaseModel):
    text: str

# Define response model
class SentimentResponse(BaseModel):
    prediction_id: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

class FeedbackRequest(BaseModel):
    prediction_id: str
    is_correct: bool
    actual_sentiment: Optional[str] = None  # If prediction was wrong
    comments: Optional[str] = ""
    model_config = ConfigDict(extra="forbid")

# Load the pre-trained model and tokenizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "Bidirectional_GRU_glove_model.h5")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "model", "tokenizer.pkl")

model = None  # --- IGNORE ---
tokenizer = None

def preprocess_text(text:str)-> str:
    """Preprocess input text for prediction"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'#(\w+)', r'#\1', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    text = re.sub(r"http\S+", "URL", text)
    text = re.sub(r"@\w+", "USER", text)
    return text

def load_model_and_tokenizer():
    """Load model and tokenizer from specified paths"""
    global model, tokenizer
    try:
        print(f"Looking for model at: {MODEL_PATH}")
        print(f"Looking for tokenizer at: {TOKENIZER_PATH}")
        for path in [MODEL_PATH, TOKENIZER_PATH]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        try:
            model = tf.keras.models.load_model(str(MODEL_PATH))
        except Exception as e:
            print(f"Model loading error: {e}")
            raise RuntimeError(f"Model loading error: {e}")

        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)

        print("Model and tokenizer loaded successfully")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "preprocess_text": preprocess_text
        }
    except Exception as e:
        model = None
        tokenizer = None
        print(f"Error loading model or tokenizer: {str(e)}")

# Load model and tokenizer at startup
load_model_and_tokenizer()

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Welcome to the Sentiment Analysis API"}

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment from input text
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model or tokenizer not loaded")

    try:
        # Validation
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=422, detail="Text is required")

        text = request.text

        # Preprocess and predict
        sequences = tokenizer.texts_to_sequences([preprocess_text(text)])
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding='post'
        )

        prediction = model.predict(padded_sequences, verbose=0)
        predicted_class = int(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0])

        sentiment_labels = {0: "negative", 1: "positive"}
        sentiment = sentiment_labels[predicted_class]

        probs_dict = {
            "negative": float(1 - prediction[0][0]),
            "positive": float(prediction[0][0])
        }

        # Generate unique prediction ID
        prediction_id = str(uuid.uuid4())

        return SentimentResponse(
            prediction_id=prediction_id,
            sentiment=sentiment,
            confidence=confidence,
            probabilities=probs_dict
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    """
    Submit user feedback about a prediction
    """
    print(f"Feedback received: {feedback.dict()}")  # Debugging log
    try:
        timestamp = datetime.now().isoformat()

        # Write feedback to CSV
        with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                feedback.prediction_id,
                timestamp,
                "",  # Text will be logged separately for privacy
                "",  # Predicted sentiment (we don't have it here)
                "",  # Confidence
                feedback.is_correct,
                feedback.actual_sentiment or "",
                feedback.comments or ""
            ])

        if feedback.is_correct is False:
            logger.warning(
                "Model misprediction feedback",
                extra={
                    "custom_dimensions": {
                        "prediction_id": feedback.prediction_id,
                        "user_comments": feedback.comments,
                        "actual_sentiment": feedback.actual_sentiment,
                    }
                }
            )

        return {
            "message": "Feedback received successfully",
            "feedback_id": feedback.prediction_id,
            "timestamp": timestamp
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")


@app.get("/feedback/stats")
def get_feedback_stats():
    """
    Get aggregated feedback statistics
    """
    try:
        if not FEEDBACK_FILE.exists():
            return {
                "total_feedback": 0,
                "correct_predictions": 0,
                "incorrect_predictions": 0,
                "accuracy": 0.0
            }

        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            feedbacks = list(reader)

        total = len(feedbacks)
        correct = sum(1 for f in feedbacks if f['is_correct'].lower() == 'true')
        incorrect = total - correct
        accuracy = (correct / total * 100) if total > 0 else 0


        return {
            "total_feedback": total,
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "accuracy": round(accuracy, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
