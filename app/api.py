from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
import pickle
import os
import re
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
from typing import Dict, Union
import tensorflow as tf
from contextlib import asynccontextmanager

load_dotenv()

# Configuration de MLflow
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
run_id = os.getenv("RUN_ID")

# Paramètres par défaut si non spécifiés dans le modèle
MAX_SEQUENCE_LENGTH = 100

# Répertoire local pour sauvegarder les artefacts du modèle
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)


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
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

class FeedbackRequest(BaseModel):
    text_text: str
    prediction: str
    confidence: float
    is_correct: bool
    corrected_sentiment: str = ""
    comments: str = ""
    model_config = ConfigDict(extra="forbid")

# Load the pre-trained model and tokenizer
MODEL_PATH = "models/Bidirectional_GRU_glove/Bidirectional_GRU_glove_model.h5"
TOKENIZER_PATH = "models/Bidirectional_GRU_glove/tokenizer.pkl"

model = None  # --- IGNORE ---
tokenizer = None

def preprocess_text(text:str)-> str:
    """Preprocess input text for prediction"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "URL", text)
    text = re.sub(r"@\w+", "USER", text)
    text = re.sub(r'#(\w+)', r'#\1', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text

def load_model_and_tokenizer():
    """Load model and tokenizer from specified paths"""
    global model, tokenizer
    try:
        for path in [MODEL_PATH, TOKENIZER_PATH]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        model = tf.keras.models.load_model(str(MODEL_PATH))

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

    Args:
        request: SentimentRequest containing text to analyze

    Returns:
        SentimentResponse with prediction results
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model or tokenizer not loaded")

    # Validation for missing or empty text
    if not hasattr(request, "text") or not isinstance(request.text, str) or not request.text.strip():
        raise HTTPException(status_code=422, detail="A text should be provided for sentiment analysis.")

    try:
        # Get the text from the request
        text = request.text

        # Tokenize and pad the text
        MAX_SEQUENCE_LENGTH = 100  # Adjust this based on your model's requirements
        sequences = tokenizer.texts_to_sequences([text])
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=MAX_SEQUENCE_LENGTH,
            padding='post'
        )

        # Make prediction
        try:
            prediction = model.predict(padded_sequences)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

        # Process the prediction (assuming binary classification)
        predicted_class = int(prediction[0][0] > 0.5)  # Threshold at 0.5
        confidence = float(prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0])

        # Map numerical class to sentiment label
        sentiment_labels = {0: "negative", 1: "positive"}
        sentiment = sentiment_labels[predicted_class]

        # Create probabilities dictionary
        probs_dict = {
            "negative": float(1 - prediction[0][0]),
            "positive": float(prediction[0][0])
        }

        return SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            probabilities=probs_dict
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
