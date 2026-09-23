"""
FastAPI app serving the fine-tuned sentiment analysis model.

Run locally:  uvicorn app.main:app --reload
Docs:         http://localhost:8000/docs
"""

import sys
import os

# allow importing from src/ when running from project root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException
from .schemas import PredictRequest, PredictResponse
from inference import SentimentModel

app = FastAPI(
    title="Sentiment Analysis API",
    description="Fine-tuned DistilBERT model for binary sentiment classification.",
    version="1.0.0",
)

# Loaded once at startup — NOT reloaded on every request.
model = None


@app.on_event("startup")
def load_model():
    global model
    model = SentimentModel()


@app.get("/")
def root():
    return {"status": "ok", "message": "Sentiment Analysis API is running."}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    result = model.predict(request.text)
    return result
