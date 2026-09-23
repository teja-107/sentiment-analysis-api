"""
Loads the fine-tuned model once and exposes a simple predict() function.
Imported by app/main.py — model is loaded at API startup, not per-request.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "./model"
LABELS = ["negative", "positive"]


class SentimentModel:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1).squeeze()

        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

        return {
            "label": LABELS[pred_idx],
            "confidence": round(confidence, 4),
        }
