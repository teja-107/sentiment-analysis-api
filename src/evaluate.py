"""
Loads the fine-tuned model and prints a full evaluation report
(accuracy, precision, recall, F1, confusion matrix) on the IMDB test set.

Run: python src/evaluate.py
"""

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix

MODEL_DIR = r"C:\Users\HP\Downloads\sentiment-analysis-api\sentiment-analysis-api\model"


def main():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("Loading test data...")
    dataset = load_dataset("stanfordnlp/imdb")
    test_dataset = dataset["test"].shuffle(seed=42).select(range(1000))

    all_preds = []
    all_labels = []

    print("Running inference...")
    with torch.no_grad():
        for example in test_dataset:
            inputs = tokenizer(
                example["text"],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            ).to(device)

            logits = model(**inputs).logits
            pred = int(np.argmax(logits.cpu().numpy(), axis=-1)[0])

            all_preds.append(pred)
            all_labels.append(example["label"])

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=["negative", "positive"]))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
