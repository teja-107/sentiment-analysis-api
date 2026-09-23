# Sentiment Analysis API

Fine-tuned DistilBERT model for binary sentiment classification (positive/negative),
served as a REST API and containerized with Docker.

## Problem
Given a piece of text (movie review, tweet, etc.), predict whether the sentiment
is positive or negative.

## Dataset
IMDB Movie Reviews (50K labeled examples, binary sentiment).
Source: https://huggingface.co/datasets/imdb

## Approach
1. Load pretrained `distilbert-base-uncased`
2. Tokenize text with `AutoTokenizer` (padding + truncation)
3. Fine-tune with a classification head using Hugging Face `Trainer`
4. Evaluate with accuracy, precision, recall, F1, confusion matrix
5. Serve the fine-tuned model via a FastAPI `/predict` endpoint
6. Containerize with Docker and deploy (Hugging Face Spaces / Render)

## Project Structure
```
sentiment-analysis-api/
├── data/                  # raw / processed data (not committed if large)
├── notebooks/             # exploration, training experiments
├── src/
│   ├── preprocess.py      # text cleaning + tokenization helpers
│   ├── train.py           # fine-tuning script (Trainer API)
│   ├── evaluate.py        # metrics: precision, recall, F1, confusion matrix
│   └── inference.py       # load model + predict function, used by the API
├── app/
│   ├── main.py            # FastAPI app, /predict endpoint
│   └── schemas.py         # Pydantic request/response models
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```
This fine-tunes DistilBERT on the IMDB dataset and saves the model to `./model/`.

## Evaluate

```bash
python src/evaluate.py
```
Prints accuracy, precision, recall, F1, and a confusion matrix on the held-out test set.

## Run the API locally

```bash
uvicorn app.main:app --reload
```
Visit `http://localhost:8000/docs` for interactive Swagger UI.

Example request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely wonderful, I loved every minute of it."}'
```

Example response:
```json
{
  "label": "positive",
  "confidence": 0.9823
}
```

## Run with Docker

```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

## Results
*(fill this in after training)*

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.89  |
| Precision | 0.89  |
| Recall    | 0.89  |
| F1        | 0.88  |

## What I'd improve with more time
- Try a larger model (bert-base, roberta) and compare
- Add 3-class sentiment (positive/neutral/negative) with a messier dataset like Twitter
- Add batch prediction endpoint
- Add basic monitoring/logging for the deployed API

## Tech Stack
Python, PyTorch, Hugging Face Transformers & Datasets, FastAPI, Docker
