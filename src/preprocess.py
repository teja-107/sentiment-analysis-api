"""
Text preprocessing and tokenization helpers for the sentiment analysis project.
"""

import re
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256


def clean_text(text: str) -> str:
    """
    Light cleaning — enough for review-style text.
    If you switch to a messier dataset (e.g. Twitter), extend this
    to strip URLs, @handles, hashtags, etc.
    """
    text = text.strip()
    text = re.sub(r"<br\s*/?>", " ", text)      # IMDB reviews contain <br /> tags
    text = re.sub(r"\s+", " ", text)             # collapse whitespace
    return text


def get_tokenizer(model_name: str = MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples, tokenizer, max_length: int = MAX_LENGTH):
    """
    Use with datasets.map(..., batched=True) during training.
    """
    cleaned = [clean_text(t) for t in examples["text"]]
    return tokenizer(
        cleaned,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


if __name__ == "__main__":
    # Quick manual check — run this file directly to see what tokenization
    # actually produces. Useful sanity check before wiring it into training.
    tokenizer = get_tokenizer()
    sample = "This movie was absolutely wonderful, I loved every minute of it!"
    cleaned = clean_text(sample)
    encoded = tokenizer(cleaned, padding="max_length", truncation=True, max_length=32)

    print("Cleaned text:", cleaned)
    print("Token IDs:", encoded["input_ids"])
    print("Attention mask:", encoded["attention_mask"])
    print("Tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
