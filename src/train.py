"""
Fine-tunes DistilBERT for binary sentiment classification on IMDB reviews.

Run: python src/train.py
Saves the fine-tuned model + tokenizer to ./model/
"""

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from preprocess import get_tokenizer, tokenize_function, MODEL_NAME

OUTPUT_DIR = "./model"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    print("Loading dataset...")
    dataset = load_dataset("imdb")

    # For a first run, subsample so training finishes in a reasonable time
    # on a free-tier GPU. Remove/raise this once your pipeline works end-to-end.
    train_dataset = dataset["train"].shuffle(seed=42).select(range(5000))
    eval_dataset = dataset["test"].shuffle(seed=42).select(range(1000))

    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    print("Tokenizing...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Final evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
