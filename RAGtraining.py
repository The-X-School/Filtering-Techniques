# RAGtraining.py
"""
Train a RAG topic classifier using DistilBERT.
- Loads RAG (positive) and non-RAG (negative) datasets
- Preprocesses and tokenizes data
- Trains a sequence classification model
- Saves the trained model and tokenizer
"""

import argparse
from tqdm import tqdm
import random
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 2. Parse command-line arguments for flexibility
def parse_args():
    parser = argparse.ArgumentParser(description="Train a RAG topic classifier using DistilBERT.")
    parser.add_argument('--max_samples_per_class', type=int, default=5000, help='Max samples per class (RAG/non-RAG)')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and eval')
    parser.add_argument('--output_dir', type=str, default='./rag_classifier_model', help='Directory to save model/tokenizer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--plot', action='store_true', help='Plot training loss after training')
    return parser.parse_args()

# 3. Load datasets
def load_rag_and_negative_datasets():
    rag_ds = load_dataset("neural-bridge/rag-dataset-12000", split="train")
    # Load wikitext-2-raw-v1.json for negatives
    import json
    with open("data/wikitext-2-raw-v1/wikitext-2-raw-v1.json", "r", encoding="utf-8") as f:
        wiki_data = json.load(f)
    wiki_texts = [ex["text"] for ex in wiki_data["instances"] if ex["text"].strip()]
    return rag_ds, wiki_texts

# 4. Preprocess and label data, with progress bar
def preprocess_and_label(rag_ds, wiki_texts, max_samples_per_class=5000):
    # RAG positives: use only the context field (text-only)
    rag_samples = rag_ds.select(range(min(len(rag_ds), max_samples_per_class)))
    rag_texts = [ex["context"] for ex in tqdm(rag_samples, desc="Preparing RAG positives") if ex["context"].strip()]
    rag_labels = [1] * len(rag_texts)
    # Non-RAG negatives: use plain text from wikitext-2-raw-v1
    wiki_texts = wiki_texts[:max_samples_per_class]
    wiki_labels = [0] * len(wiki_texts)
    texts = rag_texts + wiki_texts
    labels = rag_labels + wiki_labels
    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)

# 5. Tokenize
def tokenize(texts, labels, tokenizer, max_length=384):
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
    encodings['labels'] = list(labels)
    return Dataset.from_dict(encodings)

# 6. Compute metrics (accuracy, precision, recall, F1)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 7. Main training routine
def main():
    args = parse_args()
    set_seed(args.seed)
    rag_ds, wiki_texts = load_rag_and_negative_datasets()
    texts, labels = preprocess_and_label(rag_ds, wiki_texts, max_samples_per_class=args.max_samples_per_class)
    # Use Llama 3.2 1B tokenizer and model
    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # NOTE: Llama 3.2 1B may not have a sequence classification head by default. If not, you may need to use a text generation approach or add a classification head.
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, trust_remote_code=True)
    dataset = tokenize(texts, labels, tokenizer)
    dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_ds, eval_ds = dataset['train'], dataset['test']
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    train_output = trainer.train()
    # Save model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")

    # Plot training loss if requested
    if args.plot:
        if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
            losses = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
            plt.plot(losses)
            plt.xlabel('Logging Step')
            plt.ylabel('Training Loss')
            plt.title('Training Loss Curve')
            plt.show()
        else:
            print("No training loss history found to plot.")

if __name__ == "__main__":
    main() 