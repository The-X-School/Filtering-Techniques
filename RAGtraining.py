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
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

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
    triviaqa = load_dataset("trivia_qa", "rc", split="train")
    return rag_ds, triviaqa

# 4. Preprocess and label data, with progress bar
def preprocess_and_label(rag_ds, triviaqa, max_samples_per_class=5000):
    rag_samples = rag_ds.select(range(min(len(rag_ds), max_samples_per_class)))
    trivia_samples = triviaqa.select(range(min(len(triviaqa), max_samples_per_class)))
    # Progress bar for RAG
    rag_texts = []
    for ex in tqdm(rag_samples, desc="Preparing RAG positives"):
        rag_texts.append(f"Context: {ex['context']} Question: {ex['question']} Answer: {ex['answer']}")
    rag_labels = [1] * len(rag_texts)
    # Progress bar for negatives
    trivia_texts = []
    for ex in tqdm(trivia_samples, desc="Preparing non-RAG negatives"):
        context = ""
        search_results = ex.get('search_results', [])
        if isinstance(search_results, list) and len(search_results) > 0:
            first_result = search_results[0]
            if isinstance(first_result, dict):
                context = first_result.get('search_context', "")
        trivia_texts.append(f"Context: {context} Question: {ex.get('question', '')} Answer: {ex.get('answer', '')}")
    trivia_labels = [0] * len(trivia_texts)
    texts = rag_texts + trivia_texts
    labels = rag_labels + trivia_labels
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
    rag_ds, triviaqa = load_rag_and_negative_datasets()
    texts, labels = preprocess_and_label(rag_ds, triviaqa, max_samples_per_class=args.max_samples_per_class)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    dataset = tokenize(texts, labels, tokenizer)
    dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_ds, eval_ds = dataset['train'], dataset['test']
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
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
        # load_best_model_at_end=True,  # This would reload the best model (on eval loss) at the end of training, but requires matching save/eval strategies.
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