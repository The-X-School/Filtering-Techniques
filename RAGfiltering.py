import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2Tokenizer
import torch

OUTPUT_PATH = "RAGfiltered.json"
MAX_SAMPLES = 20000
BATCH_SIZE = 32
MODEL_PATH = "rag_classifier_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use GPT2Tokenizer for detokenization
raw_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=True)
# Use your classifier model for filtering
classifier_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

def classify_texts(texts):
    encodings = classifier_tokenizer(texts, truncation=True, padding=True, max_length=384, return_tensors="pt")
    encodings = {k: v.to(device) for k, v in encodings.items()}
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().tolist()
    return preds

def detokenize_batch(batch_data):
    results = []
    for item in batch_data:
        try:
            text = raw_tokenizer.decode(item["tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            results.append(text)
        except Exception:
            results.append("")
    return results

def detokenize_climblab():
    dataset = load_dataset("nvidia/ClimbLab", split="train", streaming=True)
    count = 0
    batch = []
    all_results = []
    sample_count = 0
    for sample in dataset:
        if MAX_SAMPLES and count >= MAX_SAMPLES:
            break
        if not isinstance(sample, dict):
            continue
        tokens = sample.get("tokens")
        if tokens is None or not isinstance(tokens, list) or len(tokens) == 0 or len(tokens) > 100000:
            continue
        batch.append({"tokens": tokens})
        sample_count += 1
        if sample_count % 1000 == 0:
            print(f"Processed {sample_count} samples so far...")
        if len(batch) >= BATCH_SIZE:
            texts = detokenize_batch(batch)
            preds = classify_texts(texts)
            for text, pred in zip(texts, preds):
                if MAX_SAMPLES and count >= MAX_SAMPLES:
                    break
                if text and len(text.strip()) > 10 and pred == 1:
                    all_results.append({"text": text})
                    count += 1
            batch = []
    if batch and (not MAX_SAMPLES or count < MAX_SAMPLES):
        texts = detokenize_batch(batch)
        preds = classify_texts(texts)
        for text, pred in zip(texts, preds):
            if MAX_SAMPLES and count >= MAX_SAMPLES:
                break
            if text and len(text.strip()) > 10 and pred == 1:
                all_results.append({"text": text})
                count += 1
    print(f"Total samples processed: {sample_count}")
    print(f"Total filtered samples: {count}")
    print(f"Total samples filtered out: {sample_count - count}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        json.dump(all_results, f_out, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    detokenize_climblab()