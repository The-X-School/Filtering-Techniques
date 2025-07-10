import json
from datasets import load_dataset
from transformers import GPT2Tokenizer
from huggingface_hub import login
HF_TOKEN = "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"
login(token=HF_TOKEN)
OUTPUT_PATH = "climblab_detokenized.json"
MAX_SAMPLES = 10000
BATCH_SIZE = 10
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=True)
def detokenize_batch(batch_data):
    results = []
    for item in batch_data:
        try:
            text = tokenizer.decode(item["tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            results.append({"text": text})
        except Exception:
            results.append({"text": ""})
    return results
def detokenize_climblab():
    dataset = load_dataset("nvidia/ClimbLab", split="train", streaming=True)
    count = 0
    batch = []
    all_results = []
    for sample in dataset:
        if MAX_SAMPLES and count >= MAX_SAMPLES:
            break
        tokens = sample.get("tokens")
        if tokens is None or not isinstance(tokens, list) or len(tokens) == 0 or len(tokens) > 100000:
            continue
        batch.append({"tokens": tokens})
        if len(batch) >= BATCH_SIZE:
            results = detokenize_batch(batch)
            for result in results:
                if MAX_SAMPLES and count >= MAX_SAMPLES:
                    break
                text = result["text"]
                if text and len(text.strip()) > 10:
                    all_results.append({"text": text})
                    count += 1
            batch = []
    if batch and (not MAX_SAMPLES or count < MAX_SAMPLES):
        results = detokenize_batch(batch)
        for result in results:
            if MAX_SAMPLES and count >= MAX_SAMPLES:
                break
            text = result["text"]
            if text and len(text.strip()) > 10:
                all_results.append({"text": text})
                count += 1
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        json.dump(all_results, f_out, ensure_ascii=False, indent=2)
if __name__ == "__main__":
    detokenize_climblab()