import argparse
import json
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
from huggingface_hub import login
import multiprocessing as mp
from functools import partial
import os
import psutil

OUTPUT_PATH = "climblab_detokenized.jsonl"
MAX_SAMPLES = 50
BATCH_SIZE = 0
NUM_WORKERS = 0
HF_TOKEN = "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"
login(token=HF_TOKEN)

def detokenize_batch(batch_data):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=True)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    results = []
    for item in batch_data:
        tokens = item["tokens"]
        try:
            text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            text = text.strip()
            results.append({"text": text})
        except Exception:
            results.append({"text": ""})
    return results

def process_samples_generator(dataset, max_samples=None):
    count = 0
    for sample in dataset:
        tokens = sample.get("tokens")
        if tokens is None or not isinstance(tokens, list) or len(tokens) == 0 or len(tokens) > 100000:
            continue
        yield {"tokens": tokens}
        count += 1
        if max_samples and count >= max_samples:
            break

def detokenize_climblab(
    output_path=OUTPUT_PATH,
    max_samples=MAX_SAMPLES,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
):
    print(f"Detokenizing using nvidia/ClimbLab")
    print(f"Target samples: {max_samples if max_samples else 'unlimited'}")
    print(f"Workers: {num_workers}, Batch size: {batch_size}")
    dataset = load_dataset("nvidia/ClimbLab", split="train", streaming=True)
    if num_workers <= 0:
        num_workers = max(1, mp.cpu_count() - 1)
    else:
        num_workers = min(num_workers, mp.cpu_count())
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    if batch_size <= 0:
        if available_memory_gb > 8:
            batch_size = 2000
        elif available_memory_gb > 4:
            batch_size = 1000
        else:
            batch_size = 500
    print(f"System: {mp.cpu_count()} CPUs, {available_memory_gb:.1f}GB RAM")
    print(f"Using {num_workers} workers, batch size {batch_size}")
    count = 0
    batch = []
    all_results = []
    try:
        pbar = tqdm(desc="Processing batches", unit="samples")
        sample_generator = process_samples_generator(dataset, max_samples)
        with mp.Pool(num_workers) as pool:
            for sample_data in sample_generator:
                batch.append(sample_data)
                if len(batch) >= batch_size:
                    detokenize_func = partial(detokenize_batch)
                    batch_results = pool.apply_async(detokenize_func, [batch])
                    results = batch_results.get()
                    for result in results:
                        text = result["text"]
                        if text and len(text.strip()) > 10:
                            all_results.append({"text": text})
                            count += 1
                    pbar.update(len(batch))
                    batch = []
                    if max_samples and count >= max_samples:
                        break
            if batch:
                detokenize_func = partial(detokenize_batch)
                batch_results = pool.apply_async(detokenize_func, [batch])
                results = batch_results.get()
                for result in results:
                    text = result["text"]
                    if text and len(text.strip()) > 10 and (not max_samples or count < max_samples):
                        all_results.append({"text": text})
                        count += 1
                pbar.update(len(batch))
        pbar.close()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
    finally:
        pass
    print(f"Detokenized {count} samples. Output: {output_path}")
    if max_samples and count < max_samples:
        print(f"Note: Processed {count} samples, requested {max_samples}. Dataset may have fewer valid samples.")
    return count, output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()
    detokenize_climblab(
        output_path=args.output,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()