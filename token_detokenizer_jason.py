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
from typing import List, Dict, Any

# =====================
# USER CONFIGURATION
# =====================
OUTPUT_PATH = "climblab_detokenized.json"
OUTPUT_JSONL = False
MAX_SAMPLES = 50  # Set to None for unlimited, or any number you want (e.g., 50, 1000, 10000)
PRINT_SAMPLES = 10  # How many samples to display in console during processing
BATCH_SIZE = 0      # Auto-detect batch size (set to positive number to override)
NUM_WORKERS = 0     # Auto-detect workers (set to positive number to override)
USE_CLIMBMIX = False  # Set to True to use ClimbMix (with clusters), False for ClimbLab
SAVE_CLUSTERS = False  # Set to True to save cluster info when available, False to skip
# =====================

# 🔐 Paste your token between the quotes (keep it private!)
HF_TOKEN = "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"
login(token=HF_TOKEN)

def detokenize_batch(batch_data: List[Dict], tokenizer_name: str = "gpt2") -> List[Dict]:
    """
    Detokenize a batch of token sequences with cluster information.
    
    Args:
        batch_data: List of dictionaries containing tokens, cluster_id, and token_count
        tokenizer_name: Name of the tokenizer to use
    
    Returns:
        List of dictionaries with detokenized text and metadata
    """
    # Initialize tokenizer in worker process with fast tokenizer if available
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name, use_fast=True)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    
    results = []
    for item in batch_data:
        tokens = item["tokens"]
        cluster_id = item["cluster_id"]
        token_count = item["token_count"]
        
        try:
            # Optimizations:
            # 1. skip_special_tokens=True for cleaner output
            # 2. clean_up_tokenization_spaces=True for better formatting
            text = tokenizer.decode(
                tokens, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            # 3. Strip whitespace for cleaner output
            text = text.strip()
            
            results.append({
                "text": text,
                "cluster_id": cluster_id,
                "token_count": token_count
            })
        except Exception:
            results.append({
                "text": "",
                "cluster_id": cluster_id,
                "token_count": token_count
            })  # Empty text for failed decodes but keep metadata
    
    return results

def process_samples_generator(dataset, max_samples: int = None, save_clusters: bool = True):
    """Generator that yields samples with optimized filtering."""
    count = 0
    for sample in dataset:
        tokens = sample.get("tokens")
        cluster_id = sample.get("cluster_id") if save_clusters else None
        token_count = sample.get("token_count")
        
        # Enhanced validation
        if (tokens is None or 
            not isinstance(tokens, list) or 
            len(tokens) == 0 or
            len(tokens) > 100000):  # Skip extremely long sequences
            continue
        
        yield {
            "tokens": tokens,
            "cluster_id": cluster_id,
            "token_count": token_count
        }
        count += 1
        
        if max_samples and count >= max_samples:
            break

def detokenize_climblab_optimized(
    output_path: str = OUTPUT_PATH,
    output_jsonl: bool = OUTPUT_JSONL,
    max_samples: int = MAX_SAMPLES,
    print_samples: int = PRINT_SAMPLES,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    use_climbmix: bool = USE_CLIMBMIX,
    save_clusters: bool = SAVE_CLUSTERS
) -> tuple:
    """
    Efficiently detokenize the NVIDIA ClimbLab/ClimbMix dataset using batch processing and multiprocessing.
    
    Args:
        output_path: Path to save the output file
        output_jsonl: If True, output as JSONL; else as a single JSON file
        max_samples: Maximum number of samples to process (None = all)
        print_samples: Number of samples to print to console
        batch_size: Number of samples to process in each batch
        num_workers: Number of parallel workers for processing
    
    Returns:
        (int, str): Number of samples processed, output file path
    """
    # Choose dataset based on configuration
    dataset_name = "nvidia/ClimbMix" if use_climbmix else "nvidia/ClimbLab"
    print(f"🚀 Starting optimized detokenization using {dataset_name}")
    print(f"🔧 Target samples: {max_samples if max_samples else 'unlimited'}")
    print(f"🔧 Workers: {num_workers}, Batch size: {batch_size}, Save clusters: {save_clusters}")
    
    # Load dataset with streaming
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    # Initialize single tokenizer for sample printing with fast tokenizer
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=True)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Auto-detect optimal number of workers
    if num_workers <= 0:
        num_workers = max(1, mp.cpu_count() - 1)
    else:
        num_workers = min(num_workers, mp.cpu_count())
    
    # Auto-adjust batch size based on available memory (rough estimation)
    import psutil
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    if batch_size <= 0:  # Auto-detect batch size
        if available_memory_gb > 8:
            batch_size = 2000
        elif available_memory_gb > 4:
            batch_size = 1000
        else:
            batch_size = 500
    
    print(f"🔧 System info: {mp.cpu_count()} CPUs, {available_memory_gb:.1f}GB RAM available")
    print(f"🚀 Using {num_workers} workers, batch size {batch_size}")
    
    count = 0
    printed = 0
    batch = []
    
    # Use buffered writing for better I/O performance
    buffer_size = 8192 * 4  # 32KB buffer
    
    # Prepare output file with optimized buffering
    if output_jsonl:
        output_file = open(output_path, "w", encoding="utf-8", buffering=buffer_size)
    else:
        all_results = []
    
    try:
        # Create progress bar
        pbar = tqdm(desc="Processing batches", unit="samples")
        
        # Process samples in batches
        sample_generator = process_samples_generator(dataset, max_samples, save_clusters)
        
        with mp.Pool(num_workers) as pool:
            for sample_data in sample_generator:
                batch.append(sample_data)
                
                # Process batch when it's full or we've reached the end
                if len(batch) >= batch_size:
                    # Process batch in parallel
                    detokenize_func = partial(detokenize_batch, tokenizer_name="gpt2")
                    batch_results = pool.apply_async(detokenize_func, [batch])
                    results = batch_results.get()
                    
                    # Write results with length filtering
                    for result in results:
                        text = result["text"]
                        cluster_id = result["cluster_id"]
                        token_count = result["token_count"]
                        
                        # Skip empty results and very short texts (likely noise)
                        if text and len(text.strip()) > 10:  
                            entry = {"text": text}
                            
                            # Add cluster info if available and enabled
                            if save_clusters and cluster_id is not None:
                                entry["cluster_id"] = cluster_id
                            if token_count is not None:
                                entry["token_count"] = token_count
                            
                            if output_jsonl:
                                output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            else:
                                all_results.append(entry)
                            
                            # Print sample if requested
                            if printed < print_samples:
                                cluster_info = f" (Cluster {cluster_id})" if save_clusters and cluster_id is not None else ""
                                print(f"Sample {count + 1}{cluster_info}: {text[:200]}{'...' if len(text) > 200 else ''}\n{'-'*40}")
                                printed += 1
                            
                            count += 1
                    
                    pbar.update(len(batch))
                    batch = []
                    
                    # Check if we've reached max_samples
                    if max_samples and count >= max_samples:
                        break
            
            # Process remaining samples in the last batch
            if batch:
                detokenize_func = partial(detokenize_batch, tokenizer_name="gpt2")
                batch_results = pool.apply_async(detokenize_func, [batch])
                results = batch_results.get()
                
                for result in results:
                    text = result["text"]
                    cluster_id = result["cluster_id"]
                    token_count = result["token_count"]
                    
                    if text and len(text.strip()) > 10 and (not max_samples or count < max_samples):
                        entry = {"text": text}
                        
                        # Add cluster info if available and enabled
                        if save_clusters and cluster_id is not None:
                            entry["cluster_id"] = cluster_id
                        if token_count is not None:
                            entry["token_count"] = token_count
                        
                        if output_jsonl:
                            output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        else:
                            all_results.append(entry)
                        
                        if printed < print_samples:
                            print(f"Sample {count + 1}: {text[:200]}{'...' if len(text) > 200 else ''}\n{'-'*40}")
                            printed += 1
                        
                        count += 1
                
                pbar.update(len(batch))
        
        pbar.close()
        
        # Write final JSON file if not using JSONL (with optimized settings)
        if not output_jsonl:
            print("💾 Writing final JSON file...")
            with open(output_path, "w", encoding="utf-8", buffering=buffer_size) as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2, separators=(',', ':'))
        
    finally:
        if output_jsonl and 'output_file' in locals():
            output_file.flush()  # Ensure all data is written
            output_file.close()
    
    target_info = f"/{max_samples}" if max_samples else ""
    print(f"✅ Successfully detokenized {count}{target_info} samples. Output saved to {output_path}")
    
    if max_samples and count < max_samples:
        print(f"⚠️  Note: Processed {count} samples, requested {max_samples}. Dataset may have fewer valid samples.")
    
    return count, output_path

# For older Python versions
try:
    from contextlib import nullcontext
except ImportError:
    class nullcontext:
        def __init__(self, enter_result=None):
            self.enter_result = enter_result
        def __enter__(self):
            return self.enter_result
        def __exit__(self, *excinfo):
            return False

def main():
    parser = argparse.ArgumentParser(description="Efficiently detokenize NVIDIA ClimbLab/ClimbMix dataset.")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help="Output file path")
    parser.add_argument("--jsonl", action="store_true", default=OUTPUT_JSONL, help="Output as JSONL")
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES, help="Maximum samples to process")
    parser.add_argument("--print_samples", type=int, default=PRINT_SAMPLES, help="Number of samples to print")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="Number of parallel workers")
    parser.add_argument("--use_climbmix", action="store_true", default=USE_CLIMBMIX, help="Use ClimbMix dataset (with clusters)")
    parser.add_argument("--save_clusters", action="store_true", default=SAVE_CLUSTERS, help="Save cluster information when available")
    parser.add_argument("--no_clusters", action="store_true", help="Don't save cluster information (overrides --save_clusters)")
    
    args = parser.parse_args()
    
    # Handle cluster saving logic
    save_clusters = args.save_clusters and not args.no_clusters
    
    detokenize_climblab_optimized(
        output_path=args.output,
        output_jsonl=args.jsonl,
        max_samples=args.max_samples,
        print_samples=args.print_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_climbmix=args.use_climbmix,
        save_clusters=save_clusters
    )

if __name__ == "__main__":
    main()