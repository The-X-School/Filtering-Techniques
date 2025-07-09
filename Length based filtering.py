#!/usr/bin/env python3

import os
import json
import logging
from typing import Optional, Iterator, Dict, Any, List, Union
from itertools import islice
from tqdm import tqdm
import numpy as np

from datasets import load_dataset, IterableDataset
from huggingface_hub import login
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
SAMPLE_SIZE = 1000  
USE_AUTH = False
CACHE_DIR = "./cache"
MIN_LENGTH = 80  
MAX_LENGTH = 3000 
FILTER_FIELD = 'tokens'

SAVE_DATA = True
OUTPUT_DIR = "./data"
OUTPUT_FORMAT = 'jsonl'

SHOW_SAMPLE_RECORDS = False 
NUM_SAMPLE_RECORDS = 3

# Production settings
BATCH_SIZE = 1000  # Process in batches for memory efficiency
MAX_RETRIES = 3

class ClimbLabLoader:
    
    def __init__(self, cache_dir: str = CACHE_DIR, use_auth: bool = USE_AUTH):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        if use_auth:
            try:
                login()
                logger.info("Successfully authenticated with HuggingFace")
            except Exception as e:
                logger.error(f"Authentication failed: {e}")
                logger.info("You may need to run: huggingface-cli login")
    
    def load_dataset_streaming(self) -> IterableDataset:
        logger.info("Loading nvidia/ClimbLab dataset in streaming mode...")
        for attempt in range(MAX_RETRIES):
            try:
                dataset = load_dataset(
                    "nvidia/ClimbLab", 
                    split="train", 
                    streaming=True,
                    cache_dir=self.cache_dir
                )
                logger.info("Dataset loaded successfully in streaming mode")
                return dataset
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed to load dataset after {MAX_RETRIES} attempts")
                    raise
    
    def get_sample(self, dataset: IterableDataset, num_samples: int) -> List[Dict[str, Any]]:
        logger.info(f"Sampling {num_samples:,} records from the dataset...")
        
        sample = []
        try:
            with tqdm(total=num_samples, desc="Sampling records") as pbar:
                for i, record in enumerate(dataset):
                    if i >= num_samples:
                        break
                    sample.append(record)
                    pbar.update(1)
                    
                    # Memory check
                    if len(sample) % 1000 == 0:
                        logger.debug(f"Sampled {len(sample):,} records so far...")
                        
        except Exception as e:
            logger.error(f"Error during sampling: {e}")
            if not sample:
                raise
            logger.warning(f"Partial sample obtained: {len(sample):,} records")
        
        logger.info(f"Successfully sampled {len(sample):,} records")
        return sample
    
    def analyze_sample(self, sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("Analyzing sample data...")
        
        if not sample:
            return {"error": "Empty sample"}
        
        # Validate data structure
        first_record = sample[0]
        available_fields = list(first_record.keys())
        
        stats = {
            "total_records": len(sample),
            "available_fields": available_fields
        }
        
        if 'tokens' in available_fields:
            token_counts = []
            invalid_records = 0
            
            for record in sample:
                if 'tokens' not in record or not record['tokens']:
                    invalid_records += 1
                    continue
                    
                # Handle different token formats
                tokens = record['tokens']
                if isinstance(tokens, list):
                    token_counts.append(len(tokens))
                elif isinstance(tokens, str):
                    # If tokens is a string, split by spaces
                    token_counts.append(len(tokens.split()))
                else:
                    invalid_records += 1
                    continue
            
            if token_counts:
                stats.update({
                    "token_analysis": {
                        "valid_records": len(token_counts),
                        "invalid_records": invalid_records,
                        "mean_tokens": float(np.mean(token_counts)),
                        "median_tokens": float(np.median(token_counts)),
                        "min_tokens": int(np.min(token_counts)),
                        "max_tokens": int(np.max(token_counts)),
                        "std_tokens": float(np.std(token_counts))
                    }
                })
            else:
                stats["token_analysis"] = {"error": "No valid token data found"}
        
        return stats
    
    def apply_length_filter(self, sample: List[Dict[str, Any]], 
                          min_length: Optional[int] = None, 
                          max_length: Optional[int] = None,
                          field: str = 'tokens') -> List[Dict[str, Any]]:
        logger.info(f"Applying ELMB-optimized length filter on '{field}' field...")
        logger.info(f"   Min length: {min_length} tokens")
        logger.info(f"   Max length: {max_length} tokens")
        logger.info(f"   Expected retention: ~87.5%")
        
        filtered_sample = []
        invalid_count = 0
        
        with tqdm(total=len(sample), desc="Filtering records") as pbar:
            for record in sample:
                if field not in record or not record[field]:
                    invalid_count += 1
                    pbar.update(1)
                    continue
                    
                # Handle different token formats
                tokens = record[field]
                if isinstance(tokens, list):
                    length = len(tokens)
                elif isinstance(tokens, str):
                    length = len(tokens.split())
                else:
                    invalid_count += 1
                    pbar.update(1)
                    continue
                
                # Apply length filters
                if min_length is not None and length < min_length:
                    pbar.update(1)
                    continue
                if max_length is not None and length > max_length:
                    pbar.update(1)
                    continue
                    
                filtered_sample.append(record)
                pbar.update(1)
        
        retention_rate = len(filtered_sample) / len(sample) * 100
        logger.info(f"Filtered from {len(sample):,} to {len(filtered_sample):,} records")
        logger.info(f"   Actual retention rate: {retention_rate:.2f}%")
        logger.info(f"   Invalid records skipped: {invalid_count:,}")
        
        return filtered_sample
    
    def save_filtered_dataset(self, data: List[Dict[str, Any]], output_path: str):
        """Save filtered dataset in the format expected by LMFlow training"""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        logger.info(f"Saving {len(data):,} records to {output_path}...")
        
        total_tokens = 0
        saved_count = 0
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                with tqdm(total=len(data), desc="Saving records") as pbar:
                    for record in data:
                        try:
                            # Create training-ready format
                            if 'tokens' in record and record['tokens']:
                                tokens = record['tokens']
                                
                                # Handle different token formats
                                if isinstance(tokens, list):
                                    # Convert token IDs back to readable format
                                    # Note: This is a simplified approach
                                    text = " ".join(str(token) for token in tokens)
                                    token_count = len(tokens)
                                elif isinstance(tokens, str):
                                    text = tokens
                                    token_count = len(tokens.split())
                                else:
                                    logger.warning(f"Unexpected token format: {type(tokens)}")
                                    continue
                                
                                training_record = {
                                    "text": text,
                                    "token_count": token_count
                                }
                                
                                total_tokens += token_count
                                
                            else:
                                # Fallback for records without tokens
                                training_record = record
                            
                            f.write(json.dumps(training_record, ensure_ascii=False) + '\n')
                            saved_count += 1
                            pbar.update(1)
                            
                        except Exception as e:
                            logger.warning(f"Error processing record: {e}")
                            pbar.update(1)
                            continue
            
            logger.info(f"Dataset saved successfully to {output_path}")
            logger.info(f"   Records saved: {saved_count:,}")
            logger.info(f"   Total tokens: {total_tokens:,}")
            logger.info(f"   Token budget usage: {total_tokens/10_000_000_000*100:.4f}% of 10B limit")
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise
    
    def print_sample_records(self, sample: List[Dict[str, Any]], num_records: int = 3):
        print(f"\nSample records (showing first {num_records}):")
        print("=" * 80)
        
        for i, record in enumerate(sample[:num_records]):
            print(f"\nRecord {i+1}:")
            print("-" * 40)
            
            for key, value in record.items():
                if key == 'tokens':
                    if isinstance(value, list):
                        print(f"{key}: [{len(value)} tokens] {value[:10]}...")
                    elif isinstance(value, str):
                        preview = value[:100] + "..." if len(value) > 100 else value
                        print(f"{key}: [{len(value.split())} tokens] {preview}")
                    else:
                        print(f"{key}: {type(value)} - {value}")
                else:
                    print(f"{key}: {value}")

    def validate_dataset_quality(self, sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the quality of the filtered dataset"""
        logger.info("Validating dataset quality...")
        
        validation_results = {
            "total_records": len(sample),
            "valid_records": 0,
            "empty_records": 0,
            "short_records": 0,
            "long_records": 0,
            "token_distribution": {}
        }
        
        token_lengths = []
        
        for record in sample:
            if 'tokens' not in record or not record['tokens']:
                validation_results["empty_records"] += 1
                continue
                
            tokens = record['tokens']
            if isinstance(tokens, list):
                length = len(tokens)
            elif isinstance(tokens, str):
                length = len(tokens.split())
            else:
                continue
            
            token_lengths.append(length)
            validation_results["valid_records"] += 1
            
            if length < MIN_LENGTH:
                validation_results["short_records"] += 1
            elif length > MAX_LENGTH:
                validation_results["long_records"] += 1
        
        if token_lengths:
            # Calculate distribution
            bins = [0, 100, 200, 500, 1000, 2000, 3000, float('inf')]
            labels = ['0-100', '100-200', '200-500', '500-1k', '1k-2k', '2k-3k', '3k+']
            hist, _ = np.histogram(token_lengths, bins=bins)
            
            for label, count in zip(labels, hist):
                percentage = count / len(token_lengths) * 100
                validation_results["token_distribution"][label] = {
                    "count": int(count),
                    "percentage": float(percentage)
                }
        
        return validation_results


def main():
    print("ELMB-Optimized ClimbLab Dataset Filtering")
    print("=" * 50)
    print(f"Configuration:")
    print(f"   Sample size: {SAMPLE_SIZE:,}")
    print(f"   Min length: {MIN_LENGTH} tokens")
    print(f"   Max length: {MAX_LENGTH} tokens")
    print(f"   Filter field: {FILTER_FIELD}")
    print(f"   Target retention: ~87.5%")
    print("=" * 50)
    
    try:
        loader = ClimbLabLoader(use_auth=USE_AUTH)
        dataset = loader.load_dataset_streaming()
        sample = loader.get_sample(dataset, SAMPLE_SIZE)
        
        if SHOW_SAMPLE_RECORDS:
            loader.print_sample_records(sample, NUM_SAMPLE_RECORDS)
        
        # Analyze original sample
        stats = loader.analyze_sample(sample)
        print(f"\nOriginal Dataset Statistics:")
        print("=" * 50)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        print(f"  {subkey}: {subvalue:.2f}" if isinstance(subvalue, float) else f"  {subkey}: {subvalue}")
                    else:
                        print(f"  {subkey}: {subvalue}")
            else:
                print(f"{key}: {value}")
        
        # Apply ELMB-optimized filtering
        filtered_sample = loader.apply_length_filter(
            sample, 
            min_length=MIN_LENGTH, 
            max_length=MAX_LENGTH,
            field=FILTER_FIELD
        )
        
        # Analyze filtered sample
        filtered_stats = loader.analyze_sample(filtered_sample)
        print(f"\nFiltered Dataset Statistics:")
        print("=" * 50)
        for key, value in filtered_stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        print(f"  {subkey}: {subvalue:.2f}" if isinstance(subvalue, float) else f"  {subkey}: {subvalue}")
                    else:
                        print(f"  {subkey}: {subvalue}")
            else:
                print(f"{key}: {value}")
        
        # Validate dataset quality
        validation_results = loader.validate_dataset_quality(filtered_sample)
        print(f"\nDataset Quality Validation:")
        print("=" * 50)
        for key, value in validation_results.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        print(f"  {subkey}: {subvalue['count']} ({subvalue['percentage']:.1f}%)")
                    else:
                        print(f"  {subkey}: {subvalue}")
            else:
                print(f"{key}: {value}")
        
        if SAVE_DATA and filtered_sample:
            output_filename = f"climblab_sample_{len(filtered_sample)}_elmb_optimized.jsonl"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            loader.save_filtered_dataset(filtered_sample, output_path)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
if __name__ == "__main__":
    main() 