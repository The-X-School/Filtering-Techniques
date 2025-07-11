#!/usr/bin/env python3

import os
import json
import logging
from typing import Optional, Iterator, Dict, Any, List, Set, Union
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from datasets import load_dataset, IterableDataset
from transformers import AutoTokenizer
from huggingface_hub import login

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClimbLabClusterStreamer:
    """
    Load ClimbLab dataset, detokenize while preserving clusters, and stream specific clusters.
    """
    
    def __init__(self, 
                 cache_dir: str = "./cache",
                 use_auth: bool = False,
                 tokenizer_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize the ClimbLab cluster streamer.
        
        Args:
            cache_dir: Directory to cache the dataset
            use_auth: Whether to use HuggingFace authentication
            tokenizer_name: Tokenizer to use for detokenization
        """
        self.cache_dir = cache_dir
        self.use_auth = use_auth
        self.tokenizer_name = tokenizer_name
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Authenticate if needed
        if use_auth:
            try:
                login()
                logger.info("Successfully authenticated with HuggingFace")
            except Exception as e:
                logger.warning(f"Authentication failed: {e}")
        
        # Initialize tokenizer for detokenization
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            self.tokenizer = None
        
        # Cluster information storage
        self.cluster_info = {}
        self.cluster_samples = defaultdict(list)
        
    def load_dataset_streaming(self) -> IterableDataset:
        """Load the ClimbLab dataset in streaming mode."""
        logger.info("Loading nvidia/ClimbLab dataset in streaming mode...")
        try:
            dataset = load_dataset(
                "nvidia/ClimbLab", 
                split="train", 
                streaming=True,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            logger.info("Dataset loaded successfully in streaming mode")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def detokenize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detokenize a sample while preserving all metadata including cluster info.
        
        Args:
            sample: Raw sample from the dataset
            
        Returns:
            Processed sample with detokenized text
        """
        processed_sample = sample.copy()
        
        # Handle different token formats
        if "tokens" in sample and sample["tokens"]:
            tokens = sample["tokens"]
            
            if self.tokenizer is not None:
                try:
                    # Try to detokenize using the tokenizer
                    if isinstance(tokens, list):
                        # Handle list of token IDs
                        if all(isinstance(t, (int, np.integer)) for t in tokens):
                            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                        else:
                            # Handle list of token strings
                            text = " ".join(str(t) for t in tokens)
                    else:
                        # Handle string tokens
                        text = str(tokens)
                    
                    processed_sample["detokenized_text"] = text
                    processed_sample["original_tokens"] = tokens
                    processed_sample["token_count"] = len(tokens) if isinstance(tokens, list) else len(str(tokens).split())
                    
                except Exception as e:
                    logger.warning(f"Failed to detokenize sample: {e}")
                    # Fallback: join tokens as strings
                    if isinstance(tokens, list):
                        text = " ".join(str(t) for t in tokens)
                    else:
                        text = str(tokens)
                    processed_sample["detokenized_text"] = text
                    processed_sample["original_tokens"] = tokens
                    processed_sample["token_count"] = len(tokens) if isinstance(tokens, list) else len(str(tokens).split())
            else:
                # No tokenizer available, use simple string conversion
                if isinstance(tokens, list):
                    text = " ".join(str(t) for t in tokens)
                else:
                    text = str(tokens)
                processed_sample["detokenized_text"] = text
                processed_sample["original_tokens"] = tokens
                processed_sample["token_count"] = len(tokens) if isinstance(tokens, list) else len(str(tokens).split())
        
        # Handle existing text field
        elif "text" in sample:
            processed_sample["detokenized_text"] = sample["text"]
            processed_sample["token_count"] = len(sample["text"].split())
        
        # Extract cluster information if available
        cluster_id = None
        for key in ["cluster", "cluster_id", "cluster_label", "group", "category"]:
            if key in sample:
                cluster_id = sample[key]
                break
        
        if cluster_id is not None:
            processed_sample["cluster_id"] = cluster_id
        else:
            # Try to infer cluster from other metadata
            processed_sample["cluster_id"] = "unknown"
        
        return processed_sample
    
    def process_and_index_dataset(self, 
                                num_samples: Optional[int] = None,
                                save_processed: bool = True,
                                output_file: str = "processed_climblab_clusters.jsonl") -> Dict[str, Any]:
        """
        Process the entire dataset, detokenize, and build cluster index.
        
        Args:
            num_samples: Number of samples to process (None for all)
            save_processed: Whether to save processed data to file
            output_file: Output file path
            
        Returns:
            Processing statistics
        """
        logger.info("Starting dataset processing and cluster indexing...")
        
        dataset = self.load_dataset_streaming()
        
        processed_count = 0
        cluster_counts = defaultdict(int)
        total_tokens = 0
        
        if save_processed:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') if save_processed else open(os.devnull, 'w') as f:
            progress_bar = tqdm(desc="Processing samples", unit="samples")
            
            for i, sample in enumerate(dataset):
                if num_samples and i >= num_samples:
                    break
                
                try:
                    # Process and detokenize sample
                    processed_sample = self.detokenize_sample(sample)
                    
                    # Update statistics
                    cluster_id = processed_sample.get("cluster_id", "unknown")
                    cluster_counts[cluster_id] += 1
                    total_tokens += processed_sample.get("token_count", 0)
                    
                    # Store in cluster index
                    self.cluster_samples[cluster_id].append({
                        "index": i,
                        "sample": processed_sample
                    })
                    
                    # Save to file if requested
                    if save_processed:
                        f.write(json.dumps(processed_sample, ensure_ascii=False) + '\n')
                    
                    processed_count += 1
                    progress_bar.update(1)
                    
                    # Log progress periodically
                    if processed_count % 1000 == 0:
                        progress_bar.set_description(f"Processed {processed_count:,} samples")
                
                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    continue
            
            progress_bar.close()
        
        # Build cluster info
        self.cluster_info = {
            "total_samples": processed_count,
            "total_tokens": total_tokens,
            "cluster_counts": dict(cluster_counts),
            "unique_clusters": len(cluster_counts),
            "average_tokens_per_sample": total_tokens / processed_count if processed_count > 0 else 0
        }
        
        logger.info(f"Processing completed!")
        logger.info(f"  Total samples: {processed_count:,}")
        logger.info(f"  Total tokens: {total_tokens:,}")
        logger.info(f"  Unique clusters: {len(cluster_counts)}")
        logger.info(f"  Average tokens per sample: {total_tokens / processed_count:.2f}")
        
        if save_processed:
            logger.info(f"  Processed data saved to: {output_file}")
        
        return self.cluster_info
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about all clusters."""
        return self.cluster_info
    
    def list_clusters(self) -> List[str]:
        """List all available cluster IDs."""
        return list(self.cluster_samples.keys())
    
    def get_cluster_stats(self, cluster_id: str) -> Dict[str, Any]:
        """Get statistics for a specific cluster."""
        if cluster_id not in self.cluster_samples:
            return {"error": f"Cluster '{cluster_id}' not found"}
        
        samples = self.cluster_samples[cluster_id]
        token_counts = [s["sample"].get("token_count", 0) for s in samples]
        
        return {
            "cluster_id": cluster_id,
            "sample_count": len(samples),
            "total_tokens": sum(token_counts),
            "avg_tokens": np.mean(token_counts) if token_counts else 0,
            "min_tokens": np.min(token_counts) if token_counts else 0,
            "max_tokens": np.max(token_counts) if token_counts else 0,
            "std_tokens": np.std(token_counts) if token_counts else 0
        }
    
    def stream_cluster(self, 
                      cluster_id: str, 
                      limit: Optional[int] = None,
                      min_tokens: Optional[int] = None,
                      max_tokens: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        Stream samples from a specific cluster with optional filtering.
        
        Args:
            cluster_id: ID of the cluster to stream
            limit: Maximum number of samples to return
            min_tokens: Minimum token count filter
            max_tokens: Maximum token count filter
            
        Yields:
            Processed samples from the specified cluster
        """
        if cluster_id not in self.cluster_samples:
            logger.error(f"Cluster '{cluster_id}' not found")
            return
        
        samples = self.cluster_samples[cluster_id]
        logger.info(f"Streaming cluster '{cluster_id}' with {len(samples)} samples")
        
        count = 0
        for sample_data in samples:
            if limit and count >= limit:
                break
            
            sample = sample_data["sample"]
            token_count = sample.get("token_count", 0)
            
            # Apply token filters
            if min_tokens and token_count < min_tokens:
                continue
            if max_tokens and token_count > max_tokens:
                continue
            
            yield sample
            count += 1
    
    def save_cluster_to_file(self, 
                           cluster_id: str, 
                           output_file: str,
                           limit: Optional[int] = None,
                           min_tokens: Optional[int] = None,
                           max_tokens: Optional[int] = None) -> int:
        """
        Save a specific cluster to a file.
        
        Args:
            cluster_id: ID of the cluster to save
            output_file: Output file path
            limit: Maximum number of samples to save
            min_tokens: Minimum token count filter
            max_tokens: Maximum token count filter
            
        Returns:
            Number of samples saved
        """
        logger.info(f"Saving cluster '{cluster_id}' to {output_file}")
        
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in self.stream_cluster(cluster_id, limit, min_tokens, max_tokens):
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                count += 1
        
        logger.info(f"Saved {count} samples from cluster '{cluster_id}' to {output_file}")
        return count


def main():
    """
    Example usage of the ClimbLabClusterStreamer.
    """
    print("ClimbLab Cluster Streamer - Example Usage")
    print("=" * 50)
    
    # Initialize streamer
    streamer = ClimbLabClusterStreamer(
        cache_dir="./cache",
        use_auth=False,  # Set to True if you need authentication
        tokenizer_name="microsoft/DialoGPT-medium"
    )
    
    # Process dataset (using small sample for demo)
    print("\n1. Processing dataset...")
    stats = streamer.process_and_index_dataset(
        num_samples=1000000,  # Process first 1000 samples for demo
        save_processed=True,
        output_file="data/climblab_processed_clusters.jsonl"
    )
    
    # Show cluster information
    print("\n2. Cluster Information:")
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Unique clusters: {stats['unique_clusters']}")
    print(f"Average tokens per sample: {stats['average_tokens_per_sample']:.2f}")
    
    # List clusters
    print("\n3. Available clusters:")
    clusters = streamer.list_clusters()
    for cluster_id in clusters[:10]:  # Show first 10 clusters
        cluster_stats = streamer.get_cluster_stats(cluster_id)
        print(f"  {cluster_id}: {cluster_stats['sample_count']} samples, "
              f"avg {cluster_stats['avg_tokens']:.1f} tokens")
    
    # Stream a specific cluster
    if clusters:
        target_cluster = clusters[0]
        print(f"\n4. Streaming cluster '{target_cluster}':")
        
        count = 0
        for sample in streamer.stream_cluster(target_cluster, limit=3):
            count += 1
            text_preview = sample.get("detokenized_text", "No text")[:100]
            print(f"  Sample {count}: {sample.get('token_count', 0)} tokens")
            print(f"    Preview: {text_preview}...")
        
        # Save cluster to file
        print(f"\n5. Saving cluster '{target_cluster}' to file...")
        saved_count = streamer.save_cluster_to_file(
            target_cluster,
            f"data/cluster_{target_cluster}.jsonl",
            limit=100,
            min_tokens=10
        )
        print(f"Saved {saved_count} samples")


if __name__ == "__main__":
    main() 
