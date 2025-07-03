#!/usr/bin/env python3

import json
import logging
from typing import Iterator, Dict, Any, List, Optional
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_reward_function(text: str, quality_threshold: float = 0.2, length_min: int = 50, length_max: int = 2000, fc_weight: float = 1.0) -> float:
    """
    Simple reward function that evaluates a single text sample.
    
    Args:
        text: Text string to evaluate
        quality_threshold: Minimum quality score to consider valid
        length_min: Minimum text length
        length_max: Maximum text length  
        fc_weight: Weight for function calling indicators
    
    Returns:
        Float reward score between 0 and 1
    """
    if not text or len(text) < length_min or len(text) > length_max:
        return 0.0
    
    # Function calling patterns to look for
    fc_patterns = ['function', 'call', 'api', 'execute', 'def', '(', ')', '=', 'return']
    
    # Calculate quality score
    words = text.lower().split()
    if not words:
        return 0.0
        
    # Length score (0-1)
    length_score = min(1.0, len(text) / 500)
    
    # Vocabulary diversity (0-1)
    unique_words = set(words)
    vocab_diversity = len(unique_words) / len(words)
    
    # Function calling score (0-1)
    fc_matches = sum(1 for pattern in fc_patterns if pattern in text.lower())
    fc_score = min(1.0, fc_matches / len(fc_patterns))
    
    # Combined quality score
    quality_score = (
        length_score * 0.3 +
        vocab_diversity * 0.4 + 
        fc_score * fc_weight * 0.3
    )
    
    return quality_score if quality_score >= quality_threshold else 0.0


def stream_climblab_with_rewards(num_samples: Optional[int] = None, min_reward: float = 0.2) -> Iterator[Dict[str, Any]]:
    """
    Stream the ClimbLab dataset and yield samples with their reward scores.
    
    Args:
        num_samples: Number of samples to process (None for all)
        min_reward: Minimum reward score to include a sample
        
    Yields:
        Dict containing sample data and reward score
    """
    logger.info("Loading ClimbLab dataset in streaming mode...")
    
    try:
        # Load dataset in streaming mode
        dataset = load_dataset(
            "nvidia/ClimbLab", 
            split="train", 
            streaming=True,
            cache_dir="./cache",
            trust_remote_code=True
        )
        
        processed_count = 0
        yielded_count = 0
        
        for i, sample in enumerate(dataset):
            if num_samples and i >= num_samples:
                break
                
            try:
                # Extract text from sample
                text = ""
                if "text" in sample:
                    text = sample["text"]
                elif "tokens" in sample:
                    # Handle tokenized text
                    tokens = sample["tokens"]
                    if isinstance(tokens, list):
                        text = " ".join(str(t) for t in tokens)
                    else:
                        text = str(tokens)
                
                if text:
                    # Calculate reward score
                    reward = simple_reward_function(text)
                    
                    processed_count += 1
                    
                    # Only yield if reward meets threshold
                    if reward >= min_reward:
                        result = {
                            "index": i,
                            "text": text,
                            "reward_score": reward,
                            "text_length": len(text),
                            "word_count": len(text.split()),
                            "metadata": {k: v for k, v in sample.items() if k not in ["text", "tokens"]}
                        }
                        yielded_count += 1
                        yield result
                        
                        if yielded_count % 100 == 0:
                            logger.info(f"Processed: {processed_count}, Yielded: {yielded_count}")
                            
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
                
        logger.info(f"Finished. Processed: {processed_count}, Yielded: {yielded_count}")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def save_filtered_samples(output_file: str = "filtered_climblab_rewards.jsonl", 
                         num_samples: Optional[int] = 1000, 
                         min_reward: float = 0.3) -> None:
    """
    Save filtered samples with high reward scores to a file.
    
    Args:
        output_file: Output file path
        num_samples: Number of samples to process
        min_reward: Minimum reward score to save
    """
    logger.info(f"Saving filtered samples to {output_file}")
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in stream_climblab_with_rewards(num_samples, min_reward):
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            count += 1
            
    logger.info(f"Saved {count} filtered samples to {output_file}")


# Simple usage example
if __name__ == "__main__":
    # Test streaming with reward function
    logger.info("Testing ClimbLab streaming with reward function...")
    
    # Process first 500 samples and show high-reward ones
    high_reward_samples = []
    for sample in stream_climblab_with_rewards(num_samples=500, min_reward=0.4):
        high_reward_samples.append(sample)
        if len(high_reward_samples) >= 10:  # Show first 10 good samples
            break
    
    print(f"\nFound {len(high_reward_samples)} high-reward samples:")
    for i, sample in enumerate(high_reward_samples[:5]):
        print(f"\nSample {i+1}:")
        print(f"Reward: {sample['reward_score']:.3f}")
        print(f"Length: {sample['text_length']}")
        print(f"Text preview: {sample['text'][:100]}...")
    
    # Save filtered samples
    save_filtered_samples("filtered_climblab_rewards.jsonl", num_samples=1000, min_reward=0.3) 