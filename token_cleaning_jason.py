import os
import torch
from huggingface_hub import login
from datasets import load_dataset, Dataset
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel,
    AutoModelForCausalLM, 
    AutoTokenizer
)
from collections import defaultdict
import time
import json
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HF_TOKEN = "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"  # Replace with your token
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

class TokenCleaner:
    def __init__(
        self,
        reference_model_name: str = "gpt2",
        influence_threshold: float = 0.1,
        rho_threshold: float = 0.7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.influence_threshold = influence_threshold
        self.rho_threshold = rho_threshold
        
        # Load reference model and tokenizer
        logger.info(f"Loading reference model: {reference_model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(reference_model_name)
        self.model = GPT2LMHeadModel.from_pretrained(reference_model_name).to(device)
        self.model.eval()

    def compute_token_influence(self, text: str) -> Tuple[List[str], List[float]]:
        """Compute influence scores for each token in the text."""
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Compute loss gradients for each token
        self.model.zero_grad()
        with torch.enable_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
        
        # Calculate influence scores based on gradient magnitudes
        influence_scores = []
        for param in self.model.parameters():
            if param.grad is not None:
                influence_scores.append(param.grad.abs().mean().item())
        
        # Normalize scores
        if influence_scores:
            influence_scores = [score / max(influence_scores) for score in influence_scores]
        
        return tokens, influence_scores

    def compute_rho_scores(self, text: str) -> Tuple[List[str], List[float]]:
        """Compute Rho-1 scores for each token."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            
            # Calculate token-level loss
            token_losses = []
            for i, token_id in enumerate(inputs["input_ids"][0]):
                token_loss = -torch.log(probs[i, token_id]).item()
                token_losses.append(token_loss)
        
        # Normalize scores
        if token_losses:
            max_loss = max(token_losses)
            rho_scores = [loss / max_loss for loss in token_losses]
        else:
            rho_scores = []
        
        return tokens, rho_scores

    def clean_text(self, text: str) -> str:
        """Clean text using both token influence and Rho-1 scoring."""
        # Get scores
        tokens, influence_scores = self.compute_token_influence(text)
        _, rho_scores = self.compute_rho_scores(text)
        
        # Combine scores (you can adjust this combination strategy)
        combined_scores = [(t, (i + r) / 2) 
                         for t, i, r in zip(tokens, influence_scores, rho_scores)]
        
        # Filter tokens based on combined scores
        cleaned_tokens = [token for token, score in combined_scores 
                        if score >= self.influence_threshold]
        
        # Detokenize
        cleaned_text = self.tokenizer.convert_tokens_to_string(cleaned_tokens)
        return cleaned_text

def process_climblab_dataset(
    output_dir: str = "cleaned_data",
    batch_size: int = 32,
    max_samples: int = None
):
    """Process the ClimbLab dataset with token-level cleaning."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize cleaner
    cleaner = TokenCleaner()
    
    # Login to Hugging Face
    login(token=HF_TOKEN)
    
    # Load ClimbLab dataset in streaming mode
    logger.info("Loading ClimbLab dataset...")
    try:
        dataset = load_dataset("nvidia/ClimbLab", split="train", streaming=True)
        logger.info("✅ Dataset loaded successfully!")
        # Print the first sample's keys for debugging
        first_sample = next(iter(dataset))
        print("First sample keys:", first_sample.keys(), flush=True)
        print("First sample:", first_sample, flush=True)
        logger.info(f"First sample keys: {list(first_sample.keys())}")
        logger.info(f"First sample: {first_sample}")
        return  # Exit after printing for debugging
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        logger.error("📦 Make sure you're connected to the internet and have access to the dataset")
        return
    
    # Process in batches
    cleaned_texts = []
    stats = {
        "total_processed": 0,
        "tokens_removed": 0,
        "avg_reduction": 0.0
    }
    
    # Create progress bar without total (since we're streaming)
    pbar = tqdm(desc="Processing samples")
    
    current_batch = []
    for sample in dataset:
        if max_samples and stats["total_processed"] >= max_samples:
            break
            
        text = sample["text"]
        current_batch.append(text)
        
        if len(current_batch) >= batch_size:
            # Process batch
            for text in current_batch:
                # Clean text
                original_tokens = len(cleaner.tokenizer.tokenize(text))
                cleaned_text = cleaner.clean_text(text)
                cleaned_tokens = len(cleaner.tokenizer.tokenize(cleaned_text))
                
                # Update stats
                stats["total_processed"] += 1
                stats["tokens_removed"] += (original_tokens - cleaned_tokens)
                stats["avg_reduction"] = stats["tokens_removed"] / stats["total_processed"]
                
                cleaned_texts.append(cleaned_text)
                
                # Update progress
                pbar.update(1)
                pbar.set_postfix({
                    "processed": stats["total_processed"],
                    "avg_reduction": f"{stats['avg_reduction']:.2%}"
                })
            
            # Save batch to avoid memory issues
            if len(cleaned_texts) >= 1000:
                batch_file = output_dir / f"cleaned_batch_{stats['total_processed']}.json"
                with open(batch_file, "w") as f:
                    json.dump(cleaned_texts, f)
                cleaned_texts = []
            
            # Clear batch
            current_batch = []
    
    # Process remaining samples in last batch
    if current_batch:
        for text in current_batch:
            original_tokens = len(cleaner.tokenizer.tokenize(text))
            cleaned_text = cleaner.clean_text(text)
            cleaned_tokens = len(cleaner.tokenizer.tokenize(cleaned_text))
            
            stats["total_processed"] += 1
            stats["tokens_removed"] += (original_tokens - cleaned_tokens)
            stats["avg_reduction"] = stats["tokens_removed"] / stats["total_processed"]
            
            cleaned_texts.append(cleaned_text)
            pbar.update(1)
    
    # Save any remaining texts
    if cleaned_texts:
        batch_file = output_dir / f"cleaned_batch_final.json"
        with open(batch_file, "w") as f:
            json.dump(cleaned_texts, f)
    
    # Save stats
    with open(output_dir / "cleaning_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    pbar.close()
    logger.info("Processing complete!")
    logger.info(f"Total samples processed: {stats['total_processed']}")
    logger.info(f"Average token reduction: {stats['avg_reduction']:.2%}")
    logger.info(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    # Process dataset with default settings
    process_climblab_dataset(
        output_dir="cleaned_climblab",
        batch_size=32,
        max_samples=1000  # Set to None to process entire dataset
    ) 