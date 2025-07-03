import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Rho1Pipeline:
    """
    Rho-1 selective language modeling pipeline for scoring and filtering training data.
    """
    
    def __init__(self, reference_model_name: str, device: str = "auto"):
        """
        Initialize the Rho-1 pipeline with a reference model.
        """
        self.reference_model_name = reference_model_name
        self.device = self._setup_device(device)
        
        logger.info(f"Loading reference model: {reference_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(reference_model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            reference_model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map="auto" if self.device.type == 'cuda' else None,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def compute_token_scores(self, text: str, max_length: int = 512) -> Tuple[List[float], List[str]]:
        """
        Compute perplexity-based scores for tokens in the input text.
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length,
            padding=False
        ).to(self.device)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        if len(tokens) <= 1:
            return [0.0], tokens
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            logits = outputs.logits[0]
            
            token_scores = []
            
            for i in range(len(tokens) - 1):
                next_token_logits = logits[i]
                next_token_id = inputs['input_ids'][0][i + 1]
                
                probs = torch.softmax(next_token_logits, dim=-1)
                token_prob = probs[next_token_id].item()
                
                score = -np.log(max(token_prob, 1e-10))
                token_scores.append(score)
            
            if token_scores:
                token_scores.append(np.mean(token_scores))
            else:
                token_scores = [0.0]
        
        return token_scores, tokens
    
    def score_dataset(self, 
                     dataset_name: str,
                     output_file: str) -> None:
        """
        Score the entire dataset using the pipeline.
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        try:
            # Load with streaming for memory efficiency
            dataset = load_dataset(dataset_name, split='train', streaming=True, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load dataset from HuggingFace: {e}")
            return
        
        logger.info(f"Processing dataset with streaming...")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        sample_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(tqdm(dataset, desc="Scoring samples")):
                try:
                    text = sample.get('text', sample.get('content', str(sample)))
                    
                    if not text or len(text.strip()) < 10:
                        continue
                    
                    token_scores, tokens = self.compute_token_scores(text)
                    
                    avg_score = np.mean(token_scores) if token_scores else 0.0
                    
                    scored_sample = {
                        'id': i,
                        'original_text': text,
                        'avg_score': float(avg_score),
                        'num_tokens': len(tokens)
                    }
                    
                    # Write immediately and discard from memory
                    f.write(json.dumps(scored_sample, ensure_ascii=False) + '\n')
                    sample_count += 1
                    
                    # Log progress periodically
                    if sample_count % 1000 == 0:
                        logger.info(f"Processed {sample_count:,} samples so far...")
                    
                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {str(e)}")
                    continue
        
        logger.info(f"Scoring completed! Processed {sample_count:,} samples total. Results saved to: {output_file}")
    
    def filter_high_quality_samples(self, 
                                   scored_file: str,
                                   output_file: str,
                                   percentile_threshold: float = 75.0) -> None:
        """
        Filter high-quality samples based on score percentiles.
        """
        logger.info(f"Loading scored samples from: {scored_file}")
        
        scored_samples = []
        with open(scored_file, 'r', encoding='utf-8') as f:
            for line in f:
                scored_samples.append(json.loads(line))
        
        if not scored_samples:
            logger.warning("No scored samples found to filter.")
            return

        scores = [sample['avg_score'] for sample in scored_samples]
        threshold = np.percentile(scores, percentile_threshold)
        
        logger.info(f"Score threshold (p{percentile_threshold}): {threshold:.4f}")
        
        high_quality_samples = [
            sample for sample in scored_samples 
            if sample['avg_score'] >= threshold
        ]
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in high_quality_samples:
                filtered_sample = {
                    'text': sample['original_text'],
                }
                f.write(json.dumps(filtered_sample, ensure_ascii=False) + '\n')
        
        logger.info(f"Filtered {len(high_quality_samples)} high-quality samples")
        logger.info(f"Results saved to: {output_file}")


def main():
    """
    Main function to run the simplified Rho-1 pipeline.
    """
    # --- Configuration ---
    MODEL_NAME = "microsoft/phi-2"
    DATASET_NAME = "nvidia/climblab"
    SCORED_OUTPUT_FILE = "data/climblab_scored.jsonl"
    FILTERED_OUTPUT_FILE = "data/climblab_filtered.jsonl"
    PERCENTILE_THRESHOLD = 75.0  # Keep the top 25% of samples
    DEVICE = "auto"
    # -------------------

    # Initialize Rho-1 pipeline
    pipeline = Rho1Pipeline(
        reference_model_name=MODEL_NAME,
        device=DEVICE
    )
    
    # Score the entire dataset
    pipeline.score_dataset(
        dataset_name=DATASET_NAME,
        output_file=SCORED_OUTPUT_FILE
    )
    
    # Filter the dataset based on scores
    pipeline.filter_high_quality_samples(
        scored_file=SCORED_OUTPUT_FILE,
        output_file=FILTERED_OUTPUT_FILE,
        percentile_threshold=PERCENTILE_THRESHOLD
    )


if __name__ == "__main__":
    main()
