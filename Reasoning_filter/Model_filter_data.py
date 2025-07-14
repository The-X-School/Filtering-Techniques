import os
import torch
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
from collections import defaultdict
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset, Dataset
from huggingface_hub import login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelBasedFilter:
    def __init__(self, model_name: str = "my_model", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
        # Filtering parameters
        self.quality_threshold = 0.5
        self.max_length = 512
        self.batch_size = 16
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'filtered_count': 0,
            'high_quality_count': 0,
            'low_quality_count': 0,
            'error_count': 0,
            'score_distribution': [],
            'processing_time': 0.0
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            
            # Add special tokens if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def calculate_quality_score(self, text: str) -> Dict[str, Any]:
        """Calculate quality score using the model"""
        try:
            if not text or len(text.strip()) < 10:
                return {
                    'score': 0.0,
                    'reason': 'text_too_short',
                    'perplexity': float('inf'),
                    'length': len(text) if text else 0
                }
            
            # Tokenize the text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Calculate perplexity as quality metric
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            # Convert perplexity to quality score (lower perplexity = higher quality)
            # Use sigmoid transformation to get score between 0 and 1
            quality_score = 1 / (1 + np.exp((perplexity - 50) / 10))
            
            # Additional quality factors
            length_score = min(1.0, len(text.split()) / 100)  # Prefer reasonable length
            
            # Combine scores
            final_score = (quality_score * 0.8) + (length_score * 0.2)
            
            return {
                'score': final_score,
                'reason': 'model_evaluation',
                'perplexity': perplexity,
                'length': len(text),
                'word_count': len(text.split()),
                'quality_factors': {
                    'perplexity_score': quality_score,
                    'length_score': length_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return {
                'score': 0.0,
                'reason': 'error',
                'error': str(e),
                'perplexity': float('inf'),
                'length': len(text) if text else 0
            }
    
    def should_keep_sample(self, sample: Dict[str, Any]) -> bool:
        """Determine if a sample should be kept based on model evaluation"""
        text = sample.get('text', '')
        
        # Basic filters
        if not text or len(text.strip()) < 20:
            return False
        
        # Calculate quality score
        quality_result = self.calculate_quality_score(text)
        
        # Store quality info in sample
        sample['quality_analysis'] = quality_result
        
        # Apply threshold
        return quality_result['score'] >= self.quality_threshold
    
    def filter_dataset(self, dataset, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Filter dataset using the model"""
        logger.info(f"Starting dataset filtering with {self.model_name}")
        logger.info(f"Quality threshold: {self.quality_threshold}")
        logger.info(f"Max samples: {max_samples or 'All'}")
        
        filtered_samples = []
        start_time = datetime.now()
        
        sample_count = 0
        
        for sample in tqdm(dataset, desc="Filtering samples"):
            if max_samples and sample_count >= max_samples:
                break
            
            try:
                # Process sample
                sample_dict = dict(sample) if hasattr(sample, 'items') else sample
                
                # Check if should keep
                if self.should_keep_sample(sample_dict):
                    filtered_samples.append(sample_dict)
                    self.stats['filtered_count'] += 1
                    self.stats['high_quality_count'] += 1
                else:
                    self.stats['low_quality_count'] += 1
                
                # Track score distribution
                if 'quality_analysis' in sample_dict:
                    self.stats['score_distribution'].append(
                        sample_dict['quality_analysis']['score']
                    )
                
                self.stats['total_processed'] += 1
                sample_count += 1
                
                # Progress update
                if sample_count % 100 == 0:
                    retention_rate = (self.stats['filtered_count'] / sample_count) * 100
                    logger.info(f"Processed {sample_count} samples, kept {self.stats['filtered_count']} ({retention_rate:.1f}%)")
                
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                self.stats['error_count'] += 1
                continue
        
        # Calculate final statistics
        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        
        logger.info(f"Filtering complete!")
        logger.info(f"Total processed: {self.stats['total_processed']}")
        logger.info(f"Samples kept: {self.stats['filtered_count']}")
        logger.info(f"Retention rate: {(self.stats['filtered_count'] / self.stats['total_processed']) * 100:.1f}%")
        
        return filtered_samples
    
    def save_filtered_data(self, filtered_samples: List[Dict[str, Any]], output_path: str):
        """Save filtered samples to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in filtered_samples:
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved {len(filtered_samples)} filtered samples to {output_path}")
        
        # Save statistics
        stats_path = output_path.with_suffix('.stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        logger.info(f"Saved statistics to {stats_path}")
    
    def print_analysis_report(self):
        """Print detailed analysis report"""
        print("\n" + "="*60)
        print(f"MODEL-BASED FILTERING REPORT")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Quality Threshold: {self.quality_threshold}")
        
        print(f"\nProcessing Statistics:")
        print(f"  Total Processed: {self.stats['total_processed']:,}")
        print(f"  Samples Kept: {self.stats['filtered_count']:,}")
        print(f"  Samples Rejected: {self.stats['low_quality_count']:,}")
        print(f"  Errors: {self.stats['error_count']:,}")
        print(f"  Processing Time: {self.stats['processing_time']:.2f}s")
        
        if self.stats['score_distribution']:
            scores = self.stats['score_distribution']
            print(f"\nQuality Score Distribution:")
            print(f"  Mean: {np.mean(scores):.3f}")
            print(f"  Median: {np.median(scores):.3f}")
            print(f"  Min: {np.min(scores):.3f}")
            print(f"  Max: {np.max(scores):.3f}")
            print(f"  Std: {np.std(scores):.3f}")
        
        retention_rate = (self.stats['filtered_count'] / self.stats['total_processed']) * 100 if self.stats['total_processed'] > 0 else 0
        print(f"\nRetention Rate: {retention_rate:.1f}%")
        
        print("="*60)


def main():
    """Main function to run the filtering"""
    # Configuration
    MODEL_NAME = "my_model"
    MAX_SAMPLES = 1000
    QUALITY_THRESHOLD = 0.5
    OUTPUT_PATH = "data/filtered_dataset/model_filtered_data.jsonl"
    
    print("🚀 Starting Model-Based Dataset Filtering")
    print(f"Model: {MODEL_NAME}")
    print(f"Max Samples: {MAX_SAMPLES}")
    print(f"Quality Threshold: {QUALITY_THRESHOLD}")
    
    try:
        # Initialize filter
        filter_system = ModelBasedFilter(
            model_name=MODEL_NAME,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        filter_system.quality_threshold = QUALITY_THRESHOLD
        
        # Load dataset
        print("\n📊 Loading dataset...")
        try:
            # Try to load ClimbLab dataset
            dataset = load_dataset("OptimalScale/ClimbLab", split="train", streaming=True)
            print("✅ ClimbLab dataset loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading ClimbLab dataset: {e}")
            print("📝 Using dummy dataset for testing")
            
            # Create dummy dataset for testing
            dummy_data = [
                {"text": "This is a high-quality text sample with good structure and meaningful content."},
                {"text": "Short text."},
                {"text": "Another well-written example that demonstrates proper language usage and contains sufficient information to be useful."},
                {"text": "Bad txt w/ poor grammar & structure..."},
                {"text": "A comprehensive explanation of machine learning concepts including supervised learning, unsupervised learning, and reinforcement learning techniques."},
            ] * 200  # Create enough samples for testing
            
            dataset = Dataset.from_list(dummy_data)
        
        # Filter dataset
        print("\n🔍 Filtering dataset...")
        filtered_samples = filter_system.filter_dataset(dataset, max_samples=MAX_SAMPLES)
        
        # Save results
        print("\n💾 Saving filtered data...")
        filter_system.save_filtered_data(filtered_samples, OUTPUT_PATH)
        
        # Print analysis
        filter_system.print_analysis_report()
        
        # Show sample results
        print(f"\n📋 Sample Filtered Results:")
        for i, sample in enumerate(filtered_samples[:3]):
            print(f"\nSample {i+1}:")
            print(f"  Text: {sample['text'][:100]}...")
            if 'quality_analysis' in sample:
                print(f"  Quality Score: {sample['quality_analysis']['score']:.3f}")
                print(f"  Perplexity: {sample['quality_analysis']['perplexity']:.2f}")
        
        print(f"\n✅ Filtering complete! Results saved to: {OUTPUT_PATH}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()

