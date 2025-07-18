# ClimbLab Dataset Preselection with Content Safety and Model Testing
import os
import torch
from huggingface_hub import login
from datasets import load_dataset, Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BartTokenizer, BartForConditionalGeneration
from collections import defaultdict, Counter
import time
import json
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path
import hashlib
from torch.nn import functional as F
from transformers import BartForSequenceClassification

# Configuration
HF_TOKEN = "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"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Paths
CACHE_DIR = "filter_cache"
EXPORT_DIR = "filtered_data"
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)

# Category definitions
CATEGORIES = {
    'roleplay': {
        'name': 'Interactive Response',
        'description': 'Content showing dialogue, persona maintenance, and contextual interaction',
        'patterns': [
            'conversation between',
            'role-playing',
            'character responds',
            'in this scenario',
            'acting as'
        ]
    },
    'reasoning': {
        'name': 'Problem Solving',
        'description': 'Content demonstrating logical steps, analysis, and problem breakdown',
        'patterns': [
            'step by step',
            'let\'s analyze',
            'first we need to',
            'the solution is',
            'here\'s how'
        ]
    },
    'function_calling': {
        'name': 'Command Execution',
        'description': 'Content with API usage, function calls, and structured commands',
        'patterns': [
            'function',
            'api call',
            'command',
            'def ',
            'return'
        ]
    },
    'rag': {
        'name': 'Knowledge Integration',
        'description': 'Content showing fact integration and knowledge synthesis',
        'patterns': [
            'according to',
            'research shows',
            'studies indicate',
            'evidence suggests',
            'source:'
        ]
    }
}

# Login to Hugging Face
login(token=HF_TOKEN)

# Load the dataset in streaming mode
print("🔄 Loading ClimbLab dataset...")
try:
    ds = load_dataset("OptimalScale/ClimbLab", split="train", streaming=True)
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("📦 Make sure you're connected to the internet and have access to the dataset")
    exit()

# Initialize toxic-bert classifier
print("🔄 Loading toxic-bert model...")
try:
    toxicity_classifier = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512
    )
    print("✅ Toxic-BERT model loaded successfully!")
    if torch.cuda.is_available():
        print("🚀 Using GPU acceleration")
    else:
        print("💻 Using CPU (slower but works)")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("📦 Make sure to install: pip install transformers torch")
    exit()

class ContentSafetyFilter:
    """Toxic-BERT based content safety filter"""
    
    def __init__(self):
        print("Loading Toxic-BERT model...")
        self.toxicity_classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512
        )
        
    def is_safe_content(self, text, threshold=0.7):
        """Check if content is safe (non-toxic)"""
        try:
            result = self.toxicity_classifier(text[:512])  # Truncate long texts
            # toxic-bert returns TOXIC/NON_TOXIC labels
            if isinstance(result, list):
                result = result[0]
            
            is_toxic = result['label'] == 'TOXIC' and result['score'] > threshold
            return not is_toxic, result['score']
        except Exception as e:
            print(f"Error in toxicity check: {e}")
            return True, 0.0  # Default to safe if error

class DatasetPreprocessor:
    """Preprocess and filter ClimbLab dataset"""
    
    def __init__(self, safety_filter):
        self.safety_filter = safety_filter
        self.stats = defaultdict(int)
        
    def preprocess_sample(self, sample):
        """Preprocess a single sample from ClimbLab"""
        # Extract text content based on ClimbLab structure
        text_content = ""
        
        # ClimbLab typically has 'conversations' or 'text' fields
        if 'conversations' in sample:
            # Handle conversation format
            for conv in sample['conversations']:
                if isinstance(conv, dict) and 'value' in conv:
                    text_content += conv['value'] + " "
        elif 'text' in sample:
            text_content = sample['text']
        elif 'instruction' in sample and 'output' in sample:
            text_content = sample['instruction'] + " " + sample['output']
        else:
            # Try to extract any text field
            for key, value in sample.items():
                if isinstance(value, str):
                    text_content += value + " "
        
        return text_content.strip()
    
    def filter_and_preselect(self, dataset_stream, max_samples=10000, quality_threshold=0.3, 
                           batch_size=50, enable_safety_filter=False):
        """Filter dataset and preselect high-quality samples with batching"""
        selected_samples = []
        processed_count = 0
        batch_texts = []
        batch_samples = []
        
        print(f"Processing ClimbLab dataset (max {max_samples} samples)...")
        print(f"Safety filtering: {'Enabled' if enable_safety_filter else 'Disabled (faster)'}")
        
        start_time = time.time()
        
        for sample in tqdm(dataset_stream, desc="Filtering samples", total=max_samples):
            if processed_count >= max_samples:
                break
                
            text_content = self.preprocess_sample(sample)
            
            # Quick length check
            if len(text_content.strip()) < 10:
                self.stats['too_short'] += 1
                processed_count += 1
                continue
            
            # Quality heuristics (fast check first)
            quality_score = self.calculate_quality_score(text_content)
            
            if quality_score < quality_threshold:
                self.stats['low_quality'] += 1
                processed_count += 1
                continue
            
            # Add to batch for safety checking
            batch_texts.append(text_content)
            batch_samples.append({
                'text': text_content,
                'quality_score': quality_score,
                'length': len(text_content),
                'original_sample': sample
            })
            
            processed_count += 1
            
            # Process batch when full or at end
            if len(batch_texts) >= batch_size or processed_count >= max_samples:
                if enable_safety_filter:
                    safe_samples = self._process_safety_batch(batch_texts, batch_samples)
                else:
                    # Skip safety filtering for speed
                    safe_samples = [(sample, 0.0) for sample in batch_samples]
                
                for sample, toxicity_score in safe_samples:
                    sample['toxicity_score'] = toxicity_score
                    selected_samples.append(sample)
                    self.stats['selected'] += 1
                
                # Clear batch
                batch_texts.clear()
                batch_samples.clear()
                
                # Progress update
                if processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed
                    print(f"Processed {processed_count} samples ({rate:.1f} samples/sec)")
        
        print(f"\nFiltering Statistics:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
        
        return selected_samples
    
    def _process_safety_batch(self, texts, samples):
        """Process a batch of texts for safety filtering"""
        safe_samples = []
        
        for text, sample in zip(texts, samples):
            try:
                is_safe, toxicity_score = self.safety_filter.is_safe_content(text)
                if is_safe:
                    safe_samples.append((sample, toxicity_score))
                else:
                    self.stats['filtered_toxic'] += 1
            except Exception as e:
                # On error, default to safe
                safe_samples.append((sample, 0.0))
        
        return safe_samples
    
    def calculate_quality_score(self, text):
        """Calculate quality score based on various heuristics"""
        score = 0.0
        
        # Length scoring (prefer moderate lengths)
        length = len(text)
        if 50 <= length <= 2000:
            score += 0.3
        elif 20 <= length < 50 or 2000 < length <= 5000:
            score += 0.1
        
        # Diversity scoring (character and word diversity)
        unique_chars = len(set(text.lower()))
        char_diversity = unique_chars / max(len(text), 1)
        score += min(char_diversity * 0.3, 0.3)
        
        words = text.split()
        if len(words) > 0:
            unique_words = len(set(word.lower() for word in words))
            word_diversity = unique_words / len(words)
            score += min(word_diversity * 0.2, 0.2)
        
        # Structure scoring (presence of proper sentences)
        sentence_endings = text.count('.') + text.count('!') + text.count('?')
        if sentence_endings > 0:
            score += 0.2
        
        return min(score, 1.0)

class ModelTester:
    """Test selected data with Llama-400M-12L and BART"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        
    def load_models(self):
        """Load Llama-400M-12L and BART models"""
        print("Loading models for testing...")
        
        try:
            # Load Llama-400M-12L
            print("Loading Llama-400M-12L...")
            self.tokenizers['llama'] = AutoTokenizer.from_pretrained("data4elm/Llama-400M-12L")
            self.models['llama'] = AutoModelForCausalLM.from_pretrained(
                "data4elm/Llama-400M-12L",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Add padding token if not present
            if self.tokenizers['llama'].pad_token is None:
                self.tokenizers['llama'].pad_token = self.tokenizers['llama'].eos_token
            
        except Exception as e:
            print(f"Error loading Llama model: {e}")
        
        try:
            # Load BART
            print("Loading BART...")
            self.tokenizers['bart'] = BartTokenizer.from_pretrained("facebook/bart-base")
            self.models['bart'] = BartForConditionalGeneration.from_pretrained(
                "facebook/bart-base",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
        except Exception as e:
            print(f"Error loading BART model: {e}")
    
    def test_sample_with_llama(self, text, max_length=100):
        """Test a sample with Llama-400M-12L"""
        if 'llama' not in self.models:
            return None
        
        try:
            # Tokenize input
            inputs = self.tokenizers['llama'].encode(
                text[:500],  # Truncate input
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(self.models['llama'].device)
            
            # Generate
            with torch.no_grad():
                outputs = self.models['llama'].generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizers['llama'].eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizers['llama'].decode(
                outputs[0][inputs.shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error in Llama generation: {e}")
            return None
    
    def test_sample_with_bart(self, text, max_length=100):
        """Test a sample with BART (summarization/generation)"""
        if 'bart' not in self.models:
            return None
        
        try:
            # Tokenize input
            inputs = self.tokenizers['bart'].encode(
                text[:1000],  # BART can handle longer inputs
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.models['bart'].device)
            
            # Generate summary/continuation
            with torch.no_grad():
                outputs = self.models['bart'].generate(
                    inputs,
                    max_length=max_length,
                    min_length=10,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
            
            # Decode
            generated_text = self.tokenizers['bart'].decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error in BART generation: {e}")
            return None
    
    def evaluate_samples(self, selected_samples, num_test_samples=5):
        """Evaluate selected samples with both models"""
        print(f"\nTesting {num_test_samples} samples with loaded models...")
        
        test_results = []
        test_samples = selected_samples[:num_test_samples]
        
        for i, sample in enumerate(test_samples):
            print(f"\n--- Sample {i+1} ---")
            input_text = sample['text'][:200] + "..."
            print(f"Input: {input_text}")
            print(f"Quality Score: {sample['quality_score']:.3f}")
            
            result = {
                'input': sample['text'],
                'quality_score': sample['quality_score'],
                'llama_output': None,
                'bart_output': None
            }
            
            # Test with Llama
            if 'llama' in self.models:
                llama_output = self.test_sample_with_llama(sample['text'])
                result['llama_output'] = llama_output
                print(f"Llama Output: {llama_output}")
            
            # Test with BART
            if 'bart' in self.models:
                bart_output = self.test_sample_with_bart(sample['text'])
                result['bart_output'] = bart_output
                print(f"BART Output: {bart_output}")
            
            test_results.append(result)
        
        return test_results

def preselect_filter(text: str) -> bool:
    """
    Quickly filter out obviously low-quality or irrelevant text.
    Returns True if the sample should be KEPT, False if it should be DISCARDED.
    """
    if not text:
        return False

    # 1. Check length
    if len(text) < 15 or len(text) > 4000:
        return False

    # 2. Check for mostly numbers (e.g., phone numbers, IDs)
    num_digits = sum(c.isdigit() for c in text)
    if num_digits / len(text) > 0.5:
        return False

    # 3. Check for low alphanumeric content (e.g., symbol spam)
    alnum_chars = sum(c.isalnum() for c in text)
    if alnum_chars / len(text) < 0.6:
        return False
        
    # 4. Check for single repeating characters (e.g., "aaaaaaa...")
    if len(set(text.lower())) <= 2 and len(text) > 10:
        return False

    return True

def classify_toxicity(text, threshold=0.5):
    """
    Use toxic-bert to classify if text is toxic
    Returns: dict with toxicity classification and confidence
    """
    try:
        # Handle empty or very short text
        if not text or len(text.strip()) < 3:
            return {
                'is_toxic': False,
                'is_safe': True,
                'confidence': 0.0,
                'label': 'NON_TOXIC',
                'text_length': len(text) if text else 0
            }
        
        # Get prediction from toxic-bert
        result = toxicity_classifier(text)
        
        # Handle different response formats
        if isinstance(result, list):
            result = result[0]
        
        is_toxic = result['label'] == 'TOXIC'
        confidence = result['score']
        
        # Apply threshold for final decision
        final_toxic = is_toxic and confidence > threshold
        
        return {
            'is_toxic': final_toxic,
            'is_safe': not final_toxic,
            'confidence': confidence,
            'label': result['label'],
            'raw_score': result['score'],
            'text_length': len(text)
        }
        
    except Exception as e:
        print(f"⚠️ Error classifying text: {e}")
        # Default to safe if there's an error
        return {
            'is_toxic': False,
            'is_safe': True,
            'confidence': 0.0,
            'label': 'ERROR',
            'error': str(e),
            'text_length': len(text) if text else 0
        }

def classify_toxicity_batch(texts: List[str], threshold=0.5, batch_size=32) -> List[Dict]:
    """
    Use toxic-bert to classify multiple texts in batches for better performance
    Returns: List of dicts with toxicity classification and confidence
    """
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            # Get predictions for the batch
            batch_results = toxicity_classifier(batch)
            
            # Process each result in the batch
            for text, result in zip(batch, batch_results):
                if isinstance(result, dict):
                    single_result = result
                else:
                    single_result = result[0]
                
                is_toxic = single_result['label'] == 'TOXIC'
                confidence = single_result['score']
                
                # Apply threshold for final decision
                final_toxic = is_toxic and confidence > threshold
                
                results.append({
                    'is_toxic': final_toxic,
                    'is_safe': not final_toxic,
                    'confidence': confidence,
                    'label': single_result['label'],
                    'raw_score': single_result['score'],
                    'text_length': len(text)
                })
                
        except Exception as e:
            print(f"⚠️ Error in batch classification: {e}")
            # Add safe defaults for failed batch
            for text in batch:
                results.append({
                    'is_toxic': False,
                    'is_safe': True,
                    'confidence': 0.0,
                    'label': 'ERROR',
                    'error': str(e),
                    'text_length': len(text) if text else 0
                })
    
    return results

def filter_dataset_with_toxic_bert(dataset_stream, toxicity_threshold=0.7, max_samples=None, batch_size=32, use_cache=True):
    """
    Filter the ClimbLab dataset using toxic-bert with batch processing
    
    Args:
        dataset_stream: Your ClimbLab dataset stream
        toxicity_threshold: Confidence threshold (0.1-0.9, higher = more strict)
        max_samples: Limit samples for testing (None = process all)
        batch_size: Number of samples to process in each batch
        use_cache: Whether to use caching for results
    
    Returns:
        safe_data: List of clean samples
        toxic_data: List of toxic samples
        stats: Processing statistics
    """
    
    # Storage lists
    safe_data = []
    toxic_data = []
    current_batch = []
    current_batch_samples = []
    
    # Statistics tracking
    stats = {
        'total_processed': 0,
        'preselect_filtered_count': 0,
        'safe_count': 0,
        'toxic_count': 0,
        'error_count': 0,
        'empty_count': 0,
        'avg_confidence_safe': 0.0,
        'avg_confidence_toxic': 0.0
    }
    
    confidence_safe_sum = 0.0
    confidence_toxic_sum = 0.0
    
    print(f"🔍 Starting toxicity filtering with threshold: {toxicity_threshold}")
    print(f"🎯 Threshold explanation: {toxicity_threshold:.1f} means {int(toxicity_threshold*100)}% confidence required to mark as toxic")
    print("📊 Processing samples...\n")
    
    start_time = time.time()
    
    for i, sample in enumerate(dataset_stream):
        text = sample.get("text", "").strip()
        
        # Apply preselect filter first
        if not preselect_filter(text):
            stats['preselect_filtered_count'] += 1
            continue
        
        # Skip completely empty samples
        if not text:
            stats['empty_count'] += 1
            continue
        
        # Add to current batch
        current_batch.append(text)
        current_batch_samples.append((i, sample))
        
        # Process batch when full or at end
        if len(current_batch) >= batch_size or (max_samples and i >= max_samples - 1):
            # Classify toxicity for the batch
            toxicity_results = classify_toxicity_batch(current_batch, threshold=toxicity_threshold)
            
            # Process results
            for (sample_idx, sample), toxicity_result in zip(current_batch_samples, toxicity_results):
                # Create enhanced sample
                enhanced_sample = {
                    **sample,
                    'toxicity_analysis': toxicity_result,
                    'sample_id': sample_idx
                }
                
                # Categorize the sample
                if toxicity_result['is_safe']:
                    safe_data.append(enhanced_sample)
                    stats['safe_count'] += 1
                    confidence_safe_sum += toxicity_result['confidence']
                else:
                    toxic_data.append(enhanced_sample)
                    stats['toxic_count'] += 1
                    confidence_toxic_sum += toxicity_result['confidence']
                
                if 'error' in toxicity_result:
                    stats['error_count'] += 1
                
                stats['total_processed'] += 1
            
            # Progress updates
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"✅ Processed {i + 1:,} samples ({rate:.1f} samples/sec)")
            print(f"   📈 Safe: {stats['safe_count']:,} | 🚨 Toxic: {stats['toxic_count']:,}")
            print(f"   🗑️ Filtered by preselect: {stats['preselect_filtered_count']:,}")
            
            # Clear batch
            current_batch = []
            current_batch_samples = []
        
        # Check if we've reached the sample limit
        if max_samples and i >= max_samples - 1:
            print(f"🔄 Stopping at {max_samples} samples (testing mode)")
            break
    
    # Calculate averages
    if stats['safe_count'] > 0:
        stats['avg_confidence_safe'] = confidence_safe_sum / stats['safe_count']
    if stats['toxic_count'] > 0:
        stats['avg_confidence_toxic'] = confidence_toxic_sum / stats['toxic_count']
    
    # Final timing
    total_time = time.time() - start_time
    print(f"\n🏁 FILTERING COMPLETE!")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"🚀 Average rate: {stats['total_processed']/total_time:.1f} samples/second")
    
    return safe_data, toxic_data, stats

def print_filtering_results(safe_data, toxic_data, stats):
    """Print detailed results of the filtering process"""
    
    total = stats['safe_count'] + stats['toxic_count']
    
    print("\n" + "="*60)
    print("📊 FILTERING RESULTS SUMMARY")
    print("="*60)
    
    print(f"📝 Total samples processed: {stats['total_processed']:,}")
    print(f"🗑️ Filtered by preselect: {stats['preselect_filtered_count']:,}")
    print(f"📋 Empty samples skipped: {stats['empty_count']:,}")
    print(f"⚠️  Processing errors: {stats['error_count']:,}")
    
    if total > 0:
        print(f"\n✅ SAFE CONTENT: {stats['safe_count']:,} samples ({stats['safe_count']/total*100:.1f}%)")
        print(f"🚨 TOXIC CONTENT: {stats['toxic_count']:,} samples ({stats['toxic_count']/total*100:.1f}%)")
    else:
        print("\n✅ No safe or toxic content found after filtering.")
    
    print(f"\n🎯 Average confidence scores:")
    print(f"   Safe content: {stats['avg_confidence_safe']:.3f}")
    print(f"   Toxic content: {stats['avg_confidence_toxic']:.3f}")
    
    # Show some examples
    print(f"\n📋 SAMPLE TOXIC CONTENT (first 3):")
    for i, sample in enumerate(toxic_data[:3]):
        analysis = sample['toxicity_analysis']
        text_preview = sample['text'][:150] + "..." if len(sample['text']) > 150 else sample['text']
        print(f"\nToxic Sample {i+1}:")
        print(f"   Confidence: {analysis['confidence']:.3f}")
        print(f"   Text: {text_preview}")
    
    print(f"\n📋 SAMPLE SAFE CONTENT (first 3):")
    for i, sample in enumerate(safe_data[:3]):
        analysis = sample['toxicity_analysis']
        text_preview = sample['text'][:150] + "..." if len(sample['text']) > 150 else sample['text']
        print(f"\nSafe Sample {i+1}:")
        print(f"   Confidence: {analysis['confidence']:.3f}")
        print(f"   Text: {text_preview}")

print("\n" + "="*60)
print("🚀 READY TO FILTER YOUR CLIMBLAB DATASET!")
print("="*60)

# 🎛️ CUSTOMIZE THESE SETTINGS:
NUM_SAMPLES_TO_PROCESS = 1000   # 👈 CHANGE THIS NUMBER!
TOXICITY_THRESHOLD = 0.7        # 👈 ADJUST STRICTNESS (0.5-0.9)
QUALITY_THRESHOLD = 0.3         # 👈 MINIMUM QUALITY SCORE
SAVE_RESULTS = False           # 👈 Set to True to save files
USE_CACHE = False               # 👈 Set to False to disable caching

print(f"\n⚙️  CURRENT SETTINGS:")
print(f"   📊 Samples to process: {NUM_SAMPLES_TO_PROCESS:,}")
print(f"   🎯 Toxicity threshold: {TOXICITY_THRESHOLD}")
print(f"   📈 Quality threshold: {QUALITY_THRESHOLD}")
print(f"   💾 Save results: {'Yes' if SAVE_RESULTS else 'No'}")
print(f"   🔄 Use cache: {'Yes' if USE_CACHE else 'No'}")
print(f"   📁 Dataset variable: ds")

print(f"\n🚀 STARTING FILTERING...")

# Initialize cache if enabled
text_cache = TextCache() if USE_CACHE else None

# Start filtering with your dataset
safe_samples, toxic_samples, filtering_stats = filter_dataset_with_toxic_bert(
    ds,                           # Your dataset variable
    toxicity_threshold=TOXICITY_THRESHOLD,
    max_samples=NUM_SAMPLES_TO_PROCESS,
    use_cache=USE_CACHE
)

# Print detailed results
print_filtering_results(safe_samples, toxic_samples, filtering_stats)

# Initialize verification system
verifier = VerificationSystem(llama_analyzer)

# Establish baseline from initial analysis
verifier.establish_baseline(analysis_results)

# Verify filtered samples
verified_samples = verifier.verify_samples(filtered_results)

# Print verification results
print("\n📊 VERIFICATION RESULTS:")
for category, samples in verified_samples.items():
    improvements = [s['overall_improvement'] for s in samples]
    avg_improvement = np.mean(improvements) if improvements else 0
    
    print(f"\n{CATEGORIES[category]['name']}:")
    print(f"   Samples: {len(samples)}")
    print(f"   Average Improvement: {avg_improvement:+.3f}")
    
    if samples:
        best_sample = samples[0]
        print(f"   Best Sample Improvement: {best_sample['overall_improvement']:+.3f}")
        print(f"   Best Sample Preview: {best_sample['text'][:100]}...")

# Export verified samples if enabled
if SAVE_RESULTS:
    export_path = verifier.export_verified_samples(verified_samples)
    print("\n💾 Results saved successfully!")
    print(f"📁 Find them in: {export_path['run_dir']}")
else:
    print("\n💾 Results not saved (SAVE_RESULTS is False)")
    print("💡 To save results, set SAVE_RESULTS = True")

print("\n✅ VERIFICATION AND EXPORT COMPLETE!")
print("\n🎯 Next Steps:")
print("   1. Review the verified samples in each category")
print("   2. Adjust thresholds if needed")
print("   3. Run with more samples for better coverage")

def main():
    """Main execution function"""
    print("🚀 Starting ClimbLab Dataset Preselection Pipeline")
    
    # Initialize components
    safety_filter = ContentSafetyFilter()
    preprocessor = DatasetPreprocessor(safety_filter)
    model_tester = ModelTester()
    
    # Filter and preselect samples
    selected_samples = preprocessor.filter_and_preselect(
        ds, 
        max_samples=1000,  # Adjust as needed
        quality_threshold=0.3
    )
    
    if not selected_samples:
        print("No samples selected. Exiting.")
        return
    
    # Sort by quality score
    selected_samples.sort(key=lambda x: x['quality_score'], reverse=True)
    
    print(f"\nSelected {len(selected_samples)} high-quality samples")
    print(f"Average quality score: {np.mean([s['quality_score'] for s in selected_samples]):.3f}")
    
    # Load models for testing
    model_tester.load_models()
    
    # Test samples with models
    test_results = model_tester.evaluate_samples(selected_samples, num_test_samples=3)
    
    # Save results
    output_data = {
        'selected_samples': selected_samples[:100],  # Save top 100
        'test_results': test_results,
        'filtering_stats': dict(preprocessor.stats),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('climblab_preselection_results.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to 'climblab_preselection_results.json'")
    print(f"📊 Top quality samples ready for model training!")

if __name__ == "__main__":
    main()

class LlamaAnalyzer:
    """Analyzes Llama model performance on different categories"""
    
    def __init__(self):
        print("🦙 Initializing Llama model for analysis...")
        self.tokenizer = AutoTokenizer.from_pretrained("data4elm/Llama-400M-12L")
        self.model = AutoModelForCausalLM.from_pretrained(
            "data4elm/Llama-400M-12L",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Llama model loaded successfully!")
    
    def analyze_performance(self, text: str) -> Dict[str, float]:
        """
        Analyze Llama's performance on a text sample for each category
        Returns scores between 0-1 for each category
        """
        scores = {}
        
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate continuation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Score each category
            for category, details in CATEGORIES.items():
                category_score = self._score_category(text, generated_text, details['patterns'])
                scores[category] = category_score
            
        except Exception as e:
            print(f"⚠️ Error in Llama analysis: {e}")
            # Default scores on error
            scores = {category: 0.0 for category in CATEGORIES.keys()}
        
        return scores
    
    def _score_category(self, input_text: str, generated_text: str, patterns: List[str]) -> float:
        """
        Score how well Llama handles a specific category
        Uses pattern matching and basic heuristics
        """
        score = 0.0
        
        # Check for pattern presence in input
        input_pattern_matches = sum(1 for p in patterns if p.lower() in input_text.lower())
        
        # Check for pattern presence in output
        output_pattern_matches = sum(1 for p in patterns if p.lower() in generated_text.lower())
        
        # Basic coherence check
        try:
            input_words = set(input_text.lower().split())
            output_words = set(generated_text.lower().split())
            coherence = len(input_words.intersection(output_words)) / len(input_words)
        except:
            coherence = 0.0
        
        # Combine scores
        pattern_score = (input_pattern_matches + output_pattern_matches) / (len(patterns) * 2)
        
        score = (pattern_score * 0.7) + (coherence * 0.3)
        return min(max(score, 0.0), 1.0)  # Normalize to 0-1

class DatasetAnalyzer:
    """Analyzes the dataset to find samples that could improve Llama's performance"""
    
    def __init__(self, llama_analyzer: LlamaAnalyzer):
        self.llama = llama_analyzer
        self.performance_cache = {}
    
    def analyze_sample(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single sample to determine its potential value
        for improving Llama's performance
        """
        # Check cache
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if text_hash in self.performance_cache:
            return self.performance_cache[text_hash]
        
        # Get Llama's performance scores
        category_scores = self.llama.analyze_performance(text)
        
        # Calculate potential value for each category
        potential_value = {}
        for category, score in category_scores.items():
            # Lower scores indicate more room for improvement
            improvement_potential = 1.0 - score
            potential_value[category] = improvement_potential
        
        result = {
            'text': text,
            'category_scores': category_scores,
            'potential_value': potential_value,
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache the result
        self.performance_cache[text_hash] = result
        return result

def save_analysis_cache(cache: Dict, filename: str = "analysis_cache.json"):
    """Save analysis results to cache file"""
    cache_path = Path(CACHE_DIR) / filename
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
        print(f"✅ Saved analysis cache to {cache_path}")
    except Exception as e:
        print(f"⚠️ Error saving cache: {e}")

def load_analysis_cache(filename: str = "analysis_cache.json") -> Dict:
    """Load analysis results from cache file"""
    cache_path = Path(CACHE_DIR) / filename
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading cache: {e}")
    return {}

# Initialize models and analyzers
print("\n" + "="*60)
print("🚀 ENHANCED DATASET FILTERING WITH LLAMA ANALYSIS")
print("="*60)

# Login to Hugging Face
login(token=HF_TOKEN)

# Load the dataset
print("\n📚 Loading ClimbLab dataset...")
try:
    ds = load_dataset("OptimalScale/ClimbLab", split="train", streaming=True)
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("📦 Make sure you're connected to the internet and have access to the dataset")
    exit()

# Initialize Llama analyzer
llama_analyzer = LlamaAnalyzer()
dataset_analyzer = DatasetAnalyzer(llama_analyzer)

# Settings
NUM_SAMPLES_TO_ANALYZE = 100  # Start with a small sample for analysis
print(f"\n⚙️ ANALYSIS SETTINGS:")
print(f"   📊 Initial samples to analyze: {NUM_SAMPLES_TO_ANALYZE}")
print(f"   🎯 Categories: {', '.join(CATEGORIES.keys())}")

# Analyze initial samples
print("\n🔍 Starting initial analysis...")
analysis_results = []
for i, sample in enumerate(ds):
    if i >= NUM_SAMPLES_TO_ANALYZE:
        break
    
    text = sample.get("text", "").strip()
    if text:
        result = dataset_analyzer.analyze_sample(text)
        analysis_results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"✓ Analyzed {i + 1} samples")

# Save analysis results
save_analysis_cache(dataset_analyzer.performance_cache)

# Print initial analysis summary
print("\n📊 INITIAL ANALYSIS SUMMARY:")
category_averages = {
    category: np.mean([r['category_scores'][category] for r in analysis_results])
    for category in CATEGORIES.keys()
}

for category, avg_score in category_averages.items():
    print(f"   {CATEGORIES[category]['name']}: {avg_score:.2f}")

print("\n✅ Initial analysis complete! Ready for BART-guided filtering...")

class BartFilter:
    """Uses BART to filter and categorize content based on Llama's needs"""
    
    def __init__(self):
        print("🔄 Initializing BART model for filtering...")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        
        # Load one model for each category
        self.category_models = {}
        for category in CATEGORIES.keys():
            try:
                model = BartForSequenceClassification.from_pretrained(
                    "facebook/bart-base",
                    num_labels=2,  # Binary classification for each category
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                self.category_models[category] = model
            except Exception as e:
                print(f"⚠️ Error loading BART model for {category}: {e}")
        
        print("✅ BART models loaded successfully!")
    
    def _prepare_examples(self, category: str, analysis_results: List[Dict]) -> List[Dict]:
        """Prepare training examples for a category based on Llama's analysis"""
        examples = []
        for result in analysis_results:
            score = result['category_scores'][category]
            text = result['text']
            
            # High scores (>0.7) are positive examples
            if score > 0.7:
                examples.append({
                    'text': text,
                    'label': 1,  # Positive example
                    'score': score
                })
            # Low scores (<0.3) with high potential are negative examples
            elif score < 0.3 and result['potential_value'][category] > 0.7:
                examples.append({
                    'text': text,
                    'label': 0,  # Negative example
                    'score': score
                })
        return examples
    
    def train_on_analysis(self, analysis_results: List[Dict]):
        """Fine-tune BART models based on Llama's analysis results"""
        print("\n🔄 Training BART models on analysis results...")
        
        for category in CATEGORIES.keys():
            examples = self._prepare_examples(category, analysis_results)
            if len(examples) < 10:  # Need minimum examples
                print(f"⚠️ Not enough examples for {category}, skipping training")
                continue
            
            print(f"📚 Training {category} model on {len(examples)} examples...")
            model = self.category_models[category]
            
            # Prepare training data
            texts = [ex['text'] for ex in examples]
            labels = torch.tensor([ex['label'] for ex in examples])
            
            # Tokenize
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            encodings = {k: v.to(model.device) for k, v in encodings.items()}
            labels = labels.to(model.device)
            
            # Train for a few steps
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            
            for epoch in range(3):  # Quick fine-tuning
                optimizer.zero_grad()
                outputs = model(**encodings, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            print(f"✅ Completed training for {category}")
    
    def filter_sample(self, text: str, category_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Filter a sample based on trained category models
        category_weights: importance of each category based on Llama's needs
        """
        results = {}
        
        # Get predictions for each category
        for category, model in self.category_models.items():
            try:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Get prediction
                model.eval()
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                    
                # Get positive class probability
                positive_prob = probs[0][1].item()
                
                # Apply category weight
                weighted_score = positive_prob * category_weights.get(category, 1.0)
                
                results[category] = {
                    'raw_score': positive_prob,
                    'weighted_score': weighted_score
                }
            
            except Exception as e:
                print(f"⚠️ Error in BART filtering for {category}: {e}")
                results[category] = {
                    'raw_score': 0.0,
                    'weighted_score': 0.0,
                    'error': str(e)
                }
        
        # Calculate overall value
        overall_score = sum(cat['weighted_score'] for cat in results.values()) / len(results)
        
        return {
            'text': text,
            'category_scores': results,
            'overall_score': overall_score,
            'timestamp': datetime.now().isoformat()
        }

# Initialize BART filter
bart_filter = BartFilter()

# Train BART on analysis results
bart_filter.train_on_analysis(analysis_results)

# Calculate category weights based on Llama's needs
category_weights = {}
for category, avg_score in category_averages.items():
    # Lower Llama scores mean higher weight for BART filtering
    weight = 1.0 + (1.0 - avg_score)  # Range: 1.0-2.0
    category_weights[category] = weight

print("\n⚖️ Category Weights for Filtering:")
for category, weight in category_weights.items():
    print(f"   {CATEGORIES[category]['name']}: {weight:.2f}")

# Start BART-guided filtering
print("\n🔍 Starting BART-guided filtering...")
filtered_results = []
filtered_count = 0

for i, sample in enumerate(ds):
    if filtered_count >= NUM_SAMPLES_TO_ANALYZE:  # Use same sample size for now
        break
    
    text = sample.get("text", "").strip()
    if not text:
        continue
    
    # Apply BART filtering
    filter_result = bart_filter.filter_sample(text, category_weights)
    
    # Keep samples with good overall scores
    if filter_result['overall_score'] > 0.5:  # Adjustable threshold
        filtered_results.append(filter_result)
        filtered_count += 1
        
        if filtered_count % 10 == 0:
            print(f"✓ Found {filtered_count} valuable samples")

# Save filtered results
filtered_cache_path = Path(CACHE_DIR) / "bart_filtered_results.json"
try:
    with open(filtered_cache_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, indent=2)
    print(f"\n✅ Saved filtered results to {filtered_cache_path}")
except Exception as e:
    print(f"⚠️ Error saving filtered results: {e}")

# Print filtering summary
print("\n📊 FILTERING SUMMARY:")
print(f"   Total samples processed: {i+1}")
print(f"   Valuable samples found: {len(filtered_results)}")

# Category distribution in filtered results
category_distribution = defaultdict(int)
for result in filtered_results:
    best_category = max(
        result['category_scores'].items(),
        key=lambda x: x[1]['weighted_score']
    )[0]
    category_distribution[best_category] += 1

print("\n📈 Category Distribution:")
for category, count in category_distribution.items():
    percentage = (count / len(filtered_results)) * 100
    print(f"   {CATEGORIES[category]['name']}: {count} samples ({percentage:.1f}%)")

print("\n✅ BART-guided filtering complete! Ready for final verification...")

class VerificationSystem:
    """Verifies that filtered samples actually improve Llama's performance"""
    
    def __init__(self, llama_analyzer: LlamaAnalyzer):
        self.llama = llama_analyzer
        self.baseline_scores = None
    
    def establish_baseline(self, initial_analysis_results: List[Dict]):
        """Calculate baseline performance from initial analysis"""
        print("\n📊 Establishing baseline performance...")
        
        category_scores = defaultdict(list)
        for result in initial_analysis_results:
            for category, score in result['category_scores'].items():
                category_scores[category].append(score)
        
        self.baseline_scores = {
            category: {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': min(scores),
                'max': max(scores)
            }
            for category, scores in category_scores.items()
        }
        
        print("✅ Baseline established:")
        for category, stats in self.baseline_scores.items():
            print(f"   {CATEGORIES[category]['name']}:")
            print(f"      Mean: {stats['mean']:.3f}")
            print(f"      Range: {stats['min']:.3f} - {stats['max']:.3f}")
    
    def verify_samples(self, filtered_results: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Verify filtered samples improve Llama's performance
        Returns samples grouped by category with verification scores
        """
        if not self.baseline_scores:
            raise ValueError("Must establish baseline before verification")
        
        print("\n🔍 Starting verification of filtered samples...")
        
        # Group samples by their best category
        categorized_samples = defaultdict(list)
        for result in filtered_results:
            # Find best category from BART scores
            best_category = max(
                result['category_scores'].items(),
                key=lambda x: x[1]['weighted_score']
            )[0]
            
            # Get new Llama score for verification
            text = result['text']
            verification_scores = self.llama.analyze_performance(text)
            
            # Calculate improvement over baseline
            improvements = {}
            for category, score in verification_scores.items():
                baseline = self.baseline_scores[category]['mean']
                improvement = score - baseline
                improvements[category] = improvement
            
            # Add verification data
            verified_sample = {
                'text': text,
                'bart_scores': result['category_scores'],
                'llama_scores': verification_scores,
                'improvements': improvements,
                'overall_improvement': improvements[best_category],  # Focus on main category
                'timestamp': datetime.now().isoformat()
            }
            
            categorized_samples[best_category].append(verified_sample)
        
        # Sort samples by improvement within each category
        for category in categorized_samples:
            categorized_samples[category].sort(
                key=lambda x: x['overall_improvement'],
                reverse=True
            )
        
        return dict(categorized_samples)
    
    def export_verified_samples(self, 
                              categorized_samples: Dict[str, List[Dict]], 
                              export_dir: str = "filtered_data"):
        """Export verified samples with detailed metadata"""
        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = Path(export_dir) / f"run_{timestamp}_preselect"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare overall metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'baseline_scores': self.baseline_scores,
            'categories': CATEGORIES,
            'stats': {
                category: {
                    'count': len(samples),
                    'avg_improvement': np.mean([s['overall_improvement'] for s in samples])
                }
                for category, samples in categorized_samples.items()
            }
        }
        
        # Save metadata
        metadata_file = run_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Export samples by category
        category_files = {}
        for category, samples in categorized_samples.items():
            category_file = run_dir / f"{category}_samples.json"
            with open(category_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2)
            category_files[category] = str(category_file.relative_to(export_dir))
        
        # Create a summary file for quick reference
        summary = {
            'run_timestamp': timestamp,
            'run_type': 'preselect_filter',
            'total_samples': sum(len(samples) for samples in categorized_samples.values()),
            'category_counts': {cat: len(samples) for cat, samples in categorized_samples.items()},
            'files': {
                'metadata': str(metadata_file.relative_to(export_dir)),
                'categories': category_files
            }
        }
        summary_file = run_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📁 Exported verified samples to {run_dir}")
        print(f"   📊 Metadata: {metadata_file}")
        print("\n   Category files:")
        for category, file_path in category_files.items():
            print(f"   • {category}: {Path(export_dir) / file_path}")
        print(f"   📑 Summary: {summary_file}")
        
        return {
            'run_dir': str(run_dir),
            'metadata_file': str(metadata_file),
            'category_files': category_files,
            'summary_file': str(summary_file)
        }