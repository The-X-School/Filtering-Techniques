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

# Configuration
HF_TOKEN = "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"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

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

def filter_dataset_with_toxic_bert(dataset_stream, toxicity_threshold=0.7, max_samples=None, batch_size=32):
    """
    Filter the ClimbLab dataset using toxic-bert with batch processing
    
    Args:
        dataset_stream: Your ClimbLab dataset stream
        toxicity_threshold: Confidence threshold (0.1-0.9, higher = more strict)
        max_samples: Limit samples for testing (None = process all)
        batch_size: Number of samples to process in each batch
    
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

print(f"\n⚙️  CURRENT SETTINGS:")
print(f"   📊 Samples to process: {NUM_SAMPLES_TO_PROCESS:,}")
print(f"   🎯 Toxicity threshold: {TOXICITY_THRESHOLD}")
print(f"   📁 Dataset variable: ds")

print(f"\n🚀 STARTING FILTERING...")

# Start filtering with your dataset
safe_samples, toxic_samples, filtering_stats = filter_dataset_with_toxic_bert(
    ds,                           # Your dataset variable
    toxicity_threshold=TOXICITY_THRESHOLD,
    max_samples=NUM_SAMPLES_TO_PROCESS
)

# Print detailed results
print_filtering_results(safe_samples, toxic_samples, filtering_stats)

print(f"\n✅ FILTERING COMPLETE!")
print(f"📦 Safe samples ready to use: {len(safe_samples)}")
print(f"🚨 Toxic samples flagged: {len(toxic_samples)}")
print(f"💾 Both lists are stored in: safe_samples, toxic_samples")

print(f"\n🔄 TO PROCESS MORE SAMPLES:")
print(f"   Just change NUM_SAMPLES_TO_PROCESS = {NUM_SAMPLES_TO_PROCESS} to a higher number")
print(f"   Then run the cell again!")

print("\n🎛️ THRESHOLD GUIDE:")
print("   0.5 = Lenient (catches obvious toxicity)")
print("   0.7 = Balanced (recommended)")
print("   0.8 = Strict (very cautious)")  
print("   0.9 = Very strict (might over-filter)")

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