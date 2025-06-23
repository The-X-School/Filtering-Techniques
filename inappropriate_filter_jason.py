import re
import json
import hashlib
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path

from huggingface_hub import login
from datasets import load_dataset
# Install dependencies (only needed once per session)
# 🔐 Paste your token between the quotes (keep it private!)
HF_TOKEN = "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"
login(token=HF_TOKEN)

# Load the dataset in streaming mode
ds = load_dataset("OptimalScale/ClimbLab", split="train", streaming=True)

# Toxic-BERT Content Safety Filter for ClimbLab Dataset
from transformers import pipeline
import torch
from collections import defaultdict
import time
import numpy as np
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

print("🔄 Loading toxic-bert model...")

# Initialize toxic-bert classifier
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

# 🔧 IMPROVEMENT: Enhanced text preprocessing
def preprocess_text(text: str) -> str:
    """
    Clean and normalize text for better classification
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove common spam patterns
    spam_patterns = [
        r'\b(click here|buy now|free offer|limited time|act now)\b',
        r'\b(www\.|http://|https://)\S+',
        r'\b[A-Z]{5,}\b',  # Excessive caps
        r'[!]{2,}',  # Multiple exclamation marks
        r'[?]{2,}',  # Multiple question marks
    ]
    
    for pattern in spam_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove excessive punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
    
    return text.strip()

# 🔧 IMPROVEMENT: Rule-based filtering
def rule_based_filter(text: str) -> Dict[str, Any]:
    """
    Apply rule-based filtering before ML classification
    """
    if not text:
        return {'is_inappropriate': False, 'reasons': [], 'confidence': 0.0}
    
    reasons = []
    confidence = 0.0
    
    # Check for obvious inappropriate content
    inappropriate_patterns = {
        'profanity': [
            r'\b(fuck|shit|bitch|asshole|dick|pussy|cunt)\b',
            r'\b(damn|hell|god damn)\b'
        ],
        'hate_speech': [
            r'\b(kill yourself|die|hate you|stupid|idiot|moron)\b',
            r'\b(racist|sexist|homophobic)\b'
        ],
        'violence': [
            r'\b(punch|hit|kill|murder|attack|fight)\b',
            r'\b(weapon|gun|knife|bomb)\b'
        ],
        'sexual_content': [
            r'\b(sex|porn|nude|naked|penis|vagina)\b',
            r'\b(erotic|sexual|intimate)\b'
        ]
    }
    
    text_lower = text.lower()
    
    for category, patterns in inappropriate_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                reasons.append(category)
                confidence += 0.3  # Each match adds confidence
    
    # Check for excessive repetition
    words = text.split()
    if len(words) > 10:
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word.lower()] += 1
        
        max_repetition = max(word_counts.values()) if word_counts else 0
        if max_repetition > len(words) * 0.3:  # More than 30% repetition
            reasons.append('excessive_repetition')
            confidence += 0.2
    
    # Check for all caps (shouting)
    if len(text) > 20 and text.isupper():
        reasons.append('excessive_caps')
        confidence += 0.1
    
    # Check for very short or very long text
    if len(text) < 10:
        reasons.append('too_short')
        confidence += 0.1
    elif len(text) > 5000:
        reasons.append('too_long')
        confidence += 0.1
    
    return {
        'is_inappropriate': len(reasons) > 0,
        'reasons': reasons,
        'confidence': min(confidence, 1.0)
    }

# Add cache functionality
class TextCache:
    """Cache for storing classification results"""
    def __init__(self, cache_file="filter_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache from file if it exists"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate a hash for the text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Dict:
        """Get cached result for text if it exists"""
        text_hash = self._get_text_hash(text)
        return self.cache.get(text_hash)
    
    def put(self, text: str, result: Dict):
        """Cache result for text"""
        text_hash = self._get_text_hash(text)
        self.cache[text_hash] = result
        # Save every 100 new entries
        if len(self.cache) % 100 == 0:
            self._save_cache()

# Modify classify_toxicity to use cache
def classify_toxicity(text, threshold=0.5):
    """
    Use toxic-bert to classify if text is toxic
    Returns: dict with toxicity classification and confidence
    """
    # Check cache first
    cached_result = text_cache.get(text)
    if cached_result:
        return cached_result
    
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
        
        # 🔧 IMPROVEMENT: Enhanced preprocessing
        cleaned_text = preprocess_text(text)
        
        # 🔧 IMPROVEMENT: Rule-based pre-filtering
        rule_result = rule_based_filter(cleaned_text)
        
        # If rule-based filter catches obvious issues, skip ML
        if rule_result['is_inappropriate'] and rule_result['confidence'] > 0.5:
            return {
                'is_toxic': True,
                'is_safe': False,
                'confidence': rule_result['confidence'],
                'label': 'TOXIC',
                'raw_score': rule_result['confidence'],
                'text_length': len(text),
                'was_truncated': len(text) > 2000,
                'rule_based_reasons': rule_result['reasons']
            }
        
        # 🔧 IMPROVEMENT: Handle very long texts (toxic-bert has 512 token limit)
        if len(cleaned_text) > 2000:  # Rough character limit
            # Take first part + last part for context
            cleaned_text = cleaned_text[:1000] + "..." + cleaned_text[-500:]
        
        # Get prediction from toxic-bert
        result = toxicity_classifier(cleaned_text)
        
        # Handle different response formats
        if isinstance(result, list):
            result = result[0]
        
        is_toxic = result['label'] == 'TOXIC'
        confidence = result['score']
        
        # 🔧 IMPROVEMENT: Combine rule-based and ML results
        if rule_result['is_inappropriate']:
            # Boost confidence if both methods agree
            confidence = min(1.0, confidence + rule_result['confidence'] * 0.2)
        
        # Apply threshold for final decision
        final_toxic = is_toxic and confidence > threshold
        
        result = {
            'is_toxic': final_toxic,
            'is_safe': not final_toxic,
            'confidence': confidence,
            'label': result['label'],
            'raw_score': result['score'],
            'text_length': len(text),
            'was_truncated': len(text) > 2000,
            'rule_based_reasons': rule_result['reasons'] if rule_result['is_inappropriate'] else [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache the result
        text_cache.put(text, result)
        return result
        
    except Exception as e:
        error_result = {
            'is_toxic': False,
            'is_safe': True,
            'confidence': 0.0,
            'label': 'ERROR',
            'error': str(e),
            'text_length': len(text) if text else 0,
            'timestamp': datetime.now().isoformat()
        }
        # Cache errors too
        text_cache.put(text, error_result)
        return error_result

# 🔧 IMPROVEMENT: Enhanced filtering with multiple criteria
def filter_dataset_with_toxic_bert(dataset_stream, toxicity_threshold=0.7, max_samples=None, 
                                   save_progress=True, batch_size=100, 
                                   enable_rule_based=True, enable_ml=True, use_cache=True):
    """
    Filter the ClimbLab dataset using toxic-bert and rule-based filtering
    
    Args:
        dataset_stream: Your ClimbLab dataset stream
        toxicity_threshold: Confidence threshold (0.1-0.9, higher = more strict)
        max_samples: Limit samples for testing (None = process all)
        save_progress: Save intermediate results every 500 samples
        batch_size: How often to print progress updates
        enable_rule_based: Use rule-based filtering
        enable_ml: Use ML-based filtering
        use_cache: Use cache for classification results
    
    Returns:
        safe_data: List of clean samples
        toxic_data: List of toxic samples
        stats: Processing statistics
    """
    
    # Storage lists
    safe_data = []
    toxic_data = []
    
    # 🔧 IMPROVEMENT: Enhanced statistics tracking
    stats = {
        'total_processed': 0,
        'safe_count': 0,
        'toxic_count': 0,
        'error_count': 0,
        'empty_count': 0,
        'avg_confidence_safe': 0.0,
        'avg_confidence_toxic': 0.0,
        'text_length_stats': {
            'min': float('inf'),
            'max': 0,
            'avg': 0,
            'total_chars': 0
        },
        'toxicity_distribution': {
            'very_low': 0,    # 0.0-0.3
            'low': 0,         # 0.3-0.5
            'medium': 0,      # 0.5-0.7
            'high': 0,        # 0.7-0.9
            'very_high': 0    # 0.9-1.0
        },
        'rule_based_catches': 0,
        'ml_catches': 0,
        'both_catches': 0
    }
    
    confidence_safe_sum = 0.0
    confidence_toxic_sum = 0.0
    
    print(f"🔍 Starting enhanced toxicity filtering with threshold: {toxicity_threshold}")
    print(f"🎯 Threshold explanation: {toxicity_threshold:.1f} means {int(toxicity_threshold*100)}% confidence required to mark as toxic")
    print(f"🔧 Rule-based filtering: {'✅ Enabled' if enable_rule_based else '❌ Disabled'}")
    print(f"🤖 ML-based filtering: {'✅ Enabled' if enable_ml else '❌ Disabled'}")
    print("📊 Processing samples...\n")
    
    start_time = time.time()
    
    for i, sample in enumerate(dataset_stream):
        text = sample.get("text", "").strip()
        
        # Skip completely empty samples
        if not text:
            stats['empty_count'] += 1
            continue
        
        # 🔧 IMPROVEMENT: Track text length statistics
        text_len = len(text)
        stats['text_length_stats']['min'] = min(stats['text_length_stats']['min'], text_len)
        stats['text_length_stats']['max'] = max(stats['text_length_stats']['max'], text_len)
        stats['text_length_stats']['total_chars'] += text_len
        
        # Classify toxicity
        toxicity_result = classify_toxicity(text, threshold=toxicity_threshold)
        
        # 🔧 IMPROVEMENT: Track filtering method statistics
        if 'rule_based_reasons' in toxicity_result and toxicity_result['rule_based_reasons']:
            stats['rule_based_catches'] += 1
        if toxicity_result['label'] == 'TOXIC' and toxicity_result['raw_score'] > toxicity_threshold:
            stats['ml_catches'] += 1
        if (toxicity_result['rule_based_reasons'] and 
            toxicity_result['label'] == 'TOXIC' and toxicity_result['raw_score'] > toxicity_threshold):
            stats['both_catches'] += 1
        
        # 🔧 IMPROVEMENT: Track toxicity score distribution
        confidence = toxicity_result['confidence']
        if confidence < 0.3:
            stats['toxicity_distribution']['very_low'] += 1
        elif confidence < 0.5:
            stats['toxicity_distribution']['low'] += 1
        elif confidence < 0.7:
            stats['toxicity_distribution']['medium'] += 1
        elif confidence < 0.9:
            stats['toxicity_distribution']['high'] += 1
        else:
            stats['toxicity_distribution']['very_high'] += 1
        
        # Create enhanced sample with toxicity info
        enhanced_sample = {
            **sample,
            'toxicity_analysis': toxicity_result,
            'sample_id': i,
            'processing_timestamp': time.time()
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
        
        # 🔧 IMPROVEMENT: Progress updates with ETA
        if (i + 1) % batch_size == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            if max_samples:
                remaining = max_samples - (i + 1)
                eta = remaining / rate if rate > 0 else 0
                eta_str = f"ETA: {eta/60:.1f}min" if eta > 60 else f"ETA: {eta:.0f}s"
            else:
                eta_str = "ETA: Unknown"
            
            print(f"✅ Processed {i + 1:,} samples ({rate:.1f}/sec, {eta_str})")
            print(f"   📈 Safe: {stats['safe_count']:,} ({stats['safe_count']/(stats['safe_count']+stats['toxic_count'])*100:.1f}%) | 🚨 Toxic: {stats['toxic_count']:,}")
        
        # 🔧 IMPROVEMENT: Save intermediate progress
        if save_progress and (i + 1) % 500 == 0:
            print(f"💾 Intermediate save point at {i + 1} samples")
        
        # Optional limit for testing
        if max_samples and i >= max_samples - 1:
            print(f"🔄 Stopping at {max_samples} samples (testing mode)")
            break
    
    # Calculate final averages
    if stats['safe_count'] > 0:
        stats['avg_confidence_safe'] = confidence_safe_sum / stats['safe_count']
    if stats['toxic_count'] > 0:
        stats['avg_confidence_toxic'] = confidence_toxic_sum / stats['toxic_count']
    
    # Calculate text length average
    if stats['total_processed'] > 0:
        stats['text_length_stats']['avg'] = stats['text_length_stats']['total_chars'] / stats['total_processed']
    
    # Final timing
    total_time = time.time() - start_time
    print(f"\n🏁 FILTERING COMPLETE!")
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"🚀 Average rate: {stats['total_processed']/total_time:.1f} samples/second")
    
    return safe_data, toxic_data, stats

def print_filtering_results(safe_data, toxic_data, stats):
    """Print detailed results of the filtering process"""
    
    total = stats['safe_count'] + stats['toxic_count']
    
    print("\n" + "="*60)
    print("📊 FILTERING RESULTS SUMMARY")
    print("="*60)
    
    print(f"📝 Total samples processed: {stats['total_processed']:,}")
    print(f"📋 Empty samples skipped: {stats['empty_count']:,}")
    print(f"⚠️  Processing errors: {stats['error_count']:,}")
    
    print(f"\n✅ SAFE CONTENT: {stats['safe_count']:,} samples ({stats['safe_count']/total*100:.1f}%)")
    print(f"🚨 TOXIC CONTENT: {stats['toxic_count']:,} samples ({stats['toxic_count']/total*100:.1f}%)")
    
    print(f"\n🎯 Average confidence scores:")
    print(f"   Safe content: {stats['avg_confidence_safe']:.3f}")
    print(f"   Toxic content: {stats['avg_confidence_toxic']:.3f}")
    
    # 🔧 IMPROVEMENT: Filtering method statistics
    print(f"\n🔧 Filtering method breakdown:")
    print(f"   Rule-based catches: {stats['rule_based_catches']:,}")
    print(f"   ML-based catches: {stats['ml_catches']:,}")
    print(f"   Both methods caught: {stats['both_catches']:,}")
    
    # 🔧 IMPROVEMENT: Text length statistics
    print(f"\n📏 Text length statistics:")
    if stats['text_length_stats']['min'] != float('inf'):
        print(f"   Min length: {stats['text_length_stats']['min']:,} chars")
        print(f"   Max length: {stats['text_length_stats']['max']:,} chars")
        print(f"   Avg length: {stats['text_length_stats']['avg']:.0f} chars")
    
    # 🔧 IMPROVEMENT: Toxicity distribution
    print(f"\n📊 Toxicity score distribution:")
    dist = stats['toxicity_distribution']
    print(f"   Very Low (0.0-0.3): {dist['very_low']:,} samples")
    print(f"   Low (0.3-0.5): {dist['low']:,} samples")
    print(f"   Medium (0.5-0.7): {dist['medium']:,} samples")
    print(f"   High (0.7-0.9): {dist['high']:,} samples")
    print(f"   Very High (0.9-1.0): {dist['very_high']:,} samples")
    
    # Show some examples
    print(f"\n📋 SAMPLE TOXIC CONTENT (first 3):")
    for i, sample in enumerate(toxic_data[:3]):
        analysis = sample['toxicity_analysis']
        text_preview = sample['text'][:150] + "..." if len(sample['text']) > 150 else sample['text']
        print(f"\nToxic Sample {i+1}:")
        print(f"   Confidence: {analysis['confidence']:.3f}")
        print(f"   Length: {analysis.get('text_length', 'unknown')} chars")
        if 'rule_based_reasons' in analysis and analysis['rule_based_reasons']:
            print(f"   Rule-based reasons: {', '.join(analysis['rule_based_reasons'])}")
        print(f"   Text: {text_preview}")
    
    print(f"\n📋 SAMPLE SAFE CONTENT (first 3):")
    for i, sample in enumerate(safe_data[:3]):
        analysis = sample['toxicity_analysis']
        text_preview = sample['text'][:150] + "..." if len(sample['text']) > 150 else sample['text']
        print(f"\nSafe Sample {i+1}:")
        print(f"   Confidence: {analysis['confidence']:.3f}")
        print(f"   Length: {analysis.get('text_length', 'unknown')} chars")
        print(f"   Text: {text_preview}")

# 🚀 CUSTOMIZABLE FILTERING SETUP
print("\n" + "="*60)
print("🚀 READY TO FILTER YOUR CLIMBLAB DATASET!")
print("="*60)

# 🎛️ CUSTOMIZE THESE SETTINGS:
NUM_SAMPLES_TO_PROCESS = 1000    # 👈 CHANGE THIS NUMBER!
TOXICITY_THRESHOLD = 0.7         # 👈 ADJUST STRICTNESS (0.5-0.9)
SAVE_RESULTS = False             # 👈 Set to True to save files
USE_CACHE = False               # 👈 Set to False to disable caching

print(f"\n⚙️  CURRENT SETTINGS:")
print(f"   📊 Samples to process: {NUM_SAMPLES_TO_PROCESS:,}")
print(f"   🎯 Toxicity threshold: {TOXICITY_THRESHOLD}")
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

# Export results if enabled
if SAVE_RESULTS:
    export_results = export_filtered_data(safe_samples, toxic_samples, use_gdrive=USE_GDRIVE, gdrive_creds=GDRIVE_CREDS)
    print("\n💾 Results saved successfully!")
    print(f"📁 Find them in: {export_results['run_dir']}")
else:
    print("\n💾 Results not saved (SAVE_RESULTS is False)")
    print("💡 To save results, set SAVE_RESULTS = True")

# Save final cache
if text_cache:
    text_cache._save_cache()

print("\n✅ FILTERING AND EXPORT COMPLETE!")

# 🔧 IMPROVEMENT: Add data quality assessment
def assess_data_quality(safe_data: List[Dict], toxic_data: List[Dict]) -> Dict[str, Any]:
    """
    Assess the quality of filtered data
    """
    quality_report = {
        'total_samples': len(safe_data) + len(toxic_data),
        'safe_ratio': len(safe_data) / (len(safe_data) + len(toxic_data)) if (len(safe_data) + len(toxic_data)) > 0 else 0,
        'avg_safe_length': 0,
        'avg_toxic_length': 0,
        'safe_confidence_stats': {'min': 1.0, 'max': 0.0, 'avg': 0.0},
        'toxic_confidence_stats': {'min': 1.0, 'max': 0.0, 'avg': 0.0}
    }
    
    # Analyze safe data
    if safe_data:
        safe_lengths = [len(sample['text']) for sample in safe_data]
        safe_confidences = [sample['toxicity_analysis']['confidence'] for sample in safe_data]
        
        quality_report['avg_safe_length'] = sum(safe_lengths) / len(safe_lengths)
        quality_report['safe_confidence_stats']['min'] = min(safe_confidences)
        quality_report['safe_confidence_stats']['max'] = max(safe_confidences)
        quality_report['safe_confidence_stats']['avg'] = sum(safe_confidences) / len(safe_confidences)
    
    # Analyze toxic data
    if toxic_data:
        toxic_lengths = [len(sample['text']) for sample in toxic_data]
        toxic_confidences = [sample['toxicity_analysis']['confidence'] for sample in toxic_data]
        
        quality_report['avg_toxic_length'] = sum(toxic_lengths) / len(toxic_lengths)
        quality_report['toxic_confidence_stats']['min'] = min(toxic_confidences)
        quality_report['toxic_confidence_stats']['max'] = max(toxic_confidences)
        quality_report['toxic_confidence_stats']['avg'] = sum(toxic_confidences) / len(toxic_confidences)
    
    return quality_report

# 🔧 IMPROVEMENT: Quick threshold testing
print(f"\n🧪 WANT TO TEST DIFFERENT THRESHOLDS?")
print(f"   Your current data can be re-analyzed without re-processing!")
print(f"   Example: reanalyze_with_threshold(safe_samples + toxic_samples, new_threshold=0.8)")

def reanalyze_with_threshold(all_samples, new_threshold=0.8):
    """Re-analyze already processed samples with a different threshold"""
    new_safe = []
    new_toxic = []
    
    for sample in all_samples:
        analysis = sample['toxicity_analysis']
        original_confidence = analysis['raw_score']
        
        # Re-apply threshold
        is_toxic_new = analysis['label'] == 'TOXIC' and original_confidence > new_threshold
        
        if is_toxic_new:
            new_toxic.append(sample)
        else:
            new_safe.append(sample)
    
    print(f"📊 Re-analysis with threshold {new_threshold}:")
    print(f"   ✅ Safe: {len(new_safe)} ({len(new_safe)/len(all_samples)*100:.1f}%)")
    print(f"   🚨 Toxic: {len(new_toxic)} ({len(new_toxic)/len(all_samples)*100:.1f}%)")
    
    return new_safe, new_toxic