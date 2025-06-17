# Install dependencies (only needed once per session)


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
                'label': 'NON_TOXIC'
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
            'raw_score': result['score']
        }
        
    except Exception as e:
        print(f"⚠️ Error classifying text: {e}")
        # Default to safe if there's an error
        return {
            'is_toxic': False,
            'is_safe': True,
            'confidence': 0.0,
            'label': 'ERROR',
            'error': str(e)
        }

def filter_dataset_with_toxic_bert(dataset_stream, toxicity_threshold=0.7, max_samples=None):
    """
    Filter the ClimbLab dataset using toxic-bert
    
    Args:
        dataset_stream: Your ClimbLab dataset stream
        toxicity_threshold: Confidence threshold (0.1-0.9, higher = more strict)
        max_samples: Limit samples for testing (None = process all)
    
    Returns:
        safe_data: List of clean samples
        toxic_data: List of toxic samples
        stats: Processing statistics
    """
    
    # Storage lists
    safe_data = []
    toxic_data = []
    
    # Statistics tracking
    stats = {
        'total_processed': 0,
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
        
        # Skip completely empty samples
        if not text:
            stats['empty_count'] += 1
            continue
        
        # Classify toxicity
        toxicity_result = classify_toxicity(text, threshold=toxicity_threshold)
        
        # Create enhanced sample with toxicity info
        enhanced_sample = {
            **sample,
            'toxicity_analysis': toxicity_result,
            'sample_id': i
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
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"✅ Processed {i + 1:,} samples ({rate:.1f} samples/sec)")
            print(f"   📈 Safe: {stats['safe_count']:,} | 🚨 Toxic: {stats['toxic_count']:,}")
        
        # Optional limit for testing
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
    print(f"📋 Empty samples skipped: {stats['empty_count']:,}")
    print(f"⚠️  Processing errors: {stats['error_count']:,}")
    
    print(f"\n✅ SAFE CONTENT: {stats['safe_count']:,} samples ({stats['safe_count']/total*100:.1f}%)")
    print(f"🚨 TOXIC CONTENT: {stats['toxic_count']:,} samples ({stats['toxic_count']/total*100:.1f}%)")
    
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

# 🚀 CUSTOMIZABLE FILTERING SETUP
print("\n" + "="*60)
print("🚀 READY TO FILTER YOUR CLIMBLAB DATASET!")
print("="*60)

# 🎛️ CUSTOMIZE THESE SETTINGS:
NUM_SAMPLES_TO_PROCESS = 500    # 👈 CHANGE THIS NUMBER!
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