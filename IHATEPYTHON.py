from datasets import load_dataset
from transformers import pipeline
import logging

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_climblab_dataset(max_samples=100):
    """Load the OptimalScale/ClimbLab dataset with streaming."""
    logger.info(f"Loading {max_samples} samples from OptimalScale/ClimbLab dataset with streaming...")
    
    try:
        # Load dataset with streaming enabled
        dataset = load_dataset(
            "OptimalScale/ClimbLab", 
            streaming=True,
            trust_remote_code=True
        )
        
        logger.info(f"✅ Successfully loaded streaming dataset OptimalScale/ClimbLab")
        logger.info(f"Dataset structure: {type(dataset)}")
        
        # Handle different dataset structures
        if isinstance(dataset, dict):
            # If dataset is a dict, try to get the train split or first available split
            available_splits = list(dataset.keys())
            logger.info(f"Available splits: {available_splits}")
            
            if "train" in available_splits:
                dataset_stream = dataset["train"]
                logger.info("Using 'train' split")
            else:
                dataset_stream = dataset[available_splits[0]]
                logger.info(f"Using '{available_splits[0]}' split")
        else:
            dataset_stream = dataset
            logger.info("Using dataset directly (no splits)")
        
        # Extract text samples from streaming dataset
        texts = []
        count = 0
        
        logger.info("Starting to extract samples from stream...")
        
        for sample in dataset_stream:
            if count >= max_samples:
                break
                
            # Log first sample structure for debugging
            if count == 0:
                logger.info(f"Sample structure: {list(sample.keys())}")
                logger.info(f"Sample types: {[(k, type(v)) for k, v in sample.items()]}")
            
            # Try different possible text field names
            text_fields = ["text", "content", "input", "prompt", "document", "passage", "instruction", "query", "response"]
            text_content = None
            
            for field in text_fields:
                if field in sample and sample[field]:
                    text_content = str(sample[field]).strip()
                    if len(text_content) > 10:  # Only accept substantial text
                        break
            
            # If no standard field found, look for any string field with substantial content
            if not text_content:
                for key, value in sample.items():
                    if isinstance(value, str) and len(str(value).strip()) > 10:
                        text_content = str(value).strip()
                        logger.info(f"Using field '{key}' as text source")
                        break
            
            if text_content:
                texts.append(text_content)
                count += 1
                
                # Log progress every 50 samples
                if count % 50 == 0:
                    logger.info(f"Extracted {count}/{max_samples} samples...")
            else:
                logger.warning(f"No suitable text found in sample {count + 1}")
        
        logger.info(f"✅ Successfully extracted {len(texts)} valid text samples")
        
        if not texts:
            raise Exception("No valid text samples could be extracted from the dataset")
            
        return texts
        
    except Exception as e:
        logger.error(f"❌ Failed to load OptimalScale/ClimbLab: {str(e)}")
        raise Exception(f"Could not load ClimbLab dataset: {str(e)}")

# Load the ClimbLab dataset
try:
    texts = load_climblab_dataset(max_samples=200)  # Change this number as needed
    print(f"✅ Successfully loaded {len(texts)} text samples from OptimalScale/ClimbLab")
    print(f"📊 Each text represents 1 data sample")
    
    # Show first few characters of first sample for verification
    if texts:
        print(f"📝 First sample preview: {texts[0][:200]}...")
        print(f"📝 Sample length: {len(texts[0])} characters")
        
except Exception as e:
    print(f"❌ {e}")
    exit(1)

# Initialize classifier
try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("✅ Classifier loaded successfully")
except Exception as e:
    print(f"❌ Failed to load classifier: {e}")
    exit(1)

# Classification setup
block_labels = ["violence", "harassment", "self-harm", "medical content"]
soft_block_labels = ["profanity"]

BLOCK_THRESHOLD = 0.7
SOFT_BLOCK_THRESHOLD = 0.5

# Storage for results
cleaned_texts = []
blocked_texts = []
soft_blocked_texts = []

print(f"\n🔍 Processing {len(texts)} text samples...")
print("=" * 80)

# Process each text sample
for i, text in enumerate(texts):
    try:
        # Show sample number and truncated text for display
        display_text = text[:300] + "..." if len(text) > 300 else text
        
        print(f"\n📄 SAMPLE #{i+1}/{len(texts)}:")
        print(f"   Length: {len(text)} characters")
        print(f"   Preview: {display_text}")
        
        # Classify the text
        result = classifier(text, block_labels + soft_block_labels)

        block_flag = False
        soft_block_flag = False
        highest_score = 0
        highest_label = ""

        # Analyze classification results
        for label, score in zip(result["labels"], result["scores"]):
            if score > highest_score:
                highest_score = score
                highest_label = label
                
            # Check for hard blocks
            if label in block_labels and score >= BLOCK_THRESHOLD:
                blocked_texts.append((text, label, score))
                block_flag = True
                print(f"   🚫 STATUS: BLOCKED")
                print(f"   🚨 REASON: {label} (confidence: {score:.3f})")
                break

            # Check for soft blocks
            if label in soft_block_labels and score >= SOFT_BLOCK_THRESHOLD:
                soft_block_flag = True
                soft_blocked_texts.append((text, label, score))

        # If not hard blocked, add to cleaned
        if not block_flag:
            cleaned_texts.append({
                "text": text,
                "soft_block": soft_block_flag
            })
            
            if soft_block_flag:
                soft_label = [label for label, score in zip(result["labels"], result["scores"]) 
                             if label in soft_block_labels and score >= SOFT_BLOCK_THRESHOLD][0]
                soft_score = [score for label, score in zip(result["labels"], result["scores"]) 
                             if label in soft_block_labels and score >= SOFT_BLOCK_THRESHOLD][0]
                print(f"   ⚠️ STATUS: SOFT-BLOCKED")
                print(f"   ⚠️ REASON: {soft_label} (confidence: {soft_score:.3f})")
            else:
                print(f"   ✅ STATUS: CLEANED")
            
            print(f"   📊 Top classification: {highest_label} ({highest_score:.3f})")

        print("-" * 60)

        # Progress checkpoint
        if (i + 1) % 25 == 0:
            clean_count = len(cleaned_texts)
            block_count = len(blocked_texts)
            soft_count = len(soft_blocked_texts)
            print(f"\n📊 PROGRESS CHECKPOINT - {i + 1}/{len(texts)} processed:")
            print(f"   ✅ Cleaned: {clean_count} | ❌ Blocked: {block_count} | ⚠️ Soft-blocked: {soft_count}")
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ Error processing sample {i+1}: {e}")
        continue

# Final comprehensive summary
print("\n" + "=" * 80)
print("📊 DATASET PROCESSING COMPLETE")
print("=" * 80)

total_processed = len(texts)
total_cleaned = len(cleaned_texts)
total_blocked = len(blocked_texts)
total_soft_blocked = len(soft_blocked_texts)

print(f"📊 Total samples processed: {total_processed}")
print(f"📊 Data structure: 1 text sample = 1 dataset row")
print()
print(f"✅ CLEANED samples: {total_cleaned}")
print(f"❌ BLOCKED samples: {total_blocked}")
print(f"⚠️ SOFT-BLOCKED samples: {total_soft_blocked}")

# Calculate percentages
if total_processed > 0:
    clean_pct = (total_cleaned / total_processed) * 100
    block_pct = (total_blocked / total_processed) * 100
    soft_pct = (total_soft_blocked / total_processed) * 100
    
    print(f"\n📈 PERCENTAGE BREAKDOWN:")
    print(f"   ✅ Clean: {clean_pct:.1f}%")
    print(f"   ❌ Blocked: {block_pct:.1f}%")
    print(f"   ⚠️ Soft-blocked: {soft_pct:.1f}%")

# Show detailed examples
if blocked_texts:
    print(f"\n❌ EXAMPLE OF BLOCKED CONTENT:")
    example = blocked_texts[0]
    print(f"   Reason: {example[1]} (confidence: {example[2]:.3f})")
    print(f"   Text preview: {example[0][:200]}...")

if soft_blocked_texts:
    print(f"\n⚠️ EXAMPLE OF SOFT-BLOCKED CONTENT:")
    example = soft_blocked_texts[0]
    print(f"   Reason: {example[1]} (confidence: {example[2]:.3f})")
    print(f"   Text preview: {example[0][:200]}...")

if cleaned_texts:
    clean_example = cleaned_texts[0]
    print(f"\n✅ EXAMPLE OF CLEANED CONTENT:")
    print(f"   Soft-blocked: {clean_example['soft_block']}")
    print(f"   Text preview: {clean_example['text'][:200]}...")

print(f"\n🎯 Content filtering of {total_processed} samples complete!")