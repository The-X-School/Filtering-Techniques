from datasets import load_dataset
import os
import hashlib
import json
from datasketch import MinHash, MinHashLSH
import logging
from typing import Dict, Any, Set, List
import re
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextDeduplicator:
    """Efficient text deduplication using exact and approximate matching."""

    def __init__(self, similarity_threshold: float = 0.5, num_perm: int = 128, min_text_length: int = 10):
        self.similarity_threshold = similarity_threshold
        self.num_perm = num_perm
        self.min_text_length = min_text_length
        self.lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perm)
        self.exact_hashes: Set[str] = set()
        self.deduplicated_samples: List[Dict[str, Any]] = []
        self.text_cleaner = re.compile(r'\s+')

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text for consistent processing."""
        text = self.text_cleaner.sub(' ', text.strip().lower())
        return text

    def create_minhash(self, text: str) -> MinHash:
        """Create MinHash signature for text."""
        minhash = MinHash(num_perm=self.num_perm)
        words = text.split()
        for word in words:
            if len(word) > 2:
                minhash.update(word.encode('utf-8'))
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            minhash.update(trigram.encode('utf-8'))
        return minhash

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate (exact or near-duplicate)."""
        if not text or len(text.strip()) < self.min_text_length:
            return True

        processed_text = self.preprocess_text(text)
        text_hash = hashlib.sha256(processed_text.encode('utf-8')).hexdigest()
        if text_hash in self.exact_hashes:
            return True

        if len(processed_text.split()) >= 5:
            minhash = self.create_minhash(processed_text)
            similar_samples = self.lsh.query(minhash)
            if similar_samples:
                return True

            self.lsh.insert(text_hash, minhash)
        self.exact_hashes.add(text_hash)
        return False

    def add_sample(self, sample: Dict[str, Any]) -> bool:
        """Add sample if not duplicate. Returns True if added."""
        text_fields = ['text', 'content', 'body', 'prompt', 'instruction', 'tokens']
        text = None

        for field in text_fields:
            if field in sample and sample[field]:
                if field == 'tokens' and isinstance(sample[field], list):
                    text = " ".join(str(token) for token in sample[field])
                else:
                    text = str(sample[field])
                break

        if not text:
            logger.warning(f"No valid text field found in sample. Available keys: {list(sample.keys())}")
            return False

        if not self.is_duplicate(text):
            self.deduplicated_samples.append(sample)
            return True
        return False

def save_samples_efficiently(samples: List[Dict[str, Any]], output_path: str, batch_size: int = 1000):
    """Save samples to JSONL with efficient batching."""
    logger.info(f"Saving {len(samples)} samples to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            batch_lines = []

            for sample in batch:
                try:
                    json_line = json.dumps(sample, ensure_ascii=False, separators=(',', ':'))
                    batch_lines.append(json_line)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Failed to serialize sample: {e}")
                    continue

            if batch_lines:
                f.write('\n'.join(batch_lines) + '\n')

def process_climblab_dataset(
    output_dir: str = "data/climblab_sample",
    max_samples: int = None,
    similarity_threshold: float = 0.5,
    batch_size: int = 1000,
    dataset = None  # Add optional dataset parameter
):
    """Main function to process and deduplicate ClimbLab dataset."""

    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load dataset only if not provided
        if dataset is None:
            logger.info("Loading ClimbLab dataset with streaming...")
            dataset = load_dataset("nvidia/ClimbLab", split="train", streaming=True)
        else:
            logger.info("Using pre-loaded dataset...")

        deduplicator = TextDeduplicator(similarity_threshold=similarity_threshold)

        logger.info("Processing samples for deduplication...")
        sample_count = 0
        duplicates_found = 0

        # Import itertools.islice here
        import itertools

        # Use itertools.islice to take only the first 'max_samples' if specified
        # If max_samples is None, it will iterate through the entire dataset
        if max_samples is not None:
            data_iterator = itertools.islice(dataset, max_samples)
            # Adjust tqdm total for better progress tracking if max_samples is set
            total_samples_for_tqdm = max_samples
        else:
            data_iterator = dataset
            total_samples_for_tqdm = None # Let tqdm figure out total if not limited by max_samples

        try:
            for sample in tqdm(data_iterator, desc="Processing samples", total=total_samples_for_tqdm):
                sample_count += 1

                if not deduplicator.add_sample(sample):
                    duplicates_found += 1

                if sample_count % batch_size == 0:
                    unique_count = len(deduplicator.deduplicated_samples)
                    logger.info(f"Processed {sample_count} samples, kept {unique_count} unique samples")

                # The max_samples check here is redundant if using itertools.islice
                # but it won't hurt. For clarity, the primary limiting is now done by islice.
                # if max_samples and len(deduplicator.deduplicated_samples) >= max_samples:
                #    logger.info(f"Reached sample limit of {max_samples}")
                #    break

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")

        if deduplicator.deduplicated_samples:
            output_path = os.path.join(output_dir, "climblab_deduplicated.jsonl")
            save_samples_efficiently(deduplicator.deduplicated_samples, output_path)

            unique_count = len(deduplicator.deduplicated_samples)
            dedup_ratio = unique_count / sample_count if sample_count > 0 else 0

            logger.info(f"Processing complete!")
            logger.info(f"Total samples processed: {sample_count}")
            logger.info(f"Unique samples saved: {unique_count}")
            logger.info(f"Duplicates found: {duplicates_found}")
            logger.info(f"Deduplication ratio: {dedup_ratio:.2%}")
            logger.info(f"Output saved to: {output_path}")
        else:
            logger.warning("No samples were processed successfully")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error("This might happen if:")
        logger.error("1. The dataset doesn't exist or is not accessible")
        logger.error("2. Network connection issues")
        logger.error("3. Insufficient memory for large datasets")
        logger.error("4. Insufficient permissions for file writing")
        raise

if __name__ == "__main__":
    OUTPUT_DIR = "data/climblab_deduplicated_output" # Changed for a cleaner output directory
    MAX_SAMPLES = 10000 # Process up to 100,000 samples for a test run, set to None for full dataset
    SIMILARITY_THRESHOLD = 0.5
    BATCH_SIZE = 1000

    process_climblab_dataset(
        output_dir=OUTPUT_DIR,
        max_samples=MAX_SAMPLES,
        similarity_threshold=SIMILARITY_THRESHOLD,
        batch_size=BATCH_SIZE
    )