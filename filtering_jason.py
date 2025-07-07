import re
import json
from datasets import load_dataset
from huggingface_hub import login
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from collections import Counter
import torch
from token_detokenizer_jason import detokenize_batch
import farmhash  # pip install pyfarmhash
from datasketch import MinHash, MinHashLSH  # pip install datasketch
from tqdm import tqdm
import logging
from typing import List, Dict, Any, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

login(token="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")

# Initialize zero-shot classifier (downloads model on first run)
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Perplexity model and tokenizer (for efficiency, load once)
perplexity_model_name = "gpt2"
perplexity_tokenizer = GPT2Tokenizer.from_pretrained(perplexity_model_name)
perplexity_model = GPT2LMHeadModel.from_pretrained(perplexity_model_name)
perplexity_model.eval()
if torch.cuda.is_available():
    perplexity_model = perplexity_model.cuda()

def compute_perplexity(text: str, model=perplexity_model, tokenizer=perplexity_tokenizer, max_length: int = 1024) -> float:
    """Compute perplexity of a text using a language model."""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()

CANDIDATE_LABELS: List[str] = [
    "roleplay",
    "function calling",
    "reasoning",
    "retrieval-augmented generation"
]

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "roleplay": [
        "User:", "Assistant:", "System:", "Bot:", "AI:", "Human:", "Customer:", "Agent:", "Doctor:", "Patient:", "Teacher:", "Student:", "Roleplay", "Let's pretend", "Imagine you are", "As your assistant", "In this scenario", "Dialogue", "Conversation", "Chat between", "Role play"
    ],
    "function_calling": [
        "call function", "execute", "run", "invoke", "API", "endpoint", "parameters", "arguments", "return value", "output", "input", "function(", "def ", "result:", "response:", "request:", "method:", "command:", "perform", "trigger", "action:", "process:", "handler", "callback", "script", "programmatically", "automatically"
    ],
    "reasoning": [
        "reasoning", "think", "plan", "strategy", "solve", "solution", "problem", "step by step", "let's think", "let's reason", "analyze", "deduce", "infer", "logic", "sequence", "if...then", "because", "therefore", "conclude", "hypothesis", "explain", "why", "how", "decision", "evaluate", "consider", "goal", "objective", "task", "approach", "method", "procedure", "process", "multi-step", "chain of thought", "reflection", "planning", "robot", "navigate", "path", "trajectory", "execute", "manipulate", "autonomous", "control"
    ],
    "rag": [
        "retrieve", "retrieved", "search", "knowledge base", "database", "document", "according to", "as found in", "reference", "cited", "source:", "from the web", "external information", "lookup", "fetch", "context", "evidence", "passage", "snippet", "article", "wikipedia", "encyclopedia", "as mentioned in", "see also", "citing", "retrieval", "RAG", "grounded in", "supporting document", "provided context", "knowledge retrieval", "search result", "knowledge graph", "fact", "quote from", "as stated in"
    ]
}

def categorize_text(text: str) -> List[str]:
    """Return a list of category names whose keywords appear in the text (case-insensitive)."""
    matches = []
    lower_text = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in lower_text:
                matches.append(category)
                break
    return matches

def zero_shot_classify(text: str, candidate_labels: List[str] = CANDIDATE_LABELS, threshold: float = 0.5) -> List[str]:
    """Classify text using zero-shot classification pipeline."""
    result = zero_shot_classifier(text, candidate_labels)
    labels = [label for label, score in zip(result["labels"], result["scores"]) if score >= threshold]
    return labels

class ContentPatterns:
    def __init__(self) -> None:
        self.spam_patterns = [
            r'(.)\1{15,}',
            r'\b(click here|buy now|free money|urgent|limited time)\b',
        ]
        self.quality_patterns = [
            r'^.{1,5}$',
            r'^\s*$',
            r'[^\w\s]{50,}',
        ]
        self.compiled_patterns = {
            'spam': [re.compile(p, re.IGNORECASE) for p in self.spam_patterns],
            'quality': [re.compile(p, re.IGNORECASE) for p in self.quality_patterns]
        }
    def rule_based_filter(self, text: str) -> bool:
        """Return True if text passes all rule-based filters."""
        for pattern in self.compiled_patterns['spam'] + self.compiled_patterns['quality']:
            if pattern.search(text):
                return False
        return True

class AdvancedQualityMetrics:
    @staticmethod
    def readability_score(text: str) -> float:
        """Compute a normalized readability score for the text."""
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        syllables = sum([AdvancedQualityMetrics._count_syllables(word) for word in text.split()])
        if sentences == 0 or words == 0:
            return 0.0
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0.0, min(100.0, score)) / 100.0
    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        if word.endswith('e'):
            syllable_count -= 1
        return max(1, syllable_count)
    @staticmethod
    def vocabulary_diversity(text: str) -> float:
        words = text.lower().split()
        if len(words) == 0:
            return 0.0
        unique_words = set(words)
        diversity = len(unique_words) / len(words)
        avg_word_length = sum(len(word) for word in unique_words) / len(unique_words)
        sophistication_bonus = min(0.3, (avg_word_length - 4) * 0.05)
        return min(1.0, diversity + sophistication_bonus)

def stream_climblab_tokens(batch_size: int = 10, max_samples: Optional[int] = None) -> List[List[int]]:
    """Yield batches of token dicts from the NVIDIA ClimbLab dataset."""
    try:
        dataset = load_dataset("nvidia/ClimbLab", split="train", streaming=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return []
    batch = []
    count = 0
    for sample in dataset:
        tokens = sample.get("tokens")
        if tokens is None or not isinstance(tokens, list) or len(tokens) == 0 or len(tokens) > 100000:
            continue
        batch.append({"tokens": tokens})
        if len(batch) >= batch_size:
            yield batch
            batch = []
        count += 1
        if max_samples and count >= max_samples:
            break
    if batch:
        yield batch

def filter_sample(text: str, patterns: ContentPatterns, metrics: AdvancedQualityMetrics, perplexity_threshold: float) -> Optional[Dict[str, Any]]:
    """Apply all filters and return (filtered_dict or None)."""
    if not text or not isinstance(text, str):
        return None
    word_count = len(text.split())
    if word_count < 20 or word_count > 1024:
        return None
    if not patterns.rule_based_filter(text):
        return None
    if metrics.readability_score(text) < 0.2:
        return None
    if metrics.vocabulary_diversity(text) < 0.2:
        return None
    try:
        perp = compute_perplexity(text)
    except Exception as e:
        logger.warning(f"Perplexity computation failed: {e}")
        return None
    if perp > perplexity_threshold:
        return None
    categories = categorize_text(text)
    if not categories:
        categories = zero_shot_classify(text)
    if categories:
        return {"text": text, "categories": categories, "perplexity": perp}
    return None

def detokenize_and_filter_batch(batch: List[Dict[str, Any]], patterns: ContentPatterns, metrics: AdvancedQualityMetrics, perplexity_threshold: float) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
    """Detokenize a batch and filter each sample. Returns list of filtered dicts."""
    detok_results = detokenize_batch(batch)
    filtered = []
    all_categories = []
    for result in detok_results:
        text = result["text"]
        filtered_dict = filter_sample(text, patterns, metrics, perplexity_threshold)
        if filtered_dict:
            filtered.append(filtered_dict)
            all_categories.append(filtered_dict["categories"])
    return filtered, all_categories

def main_filtering(
    output_path: str = "climblab_pir_filtered.json",
    max_samples: Optional[int] = None,
    batch_size: int = 10,
    perplexity_threshold: float = 50.0
) -> None:
    """Main filtering pipeline for ClimbLab dataset."""
    patterns = ContentPatterns()
    metrics = AdvancedQualityMetrics()
    filtered = []
    all_categories = []
    samples_seen = 0
    count = 0
    for batch in stream_climblab_tokens(batch_size=batch_size, max_samples=max_samples):
        batch_filtered, batch_categories = detokenize_and_filter_batch(batch, patterns, metrics, perplexity_threshold)
        filtered.extend(batch_filtered)
        all_categories.extend(batch_categories)
        samples_seen += len(batch)
        count += len(batch_filtered)
        if max_samples and count >= max_samples:
            break
    logger.info(f"Kept {len(filtered)} out of {samples_seen}")
    logger.info(f"Filtered {samples_seen - len(filtered)} samples")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        logger.info(f"Filtered dataset saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save filtered dataset: {e}")
    flat_categories = sum(all_categories, [])
    category_counts = Counter(flat_categories)
    logger.info("Category counts:")
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count}")

# =====================
# DEDUPLICATION SECTION
# =====================

def deduplicate_lsh(dataset: List[List[int]], num_perm: int = 128, threshold: float = 0.8) -> List[Tuple[int, int]]:
    """
    dataset: list of documents, each is a list of tokens (e.g., words or n-grams)
    num_perm: number of permutations (hash functions) for MinHash
    threshold: Jaccard similarity threshold for candidate pairs
    Returns:
        List of pairs (i, j) of document indices considered similar.
    """
    logger.info("Step 3.1:")
    minhashes = []
    for i, tokens in enumerate(tqdm(dataset, desc="MinHashing docs")):
        m = MinHash(num_perm=num_perm)
        for token in tokens:
            m.update(str(token).encode('utf8'))
        minhashes.append(m)
    logger.info("Step 3.2:")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, m in enumerate(tqdm(minhashes, desc="Inserting to LSH")):
        lsh.insert(i, m)
    logger.info("Step 3.3:")
    similar_pairs = []
    for i, m in enumerate(tqdm(minhashes, desc="Querying LSH")):
        result = lsh.query(m)
        for j in result:
            if i < j:
                similar_pairs.append((i, j))
    if similar_pairs:
        logger.info("\nExample similar pairs from LSH:")
        for i, j in similar_pairs[:2]:
            logger.info(f"\nPair {i} and {j}:")
    logger.info("\n")
    return similar_pairs

def dfs(adjList: List[List[int]], visited: List[bool], node: int) -> None:
    visited[node] = True
    for neighbor in adjList[node]:
        if not visited[neighbor]:
            dfs(adjList, visited, neighbor)

def deduplicate(dataset: List[List[int]], threshold: float = 0.8, ngram_size: int = 10, num_perm: int = 128) -> List[List[int]]:
    logger.info("Step 1: Generating n-grams...")
    N = ngram_size
    nGrams = [[] for _ in range(len(dataset))]
    for row in tqdm(range(len(dataset)), desc="Generating n-grams"):
        for i in range(len(dataset[row]) - N + 1):
            nGrams[row].append(dataset[row][i : i + N])
    logger.info("Step 2: Preparing token sets for MinHash...")
    hashed_ngrams = [[] for _ in range(len(dataset))]
    for row in tqdm(range(len(dataset)), desc="Hashing n-grams"):
        for ngram in nGrams[row]:
            hashed_ngrams[row].append(farmhash.hash64(" ".join(map(str, ngram))))
    logger.info("Step 3: Comparing candidate documents...")
    edgeList = deduplicate_lsh(hashed_ngrams, threshold=threshold, num_perm=num_perm)
    logger.info("Step 4: Building adjacency list...")
    adjList = [[] for _ in range(len(dataset))]
    for pair in tqdm(edgeList, desc="Building adjacency list"):
        adjList[pair[0]].append(pair[1])
        adjList[pair[1]].append(pair[0])
    logger.info("Step 5: Connected Components Algorithm...")
    filtered_dataset = []
    visited = [False] * len(dataset)
    for i in tqdm(range(len(dataset)), desc="Finding connected components"):
        if not visited[i]:
            dfs(adjList, visited, i)
            filtered_dataset.append(dataset[i])
    return filtered_dataset

def deduplicate_climblab_dataset(
    sample_size: int = 100000,
    threshold: float = 0.5,
    ngram_size: int = 2,
    num_perm: int = 128,
    output_path: str = 'filtered_tokenized.txt',
    save_to_disk: bool = True
) -> List[List[int]]:
    """
    Deduplicate the NVIDIA ClimbLab dataset using MinHash LSH and n-gram hashing.
    Returns the filtered dataset (list of token lists).
    """
    logger.info("Step 0.1: Loading dataset...")
    try:
        dataset = load_dataset("nvidia/ClimbLab", split="train", streaming=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return []
    logger.info("Step 0.2: Collecting sample...")
    sample = []
    for i, item in enumerate(tqdm(dataset, desc="Collecting sample", total=sample_size)):
        if i >= sample_size:
            break
        sample.append(item)
    logger.info("Step 0.3: Reformatting sample...")
    for i, item in enumerate(tqdm(sample, desc="Reformatting sample")):
        sample[i] = list(item["tokens"])
    logger.info(f"Loaded {len(sample)} documents")
    logger.info("Step 0.4: Running deduplication...")
    filtered_dataset = deduplicate(
        sample,
        threshold=threshold,
        ngram_size=ngram_size,
        num_perm=num_perm
    )
    logger.info(f"\nReduced from {len(sample)} to {len(filtered_dataset)} documents")
    if save_to_disk:
        try:
            logger.info("\nSaving filtered dataset...")
            with open(output_path, 'w') as f:
                json.dump(filtered_dataset, f, indent=4)
            logger.info(f"Done! Filtered dataset saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save filtered dataset: {e}")
    return filtered_dataset

if __name__ == "__main__":
    main_filtering(max_samples=1000, perplexity_threshold=50.0) 