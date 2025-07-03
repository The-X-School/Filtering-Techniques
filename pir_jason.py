import re
import json
from datasets import load_dataset
from huggingface_hub import login
from transformers import pipeline
from collections import Counter

login(token="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")

# Initialize zero-shot classifier (downloads model on first run)
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Candidate labels for zero-shot classification
CANDIDATE_LABELS = [
    "roleplay",
    "function calling",
    "reasoning",
    "retrieval-augmented generation"
]

# Category keyword lists for filtering
CATEGORY_KEYWORDS = {
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

def categorize_text(text):
    """Return a list of category names whose keywords appear in the text (case-insensitive)."""
    matches = []
    lower_text = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in lower_text:
                matches.append(category)
                break
    return matches

def zero_shot_classify(text, candidate_labels=CANDIDATE_LABELS, threshold=0.5):
    result = zero_shot_classifier(text, candidate_labels)
    labels = [label for label, score in zip(result["labels"], result["scores"]) if score >= threshold]
    return labels

class ContentPatterns:
    def __init__(self):
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
        for pattern in self.compiled_patterns['spam'] + self.compiled_patterns['quality']:
            if pattern.search(text):
                return False
        return True

class AdvancedQualityMetrics:
    @staticmethod
    def readability_score(text: str) -> float:
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

def pir_filter_dataset(dataset_name=None, input_path="climblab_detokenized.json", output_path="climblab_pir_filtered.json", max_samples=None):
    patterns = ContentPatterns()
    metrics = AdvancedQualityMetrics()
    # Load from local JSON file
    with open(input_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    filtered = []
    all_categories = []  # Collect all assigned categories
    samples_seen = 0
    for sample in dataset:
        samples_seen += 1
        text = sample.get("text")
        if not text or not isinstance(text, str):
            continue
        # Length-based filter: only keep samples with 20-1024 words
        word_count = len(text.split())
        if word_count < 20 or word_count > 1024:
            continue
        if not patterns.rule_based_filter(text):
            continue
        if metrics.readability_score(text) < 0.2:
            continue
        if metrics.vocabulary_diversity(text) < 0.2:
            continue
        # Categorize using keyword/regex
        categories = categorize_text(text)
        # If no category, use zero-shot classification
        if not categories:
            categories = zero_shot_classify(text)
        if categories:
            filtered.append({"text": text, "categories": categories})
            all_categories.append(categories)
        if max_samples and len(filtered) >= max_samples:
            break
    print(f"Kept {len(filtered)} out of {samples_seen}")
    print(f"Filtered {samples_seen - len(filtered)} samples")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    # Count how many samples relate to each category
    flat_categories = sum(all_categories, [])
    category_counts = Counter(flat_categories)
    print("Category counts:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")

if __name__ == "__main__":
    pir_filter_dataset(max_samples=100)