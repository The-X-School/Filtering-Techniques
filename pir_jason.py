import re
import json
from datasets import load_dataset
from huggingface_hub import login
login(token="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")
def pir_filter_dataset(dataset_name="nvidia/ClimbLab", output_path="climblab_pir_filtered.json", max_samples=1000):
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    patterns = ContentPatterns()
    metrics = AdvancedQualityMetrics()
    filtered = []
    count = 100
    for sample in dataset:
        text = sample.get("text")
        if not text or not isinstance(text, str):
            continue
        if not patterns.rule_based_filter(text):
            continue
        if metrics.readability_score(text) < 0.2:
            continue
        if metrics.vocabulary_diversity(text) < 0.2:
            continue
        filtered.append({"text": text})
        count += 1
        if max_samples and count >= max_samples:
            break
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
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

def pir_filter_dataset(dataset_name="nvidia/ClimbLab", output_path="climblab_pir_filtered.json", max_samples=1000):
    # Load the dataset
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    patterns = ContentPatterns()
    metrics = AdvancedQualityMetrics()
    filtered = []
    count = 0
    for sample in dataset:
        text = sample.get("text")
        if not text or not isinstance(text, str):
            continue
        if not patterns.rule_based_filter(text):
            continue
        if metrics.readability_score(text) < 0.2:
            continue
        if metrics.vocabulary_diversity(text) < 0.2:
            continue
        filtered.append({"text": text})
        count += 1
        if max_samples and count >= max_samples:
            break
    # Write filtered data to output file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    pir_filter_dataset()