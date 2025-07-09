import math
import re
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Category prompts for semantic scoring
CATEGORY_DEFINITIONS = {
    "roleplay": "Enhancing performance in interactive digital environments.",
    "reasoning": "Improving complex problem-solving for downstream applications like robotics.",
    "function_calling": "Optimizing models for mobile device interactions.",
    "rag": "Boosting capabilities in retrieval-augmented applications."
}


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def compute_semantic_score(text: str, category_text: str, vectorizer=None) -> float:
    texts = [text, category_text]
    if not vectorizer:
        vectorizer = TfidfVectorizer().fit(texts)
    vectors = vectorizer.transform(texts)
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity  # range [0,1]


def soft_length_score(length: int, min_len=50, max_len=2000) -> float:
    # Apply sigmoid normalization centered at 500 characters
    return sigmoid((length - min_len) / 150.0) * sigmoid((max_len - length) / 500.0)


def fc_score(text: str) -> float:
    patterns = [r'\bfunction\b', r'\bcall\b', r'\bapi\b', r'\bexecute\b', r'\bdef\b', r'\(', r'\)', r'=', r'\breturn\b']
    matches = sum(1 for pattern in patterns if re.search(pattern, text.lower()))
    return min(1.0, matches / len(patterns))


def category_score(text: str, category: str, vectorizer=None) -> float:
    if category not in CATEGORY_DEFINITIONS:
        return 0.0
    return compute_semantic_score(text, CATEGORY_DEFINITIONS[category], vectorizer)


def reward_function(text_samples: List[str],
                    quality_threshold=0.2,
                    fc_weight=1.0,
                    category_weights=None,
                    length_min=50,
                    length_max=2000,
                    verbose=False) -> float:

    if not text_samples:
        return 0.0

    category_weights = category_weights or {
        "function_calling": 0.3,
        "roleplay": 0.2,
        "reasoning": 0.3,
        "rag": 0.2
    }

    vectorizer = TfidfVectorizer().fit(text_samples + list(CATEGORY_DEFINITIONS.values()))

    filtered_samples = []
    total_score = 0.0

    for text in text_samples:
        if not text or len(text.strip()) == 0:
            continue

        words = text.lower().split()
        if not words:
            continue

        # Length (soft scoring)
        len_score = soft_length_score(len(text), length_min, length_max)

        # Vocab diversity
        unique_words = set(words)
        vocab_div = len(unique_words) / len(words)

        # Function calling keywords
        fc = fc_score(text)

        # Category-based semantic relevance
        category_score_total = 0.0
        for cat, weight in category_weights.items():
            score = category_score(text, cat, vectorizer)
            category_score_total += score * weight

        # Final score: weighted components
        quality_score = (
            len_score * 0.3 +
            vocab_div * 0.3 +
            fc * fc_weight * 0.2 +
            category_score_total * 0.2
        )

        if quality_score >= quality_threshold:
            filtered_samples.append(text)
            total_score += quality_score

            if verbose:
                print(f"\n✅ Accepted: {text[:60]}...")
                print(f"Length: {len(text)} | Len Score: {len_score:.2f} | "
                      f"Vocab: {vocab_div:.2f} | FC: {fc:.2f} | Cat: {category_score_total:.2f} | "
                      f"Total: {quality_score:.3f}")
        elif verbose:
            print(f"\n❌ Rejected: {text[:60]}... (Score: {quality_score:.3f})")

    if len(filtered_samples) < 3:
        return 0.0

    avg_score = total_score / len(filtered_samples)
    size_bonus = min(0.2, len(filtered_samples) / 100)

    return min(1.0, avg_score + size_bonus)


# Example usage
if __name__ == "__main__":
    sample_texts = [
        "def call_api(endpoint, params): return requests.get(endpoint, params=params)",
        "Execute database query with parameters: db.execute('SELECT * FROM users')",
        "This is just regular text without any special patterns",
        "function() method call with arguments and return values",
        "Short text",
        "API function call example with error handling and response parsing",
        "A simulated chatbot responds with empathy and role-based personality traits.",
        "To solve the robotic arm pathing problem, consider all kinematic constraints."
    ]

    reward = reward_function(sample_texts, verbose=True)
    print(f"\n🎯 Final reward score: {reward:.3f}")

    reward2 = reward_function(sample_texts, quality_threshold=0.1, fc_weight=2.0, verbose=True)
    print(f"\n🎯 Reward with different params: {reward2:.3f}")
