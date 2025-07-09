# Enhanced ClimbLab Dataset Preselection with Advanced Content Safety and Quality Assessment
import os
import re
import torch
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from huggingface_hub import login
from datasets import load_dataset, Dataset
from transformers import (
    pipeline, AutoTokenizer, AutoModelForCausalLM, 
    BartTokenizer, BartForConditionalGeneration
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FilterConfig:
    """Configuration for filtering parameters"""
    toxicity_threshold: float = 0.7
    quality_threshold: float = 0.4
    safety_threshold: float = 0.8
    batch_size: int = 32
    cache_enabled: bool = True
    context_aware: bool = True
    domain: str = "general"  # general, educational, children, medical

class ContentPatterns:
    """Enhanced pattern detection for rule-based filtering"""
    
    def __init__(self):
        # Category-specific patterns
        self.sexual_patterns = [
            r'\b(?:explicit|graphic|sexual|nsfw|xxx|porn|nude|naked)\b',
            r'\b(?:sex|intercourse|masturbat|orgasm|climax)\b',
            r'\b(?:penis|vagina|breasts?|genitals?)\b'
        ]
        
        self.violence_patterns = [
            r'\b(?:kill|murder|stab|shoot|assault|attack|violence)\b',
            r'\b(?:blood|gore|torture|abuse|harm|hurt)\b',
            r'\b(?:weapon|gun|knife|bomb|explosive)\b'
        ]
        
        self.hate_speech_patterns = [
            r'\b(?:hate|racist|nazi|fascist|supremacist)\b',
            r'\b(?:slur|derogatory|offensive|discriminat)\b'
        ]
        
        self.profanity_patterns = [
            r'\b(?:fuck|shit|damn|bitch|ass|hell)\b',
            r'\b(?:crap|piss|bastard|whore|slut)\b'
        ]
        
        # Medical/Educational exceptions
        self.medical_contexts = [
            r'\b(?:medical|anatomy|biology|health|doctor|patient)\b',
            r'\b(?:education|academic|research|study|science)\b',
            r'\b(?:textbook|clinical|therapeutic|diagnosis)\b'
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = {
            'sexual': [re.compile(p, re.IGNORECASE) for p in self.sexual_patterns],
            'violence': [re.compile(p, re.IGNORECASE) for p in self.violence_patterns],
            'hate': [re.compile(p, re.IGNORECASE) for p in self.hate_speech_patterns],
            'profanity': [re.compile(p, re.IGNORECASE) for p in self.profanity_patterns],
            'medical': [re.compile(p, re.IGNORECASE) for p in self.medical_contexts]
        }

class AdvancedQualityMetrics:
    """Enhanced quality assessment beyond basic scoring"""
    
    @staticmethod
    def readability_score(text: str) -> float:
        """Simple Flesch-Kincaid style readability score"""
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        syllables = sum([AdvancedQualityMetrics._count_syllables(word) for word in text.split()])
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Flesch Reading Ease approximation
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0.0, min(100.0, score)) / 100.0
    
    @staticmethod
    def _count_syllables(word: str) -> int:
        """Approximate syllable counting"""
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
    def engagement_score(text: str) -> float:
        """Score based on engagement indicators"""
        questions = len(re.findall(r'\?', text))
        personal_pronouns = len(re.findall(r'\b(?:you|your|we|our|I|my)\b', text, re.IGNORECASE))
        action_words = len(re.findall(r'\b(?:do|make|create|build|learn|discover)\b', text, re.IGNORECASE))
        
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        engagement = (questions * 2 + personal_pronouns + action_words) / word_count
        return min(1.0, engagement)
    
    @staticmethod
    def vocabulary_diversity(text: str) -> float:
        """Measure vocabulary sophistication and diversity"""
        words = text.lower().split()
        if len(words) == 0:
            return 0.0
        
        unique_words = set(words)
        diversity = len(unique_words) / len(words)
        
        # Bonus for longer words (sophistication indicator)
        avg_word_length = sum(len(word) for word in unique_words) / len(unique_words)
        sophistication_bonus = min(0.3, (avg_word_length - 4) * 0.05)
        
        return min(1.0, diversity + sophistication_bonus)
    
    @staticmethod
    def structural_coherence(text: str) -> float:
        """Assess logical flow and structure"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.5
        
        # Transition words indicate good flow
        transitions = re.findall(
            r'\b(?:however|therefore|furthermore|moreover|additionally|consequently|meanwhile|thus)\b',
            text, re.IGNORECASE
        )
        
        # Consistent sentence length variation is good
        sentence_lengths = [len(s.split()) for s in sentences]
        length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        transition_score = min(0.5, len(transitions) / len(sentences))
        structure_score = 0.5 - min(0.3, length_variance / 100)  # Penalize extreme variance
        
        return transition_score + structure_score

class CachedFilter:
    """Caching system for both rule-based and AI results"""
    
    def __init__(self, cache_size: int = 10000):
        self.rule_cache = {}
        self.ai_cache = {}
        self.cache_size = cache_size
    
    def _get_hash(self, text: str) -> str:
        """Generate hash for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_rule_result(self, text: str) -> Optional[Dict]:
        """Get cached rule-based result"""
        text_hash = self._get_hash(text)
        return self.rule_cache.get(text_hash)
    
    def get_ai_result(self, text: str) -> Optional[Dict]:
        """Get cached AI result"""
        text_hash = self._get_hash(text)
        return self.ai_cache.get(text_hash)
    
    def cache_rule_result(self, text: str, result: Dict):
        """Cache rule-based result"""
        if len(self.rule_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.rule_cache))
            del self.rule_cache[oldest_key]
        
        text_hash = self._get_hash(text)
        self.rule_cache[text_hash] = result
    
    def cache_ai_result(self, text: str, result: Dict):
        """Cache AI result"""
        if len(self.ai_cache) >= self.cache_size:
            oldest_key = next(iter(self.ai_cache))
            del self.ai_cache[oldest_key]
        
        text_hash = self._get_hash(text)
        self.ai_cache[text_hash] = result

class AdaptiveThresholds:
    """Domain-specific threshold adjustment"""
    
    def __init__(self, config: FilterConfig):
        self.base_config = config
        self.domain_adjustments = {
            'educational': {
                'toxicity_threshold': 0.8,  # More lenient
                'safety_threshold': 0.7,
                'quality_threshold': 0.5
            },
            'children': {
                'toxicity_threshold': 0.3,  # More strict
                'safety_threshold': 0.9,
                'quality_threshold': 0.6
            },
            'medical': {
                'toxicity_threshold': 0.85,  # Very lenient for medical terms
                'safety_threshold': 0.6,
                'quality_threshold': 0.5
            },
            'general': {
                'toxicity_threshold': 0.7,
                'safety_threshold': 0.8,
                'quality_threshold': 0.4
            }
        }
    
    def get_adjusted_config(self, domain: str) -> FilterConfig:
        """Get domain-adjusted configuration"""
        adjustments = self.domain_adjustments.get(domain, self.domain_adjustments['general'])
        
        adjusted_config = FilterConfig()
        adjusted_config.toxicity_threshold = adjustments['toxicity_threshold']
        adjusted_config.safety_threshold = adjustments['safety_threshold']
        adjusted_config.quality_threshold = adjustments['quality_threshold']
        adjusted_config.batch_size = self.base_config.batch_size
        adjusted_config.cache_enabled = self.base_config.cache_enabled
        adjusted_config.context_aware = self.base_config.context_aware
        adjusted_config.domain = domain
        
        return adjusted_config

class EnhancedContentFilter:
    """Multi-layered content filtering system"""
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.patterns = ContentPatterns()
        self.quality_metrics = AdvancedQualityMetrics()
        self.cache = CachedFilter() if config.cache_enabled else None
        self.adaptive_thresholds = AdaptiveThresholds(config)
        
        # Initialize AI models (lazy loading)
        self.toxic_classifier = None
        self.bart_model = None
        self.bart_tokenizer = None
        
    def _load_toxic_classifier(self):
        """Lazy load toxic content classifier"""
        if self.toxic_classifier is None:
            try:
                self.toxic_classifier = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Loaded toxic-bert classifier")
            except Exception as e:
                logger.error(f"Failed to load toxic-bert: {e}")
                self.toxic_classifier = None
    
    def _load_bart_model(self):
        """Lazy load BART model for quality assessment"""
        if self.bart_model is None:
            try:
                self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
                self.bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
                logger.info("Loaded BART model")
            except Exception as e:
                logger.error(f"Failed to load BART: {e}")
                self.bart_model = None
    
    def rule_based_filter(self, text: str) -> Dict[str, Any]:
        """First-pass rule-based filtering"""
        if self.cache and self.config.cache_enabled:
            cached_result = self.cache.get_rule_result(text)
            if cached_result:
                return cached_result
        
        result = {
            'violations': [],
            'warnings': [],
            'context_flags': [],
            'severity_score': 0.0,
            'is_safe': True
        }
        
        # Check for medical/educational context
        has_medical_context = any(
            pattern.search(text) for pattern in self.patterns.compiled_patterns['medical']
        )
        
        # Check each category
        for category, patterns in self.patterns.compiled_patterns.items():
            if category == 'medical':
                continue
                
            matches = []
            for pattern in patterns:
                matches.extend(pattern.findall(text))
            
            if matches:
                severity = len(matches) / len(text.split()) * 100  # Percentage of flagged words
                
                if has_medical_context and category in ['sexual']:
                    # Reduce severity for medical contexts
                    severity *= 0.3
                    result['context_flags'].append(f"Medical context detected for {category}")
                
                if severity > 5.0:  # High severity threshold
                    result['violations'].append({
                        'category': category,
                        'matches': matches[:3],  # Limit for privacy
                        'severity': severity
                    })
                    result['is_safe'] = False
                elif severity > 1.0:  # Warning threshold
                    result['warnings'].append({
                        'category': category,
                        'severity': severity
                    })
                
                result['severity_score'] = max(result['severity_score'], severity)
        
        # Cache result
        if self.cache and self.config.cache_enabled:
            self.cache.cache_rule_result(text, result)
        
        return result
    
    def ai_content_filter(self, text: str) -> Dict[str, Any]:
        """AI-based content filtering with toxic-bert"""
        if self.cache and self.config.cache_enabled:
            cached_result = self.cache.get_ai_result(text)
            if cached_result:
                return cached_result
        
        self._load_toxic_classifier()
        
        result = {
            'toxicity_score': 0.0,
            'is_toxic': False,
            'confidence': 0.0,
            'error': None
        }
        
        if self.toxic_classifier is None:
            result['error'] = "Toxic classifier not available"
            return result
        
        try:
            # Truncate text if too long
            max_length = 512
            truncated_text = text[:max_length] if len(text) > max_length else text
            
            prediction = self.toxic_classifier(truncated_text)
            
            # Handle different output formats
            if isinstance(prediction, list) and len(prediction) > 0:
                pred = prediction[0]
                if pred['label'] == 'TOXIC':
                    result['toxicity_score'] = pred['score']
                    result['confidence'] = pred['score']
                else:
                    result['toxicity_score'] = 1 - pred['score']
                    result['confidence'] = pred['score']
            
            # Adjust threshold based on domain
            adjusted_config = self.adaptive_thresholds.get_adjusted_config(self.config.domain)
            result['is_toxic'] = result['toxicity_score'] > adjusted_config.toxicity_threshold
            
        except Exception as e:
            logger.error(f"AI filtering error: {e}")
            result['error'] = str(e)
        
        # Cache result
        if self.cache and self.config.cache_enabled:
            self.cache.cache_ai_result(text, result)
        
        return result
    
    def quality_assessment(self, text: str) -> Dict[str, float]:
        """Comprehensive quality assessment"""
        return {
            'readability': self.quality_metrics.readability_score(text),
            'engagement': self.quality_metrics.engagement_score(text),
            'vocabulary_diversity': self.quality_metrics.vocabulary_diversity(text),
            'structural_coherence': self.quality_metrics.structural_coherence(text),
            'length_score': min(1.0, len(text.split()) / 50),  # Prefer moderate length
        }
    
    def comprehensive_filter(self, text: str) -> Dict[str, Any]:
        """Complete filtering pipeline"""
        # Step 0: Preselect filter (fast, cheap)
        if not preselect_filter(text):
            return {
                'text': text,
                'is_safe': False,
                'meets_quality': False,
                'should_include': False,
                'preselect_filtered': True,
                'preselect_reason': 'Failed preselect filter',
                'rule_based': None,
                'ai_based': None,
                'quality': None,
                'overall_quality': 0.0,
                'config_used': self.config.domain,
                'context_adjustments': {},
                'performance_metrics': self.performance_monitor.get_performance_report(),
                'preselect_category': categorize_sample(text),
                'inappropriate_categories': inappropriate_categories(text)
            }
        # Step 1: Rule-based pre-filter
        rule_result = self.rule_based_filter(text)
        
        # Step 2: AI-based verification (only if needed)
        ai_result = None
        if rule_result['severity_score'] > 0 or not rule_result['is_safe']:
            ai_result = self.ai_content_filter(text)
        
        # Step 3: Quality assessment
        quality_result = self.quality_assessment(text)
        overall_quality = np.mean(list(quality_result.values()))
        
        # Step 4: Final decision with context adjustments
        context_adjustments = self.conversation_analyzer.should_adjust_thresholds()
        adjusted_config = self.adaptive_thresholds.get_adjusted_config(self.config.domain)
        
        final_quality_threshold = adjusted_config.quality_threshold
        if 'quality_threshold' in context_adjustments:
            final_quality_threshold += context_adjustments['quality_threshold']
        
        is_safe = rule_result['is_safe']
        if ai_result:
            is_safe = is_safe and not ai_result['is_toxic']
        
        meets_quality = overall_quality >= final_quality_threshold
        
        # --- Categorization step ---
        preselect_category = categorize_sample(text)
        inappropriate_cats = inappropriate_categories(text)
        
        return {
            'text': text,
            'is_safe': is_safe,
            'meets_quality': meets_quality,
            'should_include': is_safe and meets_quality,
            'rule_based': rule_result,
            'ai_based': ai_result,
            'quality': quality_result,
            'overall_quality': overall_quality,
            'config_used': self.config.domain,
            'context_adjustments': context_adjustments,
            'performance_metrics': self.performance_monitor.get_performance_report(),
            'preselect_category': preselect_category,
            'inappropriate_categories': inappropriate_cats
        }

# Preselect categories and patterns for Llama model improvement
PRESELECT_CATEGORIES = {
    'roleplay': [
        'conversation between', 'role-playing', 'character responds', 'in this scenario', 'acting as'
    ],
    'reasoning': [
        'step by step', "let's analyze", 'first we need to', 'the solution is', "here's how"
    ],
    'function_calling': [
        'function', 'api call', 'command', 'def ', 'return'
    ],
    'rag': [
        'according to', 'research shows', 'studies indicate', 'evidence suggests', 'source:'
    ]
}

def categorize_sample(text: str) -> str:
    text_lower = text.lower()
    for category, patterns in PRESELECT_CATEGORIES.items():
        for pattern in patterns:
            if pattern in text_lower:
                return category
    return 'uncategorized'

# Inappropriate content categories (from inappropriate_filter_jason.py)
INAPPROPRIATE_PATTERNS = {
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

def inappropriate_categories(text: str) -> list:
    import re
    text_lower = text.lower()
    found = []
    for category, patterns in INAPPROPRIATE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                found.append(category)
                break
    # Heuristics for repetition, caps, length
    words = text.split()
    if len(words) > 10:
        from collections import defaultdict
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word.lower()] += 1
        max_repetition = max(word_counts.values()) if word_counts else 0
        if max_repetition > len(words) * 0.3:
            found.append('excessive_repetition')
    if len(text) > 20 and text.isupper():
        found.append('excessive_caps')
    if len(text) < 10:
        found.append('too_short')
    elif len(text) > 5000:
        found.append('too_long')
    return found

# Preselect filter (fast, simple filter for obvious low-quality/irrelevant text)
def preselect_filter(text: str) -> bool:
    if not text:
        return False
    if len(text) < 15 or len(text) > 4000:
        return False
    num_digits = sum(c.isdigit() for c in text)
    if num_digits / len(text) > 0.5:
        return False
    alnum_chars = sum(c.isalnum() for c in text)
    if alnum_chars / len(text) < 0.6:
        return False
    if len(set(text.lower())) <= 2 and len(text) > 10:
        return False
    return True

# Example usage and comprehensive testing
def main():
    """Comprehensive testing with all enhancements and toggles"""
    # --- TOGGLES AND SETTINGS ---
    ENABLE_AI_TOXIC_FILTER = True   # Toggle AI-based toxic filter (toxic-bert)
    ENABLE_RULE_BASED_FILTER = True # Toggle rule-based filtering
    ENABLE_PRESELECT_FILTER = True  # Toggle preselect filter (fast, simple filter)
    ENABLE_CACHE = True             # Toggle caching (stores filter results for speed)
    TOXICITY_THRESHOLD = 0.7        # Toxicity threshold (0.1-0.9)
    QUALITY_THRESHOLD = 0.4         # Quality threshold (0.0-1.0)
    DOMAIN = 'general'              # Domain: general, educational, children, medical
    BATCH_SIZE = 32                 # Batch size for processing
    NUM_SAMPLES = 5000               # <--- ADJUST THIS to control how many samples to process
    HF_TOKEN = "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" # <--- PUT YOUR HUGGING FACE TOKEN HERE

    print("\n" + "="*60)
    print("🚀 ENHANCED FILTERING TEST WITH TOGGLES")
    print("="*60)
    print(f"AI-based toxic filter:    {'ENABLED' if ENABLE_AI_TOXIC_FILTER else 'DISABLED'}")
    print(f"Rule-based filter:        {'ENABLED' if ENABLE_RULE_BASED_FILTER else 'DISABLED'}")
    print(f"Preselect filter:         {'ENABLED' if ENABLE_PRESELECT_FILTER else 'DISABLED'}")
    print(f"Caching:                  {'ENABLED' if ENABLE_CACHE else 'DISABLED'} (stores filter results for speed)")
    print(f"Toxicity threshold:       {TOXICITY_THRESHOLD}")
    print(f"Quality threshold:        {QUALITY_THRESHOLD}")
    print(f"Domain:                   {DOMAIN}")
    print(f"Batch size:               {BATCH_SIZE}")
    print(f"Num samples:              {NUM_SAMPLES}")
    print("="*60 + "\n")

    # Hugging Face login
    login(token=HF_TOKEN)

    # Configure filter system
    config = FilterConfig(
        toxicity_threshold=TOXICITY_THRESHOLD,
        quality_threshold=QUALITY_THRESHOLD,
        batch_size=BATCH_SIZE,
        cache_enabled=ENABLE_CACHE,
        context_aware=True,
        domain=DOMAIN
    )
    filter_system = EnhancedContentFilter(config)
    filter_system.config.cache_enabled = ENABLE_CACHE
    filter_system.config.domain = DOMAIN
    filter_system.config.toxicity_threshold = TOXICITY_THRESHOLD
    filter_system.config.quality_threshold = QUALITY_THRESHOLD
    filter_system.config.batch_size = BATCH_SIZE

    # Patch filter system to respect toggles
    orig_ai_content_filter = filter_system.ai_content_filter
    orig_rule_based_filter = filter_system.rule_based_filter
    orig_comprehensive_filter = filter_system.comprehensive_filter

    def ai_content_filter_toggle(text):
        if ENABLE_AI_TOXIC_FILTER:
            return orig_ai_content_filter(text)
        else:
            return {'toxicity_score': 0.0, 'is_toxic': False, 'confidence': 0.0, 'error': None}

    def rule_based_filter_toggle(text):
        if ENABLE_RULE_BASED_FILTER:
            return orig_rule_based_filter(text)
        else:
            return {'violations': [], 'warnings': [], 'context_flags': [], 'severity_score': 0.0, 'is_safe': True, 'language_detected': 'en'}

    def comprehensive_filter_toggle(text):
        # If preselect is disabled, skip preselect filter logic
        if not ENABLE_PRESELECT_FILTER:
            # Copy-paste the logic from EnhancedContentFilter.comprehensive_filter, but skip preselect
            rule_result = filter_system.rule_based_filter(text)
            ai_result = None
            if rule_result['severity_score'] > 0 or not rule_result['is_safe']:
                ai_result = filter_system.ai_content_filter(text)
            quality_result = filter_system.quality_assessment(text)
            overall_quality = np.mean(list(quality_result.values()))
            context_adjustments = filter_system.conversation_analyzer.should_adjust_thresholds()
            adjusted_config = filter_system.adaptive_thresholds.get_adjusted_config(filter_system.config.domain)
            final_quality_threshold = adjusted_config.quality_threshold
            if 'quality_threshold' in context_adjustments:
                final_quality_threshold += context_adjustments['quality_threshold']
            is_safe = rule_result['is_safe']
            if ai_result:
                is_safe = is_safe and not ai_result['is_toxic']
            meets_quality = overall_quality >= final_quality_threshold
            preselect_category = categorize_sample(text)
            inappropriate_cats = inappropriate_categories(text)
            return {
                'text': text,
                'is_safe': is_safe,
                'meets_quality': meets_quality,
                'should_include': is_safe and meets_quality,
                'rule_based': rule_result,
                'ai_based': ai_result,
                'quality': quality_result,
                'overall_quality': overall_quality,
                'config_used': filter_system.config.domain,
                'context_adjustments': context_adjustments,
                'performance_metrics': filter_system.performance_monitor.get_performance_report(),
                'preselect_category': preselect_category,
                'inappropriate_categories': inappropriate_cats,
                'preselect_filtered': False,
                'preselect_reason': None
            }
        else:
            return orig_comprehensive_filter(text)

    filter_system.ai_content_filter = ai_content_filter_toggle
    filter_system.rule_based_filter = rule_based_filter_toggle
    filter_system.comprehensive_filter = comprehensive_filter_toggle

    # --- Load and filter a sample dataset ---
    print("\nLoading ClimbLab dataset (OptimalScale/ClimbLab)...")
    try:
        dataset = load_dataset("OptimalScale/ClimbLab", split="train", streaming=True)
        print("✅ Dataset loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("📦 Make sure you're connected to the internet and have access to the dataset")
        return

    import itertools
    print(f"\nFiltering the first {NUM_SAMPLES} samples...")
    filtered_results = []
    for i, sample in enumerate(itertools.islice(dataset, NUM_SAMPLES)):
        text = sample.get("text", "").strip()
        result = filter_system.comprehensive_filter(text)
        filtered_results.append(result)
        print(f"Sample {i+1}: should_include={result['should_include']} | preselect_category={result['preselect_category']} | inappropriate={result['inappropriate_categories']}")
        if not result['should_include']:
            print(f"  Reason: {result.get('preselect_reason', '')}")

    print(f"\nTotal samples processed: {len(filtered_results)}")
    print(f"Samples passing all filters: {sum(r['should_include'] for r in filtered_results)}")

    # --- Output 5 examples that pass all filters and their category scores ---
    print("\n--- 5 Examples Passing All Filters ---")
    passing = [r for r in filtered_results if r['should_include']]
    def category_score(text, category):
        # Count pattern matches for each category
        patterns = PRESELECT_CATEGORIES[category]
        text_lower = text.lower()
        return sum(text_lower.count(pat) for pat in patterns)
    for idx, r in enumerate(passing[:5]):
        print(f"\nExample {idx+1}:")
        print(f"Text: {r['text'][:200]}{'...' if len(r['text']) > 200 else ''}")
        print(f"Preselect category: {r['preselect_category']}")
        print(f"Inappropriate categories: {r['inappropriate_categories']}")
        print("Category scores:")
        for cat in PRESELECT_CATEGORIES:
            score = category_score(r['text'], cat)
            print(f"  {cat}: {score}")

    # --- (rest of your tests, e.g. A/B, conversation context, etc. can follow) ---

if __name__ == "__main__":
    main()