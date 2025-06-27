# Enhanced ClimbLab Dataset Preselection with Advanced Content Safety and Quality Assessment
import os
import re
import torch
import hashlib
import logging
import time
import json
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
from pathlib import Path
from datetime import datetime

# Try to import sentence transformers for embedding-based preselect
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers not available. Install with: pip install sentence-transformers")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directories
CACHE_DIR = "filter_cache"
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

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

class ConversationContextAnalyzer:
    """Analyzes conversation context to adjust filtering thresholds"""
    
    def __init__(self):
        self.conversation_history = []
        self.context_patterns = {
            'educational': ['learn', 'study', 'education', 'teach', 'school'],
            'professional': ['work', 'business', 'meeting', 'project', 'office'],
            'casual': ['hey', 'hi', 'chat', 'talk', 'friend']
        }
    
    def add_message(self, text: str, is_user: bool = True):
        """Add message to conversation history"""
        self.conversation_history.append({
            'text': text,
            'is_user': is_user,
            'timestamp': time.time()
        })
        # Keep only last 10 messages
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)
    
    def should_adjust_thresholds(self) -> Dict[str, float]:
        """Determine if thresholds should be adjusted based on context"""
        if not self.conversation_history:
            return {}
        
        # Analyze recent conversation context
        recent_text = ' '.join([msg['text'] for msg in self.conversation_history[-5:]])
        recent_text_lower = recent_text.lower()
        
        adjustments = {}
        
        # Educational context - be more lenient
        if any(pattern in recent_text_lower for pattern in self.context_patterns['educational']):
            adjustments['quality_threshold'] = -0.1
            adjustments['toxicity_threshold'] = 0.1
        
        # Professional context - be more strict
        elif any(pattern in recent_text_lower for pattern in self.context_patterns['professional']):
            adjustments['quality_threshold'] = 0.1
            adjustments['toxicity_threshold'] = -0.1
        
        return adjustments

class PerformanceMonitor:
    """Monitor and track filtering performance metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.filter_calls = 0
        self.cache_hits = 0
        self.ai_model_calls = 0
        self.rule_based_calls = 0
        self.processing_times = []
    
    def record_filter_call(self):
        """Record a filter call"""
        self.filter_calls += 1
    
    def record_cache_hit(self):
        """Record a cache hit"""
        self.cache_hits += 1
    
    def record_ai_call(self):
        """Record an AI model call"""
        self.ai_model_calls += 1
    
    def record_rule_call(self):
        """Record a rule-based call"""
        self.rule_based_calls += 1
    
    def record_processing_time(self, duration: float):
        """Record processing time"""
        self.processing_times.append(duration)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance metrics"""
        uptime = time.time() - self.start_time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'uptime_seconds': uptime,
            'total_filter_calls': self.filter_calls,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.filter_calls),
            'ai_model_calls': self.ai_model_calls,
            'rule_based_calls': self.rule_based_calls,
            'avg_processing_time_ms': avg_processing_time * 1000
        }

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
        
        # Initialize conversation context analyzer and performance monitor
        self.conversation_analyzer = ConversationContextAnalyzer()
        self.performance_monitor = PerformanceMonitor()
        
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
            'is_safe': True,
            'language_detected': 'en'
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
            'inappropriate_categories': inappropriate_cats,
            'preselect_filtered': False,
            'preselect_reason': None,
            'llama_analysis': None  # Will be filled by AI preselection if enabled
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

class LlamaAnalyzer:
    """Analyzes Llama model performance on different categories"""
    
    def __init__(self):
        print("🦙 Initializing Llama model for AI-based preselection...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("data4elm/Llama-400M-12L")
            self.model = AutoModelForCausalLM.from_pretrained(
                "data4elm/Llama-400M-12L",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Llama model loaded successfully for AI preselection!")
        except Exception as e:
            print(f"⚠️ Warning: Could not load Llama model: {e}")
            self.tokenizer = None
            self.model = None
    
    def analyze_performance(self, text: str) -> Dict[str, float]:
        """
        Analyze Llama's performance on a text sample for each category
        Returns scores between 0-1 for each category
        """
        if self.model is None or self.tokenizer is None:
            # Return default scores if model not available
            return {category: 0.5 for category in PRESELECT_CATEGORIES.keys()}
        
        scores = {}
        
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate continuation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # Generate 50 new tokens instead of setting max_length
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Score each category
            for category, patterns in PRESELECT_CATEGORIES.items():
                category_score = self._score_category(text, generated_text, patterns)
                scores[category] = category_score
            
        except Exception as e:
            logger.error(f"Error in Llama analysis: {e}")
            # Default scores on error
            scores = {category: 0.5 for category in PRESELECT_CATEGORIES.keys()}
        
        return scores
    
    def _score_category(self, input_text: str, generated_text: str, patterns: List[str]) -> float:
        """
        Score how well Llama handles a specific category
        Uses pattern matching and basic heuristics
        """
        score = 0.0
        
        # Check for pattern presence in input
        input_pattern_matches = sum(1 for p in patterns if p.lower() in input_text.lower())
        
        # Check for pattern presence in output
        output_pattern_matches = sum(1 for p in patterns if p.lower() in generated_text.lower())
        
        # Basic coherence check
        try:
            input_words = set(input_text.lower().split())
            output_words = set(generated_text.lower().split())
            coherence = len(input_words.intersection(output_words)) / len(input_words)
        except:
            coherence = 0.0
        
        # Combine scores
        pattern_score = (input_pattern_matches + output_pattern_matches) / (len(patterns) * 2)
        
        score = (pattern_score * 0.7) + (coherence * 0.3)
        return min(max(score, 0.0), 1.0)  # Normalize to 0-1

class DatasetAnalyzer:
    """Analyzes the dataset to find samples that could improve Llama's performance"""
    
    def __init__(self, llama_analyzer: LlamaAnalyzer):
        self.llama = llama_analyzer
        self.performance_cache = {}
        self.cache_file = Path(CACHE_DIR) / "llama_analysis_cache.json"
        self._load_cache()
    
    def _load_cache(self):
        """Load cached analysis results"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.performance_cache = json.load(f)
                logger.info(f"Loaded {len(self.performance_cache)} cached analysis results")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
    
    def _save_cache(self):
        """Save analysis results to cache"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.performance_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def analyze_sample(self, text: str) -> Dict[str, Any]:
        """
        Analyze a single sample to determine its potential value
        for improving Llama's performance
        """
        # Check cache
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if text_hash in self.performance_cache:
            return self.performance_cache[text_hash]
        
        # Get Llama's performance scores
        category_scores = self.llama.analyze_performance(text)
        
        # Calculate potential value for each category
        potential_value = {}
        for category, score in category_scores.items():
            # Lower scores indicate more room for improvement
            improvement_potential = 1.0 - score
            potential_value[category] = improvement_potential
        
        result = {
            'text': text,
            'category_scores': category_scores,
            'potential_value': potential_value,
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache the result
        self.performance_cache[text_hash] = result
        
        # Save cache periodically
        if len(self.performance_cache) % 10 == 0:
            self._save_cache()
        
        return result

class RuleBasedPreSelect:
    """Enhanced Rule-Based PreSelect (Non-AI) with comprehensive rules"""
    
    def __init__(self):
        self.quality_indicators = [
            r'\b(?:explain|describe|analyze|demonstrate|example|because|therefore|however|moreover)\b',
            r'\b(?:solution|answer|result|conclusion|summary|key|important)\b',
            r'\b(?:first|second|third|finally|step|process|method|approach)\b'
        ]
        
        self.low_quality_patterns = [
            r'^.{1,10}$',  # Too short
            r'(.)\1{10,}',  # Excessive repetition
            r'[A-Z]{20,}',  # Excessive caps
            r'\d{50,}',     # Too many numbers
            r'[^\w\s]{20,}' # Too many special characters
        ]
        
        self.educational_indicators = [
            r'\b(?:learn|study|understand|knowledge|concept|theory|principle)\b',
            r'\b(?:research|evidence|data|analysis|findings|conclusion)\b',
            r'\b(?:question|problem|solution|answer|explanation)\b'
        ]
        
        self.code_indicators = [
            r'\b(?:function|class|def|import|return|if|else|for|while)\b',
            r'\b(?:python|javascript|java|cpp|html|css|sql)\b',
            r'[{}();]',  # Code punctuation
            r'^\s*#.*$'  # Comments
        ]
    
    def preselect(self, text: str) -> Dict[str, Any]:
        """Enhanced rule-based preselection"""
        result = {
            'passed': True,
            'reason': None,
            'quality_score': 0.0,
            'category_hints': []
        }
        
        if not text or len(text.strip()) == 0:
            result['passed'] = False
            result['reason'] = 'Empty text'
            return result
        
        # Basic length checks
        if len(text) < 15:
            result['passed'] = False
            result['reason'] = 'Too short (< 15 chars)'
            return result
        
        if len(text) > 4000:
            result['passed'] = False
            result['reason'] = 'Too long (> 4000 chars)'
            return result
        
        # Check for low quality patterns
        for pattern in self.low_quality_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                result['passed'] = False
                result['reason'] = f'Low quality pattern detected'
                return result
        
        # Calculate quality score
        quality_matches = sum(1 for pattern in self.quality_indicators 
                            if re.search(pattern, text, re.IGNORECASE))
        result['quality_score'] = quality_matches / len(self.quality_indicators)
        
        # Detect category hints
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.educational_indicators):
            result['category_hints'].append('educational')
        
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.code_indicators):
            result['category_hints'].append('code')
        
        # Word/character ratio checks
        words = text.split()
        if len(words) > 0:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length < 2 or avg_word_length > 15:
                result['passed'] = False
                result['reason'] = 'Unusual word length pattern'
                return result
        
        # Character diversity check
        unique_chars = len(set(text.lower()))
        if unique_chars < 10 and len(text) > 50:
            result['passed'] = False
            result['reason'] = 'Low character diversity'
            return result
        
        return result

class EmbeddingBasedPreSelect:
    """Zero-Shot/Embedding-Based PreSelect using semantic similarity"""
    
    def __init__(self):
        self.model = None
        self.reference_embeddings = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print("🔄 Loading sentence transformer for embedding-based preselect...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self._create_reference_embeddings()
                print("✅ Embedding-based preselect initialized!")
            except Exception as e:
                print(f"⚠️ Could not load sentence transformer: {e}")
                self.model = None
    
    def _create_reference_embeddings(self):
        """Create embeddings for high-quality reference texts"""
        reference_texts = [
            # High-quality educational content
            "This is a detailed explanation of how neural networks learn through backpropagation, adjusting weights to minimize error.",
            "Let me break down this complex problem step by step to help you understand the underlying concepts.",
            "The research findings indicate that this approach significantly improves performance across multiple metrics.",
            
            # High-quality code content
            "Here's a well-documented Python function that efficiently processes data using modern best practices.",
            "This algorithm demonstrates how to solve the problem with optimal time and space complexity.",
            
            # High-quality reasoning
            "Based on the evidence presented, we can conclude that the hypothesis is supported by the data.",
            "Let's analyze this systematically: first we identify the key variables, then we examine their relationships.",
            
            # High-quality dialogue
            "I understand your question about this topic. Let me provide a comprehensive answer that addresses your specific concerns."
        ]
        
        if self.model:
            self.reference_embeddings = self.model.encode(reference_texts)
    
    def preselect(self, text: str) -> Dict[str, Any]:
        """Embedding-based preselection using semantic similarity"""
        result = {
            'passed': True,
            'reason': None,
            'similarity_score': 0.0,
            'category_hints': []
        }
        
        if not self.model or self.reference_embeddings is None:
            # Fallback to basic checks if model not available
            result['similarity_score'] = 0.5
            return result
        
        try:
            # Encode the input text
            text_embedding = self.model.encode([text])
            
            # Calculate similarity to reference embeddings
            similarities = []
            for ref_emb in self.reference_embeddings:
                similarity = np.dot(text_embedding[0], ref_emb) / (
                    np.linalg.norm(text_embedding[0]) * np.linalg.norm(ref_emb)
                )
                similarities.append(similarity)
            
            # Take the maximum similarity as the score
            max_similarity = max(similarities)
            result['similarity_score'] = float(max_similarity)
            
            # Set threshold for passing
            threshold = 0.3  # Adjust based on testing
            if max_similarity < threshold:
                result['passed'] = False
                result['reason'] = f'Low semantic similarity ({max_similarity:.3f} < {threshold})'
            
            # Provide category hints based on which reference had highest similarity
            best_match_idx = similarities.index(max_similarity)
            if best_match_idx < 3:
                result['category_hints'].append('educational')
            elif best_match_idx < 5:
                result['category_hints'].append('code')
            elif best_match_idx < 7:
                result['category_hints'].append('reasoning')
            else:
                result['category_hints'].append('dialogue')
                
        except Exception as e:
            logger.error(f"Error in embedding-based preselect: {e}")
            result['similarity_score'] = 0.5  # Neutral score on error
        
        return result

class ModelInTheLoopPreSelect:
    """Model-in-the-Loop PreSelect using a classification model"""
    
    def __init__(self):
        self.classifier = None
        
        try:
            print("🔄 Loading classification model for model-in-the-loop preselect...")
            # Use a lightweight text classification model
            self.classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",  # You could use a different model
                device=-1,  # Use CPU to avoid memory issues
                truncation=True,
                max_length=512
            )
            print("✅ Model-in-the-loop preselect initialized!")
        except Exception as e:
            print(f"⚠️ Could not load classification model: {e}")
            try:
                # Fallback to a simpler model
                self.classifier = pipeline(
                    "text-classification",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=-1,
                    truncation=True,
                    max_length=512
                )
                print("✅ Model-in-the-loop preselect initialized with fallback model!")
            except Exception as e2:
                print(f"⚠️ Could not load fallback model: {e2}")
                self.classifier = None
    
    def preselect(self, text: str) -> Dict[str, Any]:
        """Model-based preselection using classification confidence"""
        result = {
            'passed': True,
            'reason': None,
            'confidence_score': 0.0,
            'predicted_label': None,
            'category_hints': []
        }
        
        if not self.classifier:
            # Fallback if model not available
            result['confidence_score'] = 0.5
            return result
        
        try:
            # Get classification result
            classification = self.classifier(text[:512])  # Truncate if needed
            
            if isinstance(classification, list):
                classification = classification[0]
            
            result['confidence_score'] = float(classification['score'])
            result['predicted_label'] = classification['label']
            
            # Different logic based on the model used
            if 'sentiment' in str(self.classifier.model.config).lower():
                # For sentiment models, prefer neutral/positive content
                if classification['label'] in ['NEGATIVE'] and classification['score'] > 0.8:
                    result['passed'] = False
                    result['reason'] = f'Strongly negative sentiment ({classification["score"]:.3f})'
                elif classification['label'] in ['POSITIVE', 'NEUTRAL']:
                    result['category_hints'].append('positive_content')
            else:
                # For other models, use confidence as quality indicator
                if classification['score'] < 0.6:
                    result['passed'] = False
                    result['reason'] = f'Low model confidence ({classification["score"]:.3f})'
            
        except Exception as e:
            logger.error(f"Error in model-based preselect: {e}")
            result['confidence_score'] = 0.5  # Neutral score on error
        
        return result

class MultiVariantPreSelector:
    """Combines all three preselect variants"""
    
    def __init__(self, enable_rule_based=True, enable_embedding=True, enable_model=True):
        self.enable_rule_based = enable_rule_based
        self.enable_embedding = enable_embedding
        self.enable_model = enable_model
        
        self.rule_based = RuleBasedPreSelect() if enable_rule_based else None
        self.embedding_based = EmbeddingBasedPreSelect() if enable_embedding else None
        self.model_based = ModelInTheLoopPreSelect() if enable_model else None
        
        print(f"🎯 MultiVariant PreSelector initialized:")
        print(f"   Rule-based: {'✅' if enable_rule_based else '❌'}")
        print(f"   Embedding-based: {'✅' if enable_embedding else '❌'}")
        print(f"   Model-based: {'✅' if enable_model else '❌'}")
    
    def preselect(self, text: str) -> Dict[str, Any]:
        """Run all enabled preselect variants and combine results"""
        results = {
            'rule_based': None,
            'embedding_based': None,
            'model_based': None,
            'overall_passed': True,
            'rejection_reasons': [],
            'combined_score': 0.0,
            'category_hints': set()
        }
        
        scores = []
        
        # Run rule-based preselect
        if self.rule_based:
            rule_result = self.rule_based.preselect(text)
            results['rule_based'] = rule_result
            
            if not rule_result['passed']:
                results['overall_passed'] = False
                results['rejection_reasons'].append(f"Rule-based: {rule_result['reason']}")
            else:
                scores.append(rule_result['quality_score'])
                results['category_hints'].update(rule_result['category_hints'])
        
        # Run embedding-based preselect
        if self.embedding_based and results['overall_passed']:  # Only if rule-based passed
            embed_result = self.embedding_based.preselect(text)
            results['embedding_based'] = embed_result
            
            if not embed_result['passed']:
                results['overall_passed'] = False
                results['rejection_reasons'].append(f"Embedding-based: {embed_result['reason']}")
            else:
                scores.append(embed_result['similarity_score'])
                results['category_hints'].update(embed_result['category_hints'])
        
        # Run model-based preselect
        if self.model_based and results['overall_passed']:  # Only if previous passed
            model_result = self.model_based.preselect(text)
            results['model_based'] = model_result
            
            if not model_result['passed']:
                results['overall_passed'] = False
                results['rejection_reasons'].append(f"Model-based: {model_result['reason']}")
            else:
                scores.append(model_result['confidence_score'])
                results['category_hints'].update(model_result['category_hints'])
        
        # Calculate combined score
        if scores:
            results['combined_score'] = np.mean(scores)
        
        # Convert set to list for JSON serialization
        results['category_hints'] = list(results['category_hints'])
        
        return results

class BartFilter:
    """Advanced BART-based content filtering and categorization"""
    
    def __init__(self):
        print("🔄 Initializing BART model for advanced filtering...")
        try:
            from transformers import BartForSequenceClassification
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
            
            # Load BART model for sequence classification
            self.model = BartForSequenceClassification.from_pretrained(
                "facebook/bart-base",
                num_labels=4,  # Number of categories
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            print("✅ BART model loaded successfully!")
            
        except Exception as e:
            print(f"⚠️ Error loading BART model: {e}")
            self.model = None
            self.tokenizer = None
    
    def classify_content(self, text: str) -> Dict[str, Any]:
        """Classify content using BART model"""
        if not self.model or not self.tokenizer:
            return {
                'category': 'uncategorized',
                'confidence': 0.0,
                'error': 'BART model not available'
            }
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Get prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
            # Get prediction
            predicted_id = torch.argmax(probs, dim=-1).item()
            confidence = torch.max(probs).item()
            
            # Map to category names
            category_names = ['roleplay', 'reasoning', 'function_calling', 'rag']
            predicted_category = category_names[predicted_id] if predicted_id < len(category_names) else 'uncategorized'
            
            return {
                'category': predicted_category,
                'confidence': confidence,
                'all_scores': {name: probs[0][i].item() for i, name in enumerate(category_names)}
            }
            
        except Exception as e:
            return {
                'category': 'uncategorized',
                'confidence': 0.0,
                'error': str(e)
            }

class VerificationSystem:
    """Advanced verification system for filtered samples"""
    
    def __init__(self, llama_analyzer):
        self.llama = llama_analyzer
        self.baseline_scores = None
        self.verification_cache = {}
    
    def establish_baseline(self, analysis_results: List[Dict]):
        """Calculate baseline performance metrics"""
        print("\n📊 Establishing baseline performance...")
        
        category_scores = defaultdict(list)
        for result in analysis_results:
            if 'category_scores' in result:
                for category, score in result['category_scores'].items():
                    category_scores[category].append(score)
        
        self.baseline_scores = {}
        for category, scores in category_scores.items():
            if scores:
                self.baseline_scores[category] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }
        
        print("✅ Baseline established:")
        for category, stats in self.baseline_scores.items():
            print(f"   {category}: mean={stats['mean']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")
    
    def verify_sample(self, text: str, expected_category: str) -> Dict[str, Any]:
        """Verify a single sample's quality and category fit"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.verification_cache:
            return self.verification_cache[text_hash]
        
        try:
            # Get Llama analysis
            llama_scores = self.llama.analyze_performance(text)
            
            # Calculate improvement over baseline
            improvements = {}
            if self.baseline_scores:
                for category, score in llama_scores.items():
                    if category in self.baseline_scores:
                        baseline = self.baseline_scores[category]['mean']
                        improvement = score - baseline
                        improvements[category] = improvement
            
            # Overall quality score
            quality_score = np.mean(list(llama_scores.values()))
            
            result = {
                'text': text,
                'llama_scores': llama_scores,
                'improvements': improvements,
                'quality_score': quality_score,
                'expected_category': expected_category,
                'category_match': llama_scores.get(expected_category, 0.0),
                'timestamp': datetime.now().isoformat()
            }
            
            self.verification_cache[text_hash] = result
            return result
            
        except Exception as e:
            return {
                'text': text,
                'error': str(e),
                'quality_score': 0.0
            }

class DatasetExporter:
    """Advanced dataset export with multiple formats and metadata"""
    
    def __init__(self, export_dir: str = "filtered_data_advanced"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.export_stats = defaultdict(int)
    
    def export_filtered_samples(self, samples: List[Dict], format_type: str = "jsonl") -> str:
        """Export filtered samples with comprehensive metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "jsonl":
            filename = f"filtered_samples_{timestamp}.jsonl"
            filepath = self.export_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                for sample in samples:
                    # Enhanced sample with full metadata
                    export_sample = {
                        'text': sample.get('text', ''),
                        'metadata': {
                            'category': sample.get('category', 'uncategorized'),
                            'quality_score': sample.get('quality_score', 0.0),
                            'filter_results': sample.get('filter_results', {}),
                            'llama_scores': sample.get('llama_scores', {}),
                            'improvements': sample.get('improvements', {}),
                            'timestamp': sample.get('timestamp', datetime.now().isoformat()),
                            'export_timestamp': datetime.now().isoformat()
                        }
                    }
                    
                    json.dump(export_sample, f, ensure_ascii=False)
                    f.write('\n')
                    self.export_stats['exported'] += 1
        
        elif format_type == "csv":
            filename = f"filtered_samples_{timestamp}.csv"
            filepath = self.export_dir / filename
            
            import csv
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                if samples:
                    fieldnames = ['text', 'category', 'quality_score', 'timestamp']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for sample in samples:
                        writer.writerow({
                            'text': sample.get('text', ''),
                            'category': sample.get('category', 'uncategorized'),
                            'quality_score': sample.get('quality_score', 0.0),
                            'timestamp': sample.get('timestamp', '')
                        })
                        self.export_stats['exported'] += 1
        
        # Export summary statistics
        self._export_summary(samples, timestamp)
        
        print(f"✅ Exported {len(samples)} samples to {filepath}")
        return str(filepath)
    
    def _export_summary(self, samples: List[Dict], timestamp: str):
        """Export comprehensive summary statistics"""
        summary_file = self.export_dir / f"export_summary_{timestamp}.json"
        
        # Calculate statistics
        categories = defaultdict(int)
        quality_scores = []
        
        for sample in samples:
            categories[sample.get('category', 'uncategorized')] += 1
            quality_scores.append(sample.get('quality_score', 0.0))
        
        summary = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(samples),
                'export_directory': str(self.export_dir)
            },
            'category_distribution': dict(categories),
            'quality_statistics': {
                'mean': np.mean(quality_scores) if quality_scores else 0.0,
                'std': np.std(quality_scores) if quality_scores else 0.0,
                'min': min(quality_scores) if quality_scores else 0.0,
                'max': max(quality_scores) if quality_scores else 0.0,
                'median': np.median(quality_scores) if quality_scores else 0.0
            },
            'export_stats': dict(self.export_stats)
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Summary statistics saved to {summary_file}")

# Example usage and comprehensive testing
def main():
    """Comprehensive testing with all enhancements and toggles"""
    # --- TOGGLES AND SETTINGS ---
    ENABLE_AI_TOXIC_FILTER = True   # Toggle AI-based toxic filter (toxic-bert)
    ENABLE_RULE_BASED_FILTER = True # Toggle rule-based filtering
    ENABLE_PRESELECT_FILTER = True  # Toggle original preselect filter (fast, simple filter)
    ENABLE_AI_PRESELECT = True      # Toggle AI-based preselection (Llama performance analysis)
    
    # Multi-Variant PreSelect Toggles
    ENABLE_RULE_BASED_PRESELECT = True    # Enhanced rule-based preselect (Non-AI)
    ENABLE_EMBEDDING_PRESELECT = True     # Zero-shot/embedding-based preselect
    ENABLE_MODEL_PRESELECT = True         # Model-in-the-loop preselect
    
    ENABLE_CACHE = True             # Toggle caching (stores filter results for speed)
    TOXICITY_THRESHOLD = 0.7        # Toxicity threshold (0.1-0.9)
    QUALITY_THRESHOLD = 0.4         # Quality threshold (0.0-1.0)
    LLAMA_IMPROVEMENT_THRESHOLD = 0.6  # Threshold for Llama improvement potential (0.0-1.0)
    DOMAIN = 'general'              # Domain: general, educational, children, medical
    BATCH_SIZE = 32                 # Batch size for processing
    NUM_SAMPLES = 500                # <--- ADJUST THIS to control how many samples to process
    HF_TOKEN = "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" # <--- PUT YOUR HUGGING FACE TOKEN HERE

    print("\n" + "="*60)
    print("🚀 ENHANCED FILTERING TEST WITH TOGGLES")
    print("="*60)
    print(f"AI-based toxic filter:    {'ENABLED' if ENABLE_AI_TOXIC_FILTER else 'DISABLED'}")
    print(f"Rule-based filter:        {'ENABLED' if ENABLE_RULE_BASED_FILTER else 'DISABLED'}")
    print(f"Original preselect:       {'ENABLED' if ENABLE_PRESELECT_FILTER else 'DISABLED'}")
    print(f"AI-based preselection:    {'ENABLED' if ENABLE_AI_PRESELECT else 'DISABLED'} (Llama performance analysis)")
    print(f"\n🎯 MULTI-VARIANT PRESELECT:")
    print(f"  Rule-based preselect:   {'ENABLED' if ENABLE_RULE_BASED_PRESELECT else 'DISABLED'} (Non-AI)")
    print(f"  Embedding preselect:    {'ENABLED' if ENABLE_EMBEDDING_PRESELECT else 'DISABLED'} (Zero-shot)")
    print(f"  Model preselect:        {'ENABLED' if ENABLE_MODEL_PRESELECT else 'DISABLED'} (Model-in-the-loop)")
    print(f"\nCaching:                  {'ENABLED' if ENABLE_CACHE else 'DISABLED'} (stores filter results for speed)")
    print(f"Toxicity threshold:       {TOXICITY_THRESHOLD}")
    print(f"Quality threshold:        {QUALITY_THRESHOLD}")
    print(f"Llama improvement thresh: {LLAMA_IMPROVEMENT_THRESHOLD}")
    print(f"Domain:                   {DOMAIN}")
    print(f"Batch size:               {BATCH_SIZE}")
    print(f"Num samples:              {NUM_SAMPLES}")
    print("="*60 + "\n")

    # Hugging Face login
    login(token=HF_TOKEN)

    # Initialize multi-variant preselect system
    multi_preselect = None
    if any([ENABLE_RULE_BASED_PRESELECT, ENABLE_EMBEDDING_PRESELECT, ENABLE_MODEL_PRESELECT]):
        try:
            multi_preselect = MultiVariantPreSelector(
                enable_rule_based=ENABLE_RULE_BASED_PRESELECT,
                enable_embedding=ENABLE_EMBEDDING_PRESELECT,
                enable_model=ENABLE_MODEL_PRESELECT
            )
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize multi-variant preselect: {e}")
            multi_preselect = None

    # Initialize AI-based preselection system (if enabled)
    llama_analyzer = None
    dataset_analyzer = None
    if ENABLE_AI_PRESELECT:
        try:
            llama_analyzer = LlamaAnalyzer()
            dataset_analyzer = DatasetAnalyzer(llama_analyzer)
            print("✅ AI-based preselection system initialized!")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize AI preselection: {e}")
            print("   Continuing without AI preselection...")
            ENABLE_AI_PRESELECT = False

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
    
    # Summary tracking
    total_processed = 0
    passed_filters = 0
    category_counts = {'roleplay': 0, 'reasoning': 0, 'function_calling': 0, 'rag': 0, 'uncategorized': 0}
    inappropriate_counts = {'profanity': 0, 'hate_speech': 0, 'violence': 0, 'sexual_content': 0, 'too_long': 0, 'too_short': 0, 'excessive_repetition': 0, 'excessive_caps': 0}
    preselect_filtered = 0
    ai_filtered = 0
    rule_filtered = 0
    quality_filtered = 0
    
    # Track samples cleared by each filter
    cleared_by_preselect = 0    # Passed basic preselect
    cleared_by_rules = 0        # Passed rule-based filter
    cleared_by_ai_toxic = 0     # Passed AI toxic filter
    cleared_by_quality = 0      # Passed quality assessment
    cleared_all_filters = 0     # Passed everything
    
    # AI preselection tracking
    llama_improvement_scores = {'roleplay': [], 'reasoning': [], 'function_calling': [], 'rag': []}
    high_potential_samples = []
    
    # Multi-variant preselect tracking
    multi_preselect_stats = {
        'rule_based_passed': 0,
        'embedding_passed': 0,
        'model_passed': 0,
        'overall_passed': 0,
        'rejection_reasons': defaultdict(int)
    }
    
    for i, sample in enumerate(itertools.islice(dataset, NUM_SAMPLES)):
        text = sample.get("text", "").strip()
        result = filter_system.comprehensive_filter(text)
        
        # Multi-variant preselect analysis (if enabled)
        if multi_preselect:
            try:
                preselect_results = multi_preselect.preselect(text)
                result['multi_preselect'] = preselect_results
                
                # Track multi-preselect statistics
                if preselect_results['rule_based'] and preselect_results['rule_based']['passed']:
                    multi_preselect_stats['rule_based_passed'] += 1
                if preselect_results['embedding_based'] and preselect_results['embedding_based']['passed']:
                    multi_preselect_stats['embedding_passed'] += 1
                if preselect_results['model_based'] and preselect_results['model_based']['passed']:
                    multi_preselect_stats['model_passed'] += 1
                if preselect_results['overall_passed']:
                    multi_preselect_stats['overall_passed'] += 1
                
                # Track rejection reasons
                for reason in preselect_results['rejection_reasons']:
                    multi_preselect_stats['rejection_reasons'][reason] += 1
                
                # If multi-preselect fails, override the result
                if not preselect_results['overall_passed']:
                    result['should_include'] = False
                    result['preselect_filtered'] = True
                    result['preselect_reason'] = f"Multi-preselect: {'; '.join(preselect_results['rejection_reasons'])}"
                    
            except Exception as e:
                logger.error(f"Error in multi-variant preselect: {e}")
                result['multi_preselect'] = None
        
        # AI-based preselection analysis (if enabled)
        if ENABLE_AI_PRESELECT and dataset_analyzer and result['should_include']:
            try:
                llama_analysis = dataset_analyzer.analyze_sample(text)
                result['llama_analysis'] = llama_analysis
                
                # Track improvement potential for each category
                for category, potential in llama_analysis['potential_value'].items():
                    llama_improvement_scores[category].append(potential)
                    
                    # If this sample has high improvement potential for any category
                    if potential > LLAMA_IMPROVEMENT_THRESHOLD:
                        high_potential_samples.append({
                            'text': text[:200] + '...' if len(text) > 200 else text,
                            'category': category,
                            'improvement_potential': potential,
                            'llama_scores': llama_analysis['category_scores']
                        })
                        
            except Exception as e:
                logger.error(f"Error in AI preselection analysis: {e}")
                result['llama_analysis'] = None
        
        filtered_results.append(result)
        
        total_processed += 1
        if result['should_include']:
            passed_filters += 1
        
        # Count samples cleared by each filter stage
        if not result.get('preselect_filtered', False):
            cleared_by_preselect += 1
            
            # If passed preselect, check rule-based filter
            if result.get('rule_based') and result['rule_based'].get('is_safe', True):
                cleared_by_rules += 1
                
                # If passed rules, check AI toxic filter
                if not result.get('ai_based') or not result['ai_based'].get('is_toxic', False):
                    cleared_by_ai_toxic += 1
                    
                    # If passed AI toxic, check quality
                    if result.get('meets_quality', False):
                        cleared_by_quality += 1
                        
                        # If passed quality, it cleared all filters
                        if result.get('should_include', False):
                            cleared_all_filters += 1
        
        # Count categories
        if result['preselect_category'] in category_counts:
            category_counts[result['preselect_category']] += 1
        
        # Count inappropriate categories
        for inap_cat in result['inappropriate_categories']:
            if inap_cat in inappropriate_counts:
                inappropriate_counts[inap_cat] += 1
        
        # Count filter types that rejected
        if result.get('preselect_filtered', False):
            preselect_filtered += 1
        elif not result['should_include']:
            if result.get('ai_based') and result['ai_based'].get('is_toxic', False):
                ai_filtered += 1
            elif result.get('rule_based') and not result['rule_based'].get('is_safe', True):
                rule_filtered += 1
            elif not result.get('meets_quality', False):
                quality_filtered += 1
        
        # Print summary every 100 samples
        if (i + 1) % 100 == 0:
            print(f"\n--- Summary after {i + 1} samples ---")
            print(f"📈 SAMPLES CLEARED BY EACH FILTER:")
            print(f"  1️⃣ Preselect filter:     {cleared_by_preselect}/{total_processed} ({cleared_by_preselect/total_processed*100:.1f}%)")
            print(f"  2️⃣ Rule-based filter:    {cleared_by_rules}/{total_processed} ({cleared_by_rules/total_processed*100:.1f}%)")
            print(f"  3️⃣ AI toxic filter:      {cleared_by_ai_toxic}/{total_processed} ({cleared_by_ai_toxic/total_processed*100:.1f}%)")
            print(f"  4️⃣ Quality assessment:   {cleared_by_quality}/{total_processed} ({cleared_by_quality/total_processed*100:.1f}%)")
            print(f"  ✅ ALL FILTERS PASSED:   {cleared_all_filters}/{total_processed} ({cleared_all_filters/total_processed*100:.1f}%)")
            
            print(f"\n❌ SAMPLES REJECTED BY:")
            print(f"  Preselect filter: {preselect_filtered} ({preselect_filtered/total_processed*100:.1f}%)")
            print(f"  Rule-based filter: {rule_filtered} ({rule_filtered/total_processed*100:.1f}%)")
            print(f"  AI toxic filter: {ai_filtered} ({ai_filtered/total_processed*100:.1f}%)")
            print(f"  Quality filter: {quality_filtered} ({quality_filtered/total_processed*100:.1f}%)")
            
            print(f"\n📊 Category distribution:")
            for cat, count in category_counts.items():
                if count > 0:
                    print(f"  {cat}: {count} ({count/total_processed*100:.1f}%)")
            
            # Multi-variant preselect summary
            if multi_preselect:
                print(f"\n🎯 MULTI-VARIANT PRESELECT STATS:")
                if ENABLE_RULE_BASED_PRESELECT:
                    print(f"  Rule-based passed: {multi_preselect_stats['rule_based_passed']}/{total_processed} ({multi_preselect_stats['rule_based_passed']/total_processed*100:.1f}%)")
                if ENABLE_EMBEDDING_PRESELECT:
                    print(f"  Embedding passed: {multi_preselect_stats['embedding_passed']}/{total_processed} ({multi_preselect_stats['embedding_passed']/total_processed*100:.1f}%)")
                if ENABLE_MODEL_PRESELECT:
                    print(f"  Model passed: {multi_preselect_stats['model_passed']}/{total_processed} ({multi_preselect_stats['model_passed']/total_processed*100:.1f}%)")
                print(f"  Overall passed: {multi_preselect_stats['overall_passed']}/{total_processed} ({multi_preselect_stats['overall_passed']/total_processed*100:.1f}%)")
            
            # AI preselection summary
            if ENABLE_AI_PRESELECT and llama_improvement_scores:
                print(f"\n🦙 Llama improvement potential (avg):")
                for cat, scores in llama_improvement_scores.items():
                    if scores:
                        avg_potential = np.mean(scores)
                        print(f"  {cat}: {avg_potential:.2f}")
                print(f"🎯 High-potential samples found: {len(high_potential_samples)}")
            
            print(f"\n⚠️  Top inappropriate content:")
            sorted_inap = sorted(inappropriate_counts.items(), key=lambda x: x[1], reverse=True)
            for cat, count in sorted_inap[:5]:
                if count > 0:
                    print(f"  {cat}: {count} ({count/total_processed*100:.1f}%)")

    print(f"\n" + "="*60)
    print(f"🎯 FINAL SUMMARY - {total_processed} samples processed")
    print(f"="*60)
    
    print(f"📈 FINAL FILTER CLEARANCE RATES:")
    print(f"  1️⃣ Preselect filter:     {cleared_by_preselect}/{total_processed} ({cleared_by_preselect/total_processed*100:.1f}%)")
    print(f"  2️⃣ Rule-based filter:    {cleared_by_rules}/{total_processed} ({cleared_by_rules/total_processed*100:.1f}%)")
    print(f"  3️⃣ AI toxic filter:      {cleared_by_ai_toxic}/{total_processed} ({cleared_by_ai_toxic/total_processed*100:.1f}%)")
    print(f"  4️⃣ Quality assessment:   {cleared_by_quality}/{total_processed} ({cleared_by_quality/total_processed*100:.1f}%)")
    print(f"  ✅ ALL FILTERS PASSED:   {cleared_all_filters}/{total_processed} ({cleared_all_filters/total_processed*100:.1f}%)")
    
    print(f"\n❌ FINAL REJECTION BREAKDOWN:")
    print(f"  Preselect filter: {preselect_filtered} ({preselect_filtered/total_processed*100:.1f}%)")
    print(f"  Rule-based filter: {rule_filtered} ({rule_filtered/total_processed*100:.1f}%)")
    print(f"  AI toxic filter: {ai_filtered} ({ai_filtered/total_processed*100:.1f}%)")
    print(f"  Quality filter: {quality_filtered} ({quality_filtered/total_processed*100:.1f}%)")
    
    print(f"\n📊 Final category distribution:")
    for cat, count in category_counts.items():
        if count > 0:
            print(f"  {cat}: {count} ({count/total_processed*100:.1f}%)")
    print(f"\n⚠️  Final inappropriate content breakdown:")
    sorted_inap = sorted(inappropriate_counts.items(), key=lambda x: x[1], reverse=True)
    for cat, count in sorted_inap:
        if count > 0:
            print(f"  {cat}: {count} ({count/total_processed*100:.1f}%)")

    # AI preselection final summary
    if ENABLE_AI_PRESELECT and llama_improvement_scores:
        print(f"\n🦙 LLAMA IMPROVEMENT ANALYSIS:")
        print(f"🎯 Total high-potential samples: {len(high_potential_samples)}")
        print(f"📈 Average improvement potential by category:")
        for cat, scores in llama_improvement_scores.items():
            if scores:
                avg_potential = np.mean(scores)
                print(f"  {cat}: {avg_potential:.3f} (lower Llama scores = higher improvement potential)")
        
        # Show top high-potential samples
        if high_potential_samples:
            print(f"\n🔥 TOP HIGH-POTENTIAL SAMPLES FOR LLAMA TRAINING:")
            high_potential_samples.sort(key=lambda x: x['improvement_potential'], reverse=True)
            for i, sample in enumerate(high_potential_samples[:3]):
                print(f"\n  Sample {i+1} (Category: {sample['category']}):")
                print(f"    Improvement potential: {sample['improvement_potential']:.3f}")
                print(f"    Llama scores: {sample['llama_scores']}")
                print(f"    Text preview: {sample['text']}")

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

    # --- ADVANCED FEATURES DEMONSTRATION ---
    if cleared_all_filters > 0:
        print("\n" + "="*60)
        print("🚀 ADVANCED FEATURES DEMONSTRATION")
        print("="*60)
        
        # Initialize advanced components
        try:
            print("\n🔄 Initializing advanced components...")
            
            # BART Filter
            bart_filter = BartFilter()
            
            # Verification System
            verification_system = None
            if llama_analyzer:
                verification_system = VerificationSystem(llama_analyzer)
                
                # Create some mock analysis results for baseline
                mock_analysis = []
                for result in filtered_results[:min(10, len(filtered_results))]:
                    mock_result = {
                        'category_scores': {
                            'roleplay': 0.5,
                            'reasoning': 0.6,
                            'function_calling': 0.4,
                            'rag': 0.5
                        }
                    }
                    mock_analysis.append(mock_result)
                
                if mock_analysis:
                    verification_system.establish_baseline(mock_analysis)
            
            # Dataset Exporter
            exporter = DatasetExporter(export_dir="filtered_data_advanced")
            
            # Demonstrate BART classification on first few samples
            print("\n🎯 BART Classification Results:")
            for i, result in enumerate(filtered_results[:3]):
                if 'text' in result:
                    bart_result = bart_filter.classify_content(result['text'])
                    print(f"   Sample {i+1}:")
                    print(f"      BART Category: {bart_result.get('category', 'unknown')}")
                    print(f"      Confidence: {bart_result.get('confidence', 0.0):.3f}")
                    if 'all_scores' in bart_result:
                        print(f"      All Scores: {bart_result['all_scores']}")
            
            # Demonstrate verification on samples
            if verification_system:
                print("\n🔍 Sample Verification Results:")
                for i, result in enumerate(filtered_results[:2]):
                    if 'text' in result:
                        verification = verification_system.verify_sample(
                            result['text'], 
                            result.get('preselect_category', 'uncategorized')
                        )
                        print(f"   Sample {i+1}:")
                        print(f"      Quality Score: {verification.get('quality_score', 0.0):.3f}")
                        if 'llama_scores' in verification:
                            print(f"      Llama Scores: {verification['llama_scores']}")
            
            # Export filtered samples
            if filtered_results:
                print("\n📤 Exporting filtered samples...")
                
                # Prepare samples for export
                export_samples = []
                for result in filtered_results:
                    export_sample = {
                        'text': result.get('text', ''),
                        'category': result.get('preselect_category', 'uncategorized'),
                        'quality_score': result.get('overall_quality', 0.0),
                        'filter_results': {
                            'rule_based': result.get('rule_based', {}),
                            'ai_based': result.get('ai_based', {}),
                            'quality': result.get('quality', {})
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    export_samples.append(export_sample)
                
                # Export in JSONL format
                jsonl_path = exporter.export_filtered_samples(export_samples, format_type="jsonl")
                
                # Export in CSV format
                csv_path = exporter.export_filtered_samples(export_samples, format_type="csv")
                
                print(f"✅ Exported to JSONL: {jsonl_path}")
                print(f"✅ Exported to CSV: {csv_path}")
            
            print("\n🎉 Advanced features demonstration complete!")
            
        except Exception as e:
            print(f"⚠️ Error in advanced features demonstration: {e}")
            print("   Basic filtering results are still available above.")
    
    else:
        print("\n⚠️ No samples passed all filters, skipping advanced features demonstration.")

if __name__ == "__main__":
    main()