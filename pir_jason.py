# Enhanced ClimbLab Dataset Preselection with Advanced Content Safety and Quality Assessment
import os
import re
import torch
# hashlib removed (was used for caching)
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
    pipeline,
    BartTokenizer, BartForConditionalGeneration
)
from pathlib import Path
from datetime import datetime
# inappropriate content filtering removed - handled separately if needed

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
# Cache functionality removed for simplicity

@dataclass
class FilterConfig:
    """Configuration for filtering parameters"""
    # toxicity_threshold removed (no longer using toxic-bert)
    quality_threshold: float = 0.4
    safety_threshold: float = 0.8
    batch_size: int = 32
    # cache_enabled removed
    context_aware: bool = True
    domain: str = "general"  # general, educational, children, medical

class ContentPatterns:
    """Clean pattern detection focused on quality and structure only"""
    
    def __init__(self):
        # Focus only on quality and structural issues, not content censorship
        self.spam_patterns = [
            r'(.)\1{15,}',  # Excessive character repetition (15+ chars)
            r'\b(click here|buy now|free money|urgent|limited time)\b',  # Obvious spam phrases
        ]
        
        self.quality_patterns = [
            r'^.{1,5}$',  # Extremely short content (1-5 chars)
            r'^\s*$',     # Empty or whitespace only
            r'[^\w\s]{50,}',  # Excessive special characters (50+)
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = {
            'spam': [re.compile(p, re.IGNORECASE) for p in self.spam_patterns],
            'quality': [re.compile(p, re.IGNORECASE) for p in self.quality_patterns]
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

# Cache functionality removed for simplicity

class AdaptiveThresholds:
    """Domain-specific threshold adjustment"""
    
    def __init__(self, config: FilterConfig):
        self.base_config = config
        self.domain_adjustments = {
            'educational': {
                # toxicity_threshold removed
                'safety_threshold': 0.7,
                'quality_threshold': 0.5
            },
            'children': {
                # toxicity_threshold removed
                'safety_threshold': 0.9,
                'quality_threshold': 0.6
            },
            'medical': {
                # toxicity_threshold removed
                'safety_threshold': 0.6,
                'quality_threshold': 0.5
            },
            'general': {
                # toxicity_threshold removed
                'safety_threshold': 0.8,
                'quality_threshold': 0.4
            }
        }
    
    def get_adjusted_config(self, domain: str) -> FilterConfig:
        """Get domain-adjusted configuration"""
        adjustments = self.domain_adjustments.get(domain, self.domain_adjustments['general'])
        
        adjusted_config = FilterConfig()
        # adjusted_config.toxicity_threshold removed
        adjusted_config.safety_threshold = adjustments['safety_threshold']
        adjusted_config.quality_threshold = adjustments['quality_threshold']
        adjusted_config.batch_size = self.base_config.batch_size
        # cache_enabled removed
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
            # toxicity_threshold removed
        
        # Professional context - be more strict
        elif any(pattern in recent_text_lower for pattern in self.context_patterns['professional']):
            adjustments['quality_threshold'] = 0.1
            # toxicity_threshold removed
        
        return adjustments

class PerformanceMonitor:
    """Monitor and track filtering performance metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.filter_calls = 0
        # Cache removed
        self.ai_model_calls = 0
        self.rule_based_calls = 0
        self.processing_times = []
    
    def record_filter_call(self):
        """Record a filter call"""
        self.filter_calls += 1
    
    # Cache functionality removed
    
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
            # Cache removed
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
        # Cache removed for simplicity
        self.adaptive_thresholds = AdaptiveThresholds(config)
        
        # Initialize AI models (lazy loading)
        # toxic_classifier removed
        self.bart_model = None
        self.bart_tokenizer = None
        
        # Initialize conversation context analyzer and performance monitor
        self.conversation_analyzer = ConversationContextAnalyzer()
        self.performance_monitor = PerformanceMonitor()
        
    # Toxic classifier removed - inappropriate content filtering handled separately
    
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
        """Clean rule-based filtering focused on quality and spam only"""
        # Cache removed for simplicity
        
        result = {
            'violations': [],
            'warnings': [],
            'context_flags': [],
            'severity_score': 0.0,
            'is_safe': True,
            'language_detected': 'en'
        }
        
        # Check each category (only spam and quality patterns now)
        for category, patterns in self.patterns.compiled_patterns.items():
            matches = []
            for pattern in patterns:
                matches.extend(pattern.findall(text))
            
            if matches:
                # For quality issues, severity is binary (either pass or fail)
                if category == 'quality':
                    result['violations'].append({
                        'category': category,
                        'matches': ['quality_issue'],  # Don't expose specific matches
                        'severity': 100.0  # Quality issues are binary
                    })
                    result['is_safe'] = False
                    result['severity_score'] = 100.0
                
                # For spam, calculate severity based on frequency
                elif category == 'spam':
                    severity = len(matches) / max(1, len(text.split())) * 100
                    
                    if severity > 10.0:  # High spam threshold
                        result['violations'].append({
                            'category': category,
                            'matches': matches[:2],  # Limit for privacy
                            'severity': severity
                        })
                        result['is_safe'] = False
                    elif severity > 3.0:  # Warning threshold
                        result['warnings'].append({
                            'category': category,
                            'severity': severity
                        })
                    
                    result['severity_score'] = max(result['severity_score'], severity)
        
        # Cache removed for simplicity
        
        return result
    
    # AI content filter (toxic-bert) removed - inappropriate content filtering handled separately
    
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
                # 'ai_based' removed,
                'quality': None,
                'overall_quality': 0.0,
                'config_used': self.config.domain,
                'context_adjustments': {},
                'performance_metrics': self.performance_monitor.get_performance_report(),
                'preselect_category': categorize_sample(text),
                # 'inappropriate_categories' removed
            }
        # Step 1: Rule-based pre-filter
        rule_result = self.rule_based_filter(text)
        
        # Step 2: AI-based verification removed
        
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
        # AI toxic filtering removed
        
        meets_quality = overall_quality >= final_quality_threshold
        
        # --- Categorization step ---
        preselect_category = categorize_sample(text)
        # inappropriate_categories removed
        
        # Category relevance check - reject if completely unrelated to our target categories
        is_category_relevant = preselect_category != 'uncategorized'
        
        return {
            'text': text,
            'is_safe': is_safe,
            'meets_quality': meets_quality,
            'is_category_relevant': is_category_relevant,
            'should_include': is_safe and meets_quality and is_category_relevant,
            'rule_based': rule_result,
            # 'ai_based' removed,
            'quality': quality_result,
            'overall_quality': overall_quality,
            'config_used': self.config.domain,
            'context_adjustments': context_adjustments,
            'performance_metrics': self.performance_monitor.get_performance_report(),
            'preselect_category': preselect_category,
            # 'inappropriate_categories' removed,
            'preselect_filtered': False,
            'preselect_reason': None,
            'llama_analysis': None  # Will be filled by AI preselection if enabled
        }

# Preselect categories and patterns for Llama model improvement
PRESELECT_CATEGORIES = {
    'function_calling': [
        # Programming/coding keywords
        'function', 'def ', 'return', 'import', 'class', 'method', 'parameter', 'argument',
        'api call', 'endpoint', 'request', 'response', 'json', 'post', 'get', 'put', 'delete',
        'command', 'execute', 'run', 'script', 'code', 'program', 'library', 'module',
        'variable', 'array', 'object', 'string', 'integer', 'boolean', 'null', 'undefined',
        'if ', 'else', 'elif', 'for ', 'while', 'try', 'except', 'catch', 'finally',
        'print(', 'console.log', 'println', 'cout', 'printf', 'echo',
        'python', 'javascript', 'java', 'cpp', 'c++', 'html', 'css', 'sql', 'php', 'ruby'
    ],
    'reasoning': [
        # Logical reasoning and analysis
        'step by step', 'first', 'second', 'third', 'finally', 'therefore', 'because',
        'analyze', 'analysis', 'reasoning', 'logic', 'logical', 'conclusion', 'deduce',
        'problem solving', 'solve', 'solution', 'approach', 'method', 'strategy',
        'evidence', 'proof', 'demonstrate', 'explanation', 'explain', 'understand',
        'cause', 'effect', 'result', 'consequence', 'implication', 'inference',
        'hypothesis', 'theory', 'principle', 'rule', 'pattern', 'relationship',
        'compare', 'contrast', 'difference', 'similarity', 'evaluation', 'assessment',
        'lets think', "let's analyze", 'consider', 'examine', 'investigate', 'explore'
    ],
    'roleplay': [
        # Character interaction and dialogue
        'character', 'role', 'persona', 'acting as', 'pretend', 'imagine', 'scenario',
        'conversation', 'dialogue', 'chat', 'talk', 'speak', 'say', 'respond', 'reply',
        'you are', 'i am', 'he is', 'she is', 'they are', 'we are',
        'roleplay', 'role-play', 'role playing', 'simulate', 'simulation',
        'story', 'narrative', 'plot', 'scene', 'setting', 'background',
        'interaction', 'communicate', 'express', 'feel', 'emotion', 'reaction',
        'adventure', 'journey', 'quest', 'mission', 'task', 'challenge',
        'world', 'universe', 'realm', 'environment', 'place', 'location'
    ],
    'rag': [
        # Research and information retrieval
        'according to', 'research shows', 'studies indicate', 'evidence suggests',
        'source', 'reference', 'citation', 'document', 'paper', 'article', 'report',
        'data shows', 'findings', 'results', 'statistics', 'survey', 'poll',
        'expert', 'authority', 'specialist', 'researcher', 'scientist', 'scholar',
        'published', 'journal', 'book', 'database', 'repository', 'archive',
        'information', 'knowledge', 'fact', 'detail', 'background', 'context',
        'retrieve', 'search', 'find', 'locate', 'discover', 'identify',
        'based on', 'derived from', 'extracted from', 'obtained from',
        'wikipedia', 'encyclopedia', 'documentation', 'manual', 'guide'
    ]
}

def categorize_sample(text: str) -> str:
    """Improved categorization with scoring and thresholds"""
    text_lower = text.lower()
    category_scores = {}
    
    # Calculate score for each category
    for category, patterns in PRESELECT_CATEGORIES.items():
        score = 0
        for pattern in patterns:
            if pattern in text_lower:
                # Weight longer patterns more heavily
                score += len(pattern.split())
        category_scores[category] = score
    
    # Find the category with highest score
    if category_scores:
        best_category = max(category_scores.items(), key=lambda x: x[1])
        best_score = best_category[1]
        
        # Only assign category if score meets minimum threshold
        MIN_CATEGORY_RELEVANCE = 1  # Require at least 1 pattern match (adjustable)
        if best_score >= MIN_CATEGORY_RELEVANCE:
            return best_category[0]
    
    return 'uncategorized'

# INAPPROPRIATE_PATTERNS moved to inappropriate_content_filter.py

# inappropriate_categories function moved to inappropriate_content_filter.py

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
    
    def __init__(self):
        self.baseline_scores = None
        # Cache removed for simplicity
    
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
        # Cache removed for simplicity
        
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
            
            # Cache removed for simplicity
            return result
            
        except Exception as e:
            return {
                'text': text,
                'error': str(e),
                'quality_score': 0.0
            }

class DatasetExporter:
    """Advanced dataset export with multiple formats and metadata"""
    
    def __init__(self, export_dir: str = "filtered_data_advanced", clear_previous: bool = True):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.export_stats = defaultdict(int)
        
        # Clear previous files if requested
        if clear_previous:
            self._clear_previous_files()
    
    def _clear_previous_files(self):
        """Clear previous export files to avoid accumulation"""
        try:
            # Clear main export directory files
            for file in self.export_dir.glob("*.jsonl"):
                file.unlink()
                print(f"🗑️  Cleared: {file.name}")
            for file in self.export_dir.glob("*.csv"):
                file.unlink()
                print(f"🗑️  Cleared: {file.name}")
            for file in self.export_dir.glob("*.json"):
                file.unlink()
                print(f"🗑️  Cleared: {file.name}")
            
            # Clear category training directory files
            category_dir = Path("category_training_datasets")
            if category_dir.exists():
                for file in category_dir.glob("*.jsonl"):
                    file.unlink()
                    print(f"🗑️  Cleared: category_training_datasets/{file.name}")
                for file in category_dir.glob("*.json"):
                    file.unlink()
                    print(f"🗑️  Cleared: category_training_datasets/{file.name}")
                    
            print("✅ Previous export files cleared!")
        except Exception as e:
            print(f"⚠️  Warning: Could not clear some files: {e}")
    
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
    
    def export_category_training_datasets(self, filtered_results: List[Dict], export_mode: str = "separate", 
                                         save_function_calling: bool = True, save_reasoning: bool = True,
                                         save_roleplay: bool = True, save_chatrag: bool = True, 
                                         save_combined: bool = True, save_summary: bool = False) -> Dict[str, str]:
        """Export category-specific training datasets for the 4 target categories
        
        Args:
            filtered_results: List of filtered sample dictionaries
            export_mode: "separate" (4 files), "combined" (1 file), or "none" (no export, stats only)
            save_function_calling: Whether to save function calling file
            save_reasoning: Whether to save reasoning file  
            save_roleplay: Whether to save roleplay file
            save_chatrag: Whether to save chatrag file
            save_combined: Whether to save combined file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported_files = {}
        category_stats = {}
        
        # Map 'rag' to 'chatrag' for consistency with your goal
        category_mapping = {
            'function_calling': 'function_calling',
            'reasoning': 'reasoning', 
            'roleplay': 'roleplay',
            'rag': 'chatrag'
        }
        
        # Categorize samples
        categorized_samples = {
            'function_calling': [],
            'reasoning': [],
            'roleplay': [],
            'chatrag': []
        }
        
        for result in filtered_results:
            # Include ALL samples that have a valid category (not just those passing quality filters)
            category = result.get('preselect_category', 'uncategorized')
            mapped_category = category_mapping.get(category)
            
            if mapped_category:
                # Check if sample passed all filters OR just has good category relevance
                passed_all_filters = result.get('should_include', False)
                is_category_relevant = result.get('is_category_relevant', False)
                
                # Include if it passed all filters OR if it's category-relevant with decent quality
                basic_quality_ok = result.get('quality_score', 0.0) > 0.2  # Lower threshold for category training
                
                if passed_all_filters or (is_category_relevant and basic_quality_ok):
                    # Clean training format - just text for actual training
                    training_sample = {
                        'text': result.get('text', '').strip()
                    }
                    
                    # Full sample with metadata for analysis (optional)
                    full_sample = {
                        'text': result.get('text', ''),
                        'category': mapped_category,
                        'quality_score': result.get('quality_score', 0.0),
                        'filter_passed': passed_all_filters,
                        'category_relevant': is_category_relevant,
                        'metadata': {
                            'original_category': category,
                            # 'toxicity_score' removed (no more toxic-bert),
                            'quality_metrics': result.get('quality_metrics', {}),
                            # 'inappropriate_categories' removed,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                    
                    # Store both versions
                    categorized_samples[mapped_category].append({
                        'training': training_sample,
                        'full': full_sample
                    })
        
        # Calculate category statistics first
        total_exported = 0
        for category, sample_list in categorized_samples.items():
            category_stats[category] = len(sample_list)
            total_exported += len(sample_list)
        
        # Export based on mode
        if export_mode == "separate":
            # Define toggle mapping for each category
            category_toggles = {
                'function_calling': save_function_calling,
                'reasoning': save_reasoning, 
                'roleplay': save_roleplay,
                'chatrag': save_chatrag
            }
            
            # Export each category to separate files (with toggles)
            for category, samples in categorized_samples.items():
                if not samples:
                    print(f"⚠️  No samples found for {category} category")
                    continue
                
                # Check if this category is enabled
                if not category_toggles.get(category, True):
                    print(f"⏭️  Skipping {category} export (toggle disabled) - {len(samples)} samples would be saved")
                    continue
                    
                filename = f"{category}_training_{timestamp}.jsonl"
                filepath = self.export_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    for sample_data in samples:
                        # Use clean training format (just text)
                        json.dump(sample_data['training'], f, ensure_ascii=False)
                        f.write('\n')
                
                exported_files[category] = str(filepath)
                print(f"✅ Exported {len(samples)} {category} samples to {filename}")
            
            # Also create combined file if requested (even in separate mode)
            if save_combined and total_exported > 0:
                combined_filename = f"all_categories_training_{timestamp}.jsonl"
                combined_filepath = self.export_dir / combined_filename
                
                with open(combined_filepath, 'w', encoding='utf-8') as f:
                    for category, samples in categorized_samples.items():
                        if category_toggles.get(category, True):  # Only include enabled categories
                            for sample_data in samples:
                                # Use clean training format (just text)
                                json.dump(sample_data['training'], f, ensure_ascii=False)
                                f.write('\n')
                
                exported_files['combined'] = str(combined_filepath)
                print(f"✅ Exported combined training file with {total_exported} samples: {combined_filename}")
        
        elif export_mode == "combined":
            # Create single combined training file with all categories
            if total_exported > 0:
                if save_combined:
                    combined_filename = f"all_categories_training_{timestamp}.jsonl"
                    combined_filepath = self.export_dir / combined_filename
                    
                    with open(combined_filepath, 'w', encoding='utf-8') as f:
                        for category, samples in categorized_samples.items():
                            for sample_data in samples:
                                # Use clean training format (just text)
                                json.dump(sample_data['training'], f, ensure_ascii=False)
                                f.write('\n')
                    
                    exported_files['combined'] = str(combined_filepath)
                    print(f"✅ Exported combined training file with {total_exported} samples: {combined_filename}")
                else:
                    print(f"⏭️  Skipping combined file export (toggle disabled) - {total_exported} samples would be saved")
            else:
                print("⚠️  No samples to export")
        
        elif export_mode == "none":
            # No file export, just show statistics
            print(f"📊 Export mode: NONE - Only showing statistics (no files created)")
            print(f"📈 {total_exported} samples would be exported if export was enabled")
        
        else:
            print(f"⚠️  Unknown export mode: {export_mode}. Using 'separate' mode.")
            # Fallback to separate mode
            for category, samples in categorized_samples.items():
                if samples:
                    filename = f"{category}_training_{timestamp}.jsonl"
                    filepath = self.export_dir / filename
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        for sample_data in samples:
                            # Use clean training format (just text)
                            json.dump(sample_data['training'], f, ensure_ascii=False)
                            f.write('\n')
                    
                    exported_files[category] = str(filepath)
                    print(f"✅ Exported {len(samples)} {category} samples to {filename}")
                else:
                    print(f"⚠️  No samples found for {category} category")
        
        # Create category distribution summary
        summary = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'total_samples_exported': total_exported,
                'export_directory': str(self.export_dir)
            },
            'category_statistics': category_stats,
            'files_created': exported_files,
            'training_readiness': {
                'function_calling': {
                    'samples': category_stats.get('function_calling', 0),
                    'ready_for_training': category_stats.get('function_calling', 0) >= 100,
                    'recommended_min': 1000
                },
                'reasoning': {
                    'samples': category_stats.get('reasoning', 0),
                    'ready_for_training': category_stats.get('reasoning', 0) >= 100,
                    'recommended_min': 1000
                },
                'roleplay': {
                    'samples': category_stats.get('roleplay', 0),
                    'ready_for_training': category_stats.get('roleplay', 0) >= 100,
                    'recommended_min': 1000
                },
                'chatrag': {
                    'samples': category_stats.get('chatrag', 0),
                    'ready_for_training': category_stats.get('chatrag', 0) >= 100,
                    'recommended_min': 1000
                }
            }
        }
        
        # Only save summary file if requested
        if save_summary:
            summary_file = self.export_dir / f"category_training_summary_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"📋 Summary saved to: {summary_file} 💾")
        
        print(f"\n🎯 CATEGORY TRAINING DATASET SUMMARY:")
        print(f"📊 Total samples exported: {total_exported}")
        category_emojis = {'function_calling': '📞', 'reasoning': '🧠', 'roleplay': '🎭', 'chatrag': '💬'}
        for category, count in category_stats.items():
            emoji = category_emojis.get(category, '📄')
            status = "✅ Ready" if count >= 100 else "⚠️  Need more"
            print(f"  {emoji} {category}: {count} samples ({status})")
        
        return exported_files

# Example usage and comprehensive testing
def main():
    """Comprehensive testing with all enhancements and toggles"""
    # --- TOGGLES AND SETTINGS ---
    # AI_TOXIC_FILTER REMOVED
    ENABLE_RULE_BASED_FILTER = True # Toggle rule-based filtering
    ENABLE_PRESELECT_FILTER = True  # Toggle original preselect filter (fast, simple filter)

    
    # Multi-Variant PreSelect Toggles
    ENABLE_RULE_BASED_PRESELECT = True    # Enhanced rule-based preselect (Non-AI)
    ENABLE_EMBEDDING_PRESELECT = True     # Zero-shot/embedding-based preselect
    ENABLE_MODEL_PRESELECT = False        # Model-in-the-loop preselect
    
    # CACHE FUNCTIONALITY REMOVED FOR SIMPLICITY
    # TOXICITY_THRESHOLD removed (no more toxic-bert)
    QUALITY_THRESHOLD = 0.4         # Quality threshold (0.0-1.0)
    LLAMA_IMPROVEMENT_THRESHOLD = 0.6  # Threshold for Llama improvement potential (0.0-1.0)
    DOMAIN = 'general'              # Domain: general, educational, children, medical
    BATCH_SIZE = 32                 # Batch size for processing (not currently used)
    NUM_SAMPLES = 50000               # <--- ADJUST THIS to control how many samples to process
    CLEAR_PREVIOUS_FILES = True      # <--- ADJUST THIS to clear previous output files on each run
    HF_TOKEN = "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" # <--- PUT YOUR HUGGING FACE TOKEN HERE
    
    # Export Settings
    EXPORT_MODE = os.environ.get('EXPORT_MODE', 'separate')  # "separate" (4 files), "combined" (1 file), or "none" (no export)
    
    # Individual File Toggles - Control which specific files to save
    SAVE_FUNCTION_CALLING = os.environ.get('SAVE_FUNCTION_CALLING', 'true').lower() == 'true'
    SAVE_REASONING = os.environ.get('SAVE_REASONING', 'true').lower() == 'true'
    SAVE_ROLEPLAY = os.environ.get('SAVE_ROLEPLAY', 'true').lower() == 'true'
    SAVE_CHATRAG = os.environ.get('SAVE_CHATRAG', 'true').lower() == 'true'
    SAVE_COMBINED = os.environ.get('SAVE_COMBINED', 'true').lower() == 'true'
    SAVE_SUMMARY = os.environ.get('SAVE_SUMMARY', 'false').lower() == 'true'  # Summary disabled by default

    print("\n" + "="*60)
    print("🚀 ENHANCED FILTERING TEST WITH TOGGLES")
    print("="*60)
    print(f"AI-based toxic filter:    REMOVED")
    print(f"Rule-based filter:        {'ENABLED' if ENABLE_RULE_BASED_FILTER else 'DISABLED'}")
    print(f"Original preselect:       {'ENABLED' if ENABLE_PRESELECT_FILTER else 'DISABLED'}")

    print(f"\n🎯 MULTI-VARIANT PRESELECT:")
    print(f"  Rule-based preselect:   {'ENABLED' if ENABLE_RULE_BASED_PRESELECT else 'DISABLED'} (Non-AI)")
    print(f"  Embedding preselect:    {'ENABLED' if ENABLE_EMBEDDING_PRESELECT else 'DISABLED'} (Zero-shot)")
    print(f"  Model preselect:        {'ENABLED' if ENABLE_MODEL_PRESELECT else 'DISABLED'} (Model-in-the-loop)")
    # Caching completely removed
    # Toxicity threshold removed
    print(f"Quality threshold:        {QUALITY_THRESHOLD}")
    print(f"Llama improvement thresh: {LLAMA_IMPROVEMENT_THRESHOLD}")
    print(f"Domain:                   {DOMAIN}")
    print(f"Batch size:               {BATCH_SIZE} (not used)")
    print(f"Num samples:              {NUM_SAMPLES}")
    print(f"Clear previous files:     {'ENABLED' if CLEAR_PREVIOUS_FILES else 'DISABLED'}")
    print(f"Export mode:              {EXPORT_MODE} ({'4 separate files' if EXPORT_MODE == 'separate' else '1 combined file' if EXPORT_MODE == 'combined' else 'no files (stats only)'})")
    print(f"File save toggles:")
    print(f"  📞 Function calling:     {'ENABLED' if SAVE_FUNCTION_CALLING else 'DISABLED'}")
    print(f"  🧠 Reasoning:            {'ENABLED' if SAVE_REASONING else 'DISABLED'}")
    print(f"  🎭 Roleplay:             {'ENABLED' if SAVE_ROLEPLAY else 'DISABLED'}")
    print(f"  💬 ChatRAG:              {'ENABLED' if SAVE_CHATRAG else 'DISABLED'}")
    print(f"  📄 Combined file:        {'ENABLED' if SAVE_COMBINED else 'DISABLED'}")
    print(f"  📋 Summary file:         {'ENABLED' if SAVE_SUMMARY else 'DISABLED'}")
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



    # Configure filter system
    config = FilterConfig(
        # toxicity_threshold removed,
        quality_threshold=QUALITY_THRESHOLD,
        batch_size=BATCH_SIZE,
        # cache_enabled removed,
        context_aware=True,
        domain=DOMAIN
    )
    filter_system = EnhancedContentFilter(config)
    # cache_enabled removed
    filter_system.config.domain = DOMAIN
    # filter_system.config.toxicity_threshold removed
    filter_system.config.quality_threshold = QUALITY_THRESHOLD
    filter_system.config.batch_size = BATCH_SIZE

    # Patch filter system to respect toggles
    # orig_ai_content_filter removed
    orig_rule_based_filter = filter_system.rule_based_filter
    orig_comprehensive_filter = filter_system.comprehensive_filter

    # ai_content_filter_toggle removed

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
            # ai_result removed
            quality_result = filter_system.quality_assessment(text)
            overall_quality = np.mean(list(quality_result.values()))
            context_adjustments = filter_system.conversation_analyzer.should_adjust_thresholds()
            adjusted_config = filter_system.adaptive_thresholds.get_adjusted_config(filter_system.config.domain)
            final_quality_threshold = adjusted_config.quality_threshold
            if 'quality_threshold' in context_adjustments:
                final_quality_threshold += context_adjustments['quality_threshold']
            is_safe = rule_result['is_safe']
            # AI toxic filtering removed
            meets_quality = overall_quality >= final_quality_threshold
            preselect_category = categorize_sample(text)
            # inappropriate_categories removed
            return {
                'text': text,
                'is_safe': is_safe,
                'meets_quality': meets_quality,
                'should_include': is_safe and meets_quality,
                'rule_based': rule_result,
                # 'ai_based' removed,
                'quality': quality_result,
                'overall_quality': overall_quality,
                'config_used': filter_system.config.domain,
                'context_adjustments': context_adjustments,
                'performance_metrics': filter_system.performance_monitor.get_performance_report(),
                'preselect_category': preselect_category,
                # 'inappropriate_categories' removed,
                'preselect_filtered': False,
                'preselect_reason': None
            }
        else:
            return orig_comprehensive_filter(text)

    # ai_content_filter toggle removed
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
    # inappropriate_counts removed
    preselect_filtered = 0
    # ai_filtered removed
    rule_filtered = 0
    quality_filtered = 0
    category_filtered = 0  # Rejected for being uncategorized/irrelevant
    
    # Track samples cleared by each filter
    cleared_by_preselect = 0    # Passed basic preselect
    cleared_by_rules = 0        # Passed rule-based filter
    # cleared_by_ai_toxic removed
    cleared_by_quality = 0      # Passed quality assessment
    cleared_by_category = 0     # Passed category relevance check
    cleared_all_filters = 0     # Passed everything
    

    
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
                
                # AI toxic filter removed - directly proceed to quality check
                
                # If passed rule-based, check quality
                if result.get('meets_quality', False):
                    cleared_by_quality += 1
                    
                    # If passed quality, check category relevance
                    if result.get('is_category_relevant', False):
                        cleared_by_category += 1
                        
                        # If passed category relevance, it cleared all filters
                        if result.get('should_include', False):
                            cleared_all_filters += 1
        
        # Count categories
        if result['preselect_category'] in category_counts:
            category_counts[result['preselect_category']] += 1
        
        # inappropriate_categories counting removed
        
        # Count filter types that rejected
        if result.get('preselect_filtered', False):
            preselect_filtered += 1
        elif not result['should_include']:
            if result.get('rule_based') and not result['rule_based'].get('is_safe', True):
                rule_filtered += 1
            elif not result.get('meets_quality', False):
                quality_filtered += 1
            elif not result.get('is_category_relevant', False):
                category_filtered += 1
        
        # Print summary every 100 samples
        if (i + 1) % 100 == 0:
            print(f"\n🔄 --- Summary after {i + 1} samples ---")
            print(f"📈 SAMPLES CLEARED BY EACH FILTER:")
            print(f"  1️⃣ Preselect filter:     {cleared_by_preselect}/{total_processed} ({cleared_by_preselect/total_processed*100:.1f}%) 🚪")
            print(f"  2️⃣ Rule-based filter:    {cleared_by_rules}/{total_processed} ({cleared_by_rules/total_processed*100:.1f}%) 📋")
            # AI toxic filter stats removed
            print(f"  3️⃣ Quality assessment:   {cleared_by_quality}/{total_processed} ({cleared_by_quality/total_processed*100:.1f}%) ⭐")
            print(f"  4️⃣ Category relevance:   {cleared_by_category}/{total_processed} ({cleared_by_category/total_processed*100:.1f}%) 🎯")
            print(f"  ✅ ALL FILTERS PASSED:   {cleared_all_filters}/{total_processed} ({cleared_all_filters/total_processed*100:.1f}%) 🏆")
            
            print(f"\n❌ SAMPLES REJECTED BY:")
            print(f"  🚪 Preselect filter: {preselect_filtered} ({preselect_filtered/total_processed*100:.1f}%)")
            print(f"  📋 Rule-based filter: {rule_filtered} ({rule_filtered/total_processed*100:.1f}%)")
            # AI toxic filter rejection stats removed
            print(f"  ⭐ Quality filter: {quality_filtered} ({quality_filtered/total_processed*100:.1f}%)")
            print(f"  🎯 Category filter: {category_filtered} ({category_filtered/total_processed*100:.1f}%)")
            
            print(f"\n📊 Category distribution:")
            category_emojis = {'function_calling': '📞', 'reasoning': '🧠', 'roleplay': '🎭', 'rag': '💬', 'uncategorized': '❓'}
            for cat, count in category_counts.items():
                if count > 0:
                    emoji = category_emojis.get(cat, '📄')
                    print(f"  {emoji} {cat}: {count} ({count/total_processed*100:.1f}%)")
            
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
            

            
            # inappropriate content display removed

    print(f"\n" + "="*60)
    print(f"🏁 FINAL SUMMARY - {total_processed} samples processed")
    print(f"="*60)
    
    print(f"📈 FINAL FILTER CLEARANCE RATES:")
    print(f"  1️⃣ Preselect filter:     {cleared_by_preselect}/{total_processed} ({cleared_by_preselect/total_processed*100:.1f}%) 🚪")
    print(f"  2️⃣ Rule-based filter:    {cleared_by_rules}/{total_processed} ({cleared_by_rules/total_processed*100:.1f}%) 📋")
    # AI toxic filter final stats removed
    print(f"  3️⃣ Quality assessment:   {cleared_by_quality}/{total_processed} ({cleared_by_quality/total_processed*100:.1f}%) ⭐")
    print(f"  4️⃣ Category relevance:   {cleared_by_category}/{total_processed} ({cleared_by_category/total_processed*100:.1f}%) 🎯")
    print(f"  ✅ ALL FILTERS PASSED:   {cleared_all_filters}/{total_processed} ({cleared_all_filters/total_processed*100:.1f}%) 🏆")
    
    print(f"\n❌ FINAL REJECTION BREAKDOWN:")
    print(f"  🚪 Preselect filter: {preselect_filtered} ({preselect_filtered/total_processed*100:.1f}%)")
    print(f"  📋 Rule-based filter: {rule_filtered} ({rule_filtered/total_processed*100:.1f}%)")
    # AI toxic filter final rejection stats removed
    print(f"  ⭐ Quality filter: {quality_filtered} ({quality_filtered/total_processed*100:.1f}%)")
    print(f"  🎯 Category filter: {category_filtered} ({category_filtered/total_processed*100:.1f}%)")
    
    print(f"\n📊 Final category distribution:")
    category_emojis = {'function_calling': '📞', 'reasoning': '🧠', 'roleplay': '🎭', 'rag': '💬', 'uncategorized': '❓'}
    for cat, count in category_counts.items():
        if count > 0:
            emoji = category_emojis.get(cat, '📄')
            print(f"  {emoji} {cat}: {count} ({count/total_processed*100:.1f}%)")
    # Final inappropriate content breakdown removed



    # --- Output 5 examples that pass all filters and their category scores ---
    print("\n--- 5 Examples Passing All Filters ---")
    passing = [r for r in filtered_results if r['should_include']]
    def category_score(text, category):
        # Improved category scoring with weighted patterns
        patterns = PRESELECT_CATEGORIES[category]
        text_lower = text.lower()
        score = 0
        for pattern in patterns:
            if pattern in text_lower:
                # Weight longer patterns more heavily and count occurrences
                pattern_weight = len(pattern.split())
                occurrences = text_lower.count(pattern)
                score += pattern_weight * occurrences
        return score
    for idx, r in enumerate(passing[:5]):
        print(f"\nExample {idx+1}:")
        print(f"Text: {r['text'][:200]}{'...' if len(r['text']) > 200 else ''}")
        print(f"Preselect category: {r['preselect_category']}")
        # inappropriate_categories output removed
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
            verification_system = VerificationSystem()
            
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
            exporter = DatasetExporter(export_dir="filtered_data_advanced", clear_previous=CLEAR_PREVIOUS_FILES)
            
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
    
    # --- EXPORT CATEGORY-SPECIFIC TRAINING DATASETS ---
    if cleared_all_filters > 0:
        print("\n" + "="*60)
        print("📚 EXPORTING CATEGORY-SPECIFIC TRAINING DATASETS")
        print("="*60)
        
        # Initialize category training dataset exporter
        category_exporter = DatasetExporter(export_dir="category_training_datasets", clear_previous=CLEAR_PREVIOUS_FILES)
        
        # Export category-specific training datasets
        exported_files = category_exporter.export_category_training_datasets(
        filtered_results, 
        export_mode=EXPORT_MODE,
        save_function_calling=SAVE_FUNCTION_CALLING,
        save_reasoning=SAVE_REASONING,
        save_roleplay=SAVE_ROLEPLAY,
        save_chatrag=SAVE_CHATRAG,
        save_combined=SAVE_COMBINED,
        save_summary=SAVE_SUMMARY
    )
        
        print(f"\n🎯 TRAINING DATASETS CREATED FOR YOUR 4 TARGET CATEGORIES:")
        print(f"Your goal: Train a model that excels at these 4 categories")
        print(f"📁 Export directory: category_training_datasets/")
        
        for category, filepath in exported_files.items():
            if category != 'combined':
                print(f"  📄 {category}_training_TIMESTAMP.jsonl")
        
        if exported_files.get('combined'):
            print(f"  📄 all_categories_training_TIMESTAMP.jsonl (combined)")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"1. Use these training files to fine-tune your model")
        print(f"2. Each file contains samples that passed all quality filters")
        print(f"3. Files are in JSONL format ready for training pipelines")
        print(f"4. Compare with your target datasets:")
        print(f"   - Function calling: https://huggingface.co/datasets/data4elm/ELMB-FunctionCalling")
        print(f"   - Reasoning: https://huggingface.co/datasets/data4elm/ELMB-Reasoning")
        print(f"   - Roleplay: https://huggingface.co/datasets/data4elm/ELMB-RolePlay") 
        print(f"   - ChatRAG: https://huggingface.co/datasets/data4elm/ELMB-ChatRAG")
        
        print(f"\n🚀 OPTIMAL WORKFLOW FOR YOUR GOAL:")
        print(f"Step 1: Get more raw data")
        print(f"   python token_detokenizer_jason.py --max_samples 50000")
        print(f"Step 2: Filter into categories") 
        print(f"   python pir_jason.py (this script)")
        print(f"Step 3: Train your model on the category-specific datasets")

if __name__ == "__main__":
    main()