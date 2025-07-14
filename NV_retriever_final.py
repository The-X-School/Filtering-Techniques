import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from contextlib import nullcontext
from typing import List, Optional, Dict, Union, Tuple, Any
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import fasttext
import re
from collections import defaultdict, Counter
import math
from pathlib import Path
import time
from datetime import datetime
from dataclasses import dataclass
import warnings

from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PretrainedConfig
import numpy as np
import os
import json
from contextlib import contextmanager
import sys

@contextmanager
def suppress_stderr():
    """A context manager to temporarily suppress stderr."""
    with open(os.devnull, 'w') as devnull:
        original_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = original_stderr

def load_jsonl(file_path: str) -> List[Dict]:
    """Load a JSONL dataset from the specified file path."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

class NVEmbedConfig(PretrainedConfig):
    model_type = "nvembed"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class LatentAttentionConfig(PretrainedConfig):
    model_type = "latent_attention"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BidirectionalMistralConfig(PretrainedConfig):
    model_type = "bidirectional_mistral"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

def input_transform_func(tokenizer, batch_dict, always_add_eos, max_length, instruction):
    """Transform input texts for model processing."""
    texts = [f"{instruction}{text}" if instruction else text for text in batch_dict["input_texts"]]
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding='longest',
        max_length=max_length,
        return_tensors='pt'
    )
    return tokenized_inputs

class NVEmbedFeatures(dict):
    pass

class LatentAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.config = config

    def forward(self, last_hidden_state, pool_mask):
        if pool_mask is not None:
            masked_hidden_state = last_hidden_state * pool_mask.unsqueeze(-1)
            sum_masked = masked_hidden_state.sum(dim=1)
            num_unmasked = pool_mask.sum(dim=1).unsqueeze(-1)
            embeds = sum_masked / (num_unmasked + 1e-9)
        else:
            embeds = last_hidden_state.mean(dim=1)
        return self.linear(embeds)

class BidirectionalMistralModel(PreTrainedModel):
    config_class = BidirectionalMistralConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.decoder = nn.ModuleDict({
            "block": nn.ModuleList([
                nn.Linear(config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)
            ])
        })
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: bool = True,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")
        elif input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        causal_mask = None
        if attention_mask is not None:
            seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            causal_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        for idx, decoder_layer in enumerate(self.decoder.block):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            current_layer_outputs = [decoder_layer(hidden_states)]

            if use_cache:
                dummy_past_key_value = torch.zeros(1, 1, 1, 1, device=hidden_states.device, dtype=hidden_states.dtype)
                current_layer_outputs.append(dummy_past_key_value)

            if output_attentions:
                dummy_attention = torch.zeros(1, 1, 1, 1, device=hidden_states.device, dtype=hidden_states.dtype)
                current_layer_outputs.append(dummy_attention)
                if all_self_attentions is not None:
                    all_self_attentions += (dummy_attention,)

            layer_outputs = tuple(current_layer_outputs)
            hidden_states = layer_outputs[0]

            current_next_decoder_cache = None
            if use_cache:
                if output_attentions and len(layer_outputs) > 2:
                    current_next_decoder_cache = layer_outputs[2]
                elif not output_attentions and len(layer_outputs) > 1:
                    current_next_decoder_cache = layer_outputs[1]

            if use_cache:
                if next_decoder_cache == ():
                    next_decoder_cache = (current_next_decoder_cache,) if current_next_decoder_cache is not None else None
                elif next_decoder_cache is not None:
                    if current_next_decoder_cache is not None:
                        next_decoder_cache += (current_next_decoder_cache,)
                    else:
                        next_decoder_cache += (None,)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        final_next_cache = None
        if next_decoder_cache is not None:
            use_legacy_cache = getattr(self.config, 'use_legacy_cache', False)
            if use_legacy_cache:
                final_next_cache = next_decoder_cache.to_legacy_cache() if hasattr(next_decoder_cache, 'to_legacy_cache') else next_decoder_cache
            else:
                final_next_cache = next_decoder_cache

        if not return_dict:
            return (hidden_states, final_next_cache, all_hidden_states, all_self_attentions)

        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=final_next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class NVEmbedModel(PreTrainedModel):
    config_class = NVEmbedConfig
    base_model_prefix = "embedding_model"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedding_model = BidirectionalMistralModel(config.embedding_model_config)
        self.latent_attention_model = LatentAttentionModel(config.latent_attention_config)
        self.tokenizer = None
        self.padding_side = "right"
        self.is_mask_instruction = True
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def prepare_kwargs_from_batch(self, batch_dict, instruction_lens, device):
        """Prepares keyword arguments for the model's forward pass."""
        features = NVEmbedFeatures()
        features["input_ids"] = batch_dict["input_ids"].to(device)
        features["attention_mask"] = batch_dict["attention_mask"].to(device)
        features["pool_mask"] = None
        return features

    @torch.no_grad()
    def encode(self, prompts: List[str], instruction: str="", max_length: int=4096, **kwargs):
        if self.padding_side == "right" and self.is_mask_instruction == True and len(instruction) > 0:
            instruction_lens = len(self.tokenizer.tokenize(instruction))
        else:
            instruction_lens = 0
        
        device = next(self.embedding_model.parameters()).device
        batch_dict = input_transform_func(self.tokenizer,
                                          {"input_texts": [prompt for prompt in prompts]},
                                          always_add_eos=True,
                                          max_length=max_length,
                                          instruction=instruction)

        features: NVEmbedFeatures = self.prepare_kwargs_from_batch(batch_dict, instruction_lens, device=device)
        return self(**features)["sentence_embeddings"].squeeze(1)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                pool_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: bool = True):
        if inputs_embeds is not None:
            raise NotImplementedError("NVEmbedModel does not support 'inputs_embeds'. Please use 'input_ids' instead.")
        
        autocast_ctx = torch.autocast if torch.cuda.is_available() else nullcontext
        with autocast_ctx("cuda"):
            outputs = self.embedding_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            embeds = self.latent_attention_model(
                outputs.last_hidden_state,
                pool_mask,
            )
        if not return_dict:
            return (embeds,)
        return {"sentence_embeddings": embeds}

@dataclass
class FilteringConfig:
    """Configuration for the filtering pipeline"""
    min_text_length: int = 50
    max_text_length: int = 8192
    min_quality_score: float = 0.3
    toxicity_threshold: float = 0.7
    min_informativeness_score: float = 0.4
    diversity_threshold: float = 0.6
    n_clusters: int = 50
    outlier_threshold: float = 0.8
    quality_weight: float = 0.3
    informativeness_weight: float = 0.3
    diversity_weight: float = 0.2
    safety_weight: float = 0.2
    batch_size: int = 32
    use_cache: bool = True
    enable_preselect: bool = True
    enable_toxicity_filter: bool = True
    enable_length_filter: bool = True
    enable_informativeness_filter: bool = True

class ComprehensiveDataFilter:
    """Multi-stage data filtering system for quality, safety, and informativeness."""
    
    def __init__(self, config: FilteringConfig = None):
        self.config = config or FilteringConfig()
        self.preselect_model = None
        self.toxicity_patterns = self._compile_toxicity_patterns()
        self.quality_metrics = QualityMetrics()
        self.informativeness_scorer = InformativenessScorer()
        self.clustering_filter = ClusteringFilter(
            n_clusters=self.config.n_clusters,
            outlier_threshold=self.config.outlier_threshold
        )
        
        self.stats = {
            'total_processed': 0,
            'passed_preselect': 0,
            'passed_safety': 0,
            'passed_quality': 0,
            'passed_informativeness': 0,
            'passed_clustering': 0,
            'final_filtered': 0,
            'stage_timings': defaultdict(list)
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize filtering models"""
        if self.config.enable_preselect:
            self._load_preselect_model()
    
    def _load_preselect_model(self):
        """Load FastText preselect model"""
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="hkust-nlp/preselect-fasttext-classifier",
            filename="PreSelect-classifier.bin"
        )
        with suppress_stderr():
            self.preselect_model = fasttext.load_model(model_path)
    
    def _compile_toxicity_patterns(self):
        """Compile regex patterns for toxicity detection"""
        patterns = {
            'profanity': [
                r'\b(fuck|shit|bitch|asshole|dick|pussy|cunt)\b',
                r'\b(damn|hell|goddamn)\b'
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
        
        compiled_patterns = {}
        for category, pattern_list in patterns.items():
            compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in pattern_list
            ]
        
        return compiled_patterns
    
    def preselect_filter(self, text: str) -> Dict[str, Any]:
        """Stage 1: Preselect filtering using FastText"""
        if not self.config.enable_preselect or not self.preselect_model:
            return {'passed': True, 'score': 1.0, 'reason': 'preselect_disabled'}
        
        cleaned_text = text.replace('\n', ' ').replace('\t', ' ')
        predictions = self.preselect_model.predict(cleaned_text, k=1)
        label = predictions[0][0]
        confidence = predictions[1][0]
        
        passed = label == "__label__1" and confidence >= 0.99
        
        return {
            'passed': passed,
            'score': confidence if passed else 1.0 - confidence,
            'label': label,
            'confidence': confidence,
            'reason': 'preselect_pass' if passed else 'preselect_fail'
        }
    
    def safety_filter(self, text: str) -> Dict[str, Any]:
        """Stage 2: Safety filtering (toxicity detection)"""
        if not self.config.enable_toxicity_filter:
            return {'passed': True, 'score': 1.0, 'reason': 'safety_disabled'}
        
        toxicity_score = 0.0
        detected_categories = []
        text_lower = text.lower()
        
        for category, patterns in self.toxicity_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    toxicity_score += 0.3
                    detected_categories.append(category)
        
        words = text.split()
        if len(words) > 10:
            word_counts = Counter(words)
            max_repetition = max(word_counts.values()) if word_counts else 0
            if max_repetition > len(words) * 0.3:
                toxicity_score += 0.2
                detected_categories.append('excessive_repetition')
        
        if len(text) > 20 and text.isupper():
            toxicity_score += 0.1
            detected_categories.append('excessive_caps')
        
        toxicity_score = min(toxicity_score, 1.0)
        safety_score = 1.0 - toxicity_score
        passed = toxicity_score < self.config.toxicity_threshold
        
        return {
            'passed': passed,
            'score': safety_score,
            'toxicity_score': toxicity_score,
            'detected_categories': detected_categories,
            'reason': 'safety_pass' if passed else 'safety_fail'
        }
    
    def quality_assessment(self, text: str) -> Dict[str, Any]:
        """Stage 3: Quality assessment"""
        if self.config.enable_length_filter:
            if len(text) < self.config.min_text_length:
                return {'passed': False, 'score': 0.0, 'reason': 'too_short'}
            if len(text) > self.config.max_text_length:
                return {'passed': False, 'score': 0.0, 'reason': 'too_long'}
        
        quality_scores = self.quality_metrics.assess_quality(text)
        overall_quality = sum(quality_scores.values()) / len(quality_scores)
        passed = overall_quality >= self.config.min_quality_score
        
        return {
            'passed': passed,
            'score': overall_quality,
            'breakdown': quality_scores,
            'reason': 'quality_pass' if passed else 'quality_fail'
        }
    
    def informativeness_assessment(self, text: str, corpus_stats: Dict = None) -> Dict[str, Any]:
        """Stage 4: Informativeness assessment"""
        if not self.config.enable_informativeness_filter:
            return {'passed': True, 'score': 1.0, 'reason': 'informativeness_disabled'}
        
        informativeness_scores = self.informativeness_scorer.score_text(text, corpus_stats)
        overall_informativeness = sum(informativeness_scores.values()) / len(informativeness_scores)
        passed = overall_informativeness >= self.config.min_informativeness_score
        
        return {
            'passed': passed,
            'score': overall_informativeness,
            'breakdown': informativeness_scores,
            'reason': 'informativeness_pass' if passed else 'informativeness_fail'
        }
    
    def filter_sample(self, text: str, sample_id: str = None, corpus_stats: Dict = None) -> Dict[str, Any]:
        """Filter a single sample through all stages"""
        result = {
            'sample_id': sample_id,
            'text': text,
            'passed_overall': False,
            'final_score': 0.0,
            'stage_results': {},
            'filtering_reason': None
        }
        
        # Stage 1: Preselect
        preselect_result = self.preselect_filter(text)
        result['stage_results']['preselect'] = preselect_result
        
        if not preselect_result['passed']:
            result['filtering_reason'] = preselect_result['reason']
            return result
        
        self.stats['passed_preselect'] += 1
        
        # Stage 2: Safety
        safety_result = self.safety_filter(text)
        result['stage_results']['safety'] = safety_result
        
        if not safety_result['passed']:
            result['filtering_reason'] = safety_result['reason']
            return result
        
        self.stats['passed_safety'] += 1
        
        # Stage 3: Quality
        quality_result = self.quality_assessment(text)
        result['stage_results']['quality'] = quality_result
        
        if not quality_result['passed']:
            result['filtering_reason'] = quality_result['reason']
            return result
        
        self.stats['passed_quality'] += 1
        
        # Stage 4: Informativeness
        informativeness_result = self.informativeness_assessment(text, corpus_stats)
        result['stage_results']['informativeness'] = informativeness_result
        
        if not informativeness_result['passed']:
            result['filtering_reason'] = informativeness_result['reason']
            return result
        
        self.stats['passed_informativeness'] += 1
        
        # Calculate final score
        final_score = (
            self.config.quality_weight * quality_result['score'] +
            self.config.informativeness_weight * informativeness_result['score'] +
            self.config.safety_weight * safety_result['score'] +
            0.1 * preselect_result['score']
        )
        
        result['passed_overall'] = True
        result['final_score'] = final_score
        result['filtering_reason'] = 'passed_all_stages'
        self.stats['final_filtered'] += 1
        
        return result
    
    def filter_dataset(self, dataset: List[Dict], max_samples: int = None) -> List[Dict]:
        """Filter entire dataset and return filtered samples"""
        filtered_samples = []
        
        sample_count = 0
        for i, sample in enumerate(dataset):
            if max_samples and sample_count >= max_samples:
                break
            
            text = sample.get('text', '')
            if not text:
                continue
            
            filter_result = self.filter_sample(text, sample_id=str(i))
            
            if filter_result['passed_overall']:
                filtered_samples.append({
                    'original_sample': sample,
                    'filter_result': filter_result
                })
            else:
                print(f"Sample {i} filtered out. Reason: {filter_result.get('filtering_reason', 'Unknown')}")
            
            sample_count += 1
            self.stats['total_processed'] += 1
        
        # Apply clustering-based filtering
        if filtered_samples:
            clustering_results = self.clustering_filter.filter_samples(filtered_samples)
            
            final_filtered = []
            for sample, cluster_result in zip(filtered_samples, clustering_results):
                if cluster_result['keep']:
                    sample['cluster_info'] = cluster_result
                    final_filtered.append(sample)
            
            self.stats['passed_clustering'] = len(final_filtered)
            filtered_samples = final_filtered
        
        return filtered_samples

class QualityMetrics:
    """Text quality assessment metrics"""
    
    def assess_quality(self, text: str) -> Dict[str, float]:
        """Assess various quality aspects of text"""
        return {
            'readability': self._readability_score(text),
            'coherence': self._coherence_score(text),
            'vocabulary_diversity': self._vocabulary_diversity(text),
            'grammatical_correctness': self._grammatical_score(text),
            'information_density': self._information_density(text)
        }
    
    def _readability_score(self, text: str) -> float:
        """Simple readability score based on sentence and word length"""
        sentences = text.split('.')
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        sentence_score = max(0, 1 - abs(avg_sentence_length - 17.5) / 17.5)
        word_score = max(0, 1 - abs(avg_word_length - 5) / 5)
        
        return (sentence_score + word_score) / 2
    
    def _coherence_score(self, text: str) -> float:
        """Assess text coherence based on word repetition and flow"""
        words = text.lower().split()
        if len(words) < 10:
            return 0.0
        
        word_counts = Counter(words)
        total_words = len(words)
        unique_words = len(word_counts)
        
        diversity_ratio = unique_words / total_words
        coherence_score = 1 - abs(diversity_ratio - 0.75) / 0.75
        
        return max(0, coherence_score)
    
    def _vocabulary_diversity(self, text: str) -> float:
        """Measure vocabulary diversity (TTR - Type Token Ratio)"""
        words = text.lower().split()
        if len(words) < 5:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        ttr = unique_words / math.sqrt(total_words)
        return min(1.0, ttr / 10)
    
    def _grammatical_score(self, text: str) -> float:
        """Simple grammatical correctness heuristics"""
        score = 1.0
        
        if not re.search(r'[.!?]', text):
            score -= 0.3
        
        sentences = re.split(r'[.!?]', text)
        capitalized_sentences = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
        if sentences and capitalized_sentences / len(sentences) < 0.5:
            score -= 0.2
        
        if re.search(r'[.!?]{3,}', text):
            score -= 0.2
        
        return max(0, score)
    
    def _information_density(self, text: str) -> float:
        """Measure information density (content words vs function words)"""
        words = text.lower().split()
        if not words:
            return 0.0
        
        function_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
        }
        
        content_words = sum(1 for word in words if word not in function_words)
        density = content_words / len(words)
        
        return min(1.0, density * 2)

class InformativenessScorer:
    """Assess text informativeness and novelty"""
    
    def score_text(self, text: str, corpus_stats: Dict = None) -> Dict[str, float]:
        """Score text informativeness"""
        return {
            'novelty': self._novelty_score(text, corpus_stats),
            'specificity': self._specificity_score(text),
            'educational_value': self._educational_value(text),
            'technical_depth': self._technical_depth(text),
            'conceptual_richness': self._conceptual_richness(text)
        }
    
    def _novelty_score(self, text: str, corpus_stats: Dict = None) -> float:
        """Assess novelty compared to corpus (if available)"""
        if not corpus_stats:
            return 0.5
        return 0.7
    
    def _specificity_score(self, text: str) -> float:
        """Measure specificity vs generality"""
        words = text.lower().split()
        if not words:
            return 0.0
        
        specific_indicators = [
            r'\b\d+\b',
            r'\b[A-Z][a-z]*[A-Z]\w*\b',
            r'\b\w+\.\w+\b',
            r'\b\w+://\w+\b',
            r'\b\w+@\w+\.\w+\b',
        ]
        
        specific_count = 0
        for pattern in specific_indicators:
            specific_count += len(re.findall(pattern, text))
        
        return min(1.0, specific_count / len(words) * 10)
    
    def _educational_value(self, text: str) -> float:
        """Assess educational/instructional value"""
        educational_patterns = [
            r'\b(learn|teach|explain|understand|demonstrate|example|tutorial)\b',
            r'\b(how to|step by step|first|second|third|finally)\b',
            r'\b(definition|concept|principle|theory|method)\b',
            r'\b(because|therefore|thus|hence|consequently)\b',
        ]
        
        educational_score = 0
        for pattern in educational_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            educational_score += matches
        
        return min(1.0, educational_score / len(text.split()) * 5)
    
    def _technical_depth(self, text: str) -> float:
        """Measure technical content depth"""
        technical_indicators = [
            r'\b(algorithm|function|class|method|variable|parameter)\b',
            r'\b(implementation|optimization|architecture|framework)\b',
            r'\b(data|analysis|model|system|process|structure)\b',
            r'\b(performance|efficiency|scalability|reliability)\b',
        ]
        
        technical_score = 0
        for pattern in technical_indicators:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            technical_score += matches
        
        return min(1.0, technical_score / len(text.split()) * 3)
    
    def _conceptual_richness(self, text: str) -> float:
        """Measure conceptual richness and complexity"""
        sentences = text.split('.')
        if not sentences:
            return 0.0
        
        complex_patterns = [
            r'\b(relationship|connection|correlation|causation)\b',
            r'\b(abstract|concrete|theoretical|practical)\b',
            r'\b(analysis|synthesis|evaluation|comparison)\b',
            r'\b(implication|consequence|significance|importance)\b',
        ]
        
        conceptual_score = 0
        for pattern in complex_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            conceptual_score += matches
        
        return min(1.0, conceptual_score / len(sentences) * 2)

class ClusteringFilter:
    """Clustering-based outlier detection and diversity filtering"""
    
    def __init__(self, n_clusters: int = 50, outlier_threshold: float = 0.8):
        self.n_clusters = n_clusters
        self.outlier_threshold = outlier_threshold
        self.scaler = StandardScaler()
        self.kmeans = None
        
    def filter_samples(self, samples: List[Dict]) -> List[Dict]:
        """Filter samples using clustering-based outlier detection"""
        if len(samples) < self.n_clusters:
            return [{'keep': True, 'reason': 'insufficient_samples', 'cluster_id': -1, 'distance': 0.0}] * len(samples)
        
        features = []
        for sample in samples:
            filter_result = sample['filter_result']
            feature_vector = self._extract_features(filter_result)
            features.append(feature_vector)
        
        features_normalized = self.scaler.fit_transform(features)
        
        n_clusters = min(self.n_clusters, len(samples) // 2)
        if n_clusters <= 0:
            return [{'keep': True, 'reason': 'insufficient_samples_for_clustering', 'cluster_id': -1, 'distance': 0.0}] * len(samples)
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(features_normalized)
        
        distances = []
        for i, (features_norm, label) in enumerate(zip(features_normalized, cluster_labels)):
            center = self.kmeans.cluster_centers_[label]
            distance = np.linalg.norm(features_norm - center)
            distances.append(distance)
        
        distance_threshold = np.percentile(distances, self.outlier_threshold * 100)
        
        results = []
        for i, (distance, cluster_id) in enumerate(zip(distances, cluster_labels)):
            keep = distance <= distance_threshold
            results.append({
                'keep': keep,
                'reason': 'cluster_inlier' if keep else 'cluster_outlier',
                'cluster_id': int(cluster_id),
                'distance': float(distance),
                'distance_threshold': float(distance_threshold)
            })
        
        return results
    
    def _extract_features(self, filter_result: Dict) -> List[float]:
        """Extract numerical features from filter results"""
        features = []
        stage_results = filter_result.get('stage_results', {})
        
        features.append(stage_results.get('preselect', {}).get('score', 0.0))
        features.append(stage_results.get('safety', {}).get('score', 0.0))
        features.append(stage_results.get('quality', {}).get('score', 0.0))
        features.append(stage_results.get('informativeness', {}).get('score', 0.0))
        
        quality_breakdown = stage_results.get('quality', {}).get('breakdown', {})
        features.extend([
            quality_breakdown.get('readability', 0.0),
            quality_breakdown.get('coherence', 0.0),
            quality_breakdown.get('vocabulary_diversity', 0.0),
            quality_breakdown.get('grammatical_correctness', 0.0),
            quality_breakdown.get('information_density', 0.0)
        ])
        
        info_breakdown = stage_results.get('informativeness', {}).get('breakdown', {})
        features.extend([
            info_breakdown.get('novelty', 0.0),
            info_breakdown.get('specificity', 0.0),
            info_breakdown.get('educational_value', 0.0),
            info_breakdown.get('technical_depth', 0.0),
            info_breakdown.get('conceptual_richness', 0.0)
        ])
        
        text_length = len(filter_result.get('text', ''))
        features.extend([
            min(1.0, text_length / 1000),
            filter_result.get('final_score', 0.0)
        ])
        
        return features

def get_embeddings(model, inputs: Dict[str, torch.Tensor], batch_size: int = 32):
    """Generate embeddings for filtered text samples."""
    all_embeddings = []
    num_samples = inputs['input_ids'].shape[0]
    device = next(model.parameters()).device

    for i in range(0, num_samples, batch_size):
        batch_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                batch_inputs[key] = value[i:i + batch_size].to(device)
            else:
                batch_inputs[key] = value

        filtered_batch_inputs = {
            k: v for k, v in batch_inputs.items()
            if k in ['input_ids', 'attention_mask', 'pool_mask']
        }
        
        outputs = model(**filtered_batch_inputs)

        if isinstance(outputs, dict) and 'sentence_embeddings' in outputs:
            batch_emb = outputs['sentence_embeddings']
        else:
            batch_emb = outputs[0]

        all_embeddings.append(batch_emb.cpu())
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if all_embeddings:
        return torch.cat(all_embeddings, dim=0)
    else:
        return torch.empty(0, 768)

if __name__ == "__main__":
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class DummyConfig(PretrainedConfig):
        def __init__(self, vocab_size, **kwargs):
            super().__init__(**kwargs)
            self.hidden_size = 768
            self.num_hidden_layers = 2
            self.vocab_size = vocab_size
            self.initializer_range = 0.02
            self.embedding_model_config = BidirectionalMistralConfig(
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                vocab_size=self.vocab_size,
                initializer_range=self.initializer_range
            )
            self.latent_attention_config = LatentAttentionConfig(
                hidden_size=self.hidden_size
            )
            self.use_legacy_cache = False

    config = DummyConfig(vocab_size=len(tokenizer))
    model = NVEmbedModel(config)
    model.tokenizer = tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Configuration
    max_length = 512
    output_path = "./artifacts/embeddings_output.json"
    dataset_path = "data/climblab_processed_clusters.jsonl"
    text_field = "detokenized_text"
    
    # Load dataset
    if dataset_path and os.path.exists(dataset_path):
        sample = load_jsonl(dataset_path)
    else:
        print(f"Dataset not found at {dataset_path}. Using a default sample dataset.")
        sample = [
            {"text": "This is a high-quality sentence that should pass the filters. It is long enough and has good vocabulary."},
            {"text": "short"},
            {"text": "This is another good sentence. It talks about machine learning and data science, which is informative."},
            {"text": "I hate you, you are stupid and I will kill you."},
            {"text": "This sentence is just a bunch of repeated words repeated words repeated words repeated words repeated words repeated words."},
        ]
    
    # Initialize filtering system
    filtering_config = FilteringConfig(
        min_text_length=10,
        max_text_length=4096,
        min_quality_score=0.1,
        toxicity_threshold=0.7,
        min_informativeness_score=0.1,
        n_clusters=min(10, len(sample) // 3),
        outlier_threshold=0.8
    )
    
    data_filter = ComprehensiveDataFilter(config=filtering_config)
    
    # Apply filtering
    filtered_samples = data_filter.filter_dataset(sample, max_samples=1000)
    
    # Extract text from filtered samples
    docs = []
    for sample_data in filtered_samples:
        if "tokens" in sample_data['original_sample']:
            docs.append(tokenizer.decode(sample_data['original_sample']["tokens"], skip_special_tokens=True))
        else:
            text_content = sample_data['original_sample'].get(text_field, sample_data['original_sample'].get("text", ""))
            docs.append(text_content)
    
    # Generate embeddings for filtered data
    if docs:
        model.eval()
        all_doc_inputs = tokenizer(
            docs,
            truncation=True,
            padding='longest',
            max_length=max_length,
            return_tensors='pt'
        )
        
        all_doc_inputs = {k: v.to(device) for k, v in all_doc_inputs.items()}
        
        with torch.no_grad():
            embeddings_tensor = get_embeddings(model, all_doc_inputs, batch_size=4)
            embeddings_np = embeddings_tensor.cpu().numpy()
        
        # Save results
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            embeddings_file = output_path.replace('.json', '_embeddings.npy')
            np.save(embeddings_file, embeddings_np)
            
            with open(output_path, 'w') as f:
                data = {
                    'texts': docs,
                    'embedding_shape': embeddings_np.shape,
                    'embedding_file': embeddings_file,
                    'filtered_samples': len(filtered_samples),
                    'total_processed': data_filter.stats['total_processed'],
                    'retention_rate': len(filtered_samples) / data_filter.stats['total_processed'] if data_filter.stats['total_processed'] > 0 else 0,
                    'timestamp': datetime.now().isoformat()
                }
                json.dump(data, f, indent=2)
    else:
        print("All documents were filtered out. No embeddings will be generated and no output file will be created.")