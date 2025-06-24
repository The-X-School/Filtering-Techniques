import re
import json
import os
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MobileFunctionCallFilter:
    """
    Mobile Function Calling Filter specifically designed for the Data Filtering Challenge.
    Extracts content that teaches edge LMs to perform function calling on mobile devices.
    """
    
    def __init__(self):
        """Initialize the filter with mobile function calling patterns"""
        self.command_patterns = self._compile_command_patterns()      # Direct commands
        self.action_patterns = self._compile_action_patterns()        # Action-oriented language
        self.device_patterns = self._compile_device_patterns()        # Mobile/IoT device interactions
        self.conversation_patterns = self._compile_conversation_patterns()  # Conversational function calls
        
        logger.info("🚀 Mobile Function Calling Filter initialized for Data Filtering Challenge")
        logger.info(f"📊 Pattern counts: Commands={len(self.command_patterns)}, Actions={len(self.action_patterns)}, Devices={len(self.device_patterns)}, Conversations={len(self.conversation_patterns)}")
    
    def _compile_command_patterns(self) -> List[Tuple[re.Pattern, str, int]]:
        """
        TIER 1: DIRECT COMMANDS (Weight: 30-35x)
        These are explicit function calling commands for mobile devices
        """
        patterns = [
            # Voice assistant commands
            (re.compile(r'\b(?:turn on|turn off|switch on|switch off|enable|disable)\s+(?:the\s+)?(?:lights?|fan|ac|air conditioning|heating|music|tv|television|radio)', re.IGNORECASE), 
             "Device control commands", 35),
            
            # Time and scheduling
            (re.compile(r'\b(?:set|create|schedule)\s+(?:an?\s+)?(?:alarm|reminder|timer|appointment|meeting)\s+(?:for|at|in)\s+', re.IGNORECASE), 
             "Time scheduling commands", 34),
            
            # Communication actions
            (re.compile(r'\b(?:call|text|message|email|send)\s+(?:to\s+)?(?:\w+|mom|dad|john|sarah|my\s+\w+)', re.IGNORECASE), 
             "Communication commands", 33),
            
            # Media and entertainment
            (re.compile(r'\b(?:play|pause|stop|skip|shuffle|repeat)\s+(?:music|song|video|podcast|playlist)', re.IGNORECASE), 
             "Media control commands", 32),
            
            # Navigation and location
            (re.compile(r'\b(?:navigate to|directions to|find|locate|search for)\s+(?:nearest|nearby)?\s*\w+', re.IGNORECASE), 
             "Navigation commands", 31),
            
            # Smart home automation
            (re.compile(r'\b(?:lock|unlock|open|close)\s+(?:the\s+)?(?:door|window|garage|gate)', re.IGNORECASE), 
             "Home automation commands", 30),
        ]
        return patterns
    
    def _compile_action_patterns(self) -> List[Tuple[re.Pattern, str, int]]:
        """
        TIER 2: ACTION-ORIENTED LANGUAGE (Weight: 20-25x)
        Patterns that indicate function calling context
        """
        patterns = [
            # Imperative verbs for mobile actions
            (re.compile(r'\b(?:calculate|compute|convert|translate|check|verify|confirm|book|order|buy|purchase)\s+', re.IGNORECASE), 
             "Action verbs for mobile functions", 25),
            
            # Weather and information requests
            (re.compile(r'\b(?:what\'s|check|get)\s+(?:the\s+)?(?:weather|temperature|forecast|news|traffic|stock price)', re.IGNORECASE), 
             "Information request patterns", 24),
            
            # Task management
            (re.compile(r'\b(?:add to|remove from|update|complete|mark as done)\s+(?:my\s+)?(?:calendar|todo|list|notes)', re.IGNORECASE), 
             "Task management actions", 23),
            
            # Shopping and e-commerce
            (re.compile(r'\b(?:order|buy|purchase|add to cart|checkout|pay for)\s+.{1,50}(?:from|at|on)\s+\w+', re.IGNORECASE), 
             "E-commerce function calls", 22),
            
            # App-specific actions
            (re.compile(r'\b(?:open|launch|start|close|exit)\s+(?:the\s+)?(?:app|application|\w+\s+app)', re.IGNORECASE), 
             "App control commands", 21),
            
            # Settings and configuration
            (re.compile(r'\b(?:change|adjust|set|modify)\s+(?:the\s+)?(?:volume|brightness|settings|preferences)', re.IGNORECASE), 
             "Device settings commands", 20),
        ]
        return patterns
    
    def _compile_device_patterns(self) -> List[Tuple[re.Pattern, str, int]]:
        """
        TIER 3: DEVICE INTERACTION PATTERNS (Weight: 15-18x)
        Mobile and IoT device specific patterns
        """
        patterns = [
            # Mobile device features
            (re.compile(r'\b(?:camera|photo|picture|video|flashlight|torch|bluetooth|wifi|cellular|gps)', re.IGNORECASE), 
             "Mobile device features", 18),
            
            # Smart home devices
            (re.compile(r'\b(?:smart\s+)?(?:thermostat|doorbell|security\s+camera|smoke\s+detector|alexa|google\s+home|siri)', re.IGNORECASE), 
             "Smart home devices", 17),
            
            # Notification and alert patterns
            (re.compile(r'\b(?:notification|alert|ping|buzz|vibrate|silent\s+mode|do\s+not\s+disturb)', re.IGNORECASE), 
             "Notification management", 16),
            
            # Location-based services
            (re.compile(r'\b(?:gps|location|nearby|within\s+\d+\s+(?:miles|km|meters))', re.IGNORECASE), 
             "Location services", 15),
        ]
        return patterns
    
    def _compile_conversation_patterns(self) -> List[Tuple[re.Pattern, str, int]]:
        """
        TIER 4: CONVERSATIONAL FUNCTION CALLING (Weight: 10-12x)
        Natural language patterns that indicate function calling
        """
        patterns = [
            # Question patterns that trigger functions
            (re.compile(r'\b(?:can you|could you|please|would you)\s+(?:help me|assist me)?\s*(?:with)?\s*(?:turn|set|call|send|play|find)', re.IGNORECASE), 
             "Polite command patterns", 12),
            
            # Time-based triggers
            (re.compile(r'\b(?:at|when|in|after|before)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|o\'clock)?', re.IGNORECASE), 
             "Time-based triggers", 11),
            
            # Conditional commands
            (re.compile(r'\b(?:if|when|whenever|as soon as)\s+.{5,30}(?:then|please|can you)', re.IGNORECASE), 
             "Conditional function calls", 10),
        ]
        return patterns
    
    def calculate_function_calling_score(self, text: str) -> Dict[str, Any]:
        """
        Calculate function calling score optimized for mobile device commands
        """
        if not text:
            return {"total_score": 0.0, "breakdown": {}, "quality_indicators": {}}
        
        text_length = len(text)
        breakdown = {"commands": 0, "actions": 0, "devices": 0, "conversations": 0, "keywords": 0}
        pattern_matches = {}
        
        # Apply Command patterns (highest value)
        for pattern, description, weight in self.command_patterns:
            matches = pattern.findall(text)
            if matches:
                score = len(matches) * weight
                breakdown["commands"] += score
                pattern_matches[description] = len(matches)
        
        # Apply Action patterns
        for pattern, description, weight in self.action_patterns:
            matches = pattern.findall(text)
            if matches:
                score = len(matches) * weight
                breakdown["actions"] += score
                pattern_matches[description] = len(matches)
        
        # Apply Device patterns
        for pattern, description, weight in self.device_patterns:
            matches = pattern.findall(text)
            if matches:
                score = len(matches) * weight
                breakdown["devices"] += score
                pattern_matches[description] = len(matches)
        
        # Apply Conversation patterns
        for pattern, description, weight in self.conversation_patterns:
            matches = pattern.findall(text)
            if matches:
                score = len(matches) * weight
                breakdown["conversations"] += score
                pattern_matches[description] = len(matches)
        
        # Function calling keywords bonus
        function_keywords = {
            'command', 'execute', 'perform', 'trigger', 'activate', 'control', 'operate',
            'assistant', 'siri', 'alexa', 'google', 'voice', 'speak', 'listen', 'respond',
            'app', 'application', 'mobile', 'phone', 'device', 'smart', 'iot', 'automation',
            'schedule', 'remind', 'notify', 'alert', 'calendar', 'timer', 'alarm',
            'call', 'text', 'message', 'email', 'contact', 'dial', 'send', 'receive'
        }
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in function_keywords if keyword in text_lower)
        breakdown["keywords"] = keyword_count * 3
        
        # Calculate total score with normalization
        raw_score = sum(breakdown.values())
        normalized_score = raw_score / (text_length / 1000) if text_length > 0 else 0
        final_score = min(normalized_score, 200.0)  # Cap at 200 for mobile function calling
        
        # Quality indicators for mobile function calling
        quality_indicators = {
            "has_direct_commands": any("commands" in desc.lower() for desc in pattern_matches.keys()),
            "has_device_interactions": any("device" in desc.lower() or "mobile" in desc.lower() for desc in pattern_matches.keys()),
            "has_time_scheduling": any("time" in desc.lower() or "schedule" in desc.lower() for desc in pattern_matches.keys()),
            "has_conversation_patterns": any("conversation" in desc.lower() or "polite" in desc.lower() for desc in pattern_matches.keys()),
            "action_diversity": len(pattern_matches),
            "is_mobile_optimized": final_score > 30,
            "command_density": breakdown["commands"] / text_length if text_length > 0 else 0
        }
        
        return {
            "total_score": final_score,
            "raw_score": raw_score,
            "breakdown": breakdown,
            "pattern_matches": pattern_matches,
            "quality_indicators": quality_indicators,
            "text_length": text_length
        }
    
    def extract_function_calling_segments(self, text: str, min_score: float = 10.0) -> List[Dict[str, Any]]:
        """
        Extract segments that contain function calling patterns
        """
        segments = []
        
        # Extract sentences with commands
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        for sentence in sentences:
            if len(sentence) > 20:  # Minimum meaningful length
                score_data = self.calculate_function_calling_score(sentence)
                if score_data["total_score"] >= min_score:
                    segments.append({
                        "content": sentence,
                        "type": "command_sentence",
                        "score": score_data["total_score"],
                        "patterns": score_data["pattern_matches"]
                    })
        
        # Extract dialogue exchanges
        dialogues = [p.strip() for p in text.split('\n\n') if p.strip()]
        for dialogue in dialogues:
            if len(dialogue) > 50:
                score_data = self.calculate_function_calling_score(dialogue)
                if score_data["total_score"] >= min_score * 0.7:
                    segments.append({
                        "content": dialogue,
                        "type": "dialogue_exchange",
                        "score": score_data["total_score"],
                        "patterns": score_data["pattern_matches"]
                    })
        
        # Sort by score and return top segments
        segments.sort(key=lambda x: x["score"], reverse=True)
        return segments[:30]  # Top 30 function calling segments
    
    def process_record(self, record_data: Dict[str, Any], threshold: float = 15.0) -> Optional[Dict[str, Any]]:
        """
        Process a single record for mobile function calling content
        """
        text = record_data.get('text', '')
        if not text or len(text) < 30:
            return None
        
        # Calculate function calling score
        score_data = self.calculate_function_calling_score(text)
        
        if score_data["total_score"] >= threshold:
            # Extract function calling segments
            segments = self.extract_function_calling_segments(text)
            
            if segments:
                # Combine top segments for mobile function calling training
                top_segments = segments[:15]  # Top 15 segments
                combined_content = '\n\n'.join([seg["content"] for seg in top_segments])
                
                return {
                    "text": combined_content,
                    "function_calling_score": score_data["total_score"],
                    "score_breakdown": score_data["breakdown"],
                    "pattern_matches": score_data["pattern_matches"],
                    "quality_indicators": score_data["quality_indicators"],
                    "segments_extracted": len(top_segments),
                    "command_types": {seg["type"]: 1 for seg in top_segments},
                    "original_length": len(text),
                    "filtered_length": len(combined_content),
                    "mobile_optimization_score": score_data["quality_indicators"]["command_density"] * 1000,
                    "top_command_scores": [seg["score"] for seg in top_segments[:5]]
                }
        
        return None
    
    def filter_dataset(self, input_path: str, output_path: str, 
                      threshold: float = 15.0, max_records: Optional[int] = None) -> Dict[str, Any]:
        """
        Filter dataset for mobile function calling content
        """
        logger.info(f"🎯 Starting Mobile Function Calling Filter for Data Filtering Challenge")
        logger.info(f"📂 Input: {input_path}")
        logger.info(f"📂 Output: {output_path}")
        logger.info(f"⚡ Threshold: {threshold}")
        logger.info(f"📊 Max records: {max_records or 'All'}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        stats = {
            'total_processed': 0,
            'total_kept': 0,
            'total_segments': 0,
            'score_distribution': [],
            'command_analysis': {
                'direct_commands': 0,
                'device_interactions': 0,
                'time_scheduling': 0,
                'conversation_patterns': 0,
                'mobile_optimized': 0
            },
            'pattern_usage': {},
            'processing_time': 0.0
        }
        
        start_time = datetime.now()
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(tqdm(infile, desc="🔄 Processing for mobile function calling")):
                if max_records and stats['total_processed'] >= max_records:
                    break
                
                try:
                    record_data = json.loads(line.strip())
                    stats['total_processed'] += 1
                    
                    result = self.process_record(record_data, threshold)
                    
                    if result:
                        # Write filtered record
                        outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                        stats['total_kept'] += 1
                        stats['total_segments'] += result['segments_extracted']
                        stats['score_distribution'].append(result['function_calling_score'])
                        
                        # Update command analysis
                        quality = result['quality_indicators']
                        for key in stats['command_analysis']:
                            if key in quality and quality.get(key, False):
                                stats['command_analysis'][key] += 1
                        
                        # Update pattern usage
                        for pattern, count in result['pattern_matches'].items():
                            stats['pattern_usage'][pattern] = stats['pattern_usage'].get(pattern, 0) + count
                
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Invalid JSON on line {line_num + 1}")
                    continue
                except Exception as e:
                    logger.error(f"❌ Error processing line {line_num + 1}: {e}")
                    continue
        
        # Calculate final statistics
        end_time = datetime.now()
        stats['processing_time'] = (end_time - start_time).total_seconds()
        stats['retention_rate'] = (stats['total_kept'] / stats['total_processed']) * 100 if stats['total_processed'] > 0 else 0
        
        if stats['score_distribution']:
            stats['score_stats'] = {
                'mean': np.mean(stats['score_distribution']),
                'median': np.median(stats['score_distribution']),
                'std': np.std(stats['score_distribution']),
                'min': np.min(stats['score_distribution']),
                'max': np.max(stats['score_distribution'])
            }
        
        # Log final results
        logger.info(f"✅ Mobile function calling filtering complete!")
        logger.info(f"📊 Total processed: {stats['total_processed']:,}")
        logger.info(f"📊 Total kept: {stats['total_kept']:,}")
        logger.info(f"📊 Retention rate: {stats['retention_rate']:.2f}%")
        logger.info(f"📊 Command segments: {stats['total_segments']:,}")
        logger.info(f"⏱️ Processing time: {stats['processing_time']:.2f} seconds")
        
        return stats
    
    def print_filter_explanation(self):
        """
        Print detailed explanation of mobile function calling filters
        """
        print("\n" + "=" * 80)
        print("📱 MOBILE FUNCTION CALLING FILTER - DATA FILTERING CHALLENGE")
        print("=" * 80)
        
        print("\n🏆 TIER 1: DIRECT COMMANDS (Weight: 30-35x) - MOBILE DEVICE ACTIONS")
        print("-" * 60)
        for pattern, description, weight in self.command_patterns:
            print(f"  • {description} (Weight: {weight}x)")
        
        print("\n⭐ TIER 2: ACTION PATTERNS (Weight: 20-25x) - FUNCTION TRIGGERS")
        print("-" * 60)
        for pattern, description, weight in self.action_patterns:
            print(f"  • {description} (Weight: {weight}x)")
        
        print("\n🔧 TIER 3: DEVICE INTERACTIONS (Weight: 15-18x) - MOBILE/IoT CONTEXT")
        print("-" * 60)
        for pattern, description, weight in self.device_patterns:
            print(f"  • {description} (Weight: {weight}x)")
        
        print("\n🗣️ TIER 4: CONVERSATION PATTERNS (Weight: 10-12x) - NATURAL LANGUAGE")
        print("-" * 60)
        for pattern, description, weight in self.conversation_patterns:
            print(f"  • {description} (Weight: {weight}x)")
        
        print("\n🎯 OPTIMIZATION FOR DATA FILTERING CHALLENGE")
        print("-" * 60)
        print("  • Voice assistant commands (Siri, Alexa, Google)")
        print("  • Mobile app function calls")
        print("  • Smart home automation")
        print("  • Task scheduling and reminders")
        print("  • Communication commands")
        print("  • Media control and navigation")
        print("  • IoT device interactions")
        print("  • Conversational function calling")

def main():
    """
    Main function optimized for Data Filtering Challenge
    """
    parser = argparse.ArgumentParser(
        description='Mobile Function Calling Filter for Data Filtering Challenge',
        epilog='Challenge: https://sites.google.com/view/datafilteringchallenge/home'
    )
    parser.add_argument('input', help='Input JSONL file path')
    parser.add_argument('output', help='Output JSONL file path')
    parser.add_argument('--threshold', type=float, default=15.0, 
                       help='Minimum function calling score (default: 15.0, recommended: 10-25)')
    parser.add_argument('--max-records', type=int, 
                       help='Maximum number of records to process')
    parser.add_argument('--explain', action='store_true', 
                       help='Show detailed explanation of mobile function calling filters')
    
    args = parser.parse_args()
    
    # Create the mobile function calling filter
    filter_engine = MobileFunctionCallFilter()
    
    # Show explanation if requested
    if args.explain:
        filter_engine.print_filter_explanation()
        return
    
    # Process the dataset for mobile function calling
    stats = filter_engine.filter_dataset(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        max_records=args.max_records
    )
    
    # Print comprehensive summary for Data Filtering Challenge
    print("\n" + "=" * 80)
    print("🏆 MOBILE FUNCTION CALLING FILTER RESULTS")
    print("=" * 80)
    print(f"📂 Input file: {args.input}")
    print(f"📂 Output file: {args.output}")
    print(f"⚡ Threshold: {args.threshold}")
    print(f"📊 Records processed: {stats['total_processed']:,}")
    print(f"📊 Records kept: {stats['total_kept']:,}")
    print(f"📊 Retention rate: {stats['retention_rate']:.2f}%")
    print(f"📊 Command segments: {stats['total_segments']:,}")
    print(f"⏱️ Processing time: {stats['processing_time']:.2f} seconds")
    
    if 'score_stats' in stats:
        print(f"\n📈 FUNCTION CALLING SCORES")
        print(f"   Mean: {stats['score_stats']['mean']:.2f}")
        print(f"   Median: {stats['score_stats']['median']:.2f}")
        print(f"   Range: {stats['score_stats']['min']:.2f} - {stats['score_stats']['max']:.2f}")
    
    print(f"\n🎯 MOBILE FUNCTION CALLING ANALYSIS")
    for metric, count in stats['command_analysis'].items():
        percentage = (count / stats['total_kept']) * 100 if stats['total_kept'] > 0 else 0
        print(f"   {metric.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    print(f"\n🔧 TOP COMMAND PATTERNS DETECTED")
    sorted_patterns = sorted(stats['pattern_usage'].items(), key=lambda x: x[1], reverse=True)
    for pattern, count in sorted_patterns[:5]:
        print(f"   {pattern}: {count} occurrences")
    
    print("\n" + "=" * 80)
    print("🎉 READY FOR DATA FILTERING CHALLENGE SUBMISSION!")
    print("Challenge: https://sites.google.com/view/datafilteringchallenge/home")
    print("=" * 80)

if __name__ == "__main__":
    # Handle case with no arguments - optimized for mobile function calling
    if len(sys.argv) == 1:
        print("📱 Mobile Function Calling Filter for Data Filtering Challenge")
        print("Usage: python regex_stuff.py input.jsonl output.jsonl [options]")
        print("Use --help for detailed options")
        
        # Try to run on default file if available
        if os.path.exists("climblab_sample.jsonl"):
            print("\n🔄 Found climblab_sample.jsonl, running mobile function calling filter...")
            filter_engine = MobileFunctionCallFilter()
            filter_engine.print_filter_explanation()
            
            stats = filter_engine.filter_dataset(
                input_path="climblab_sample.jsonl",
                output_path="data/mobile_function_calling_filtered.jsonl",
                threshold=15.0,
                max_records=1000
            )
            print(f"\n✅ Results saved to: data/mobile_function_calling_filtered.jsonl")
            print(f"📊 Retention rate: {stats['retention_rate']:.2f}%")
            print("🎯 Optimized for mobile device function calling!")
    else:
        main() 