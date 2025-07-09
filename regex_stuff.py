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

class NaturalLanguageCommandFilter:
    """
    Natural Language Command Filter - Finds sentences that exhibit function calling behavior
    like "turn on the lights at 8pm", "set a reminder for tomorrow", etc.
    
    Focuses on imperative commands and action-oriented natural language that would
    typically trigger function calls in smart assistants or automation systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the natural language command filter with configurable thresholds"""
        # Set configurable parameters with defaults
        self.config = config or {}
        self.acceptance_threshold = self.config.get('acceptance_threshold', 8.0)  # Lower threshold for natural language
        self.min_text_length = self.config.get('min_text_length', 20)
        self.min_command_ratio = self.config.get('min_command_ratio', 0.05)  # Min 5% command-like content
        
        # Pattern categories for command detection
        self.action_patterns = self._compile_action_patterns()               # Action verbs and commands
        self.device_patterns = self._compile_device_patterns()               # Smart device and IoT references
        self.time_patterns = self._compile_time_patterns()                   # Time and scheduling expressions
        self.location_patterns = self._compile_location_patterns()           # Location and spatial references
        self.service_patterns = self._compile_service_patterns()             # Service and app commands
        self.imperative_patterns = self._compile_imperative_patterns()       # Imperative sentence structures
        
        logger.info("Natural Language Command Filter initialized")
        logger.info(f"Configuration: {self.config}")
        logger.info(f"Pattern counts: Actions={len(self.action_patterns)}, Devices={len(self.device_patterns)}, Time={len(self.time_patterns)}, Location={len(self.location_patterns)}, Services={len(self.service_patterns)}, Imperatives={len(self.imperative_patterns)}")
    
    def _compile_action_patterns(self) -> List[Tuple[re.Pattern, str, int]]:
        """
        Action verb patterns that indicate commands/instructions
        More precise patterns to avoid false positives
        """
        patterns = [
            # Direct imperative commands (sentence beginning)
            (re.compile(r'^\s*(?:please\s+)?(?:turn on|turn off|switch on|switch off|toggle|turn)\s+(?:the\s+)?(?:lights?|lamp|tv|music|radio|system)', re.IGNORECASE), 
             "Direct device control commands", 50),
            
            # Smart home specific actions (sentence beginning)
            (re.compile(r'^\s*(?:please\s+)?(?:dim|brighten|adjust|set|change)\s+(?:the\s+)?(?:lights?|brightness|volume|temperature|thermostat)', re.IGNORECASE), 
             "Smart home adjustment commands", 45),
            
            # Media control at sentence start
            (re.compile(r'^\s*(?:please\s+)?(?:play|pause|stop|skip|resume)\s+(?:the\s+)?(?:music|song|video|podcast|movie|playlist)', re.IGNORECASE), 
             "Media control commands", 45),
            
            # Communication commands at sentence start
            (re.compile(r'^\s*(?:please\s+)?(?:send|text|call|email)\s+(?:a\s+)?(?:message|text|email)\s+to\s+\w+', re.IGNORECASE), 
             "Communication commands", 50),
            
            # Scheduling commands at sentence start
            (re.compile(r'^\s*(?:please\s+)?(?:set|create|add|schedule)\s+(?:a\s+)?(?:reminder|alarm|timer|appointment|meeting)\s+(?:for|at)', re.IGNORECASE), 
             "Scheduling commands", 50),
            
            # Information requests as commands
            (re.compile(r'^\s*(?:please\s+)?(?:check|get|find|look up|search for|tell me)\s+(?:the\s+)?(?:weather|time|news|traffic|forecast)', re.IGNORECASE), 
             "Information request commands", 40),
            
            # Shopping/ordering commands with context
            (re.compile(r'^\s*(?:please\s+)?(?:order|buy|purchase)\s+(?:some\s+|a\s+)?(?:food|groceries|pizza|coffee|lunch|dinner)', re.IGNORECASE), 
             "Shopping and ordering commands", 45),
            
            # Navigation commands
            (re.compile(r'^\s*(?:please\s+)?(?:navigate to|drive to|directions to|take me to|find route to)\s+', re.IGNORECASE), 
             "Navigation commands", 45),
            
            # Voice assistant style commands
            (re.compile(r'\b(?:hey|ok|okay)\s+(?:google|alexa|siri|assistant),?\s+(?:turn|set|play|call|send|order|check|find)', re.IGNORECASE), 
             "Voice assistant commands", 60),
        ]
        return patterns
    
    def _compile_device_patterns(self) -> List[Tuple[re.Pattern, str, int]]:
        """
        Smart device and IoT device patterns - only when in command context
        """
        patterns = [
            # Lighting devices in command context
            (re.compile(r'\b(?:turn|set|dim|brighten|adjust|change)\s+(?:the\s+)?(?:lights?|lamp|bulbs?)', re.IGNORECASE), 
             "Lighting device commands", 35),
            
            # Climate control in command context
            (re.compile(r'\b(?:set|adjust|turn|change)\s+(?:the\s+)?(?:thermostat|temperature|heater|ac|air conditioning)', re.IGNORECASE), 
             "Climate control commands", 35),
            
            # Media devices in command context
            (re.compile(r'\b(?:turn|play|stop|pause|change)\s+(?:the\s+)?(?:tv|television|music|speaker|radio)', re.IGNORECASE), 
             "Media device commands", 30),
        ]
        return patterns
    
    def _compile_time_patterns(self) -> List[Tuple[re.Pattern, str, int]]:
        """
        Time expressions in command context only
        """
        patterns = [
            # Specific times in command context
            (re.compile(r'\b(?:at|for)\s+(?:\d{1,2}:\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm|o\'?clock))', re.IGNORECASE), 
             "Specific scheduled times", 30),
            
            # Command with time context
            (re.compile(r'\b(?:set|schedule|remind me|wake me up)\s+.*?\b(?:at|for|in)\s+(?:\d+\s*(?:minutes?|hours?)|tomorrow|tonight)', re.IGNORECASE), 
             "Time-based scheduling commands", 40),
            
            # Timer commands
            (re.compile(r'\b(?:set|start)\s+(?:a\s+)?timer\s+(?:for\s+)?\d+\s*(?:minutes?|hours?)', re.IGNORECASE), 
             "Timer setting commands", 45),
        ]
        return patterns
    
    def _compile_location_patterns(self) -> List[Tuple[re.Pattern, str, int]]:
        """
        Location references in command context
        """
        patterns = [
            # Room-specific commands
            (re.compile(r'\b(?:turn on|dim|play music|set temperature)\s+.*?\bin the\s+(?:living room|bedroom|kitchen|bathroom)', re.IGNORECASE), 
             "Room-specific commands", 40),
            
            # Navigation to locations
            (re.compile(r'\b(?:navigate to|drive to|directions to|take me to)\s+(?:the\s+)?(?:store|restaurant|home|work|office)', re.IGNORECASE), 
             "Navigation location commands", 35),
        ]
        return patterns
    
    def _compile_service_patterns(self) -> List[Tuple[re.Pattern, str, int]]:
        """
        Service commands - only in command context
        """
        patterns = [
            # Music service commands
            (re.compile(r'\b(?:play|stop|pause)\s+.*?\bon\s+(?:spotify|apple music|youtube|pandora)', re.IGNORECASE), 
             "Music service commands", 40),
            
            # Communication app commands
            (re.compile(r'\b(?:send|call|message)\s+.*?\b(?:on|via|through)\s+(?:whatsapp|slack|skype|zoom)', re.IGNORECASE), 
             "Communication app commands", 40),
            
            # Food delivery commands
            (re.compile(r'\b(?:order|get)\s+.*?\b(?:from|on|via)\s+(?:uber eats|doordash|grubhub)', re.IGNORECASE), 
             "Food delivery commands", 40),
        ]
        return patterns
    
    def _compile_imperative_patterns(self) -> List[Tuple[re.Pattern, str, int]]:
        """
        Imperative sentence structure patterns - very specific to commands
        """
        patterns = [
            # Direct imperative at sentence start with specific targets
            (re.compile(r'^\s*(?:please\s+)?(?:turn|set|play|stop|start|open|close|send|call|order|check|find|show|tell|dim|brighten)\s+(?:the\s+|a\s+|my\s+)?(?:lights?|music|tv|alarm|reminder|volume|temperature)', re.IGNORECASE), 
             "Direct imperative commands", 50),
            
            # Polite requests for specific actions
            (re.compile(r'\b(?:can you|could you|would you please)\s+(?:turn|set|play|send|order|check|find)', re.IGNORECASE), 
             "Polite command requests", 45),
            
            # Want/need for actions (not things)
            (re.compile(r'\b(?:I want to|I need to|I\'d like to)\s+(?:turn|set|play|send|order|check|book|schedule)', re.IGNORECASE), 
             "Want/need action patterns", 35),
            
            # Commands with specific objects and actions
            (re.compile(r'\b(?:turn|set|adjust|change)\s+(?:the\s+)?(?:lights?|volume|temperature|thermostat)\s+(?:to|on|off|up|down)', re.IGNORECASE), 
             "Command with object and action", 45),
        ]
        return patterns
    
    def calculate_command_score(self, text: str) -> Dict[str, Any]:
        """
        Calculate how command-like the text is (higher score = more command-like = should be KEPT)
        """
        if not text:
            return {"total_score": 0.0, "breakdown": {}, "detected_patterns": {}}
        
        text_length = len(text)
        breakdown = {"actions": 0, "devices": 0, "time": 0, "location": 0, "services": 0, "imperatives": 0}
        detected_patterns = {}
        
        # Check action patterns
        for pattern, description, weight in self.action_patterns:
            matches = pattern.findall(text)
            if matches:
                score = len(matches) * weight
                breakdown["actions"] += score
                detected_patterns[description] = len(matches)
        
        # Check device patterns
        for pattern, description, weight in self.device_patterns:
            matches = pattern.findall(text)
            if matches:
                score = len(matches) * weight
                breakdown["devices"] += score
                detected_patterns[description] = len(matches)
        
        # Check time patterns
        for pattern, description, weight in self.time_patterns:
            matches = pattern.findall(text)
            if matches:
                score = len(matches) * weight
                breakdown["time"] += score
                detected_patterns[description] = len(matches)
        
        # Check location patterns
        for pattern, description, weight in self.location_patterns:
            matches = pattern.findall(text)
            if matches:
                score = len(matches) * weight
                breakdown["location"] += score
                detected_patterns[description] = len(matches)
        
        # Check service patterns
        for pattern, description, weight in self.service_patterns:
            matches = pattern.findall(text)
            if matches:
                score = len(matches) * weight
                breakdown["services"] += score
                detected_patterns[description] = len(matches)
        
        # Check imperative patterns
        for pattern, description, weight in self.imperative_patterns:
            matches = pattern.findall(text)
            if matches:
                score = len(matches) * weight
                breakdown["imperatives"] += score
                detected_patterns[description] = len(matches)
        
        # Calculate total score with normalization
        raw_score = sum(breakdown.values())
        normalized_score = raw_score / (text_length / 1000) if text_length > 0 else 0  # Per 1000 characters
        final_score = max(min(normalized_score, 100.0), 0.0)  # Cap between 0 and 100
        
        # Command indicators
        command_indicators = {
            "has_action_verbs": breakdown["actions"] > 10,
            "has_device_references": breakdown["devices"] > 10,
            "has_time_expressions": breakdown["time"] > 15,
            "has_location_context": breakdown["location"] > 10,
            "has_service_references": breakdown["services"] > 15,
            "has_imperative_structure": breakdown["imperatives"] > 20,
            "multiple_command_types": len([k for k in breakdown.keys() if breakdown[k] > 0]) >= 2,
            "high_command_density": final_score > self.acceptance_threshold,
            "command_diversity": len([p for p in detected_patterns.keys()]) >= 2
        }
        
        return {
            "total_score": final_score,
            "raw_score": raw_score,
            "breakdown": breakdown,
            "detected_patterns": detected_patterns,
            "command_indicators": command_indicators,
            "text_length": text_length
        }
    
    def should_keep_sample(self, score_data: Dict[str, Any], text: str) -> Tuple[bool, str]:
        """
        Determine if a sample should be KEPT based on command detection
        Returns (should_keep, reason)
        """
        indicators = score_data["command_indicators"]
        
        # KEEP if total score is high enough (command-like)
        if score_data["total_score"] >= self.acceptance_threshold:
            return True, f"High command score: {score_data['total_score']:.1f}"
        
        # KEEP if has clear imperative structure (direct commands)
        if indicators["has_imperative_structure"] and score_data["breakdown"]["imperatives"] > 25:
            return True, "Strong imperative/command structure detected"
        
        # KEEP if has action verbs with device references (smart home commands)
        if indicators["has_action_verbs"] and indicators["has_device_references"]:
            return True, "Action verbs with device references detected"
        
        # KEEP if has action verbs with time expressions (scheduled commands)
        if indicators["has_action_verbs"] and indicators["has_time_expressions"]:
            return True, "Action verbs with time expressions detected"
        
        # KEEP if has multiple command types (diverse command content)
        if indicators["multiple_command_types"] and indicators["command_diversity"]:
            return True, "Multiple command pattern types detected"
        
        # KEEP if has service references with actions (app/service commands)
        if indicators["has_service_references"] and indicators["has_action_verbs"]:
            return True, "Service references with action verbs detected"
        
        # Calculate command-specific ratio
        command_matches = sum([
            score_data["detected_patterns"].get(pattern, 0) 
            for pattern in score_data["detected_patterns"].keys()
        ])
        command_ratio = command_matches / (len(text) / 100) if len(text) > 0 else 0
        
        if command_ratio >= self.min_command_ratio:
            return True, f"Sufficient command content ratio: {command_ratio:.1%}"
        
        return False, "Insufficient command content - likely descriptive or narrative text"
    
    def process_record(self, record_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single record and determine if it should be kept (contains commands) or rejected
        """
        text = record_data.get('text', '')
        if not text or len(text) < self.min_text_length:
            return None
        
        # Calculate command score
        score_data = self.calculate_command_score(text)
        
        # Check if should be kept
        should_keep, reason = self.should_keep_sample(score_data, text)
        
        if not should_keep:
            return None  # Reject this sample
        
        # Keep the sample (it contains command-like content)
        return {
            "text": text,
            "command_score": score_data["total_score"],
            "keep_reason": "KEPT - " + reason,
            "detected_patterns": score_data["detected_patterns"],
            "command_indicators": score_data["command_indicators"],
            "original_length": len(text)
        }
    
    def filter_dataset(self, input_path: str, output_path: str, max_records: Optional[int] = None) -> Dict[str, Any]:
        """
        Filter dataset to keep only natural language commands and instructions
        """
        logger.info(f"🎯 Starting Natural Language Command Filter")
        logger.info(f"📂 Input: {input_path}")
        logger.info(f"📂 Output: {output_path}")
        logger.info(f"🎯 Acceptance threshold: {self.acceptance_threshold}")
        logger.info(f"📊 Max records: {max_records or 'All'}")
        logger.info(f"🎯 Goal: Find sentences like 'turn on the lights at 8pm'")
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        stats = {
            'total_processed': 0,
            'total_kept': 0,
            'total_rejected': 0,
            'keep_reasons': {},
            'pattern_detections': {},
            'command_score_distribution': [],
            'processing_time': 0.0
        }
        
        start_time = datetime.now()
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(tqdm(infile, desc="🎯 Filtering for command-like sentences")):
                if max_records and stats['total_processed'] >= max_records:
                    break
                
                try:
                    record_data = json.loads(line.strip())
                    stats['total_processed'] += 1
                    
                    # Calculate command score for statistics
                    text = record_data.get('text', '')
                    if text:
                        score_data = self.calculate_command_score(text)
                        stats['command_score_distribution'].append(score_data['total_score'])
                        
                        # Track pattern detections
                        for pattern, count in score_data['detected_patterns'].items():
                            stats['pattern_detections'][pattern] = stats['pattern_detections'].get(pattern, 0) + count
                    
                    result = self.process_record(record_data)
                    
                    if result:
                        # Keep this sample (contains commands)
                        outfile.write(json.dumps(record_data, ensure_ascii=False) + '\n')  # Keep original format
                        stats['total_kept'] += 1
                        should_keep, reason = self.should_keep_sample(score_data, text)
                        stats['keep_reasons'][reason] = stats['keep_reasons'].get(reason, 0) + 1
                    else:
                        # Rejected (no command content)
                        stats['total_rejected'] += 1
                
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
        stats['rejection_rate'] = (stats['total_rejected'] / stats['total_processed']) * 100 if stats['total_processed'] > 0 else 0
        
        if stats['command_score_distribution']:
            stats['score_stats'] = {
                'mean': np.mean(stats['command_score_distribution']),
                'median': np.median(stats['command_score_distribution']),
                'std': np.std(stats['command_score_distribution']),
                'min': np.min(stats['command_score_distribution']),
                'max': np.max(stats['command_score_distribution'])
            }
        
        # Log final results
        logger.info(f"🎯 Natural language command filtering complete!")
        logger.info(f"📊 Total processed: {stats['total_processed']:,}")
        logger.info(f"📊 Total kept (command content): {stats['total_kept']:,}")
        logger.info(f"📊 Total rejected (non-command): {stats['total_rejected']:,}")
        logger.info(f"📊 Retention rate: {stats['retention_rate']:.1f}%")
        logger.info(f"📊 Rejection rate: {stats['rejection_rate']:.1f}%")
        logger.info(f"⏱️ Processing time: {stats['processing_time']:.2f} seconds")
        
        return stats
    
    @classmethod
    def create_strict_config(cls) -> Dict[str, Any]:
        """Create a strict configuration that only keeps very clear commands"""
        return {
            'acceptance_threshold': 15.0,  # Higher threshold = only obvious commands
            'min_text_length': 30,
            'min_command_ratio': 0.1       # Min 10% command content required
        }
    
    @classmethod
    def create_moderate_config(cls) -> Dict[str, Any]:
        """Create a moderate configuration balancing command detection"""
        return {
            'acceptance_threshold': 8.0,   # Default threshold
            'min_text_length': 20,
            'min_command_ratio': 0.05      # Min 5% command content required
        }
    
    @classmethod
    def create_lenient_config(cls) -> Dict[str, Any]:
        """Create a lenient configuration that keeps more command-adjacent content"""
        return {
            'acceptance_threshold': 5.0,   # Lower threshold = keeps more
            'min_text_length': 15,
            'min_command_ratio': 0.02      # Min 2% command content required
        }
    
    def print_filter_explanation(self):
        """
        Print detailed explanation of natural language command detection patterns
        """
        print("\n" + "=" * 80)
        print("🎯 NATURAL LANGUAGE COMMAND FILTER")
        print("=" * 80)
        print("🎯 Goal: Find sentences like 'turn on the lights at 8pm'")
        print("📝 Target: Natural language commands and instructions")
        
        print("\n🎬 ACTION PATTERNS (Weight: 20-35x) - COMMAND VERBS")
        print("-" * 60)
        for pattern, description, weight in self.action_patterns:
            print(f"  • {description} (Weight: {weight}x)")
        
        print("\n🏠 DEVICE PATTERNS (Weight: 15-30x) - SMART DEVICES & IOT")
        print("-" * 60)
        for pattern, description, weight in self.device_patterns:
            print(f"  • {description} (Weight: {weight}x)")
        
        print("\n⏰ TIME PATTERNS (Weight: 15-30x) - SCHEDULING & TIMING")
        print("-" * 60)
        for pattern, description, weight in self.time_patterns:
            print(f"  • {description} (Weight: {weight}x)")
        
        print("\n📍 LOCATION PATTERNS (Weight: 10-20x) - SPATIAL REFERENCES")
        print("-" * 60)
        for pattern, description, weight in self.location_patterns:
            print(f"  • {description} (Weight: {weight}x)")
        
        print("\n📱 SERVICE PATTERNS (Weight: 15-25x) - APPS & SERVICES")
        print("-" * 60)
        for pattern, description, weight in self.service_patterns:
            print(f"  • {description} (Weight: {weight}x)")
        
        print("\n📢 IMPERATIVE PATTERNS (Weight: 20-30x) - COMMAND STRUCTURE")
        print("-" * 60)
        for pattern, description, weight in self.imperative_patterns:
            print(f"  • {description} (Weight: {weight}x)")
        
        print(f"\n✅ COMMAND ACCEPTANCE CRITERIA")
        print("-" * 60)
        print(f"  • Command score threshold: {self.acceptance_threshold}")
        print(f"  • Minimum text length: {self.min_text_length}")
        print(f"  • Minimum command content ratio: {self.min_command_ratio:.0%}")
        print("  • KEEPS sentences with imperative structures")
        print("  • KEEPS sentences with action verbs + devices/time")
        print("  • KEEPS sentences with service references + actions")
        print("  • TARGETS natural language that would trigger function calls")
        
        print(f"\n🎯 EXAMPLE TARGETS:")
        print("-" * 60)
        print("  • 'turn on the lights at 8pm'")
        print("  • 'set a reminder for tomorrow at 9am'")
        print("  • 'play music in the living room'")
        print("  • 'send a message to John'")
        print("  • 'order groceries for delivery'")
        print("  • 'check the weather forecast'")
        print("  • 'dim the bedroom lights to 50%'")
        print("  • 'book a table for two at 7pm'")

def main():
    """
    Main function for natural language command filtering
    """
    parser = argparse.ArgumentParser(
        description='Natural Language Command Filter - Find Command-like Sentences',
        epilog='Finds sentences like "turn on the lights at 8pm" that exhibit function calling behavior'
    )
    parser.add_argument('input', help='Input JSONL file path')
    parser.add_argument('output', help='Output JSONL file path')
    parser.add_argument('--threshold', type=float, default=8.0, 
                       help='Command acceptance threshold (default: 8.0, lower = more permissive)')
    parser.add_argument('--max-records', type=int, 
                       help='Maximum number of records to process')
    parser.add_argument('--explain', action='store_true', 
                       help='Show detailed explanation of command filtering patterns')
    parser.add_argument('--strictness', choices=['strict', 'moderate', 'lenient'], 
                       default='moderate', help='Filter strictness level (default: moderate)')
    
    args = parser.parse_args()
    
    # Create the filter with appropriate configuration
    if args.strictness == 'strict':
        config = NaturalLanguageCommandFilter.create_strict_config()
    elif args.strictness == 'lenient':
        config = NaturalLanguageCommandFilter.create_lenient_config()
    else:  # moderate
        config = NaturalLanguageCommandFilter.create_moderate_config()
    
    # Override threshold if provided
    if args.threshold != 8.0:
        config['acceptance_threshold'] = args.threshold
    
    filter_engine = NaturalLanguageCommandFilter(config)
    
    # Show explanation if requested
    if args.explain:
        filter_engine.print_filter_explanation()
        return
    
    # Process the dataset
    stats = filter_engine.filter_dataset(
        input_path=args.input,
        output_path=args.output,
        max_records=args.max_records
    )
    
    # Print comprehensive summary
    print("\n" + "=" * 80)
    print("🎯 NATURAL LANGUAGE COMMAND FILTER RESULTS")
    print("=" * 80)
    print(f"📂 Input file: {args.input}")
    print(f"📂 Output file: {args.output}")
    print(f"🎯 Acceptance threshold: {config['acceptance_threshold']}")
    print(f"📊 Records processed: {stats['total_processed']:,}")
    print(f"📊 Records kept (command content): {stats['total_kept']:,}")
    print(f"📊 Records rejected (non-command): {stats['total_rejected']:,}")
    print(f"📊 Retention rate: {stats['retention_rate']:.1f}%")
    print(f"📊 Rejection rate: {stats['rejection_rate']:.1f}%")
    print(f"⏱️ Processing time: {stats['processing_time']:.2f} seconds")
    
    if 'score_stats' in stats:
        print(f"\n📈 COMMAND SCORES (Higher = More Command-like)")
        print(f"   Mean: {stats['score_stats']['mean']:.2f}")
        print(f"   Median: {stats['score_stats']['median']:.2f}")
        print(f"   Range: {stats['score_stats']['min']:.2f} - {stats['score_stats']['max']:.2f}")
    
    print(f"\n✅ TOP KEEP REASONS")
    sorted_reasons = sorted(stats['keep_reasons'].items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_reasons[:5]:
        print(f"   {reason}: {count} samples")
    
    print(f"\n🔍 TOP DETECTED PATTERNS")
    sorted_patterns = sorted(stats['pattern_detections'].items(), key=lambda x: x[1], reverse=True)
    for pattern, count in sorted_patterns[:5]:
        print(f"   {pattern}: {count} occurrences")
    
    print("\n" + "=" * 80)
    print("🎯 NATURAL LANGUAGE COMMAND FILTER COMPLETE!")
    print("Dataset now contains sentences that exhibit function calling behavior.")
    print("Example targets: 'turn on the lights at 8pm', 'set a reminder', etc.")
    print("=" * 80)

if __name__ == "__main__":
    # Handle case with no arguments
    if len(sys.argv) == 1:
        print("🎯 Natural Language Command Filter")
        print("Usage: python regex_stuff.py input.jsonl output.jsonl [options]")
        print("Use --help for detailed options")
        print("🎯 Goal: Find sentences like 'turn on the lights at 8pm'")
        
        # Try to run on default file if available
        if os.path.exists("climblab_sample.jsonl"):
            print("\n🔄 Found climblab_sample.jsonl, running command filter...")
            
            # Use moderate config for demo
            config = NaturalLanguageCommandFilter.create_moderate_config()
            filter_engine = NaturalLanguageCommandFilter(config)
            filter_engine.print_filter_explanation()
            
            stats = filter_engine.filter_dataset(
                input_path="climblab_sample.jsonl",
                output_path="data/command_filtered.jsonl",
                max_records=1000
            )
            print(f"\n✅ Results saved to: data/command_filtered.jsonl")
            print(f"📊 Retention rate: {stats['retention_rate']:.1f}%")
            print(f"📊 Rejection rate: {stats['rejection_rate']:.1f}%")
            print("🎯 Natural language command filtering complete!")
    else:
        main() 