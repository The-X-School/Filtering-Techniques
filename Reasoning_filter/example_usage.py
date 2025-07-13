#!/usr/bin/env python3
"""
Example usage of the Model-Based Dataset Filtering System
"""

from Model_filter_data import ModelBasedFilter
from datasets import Dataset
import json

def example_basic_usage():
    """Basic usage example with custom data"""
    print("🔍 Example 1: Basic Usage")
    print("=" * 50)
    
    # Create sample data
    sample_data = [
        {"text": "This is a high-quality educational text about machine learning algorithms and their applications in real-world scenarios."},
        {"text": "Short text."},
        {"text": "A comprehensive guide to understanding neural networks, including backpropagation, gradient descent, and optimization techniques."},
        {"text": "Bad txt."},
        {"text": "Advanced topics in deep learning include convolutional neural networks, recurrent neural networks, and transformer architectures."},
        {"text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit."},
        {"text": "Mathematical foundations of statistics provide the theoretical framework for understanding probability distributions and hypothesis testing."},
        {"text": "xyz"},
        {"text": "Data preprocessing is a crucial step in machine learning pipelines that involves cleaning, transforming, and preparing data for model training."},
        {"text": "The impact of artificial intelligence on modern society extends beyond technology to influence economics, education, and social structures."}
    ]
    
    # Create dataset
    dataset = Dataset.from_list(sample_data)
    
    # Initialize filter with custom settings
    filter_system = ModelBasedFilter(
        model_name="my_model",
        device="cpu"  # Force CPU for this example
    )
    
    # Set quality threshold (lower = more permissive)
    filter_system.quality_threshold = 0.3
    
    # Filter the dataset
    filtered_samples = filter_system.filter_dataset(dataset)
    
    # Print results
    print(f"\nResults:")
    print(f"Original samples: {len(sample_data)}")
    print(f"Filtered samples: {len(filtered_samples)}")
    print(f"Retention rate: {len(filtered_samples)/len(sample_data)*100:.1f}%")
    
    # Show some examples
    print(f"\nSample filtered results:")
    for i, sample in enumerate(filtered_samples[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Text: {sample['text'][:80]}...")
        print(f"  Quality Score: {sample['quality_analysis']['score']:.3f}")
        print(f"  Perplexity: {sample['quality_analysis']['perplexity']:.2f}")

def example_strict_filtering():
    """Example with strict filtering settings"""
    print("\n\n🔍 Example 2: Strict Filtering")
    print("=" * 50)
    
    # Create sample data with varying quality
    sample_data = [
        {"text": "Advanced machine learning techniques require deep understanding of mathematical concepts including linear algebra, calculus, and statistics."},
        {"text": "AI good."},
        {"text": "The comprehensive analysis of large-scale distributed systems necessitates understanding of network protocols, database optimization, and concurrent programming paradigms."},
        {"text": "Nice weather today."},
        {"text": "Quantum computing represents a paradigm shift in computational capabilities, leveraging quantum mechanical phenomena such as superposition and entanglement."},
    ]
    
    dataset = Dataset.from_list(sample_data)
    
    # Initialize filter with strict settings
    filter_system = ModelBasedFilter(model_name="my_model")
    filter_system.quality_threshold = 0.8  # Very strict threshold
    
    # Filter the dataset
    filtered_samples = filter_system.filter_dataset(dataset)
    
    print(f"\nStrict Filtering Results:")
    print(f"Original samples: {len(sample_data)}")
    print(f"Filtered samples: {len(filtered_samples)}")
    print(f"Retention rate: {len(filtered_samples)/len(sample_data)*100:.1f}%")
    
    # Show quality scores for all samples
    print(f"\nQuality scores for all samples:")
    for sample in dataset:
        quality_result = filter_system.calculate_quality_score(sample['text'])
        kept = "✅ KEPT" if quality_result['score'] >= filter_system.quality_threshold else "❌ REJECTED"
        print(f"  Score: {quality_result['score']:.3f} - {kept}")
        print(f"    Text: {sample['text'][:60]}...")

def example_batch_processing():
    """Example showing batch processing capabilities"""
    print("\n\n🔍 Example 3: Batch Processing")
    print("=" * 50)
    
    # Simulate larger dataset
    base_texts = [
        "Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning approaches.",
        "Data visualization techniques help in understanding patterns and relationships within complex datasets.",
        "Short.",
        "Natural language processing combines computational linguistics with machine learning to enable computers to understand human language.",
        "Big data analytics involves processing and analyzing large volumes of data to extract meaningful insights.",
        "Bad text here.",
        "Statistical inference provides methods for drawing conclusions about populations based on sample data.",
        "Computer vision algorithms enable machines to interpret and understand visual information from images and videos.",
    ]
    
    # Create larger dataset by repeating
    large_dataset = []
    for i in range(50):
        for text in base_texts:
            large_dataset.append({"text": f"{text} (Sample {i+1})"})
    
    dataset = Dataset.from_list(large_dataset)
    
    # Initialize filter
    filter_system = ModelBasedFilter(model_name="my_model")
    filter_system.quality_threshold = 0.5
    
    # Process in batches
    print(f"Processing {len(large_dataset)} samples...")
    filtered_samples = filter_system.filter_dataset(dataset, max_samples=100)
    
    # Print batch processing results
    filter_system.print_analysis_report()
    
    # Save results
    output_path = "data/filtered_dataset/batch_example_results.jsonl"
    filter_system.save_filtered_data(filtered_samples, output_path)
    print(f"\nBatch results saved to: {output_path}")

def example_custom_scoring():
    """Example showing how to extend the filter for custom scoring"""
    print("\n\n🔍 Example 4: Custom Scoring Extension")
    print("=" * 50)
    
    class CustomFilter(ModelBasedFilter):
        def calculate_quality_score(self, text: str):
            # Get base score from parent class
            base_result = super().calculate_quality_score(text)
            
            # Add custom scoring factors
            word_count = len(text.split())
            sentence_count = len([s for s in text.split('.') if s.strip()])
            
            # Custom quality factors
            word_diversity = len(set(text.lower().split())) / word_count if word_count > 0 else 0
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Combine with base score
            custom_score = (
                base_result['score'] * 0.6 +
                min(word_diversity, 1.0) * 0.2 +
                min(avg_sentence_length / 15, 1.0) * 0.2
            )
            
            # Update result
            base_result['score'] = custom_score
            base_result['custom_factors'] = {
                'word_diversity': word_diversity,
                'avg_sentence_length': avg_sentence_length,
                'sentence_count': sentence_count
            }
            
            return base_result
    
    # Test custom filter
    sample_data = [
        {"text": "Advanced machine learning techniques require understanding of mathematical concepts and algorithmic implementations."},
        {"text": "AI is good. AI is nice. AI is great."},  # Low diversity
        {"text": "The quick brown fox jumps over the lazy dog multiple times in this sentence."},
    ]
    
    dataset = Dataset.from_list(sample_data)
    custom_filter = CustomFilter(model_name="my_model")
    custom_filter.quality_threshold = 0.5
    
    filtered_samples = custom_filter.filter_dataset(dataset)
    
    print(f"\nCustom Scoring Results:")
    for i, sample in enumerate(filtered_samples):
        print(f"\nSample {i+1}:")
        print(f"  Text: {sample['text'][:60]}...")
        print(f"  Quality Score: {sample['quality_analysis']['score']:.3f}")
        if 'custom_factors' in sample['quality_analysis']:
            print(f"  Word Diversity: {sample['quality_analysis']['custom_factors']['word_diversity']:.3f}")
            print(f"  Avg Sentence Length: {sample['quality_analysis']['custom_factors']['avg_sentence_length']:.1f}")

def main():
    """Run all examples"""
    print("🚀 Model-Based Dataset Filtering Examples")
    print("=" * 80)
    
    try:
        example_basic_usage()
        example_strict_filtering()
        example_batch_processing()
        example_custom_scoring()
        
        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("💡 You can now use the ModelBasedFilter class for your own datasets")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        raise

if __name__ == "__main__":
    main() 