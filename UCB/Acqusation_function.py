#!/usr/bin/env python3

import os
import json
import shutil
from pathlib import Path
import fasttext
from huggingface_hub import hf_hub_download
from Borf import main as Borf
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound

def run_fasttext_filtering(threshold=0.99, num_samples=10000):
    """Run FastText filtering on sample data"""
    try:
        # Download FastText model
        model_path = hf_hub_download(
            repo_id="hkust-nlp/preselect-fasttext-classifier", 
            filename="PreSelect-classifier.bin"
        )
        
        # Load model
        model = fasttext.load_model(model_path)
    except Exception as e:
        print(f"Warning: Could not load FastText model: {e}")
        print("Using mock filtering for testing")
        return run_mock_filtering(threshold, num_samples)
    
    # Clean up existing temp directories
    temp_output = "temp_filtered_output"
    if os.path.exists(temp_output):
        shutil.rmtree(temp_output)
    Path(temp_output).mkdir(parents=True, exist_ok=True)
    
    # Read sample data
    climblab_samples_path = os.path.join("..", "climblab_samples", "00001.jsonl")
    sample_texts = []
    
    if os.path.exists(climblab_samples_path):
        with open(climblab_samples_path, "r") as f:
            lines = f.readlines()[:num_samples]
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    if "text" in data:
                        sample_texts.append(data["text"])
                except json.JSONDecodeError:
                    continue
    
    # Generate dummy samples if needed
    while len(sample_texts) < num_samples:
        sample_texts.append(f"Sample text {len(sample_texts)} for testing machine learning models and natural language processing.")
    
    sample_texts = sample_texts[:num_samples]
    
    # Run filtering
    filtered_texts = []
    for text in sample_texts:
        clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
        if not clean_text:
            continue
            
        try:
            predictions = model.predict(clean_text)
            labels, scores = predictions
            
            for label, score in zip(labels, scores):
                if label == "__label__1" and score >= threshold:
                    filtered_texts.append(text)
                    break
        except Exception as e:
            print(f"Error filtering text: {e}")
            continue
    
    filtered_count = len(filtered_texts)
    retention_rate = filtered_count / num_samples if num_samples > 0 else 0.0
    
    # Save filtered results
    output_file = os.path.join(temp_output, "00001.jsonl")
    with open(output_file, "w") as f:
        for text in filtered_texts:
            f.write(json.dumps({"text": text}) + "\n")
    
    return {
        "filtered_count": filtered_count,
        "total_count": num_samples,
        "retention_rate": retention_rate
    }

def run_mock_filtering(threshold=0.99, num_samples=10000):
    """Mock filtering function for testing when FastText is not available"""
    temp_output = "temp_filtered_output"
    if os.path.exists(temp_output):
        shutil.rmtree(temp_output)
    Path(temp_output).mkdir(parents=True, exist_ok=True)
    
    # Generate mock filtered data
    import random
    random.seed(42)
    retention_rate = max(0.1, min(0.9, threshold * 0.8))  # Simulate filtering effect
    filtered_count = int(num_samples * retention_rate)
    
    # Create mock filtered data
    output_file = os.path.join(temp_output, "00001.jsonl")
    with open(output_file, "w") as f:
        for i in range(filtered_count):
            text = f"High-quality sample text {i} about machine learning and natural language processing."
            f.write(json.dumps({"text": text}) + "\n")
    
    return {
        "filtered_count": filtered_count,
        "total_count": num_samples,
        "retention_rate": retention_rate
    }

def fasttext_objective(threshold=0.99, num_samples=10000):
    """Objective function for Bayesian optimization"""
    print(f"Testing threshold={threshold:.3f}, num_samples={int(num_samples)}")
    
    # Run filtering
    filter_stats = run_fasttext_filtering(threshold, int(num_samples))
    
    print(f"Filtered {filter_stats['filtered_count']} out of {filter_stats['total_count']} samples")
    
    # Check if enough samples
    if filter_stats["filtered_count"] < 50:
        print("Not enough filtered samples, returning low score")
        return 0.1
    
    # Run training with Borf
    try:
        accuracy = Borf()
        print(f"Training accuracy: {accuracy:.3f}")
    except Exception as e:
        print(f"Training failed: {e}")
        return 0.1
    
    # Combine accuracy with retention rate
    retention_rate = filter_stats["retention_rate"]
    
    if retention_rate < 0.05:
        combined_score = accuracy * 0.1
    elif retention_rate < 0.10:
        combined_score = accuracy * 0.5
    else:
        combined_score = accuracy * (0.8 + 0.2 * retention_rate)
    
    print(f"Combined score: {combined_score:.3f}")
    return combined_score

# Set up optimization
optimizer = BayesianOptimization(
    f=fasttext_objective,
    pbounds={
        'threshold': (0.5, 0.99),
        'num_samples': (1000, 20000)
    },
    acquisition_function=UpperConfidenceBound(kappa=2.576), 
    random_state=42
)

def get_optimizer():
    """Return the configured optimizer"""
    return optimizer

if __name__ == "__main__":
    # Test the objective function
    print("Testing acquisition function...")
    test_score = fasttext_objective(threshold=0.95, num_samples=5000)
    print(f"Test score: {test_score:.3f}")
