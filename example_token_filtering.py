#!/usr/bin/env python3
"""
Example script showing how to use token-based filtering.

This demonstrates how to filter a specific number of tokens instead of 
a fixed number of documents.
"""

from NV_retriever import filter_and_save

def main():
    # Example 1: Filter to get exactly 1 million tokens
    print("=== Example 1: 1 Million Tokens ===")
    filter_and_save(
        output_path="data/filtered_1M_tokens.jsonl",
        target_tokens=1_000_000,  # 1M tokens
        eps=0.4,
        norm_percentile=30,
        similarity_percentile=30,
        length_percentile=20
    )
    
    # Example 2: Filter to get 10 million tokens (good for testing)
    print("\n=== Example 2: 10 Million Tokens ===")
    filter_and_save(
        output_path="data/filtered_10M_tokens.jsonl",
        target_tokens=10_000_000,  # 10M tokens
        eps=0.5,
        norm_percentile=25,
        similarity_percentile=25,
        length_percentile=15
    )
    
    # Example 3: Filter to get 1 billion tokens (realistic for competition)
    print("\n=== Example 3: 1 Billion Tokens ===")
    filter_and_save(
        output_path="data/filtered_1B_tokens.jsonl",
        target_tokens=1_000_000_000,  # 1B tokens
        eps=0.6,
        norm_percentile=20,
        similarity_percentile=20,
        length_percentile=10
    )
    
    # Example 4: Use traditional document count (fallback)
    print("\n=== Example 4: Traditional Document Count ===")
    filter_and_save(
        output_path="data/filtered_1000_docs.jsonl",
        sample_size=1000,  # 1000 documents
        target_tokens=None,  # Use sample_size instead
        eps=0.4,
        norm_percentile=30,
        similarity_percentile=30,
        length_percentile=20
    )

if __name__ == "__main__":
    main() 