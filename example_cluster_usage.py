#!/usr/bin/env python3

"""
Simple example of using ClimbLabClusterStreamer to load, detokenize, and stream clusters.
"""

from climblab_cluster_streamer import ClimbLabClusterStreamer

def main():
    print("🔧 ClimbLab Cluster Streaming Example")
    print("=" * 50)
    
    # 1. Initialize the streamer
    print("1. Initializing ClimbLab Cluster Streamer...")
    streamer = ClimbLabClusterStreamer(
        cache_dir="./cache",
        use_auth=False,  # Set to True if you need HF authentication
        tokenizer_name="microsoft/DialoGPT-medium"  # Change tokenizer if needed
    )
    
    # 2. Process a sample of the dataset
    print("\n2. Processing dataset sample...")
    stats = streamer.process_and_index_dataset(
        num_samples=500,  # Adjust sample size as needed
        save_processed=True,
        output_file="data/my_climblab_clusters.jsonl"
    )
    
    print(f"✅ Processed {stats['total_samples']:,} samples")
    print(f"   Total tokens: {stats['total_tokens']:,}")
    print(f"   Unique clusters: {stats['unique_clusters']}")
    
    # 3. Explore clusters
    print("\n3. Available clusters:")
    clusters = streamer.list_clusters()
    
    for cluster_id in clusters:
        cluster_stats = streamer.get_cluster_stats(cluster_id)
        print(f"   📁 {cluster_id}: {cluster_stats['sample_count']} samples, "
              f"avg {cluster_stats['avg_tokens']:.1f} tokens")
    
    # 4. Stream specific cluster samples
    if clusters:
        target_cluster = clusters[0]  # Pick first cluster
        print(f"\n4. Streaming samples from cluster '{target_cluster}':")
        
        # Stream with filters
        sample_count = 0
        for sample in streamer.stream_cluster(
            cluster_id=target_cluster,
            limit=5,  # Only get 5 samples
            min_tokens=100,  # Only samples with 100+ tokens
            max_tokens=1000  # Only samples with <1000 tokens
        ):
            sample_count += 1
            text = sample.get("detokenized_text", "No text available")
            token_count = sample.get("token_count", 0)
            
            print(f"\n   📄 Sample {sample_count} ({token_count} tokens):")
            print(f"      {text[:150]}...")
    
    # 5. Save a cluster to file
    if clusters:
        print(f"\n5. Saving cluster '{target_cluster}' to file...")
        saved_count = streamer.save_cluster_to_file(
            cluster_id=target_cluster,
            output_file=f"data/my_cluster_{target_cluster}.jsonl",
            limit=50,  # Save up to 50 samples
            min_tokens=50  # Only samples with 50+ tokens
        )
        print(f"✅ Saved {saved_count} samples to file")
    
    print("\n🎉 Done! Your cluster data is ready to use.")


if __name__ == "__main__":
    main() 