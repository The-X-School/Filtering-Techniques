import farmhash #pip install pyfarmhash
from collections import defaultdict
from datasketch import MinHash, MinHashLSH #pip install datasketch
from datasets import load_dataset, IterableDataset
import json
from tqdm import tqdm  # Added tqdm import
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

"""
conda activate py310
huggingface-cli login
~/miniconda3/envs/py310/bin/python deduplicate_streaming.py
"""

def deduplicate_lsh(dataset, num_perm=128, threshold=0.8):
    """
    dataset: list of documents, each is a list of tokens (e.g., words or n-grams)
    num_perm: number of permutations (hash functions) for MinHash
    threshold: Jaccard similarity threshold for candidate pairs
    
    Returns:
        List of pairs (i, j) of document indices considered similar.
    """
    print("Step 3.1:")
    # 1. Create MinHash objects for each document
    minhashes = []
    for i, tokens in enumerate(tqdm(dataset, desc="MinHashing docs")):
        m = MinHash(num_perm=num_perm)
        for token in tokens:
            m.update(str(token).encode('utf8'))
        minhashes.append(m)
    
    print("Step 3.2:")
    # 2. Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, m in enumerate(tqdm(minhashes, desc="Inserting to LSH")):
        lsh.insert(i, m)
    
    print("Step 3.3:")
    # 3. Find similar pairs
    similar_pairs = []
    for i, m in enumerate(tqdm(minhashes, desc="Querying LSH")):
        result = lsh.query(m)
        for j in result:
            if i < j:  # avoid duplicates and self-pairs
                similar_pairs.append((i, j))
    
    # Print example similar pairs
    if similar_pairs:
        print("\nExample similar pairs from LSH:")
        for i, j in similar_pairs[:2]:  # Show first 2 pairs
            print(f"\nPair {i} and {j}:")
    print("\n")
    
    return similar_pairs

def dfs(adjList, visited, node):
    visited[node] = True
    for neighbor in adjList[node]:
        if not visited[neighbor]:
            dfs(adjList, visited, neighbor)

def deduplicate(dataset, threshold=0.8, ngram_size=10, num_perm = 128):
    print("Step 1: Generating n-grams...")
    N = ngram_size
    nGrams = [[] for _ in range(len(dataset))]
    for row in tqdm(range(len(dataset)), desc="Generating n-grams"):
        for i in range(len(dataset[row]) - N + 1):
            nGrams[row].append(dataset[row][i : i + N])

    print("Step 2: Preparing token sets for MinHash...")
    hashed_ngrams = [[] for _ in range(len(dataset))]
    for row in tqdm(range(len(dataset)), desc="Hashing n-grams"):
        for ngram in nGrams[row]:
            hashed_ngrams[row].append(farmhash.hash64(" ".join(map(str, ngram))))

    print("Step 3: Comparing candidate documents...")
    edgeList = deduplicate_lsh(hashed_ngrams, threshold=threshold, num_perm = num_perm)

    print("Step 4: Building adjacency list...")
    adjList = [[] for _ in range(len(dataset))]
    for pair in tqdm(edgeList, desc="Building adjacency list"):
        adjList[pair[0]].append(pair[1])
        adjList[pair[1]].append(pair[0])
    
    print("Step 5: Connected Components Algorithm...")
    filtered_dataset = []
    visited = [False] * len(dataset)
    for i in tqdm(range(len(dataset)), desc="Finding connected components"):
        if not visited[i]:
            dfs(adjList, visited, i)
            filtered_dataset.append(dataset[i]) # normally pick the highest quality document
    return filtered_dataset

def main():
    # Load the processed data   
    print("Step 0.1: Loading dataset...")
    dataset = load_dataset("nvidia/ClimbLab", split="train", streaming=True)

    # Convert to list of token lists
    print("Step 0.2: Collecting sample...")
    total_sample = 100000
    sample = []
    for i, item in enumerate(tqdm(dataset, desc="Collecting sample", total=total_sample)):
        if i >= total_sample:
            break
        sample.append(item)
    print("Step 0.3: Reformatting sample...")
    for i, item in enumerate(tqdm(sample, desc="Reformatting sample")):
        sample[i] = list(item["tokens"])
    
    print(f"Loaded {len(sample)} documents")
    
    # Run deduplication
    print("Step 0.4: Running deduplication...")
    filtered_dataset = deduplicate(sample, threshold=0.5, ngram_size=2, num_perm = 128)
    
    print(f"\nReduced from {len(sample)} to {len(filtered_dataset)} documents")
    
    # Save the filtered dataset
    print("\nSaving filtered dataset...")
    with open('filtered_tokenized.txt', 'w') as f:
        json.dump(filtered_dataset, f, indent=4)
    
    print("Done! Filtered dataset saved to filtered_tokenized.txt")

if __name__ == "__main__":
    main()
