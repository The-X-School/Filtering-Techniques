import os
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
import warnings
import json
import gc

warnings.filterwarnings("ignore")
os.environ["TORCH_LOGS"] = "0"

def generate_embeddings(sample_size=1000, target_tokens=None, output_path=None):
    """
    Loads the ClimbLab dataset, generates embeddings, and optionally saves them.
    
    Args:
        sample_size: Number of documents to sample for embedding generation (used if target_tokens is None)
        target_tokens: Target number of tokens to process (overrides sample_size)
        output_path: Optional path to save the embeddings and texts
    
    Returns:
        tuple: (embeddings_np, texts, token_count)
    """
    login()

    # --- 1. Load Model ---
    print("Loading embedding model...")
    # Install and load sentence-transformers
    os.system("pip install -q sentence-transformers")
    from sentence_transformers import SentenceTransformer
    
    print("Loading embedding model: all-MiniLM-L6-v2")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # --- 2. Load and Sample Dataset ---
    dataset = load_dataset("nvidia/ClimbLab", streaming=True, cache_dir="./cache")
    
    if target_tokens:
        print(f"Loading documents to reach target of {target_tokens:,} tokens from nvidia/ClimbLab...")
        sample = []
        total_tokens = 0
        
        for row in dataset["train"]:
            # Count tokens in this document
            if "tokens" in row:
                doc_tokens = len(row["tokens"])
            else:
                # Fallback: estimate tokens as roughly 0.75 * word count
                text = row.get("text", "")
                doc_tokens = int(len(text.split()) * 0.75)
            
            sample.append(row)
            total_tokens += doc_tokens
            
            if total_tokens >= target_tokens:
                print(f"Reached target tokens: {total_tokens:,} tokens with {len(sample)} documents")
                break
    else:
        print(f"Loading and sampling {sample_size} items from nvidia/ClimbLab...")
        sample = list(dataset["train"].take(sample_size))

    # Convert token arrays to text for sentence-transformers
    docs = []
    for row in sample:
        if "tokens" in row:
            # Try to decode tokens if available
            try:
                # Use a simple tokenizer for decoding
                from transformers import AutoTokenizer
                temp_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                text = temp_tokenizer.decode(row["tokens"], skip_special_tokens=True)
                docs.append(text)
            except:
                # Fallback: join tokens as string
                docs.append(" ".join(map(str, row["tokens"])))
        else:
            # Use text field if available
            text = row.get("text", "")
            docs.append(text)
    
    print(f"Processed {len(docs)} documents.")

    # --- 3. Generate Embeddings ---
    print("Generating embeddings...")
    embeddings_np = model.encode(docs, convert_to_numpy=True)
    print(f"Generated embeddings with shape: {embeddings_np.shape}")
    
    # Calculate total token count
    total_tokens = 0
    for row in sample:
        if "tokens" in row:
            total_tokens += len(row["tokens"])
        else:
            # Fallback: estimate tokens
            text = row.get("text", "")
            total_tokens += int(len(text.split()) * 0.75)
    
    print(f"Total tokens processed: {total_tokens:,}")

    # --- 4. Save if requested ---
    if output_path:
        print(f"Saving embeddings and texts to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save embeddings as numpy array
        embeddings_file = output_path.replace('.json', '_embeddings.npy')
        np.save(embeddings_file, embeddings_np)
        
        # Save texts and metadata
        with open(output_path, 'w') as f:
            data = {
                'texts': docs,
                'token_count': total_tokens,
                'embedding_shape': embeddings_np.shape,
                'embedding_file': embeddings_file
            }
            json.dump(data, f, indent=2)
        
        print(f"Embeddings saved to {embeddings_file}")
        print(f"Metadata saved to {output_path}")
    
    return embeddings_np, docs, total_tokens

def load_embeddings(metadata_path):
    """
    Load previously generated embeddings and texts.
    
    Args:
        metadata_path: Path to the metadata JSON file
    
    Returns:
        tuple: (embeddings_np, texts, token_count)
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    embeddings_np = np.load(metadata['embedding_file'])
    texts = metadata['texts']
    token_count = metadata['token_count']
    
    return embeddings_np, texts, token_count

if __name__ == '__main__':
    # This part allows the script to be run directly for a quick test.
    print("Running a standalone test of the embedding generation process...")
    embeddings, texts, token_count = generate_embeddings(
        sample_size=200,  # Using a small sample for quick testing
        output_path="data/embeddings/test_embeddings.json"
    )
    print(f"Generated {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions")
    print(f"Processed {token_count:,} tokens")
    print("Standalone test finished.") 