

import torch
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np

def load_jsonl(file_path: str):
    """Load a JSONL dataset from the specified file path."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def main():
    # Configuration
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dataset_path = "data/climblab_processed_clusters.jsonl"
    output_dir = "data/filtered_output"
    output_file = os.path.join(output_dir, "embeddings.jsonl")
    text_field = "detokenized_text"
    batch_size = 32

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    print(f"Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model.half()  # Convert model to half-precision (float16)
    
    print("Model loaded successfully.")

    # Load the dataset
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset from: {dataset_path}")
    dataset = load_jsonl(dataset_path)
    print(f"Loaded {len(dataset)} samples.")

    # Extract texts
    texts = [sample.get(text_field, "") for sample in dataset]
    texts = [text for text in texts if text]
    print(f"Found {len(texts)} non-empty texts to process.")

    if not texts:
        print("No texts to process. Exiting.")
        return

    # Generate embeddings in chunks
    print("Generating embeddings in chunks...")
    chunk_size = 100  # Define chunk size
    total_texts = len(texts)

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(0, total_texts, chunk_size):
            chunk_texts = texts[i:i + chunk_size]
            print(f"Processing chunk {i // chunk_size + 1}/{(total_texts + chunk_size - 1) // chunk_size} ({len(chunk_texts)} texts)...")
            chunk_embeddings = model.encode(chunk_texts, batch_size=batch_size, show_progress_bar=False)

            for j in range(len(chunk_texts)):
                output_record = {
                    "text": chunk_texts[j],
                    "embedding": chunk_embeddings[j].tolist()
                }
                f.write(json.dumps(output_record) + '\n')
    print("Embeddings generated and saved successfully.")

    print("Processing complete.")

if __name__ == "__main__":
    main()

