import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
import warnings

# Suppress torchvision warnings and compatibility issues
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TORCH_LOGS"] = "0"

# Check for torchvision compatibility issues
def check_torch_compatibility():
    """Check and fix torch/torchvision compatibility"""
    try:
        torch_version = torch.__version__
        print(f"PyTorch version: {torch_version}")
        return True
    except Exception as e:
        print(f"PyTorch compatibility issue: {e}")
        return False

# Comprehensive monkey patch to fix the _attn_implementation issue
def patch_model_if_needed(model):
    """Add missing _attn_implementation attribute recursively to all model components"""
    def patch_recursive(module):
        if not hasattr(module, '_attn_implementation'):
            module._attn_implementation = "eager"
        
        for attr_name in ['encoder', 'decoder', 'model', 'transformer', 'bert', 'roberta']:
            if hasattr(module, attr_name):
                attr = getattr(module, attr_name)
                if hasattr(attr, '__dict__'):
                    patch_recursive(attr)
        
        for child in module.children():
            patch_recursive(child)
    
    if not hasattr(model, '_attn_implementation'):
        model._attn_implementation = "eager"
    patch_recursive(model)
    return model

# Check PyTorch compatibility
if not check_torch_compatibility():
    print("PyTorch compatibility issues detected. Continuing with caution...")

# Login to Hugging Face
login()

try:
    # Try NV-Retriever first
    tokenizer = AutoTokenizer.from_pretrained("nvidia/NV-Retriever-v1")
    model = AutoModel.from_pretrained("nvidia/NV-Retriever-v1",
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model = patch_model_if_needed(model)
    model_type = "nv-retriever"
except:
    # Fallback to sentence-transformers
    os.system("pip install sentence-transformers")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = None
    model_type = "sentence-transformers"

# Load dataset
dataset = load_dataset("OptimalScale/ClimbLab", streaming=True, cache_dir="./cache")
sample = []
for i, item in enumerate(dataset["train"]):
    if i >= 30:
        break
    sample.append(item)

docs = ["passage: " + row["text"] for row in sample]
original_count = len(docs)

# Generate embeddings
if model_type == "sentence-transformers":
    embeddings_np = model.encode(docs, convert_to_numpy=True)
else:
    try:
        inputs = tokenizer(docs, padding=True, truncation=True, return_tensors="pt")
        if next(model.parameters()).device.type == 'cpu':
            inputs = {k: v.cpu() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
        
        mask = inputs["attention_mask"].float()
        masked = last_hidden * mask.unsqueeze(-1)
        embeds = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        embeddings_np = embeds.cpu().numpy()
    except:
        # Fallback to sentence-transformers if NV-Retriever fails during inference
        os.system("pip install sentence-transformers")
        from sentence_transformers import SentenceTransformer
        fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings_np = fallback_model.encode(docs, convert_to_numpy=True)

# Quality filtering based on embeddings
filtered_indices = []

# Remove outliers using DBSCAN clustering
clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
cluster_labels = clustering.fit_predict(embeddings_np)
non_outliers = [i for i, label in enumerate(cluster_labels) if label != -1]

# Filter by embedding magnitude
embedding_norms = np.linalg.norm(embeddings_np, axis=1)
norm_threshold = np.percentile(embedding_norms, 25)

# Filter by cosine similarity to centroid
centroid = np.mean(embeddings_np, axis=0)
similarities = cosine_similarity(embeddings_np, centroid.reshape(1, -1)).flatten()
similarity_threshold = np.percentile(similarities, 25)

# Combine all filtering criteria
for i in range(len(docs)):
    if (i in non_outliers and 
        embedding_norms[i] > norm_threshold and 
        similarities[i] > similarity_threshold):
        filtered_indices.append(i)

filtered_count = original_count - len(filtered_indices)

print(f"Filtered out {filtered_count} files out of {original_count} total files")