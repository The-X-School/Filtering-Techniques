import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
import warnings

warnings.filterwarnings("ignore")
os.environ["TORCH_LOGS"] = "0"

def patch_model(model):
    def patch_recursive(module):
        if not hasattr(module, '_attn_implementation'):
            module._attn_implementation = "eager"
        for attr_name in ['encoder', 'decoder', 'model', 'transformer']:
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

login()

try:
    tokenizer = AutoTokenizer.from_pretrained("nvidia/NV-Retriever-v1")
    model = AutoModel.from_pretrained("nvidia/NV-Retriever-v1", trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu")
    model = patch_model(model)
    model_type = "nv-retriever"
except:
    os.system("pip install sentence-transformers")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = None
    model_type = "sentence-transformers"

dataset = load_dataset("OptimalScale/ClimbLab", streaming=True, cache_dir="./cache")
sample = []
for i, item in enumerate(dataset["train"]):
    if i >= 1000000:
        break
    sample.append(item)

docs = ["passage: " + row["text"] for row in sample]
original_count = len(docs)

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
        os.system("pip install sentence-transformers")
        from sentence_transformers import SentenceTransformer
        fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings_np = fallback_model.encode(docs, convert_to_numpy=True)

filtered_indices = []

# Stricter clustering (lower eps = tighter clusters)
clustering = DBSCAN(eps=0.4, min_samples=2, metric='cosine')
cluster_labels = clustering.fit_predict(embeddings_np)
outliers = [i for i, label in enumerate(cluster_labels) if label == -1]

# Remove bottom 30% by embedding magnitude (stricter)
embedding_norms = np.linalg.norm(embeddings_np, axis=1)
norm_threshold = np.percentile(embedding_norms, 30)

# Remove bottom 30% by similarity to centroid (stricter)
centroid = np.mean(embeddings_np, axis=0)
similarities = cosine_similarity(embeddings_np, centroid.reshape(1, -1)).flatten()
similarity_threshold = np.percentile(similarities, 30)

# Additional filter: document length
doc_lengths = [len(doc.split()) for doc in docs]
length_threshold = np.percentile(doc_lengths, 20)  # Remove shortest 20%

for i in range(len(docs)):
    is_outlier = i in outliers
    low_magnitude = embedding_norms[i] <= norm_threshold
    low_similarity = similarities[i] <= similarity_threshold
    too_short = doc_lengths[i] <= length_threshold
    
    if is_outlier or low_magnitude or low_similarity or too_short:
        filtered_indices.append(i)

filtered_count = len(filtered_indices)

print(f"Filtered out {filtered_count} files out of {original_count} total files") 