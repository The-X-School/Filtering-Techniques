import json, os
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from collections import deque

# Lo ad dataset and generate text chunks
items = [json.loads(l) for l in open("climblab_sample.jsonl")]
chunks = [item.get("text") for item in items if item.get("text") is not None and isinstance(item.get("text"), str)]

# Load embedding model
embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Compute chunk embeddings
inputs = tokenizer(chunks, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    embeddings = embed_model(**inputs).last_hidden_state.mean(1).cpu().numpy()

# Build FAISS IVF index with pruning (store only centroids)
d = embeddings.shape[1]
nlist, nprobe = 100, 10
quantizer = faiss.IndexFlatL2(d)
ivf_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
ivf_index.train(embeddings)
ivf_index.add(embeddings)

# Setup cache and metadata
cache, CACHE_SIZE = {}, 100
cache_order = deque()

# Query loop
def embed_text(text):
    inp = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        return embed_model(**inp).last_hidden_state.mean(1).cpu().numpy()

def retrieve(query, topk=5):
    q_emb = embed_text(query)
    ivf_index.nprobe = nprobe
    D, I = ivf_index.search(q_emb, topk)
    chunks_out = []
    for idx in I[0]:
        if idx in cache:
            emb = cache[idx]
        else:
            emb = embeddings[idx]  # pre-computed
            cache[idx] = emb
            cache_order.append(idx)
            if len(cache_order) > CACHE_SIZE:
                evict = cache_order.popleft()
                del cache[evict]
        chunks_out.append(chunks[idx])
    return chunks_out

# Use RAG: feed retrieved chunks into LLM
from transformers import pipeline
llm = pipeline("text-generation", model="gpt2")

def rag_answer(query):
    relevant_chunks = retrieve(query)
    context = "\n".join(relevant_chunks)
    prompt = f"Context:\n{context}\n\nUser: {query}\nAssistant:"
    return llm(prompt, max_length=200)[0]["generated_text"]

# Run on sample queries
for q in ["What is climbLAB?", "Define helpful instruction for AI dataset"]:
    print(rag_answer(q))
