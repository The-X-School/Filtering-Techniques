from huggingface_hub import login
# login()  # Comment out for testing - you can uncomment and run when ready to authenticate

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

try:
    # Load tokenizer & model with correct attention backend
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("nvidia/NV-Retriever-v1")
    model = AutoModel.from_pretrained(
        "nvidia/NV-Retriever-v1",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    print("✓ Model and tokenizer loaded successfully!")
    
    # Load and sample dataset
    print("Loading dataset...")
    dataset = load_dataset("OptimalScale/ClimbLab", streaming=True)
    sample = list(dataset["train"].take(10))
    docs = ["passage: " + row["text"] for row in sample]
    print(f"✓ Loaded {len(docs)} document samples")
    
    # Tokenize and run
    print("Tokenizing and generating embeddings...")
    inputs = tokenizer(docs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
    
    # Mean-pool for embeddings
    mask = inputs["attention_mask"].float()
    masked = last_hidden * mask.unsqueeze(-1)
    embeds = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)
    
    print("✓ Embeddings generated successfully!")
    print("Embeddings shape:", embeds.shape)
    print("First embedding vector:", embeds[0][:10])
    
except Exception as e:
    print(f"❌ Error occurred: {e}")
    print("\nIf you see a 'GatedRepoError' or 'Unauthorized', you need to:")
    print("1. Uncomment the login() line at the top")
    print("2. Run the script and enter your HuggingFace token")
    print("3. Make sure you have access to the nvidia/NV-Retriever-v1 model")