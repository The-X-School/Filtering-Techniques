import os
import torch
import traceback # Import traceback for detailed error info
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login, get_token # Changed HfHub to get_token for direct use
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

# --- Upgrade huggingface_hub library ---
# This ensures you have the latest version that includes the necessary functions.
print("Upgrading huggingface_hub to the latest version...")
os.system("pip install --upgrade huggingface_hub")
print("✓ huggingface_hub upgrade complete.")

# --- Authentication ---
# Uncomment and run this part to log in.
# If you've already logged in previously in this environment and your token is valid
# (e.g., stored in ~/.cache/huggingface/token), you might not need to run login() again.
print("Attempting to log in to Hugging Face...")
try:
    login()
    print("✓ Successfully logged in to Hugging Face!")
except Exception as e:
    print(f"❌ Hugging Face login failed: {e}")
    print("Please ensure you have entered a valid token and have internet access.")
    # You might want to exit here if login is critical
    # import sys
    # sys.exit(1)

# Verify token presence (optional, but good for debugging)
if get_token(): # Used get_token() directly from huggingface_hub
    print("Hugging Face token detected.")
else:
    print("No Hugging Face token detected after login attempt. This might cause issues.")
    print("Please make sure your token is correctly set via `login()` or `HF_TOKEN` environment variable.")


try:
    # Load tokenizer & model with correct attention backend
    print("Loading tokenizer and model 'nvidia/NV-Retriever-v1'...")
    tokenizer = AutoTokenizer.from_pretrained("nvidia/NV-Retriever-v1")
    model = AutoModel.from_pretrained(
        "nvidia/NV-Retriever-v1",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    print("✓ Model and tokenizer loaded successfully!")
    
    # Load and sample dataset
    print("Loading dataset 'OptimalScale/ClimbLab'...")
    dataset = load_dataset("OptimalScale/ClimbLab", streaming=True)
    # Take more samples to ensure streaming works with a larger dataset if possible
    sample = list(dataset["train"].take(50)) # Increased sample size
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
    print("First embedding vector (first 10 elements):", embeds[0][:10])
    
except Exception as e:
    print(f"❌ Error occurred: {e}")
    print("\nDetailed traceback:")
    traceback.print_exc()
    print("\nIf you see a 'GatedRepoError' or 'Unauthorized', you need to:")
    print("1. Make sure you have access to the nvidia/NV-Retriever-v1 model")
    print("2. Ensure your HuggingFace token has the correct permissions")
    print("3. Check if the dataset 'OptimalScale/ClimbLab' is accessible")