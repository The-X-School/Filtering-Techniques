import json
from transformers import GPT2Tokenizer, AutoTokenizer
import fasttext
import os

def train():
    with open('scores.json', 'r') as f:
        data = json.load(f)
    
    # Initialize GPT2 tokenizer
    gpt2tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=True)
    
    # Prepare training data for FastText
    # FastText expects text files with labels prefixed with __label__
    training_file = "fasttext_training.txt"
    
    with open(training_file, 'w', encoding='utf-8') as f:
        for score, token_ids in data:
            # Detokenize the token IDs back to text
            text = gpt2tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            # Write in FastText format: __label__score text
            f.write(f"__label__{score} {text}\n")
    
    # Train FastText model
    model = fasttext.train_supervised(
        input=training_file,
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        minCount=1,
        loss='ova'  # One-vs-all for multi-class classification
    )
    
    # Save the trained model
    model.save_model("trained_model.bin")
    
    # Clean up training file
    os.remove(training_file)
    
    print(f"FastText model trained and saved as 'trained_model.bin'")
    print(f"Number of training examples: {len(data)}")
    
    return model
    
    
