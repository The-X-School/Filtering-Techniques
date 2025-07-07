import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from huggingface_hub import login
import warnings
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from peft import get_peft_model, LoraConfig, TaskType
import random

warnings.filterwarnings("ignore")
os.environ["TORCH_LOGS"] = "0"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

sample_size = 5
output_path = "data/embeddings/dora_finetuned_embeddings.json"
model_name = 'nvidia/NV-Embed-v2'
dora_dataset = 'data4elm/ELMB-FunctionCalling'
decoder_tokenizer = "microsoft/DialoGPT-medium"
cache_dir = "./cache"
search_top_k = 3
test_query = "call a function"

num_epochs = 3
learning_rate = 2e-5
batch_size = 8
max_length = 512

login()
dora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    use_dora=True,
    task_type=TaskType.FEATURE_EXTRACTION
)

base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision="main")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = get_peft_model(base_model, dora_config)
model.print_trainable_parameters()

class ContrastiveDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        query = item.get('query', item.get('instruction', ''))
        positive = item.get('positive', item.get('response', ''))
        
        if not query or not positive:
            query = "default query"
            positive = "default response"
        
        neg_idx = random.randint(0, len(self.data) - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.data) - 1)
        negative = self.data[neg_idx].get('positive', self.data[neg_idx].get('response', ''))
        
        if not negative:
            negative = "default negative"
        
        query_tokens = self.tokenizer(query, truncation=True, padding='max_length', 
                                    max_length=self.max_length, return_tensors='pt')
        pos_tokens = self.tokenizer(positive, truncation=True, padding='max_length',
                                  max_length=self.max_length, return_tensors='pt')
        neg_tokens = self.tokenizer(negative, truncation=True, padding='max_length',
                                  max_length=self.max_length, return_tensors='pt')
        
        return {
            'query': {k: v.squeeze(0) for k, v in query_tokens.items()},
            'positive': {k: v.squeeze(0) for k, v in pos_tokens.items()},
            'negative': {k: v.squeeze(0) for k, v in neg_tokens.items()}
        }

def get_embeddings(model, input_dict):
    outputs = model(**input_dict)
    if hasattr(outputs, 'last_hidden_state'):
        embeddings = outputs.last_hidden_state.mean(dim=1)
    elif hasattr(outputs, 'pooler_output'):
        embeddings = outputs.pooler_output
    else:
        embeddings = outputs[0].mean(dim=1)
    return F.normalize(embeddings, p=2, dim=1)

def contrastive_loss(query_emb, pos_emb, neg_emb, temperature=0.05):
    pos_sim = torch.sum(query_emb * pos_emb, dim=1) / temperature
    neg_sim = torch.sum(query_emb * neg_emb, dim=1) / temperature
    
    logits = torch.stack([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
    return F.cross_entropy(logits, labels)

dataset = load_dataset("nvidia/ClimbLab", streaming=True, cache_dir=cache_dir)
dora_datasets = load_dataset(dora_dataset, streaming=True, cache_dir=cache_dir)

sample = list(dataset["train"].take(sample_size))
dora_sample = list(dora_datasets["train"].take(sample_size))

temp_tokenizer = AutoTokenizer.from_pretrained(decoder_tokenizer)
training_data = []

for row in sample:
    if "tokens" in row:
        text = temp_tokenizer.decode(row["tokens"], skip_special_tokens=True)
    else:
        text = row.get("text", "")
    
    if text:
        sentences = text.split('. ')
        if len(sentences) >= 2:
            training_data.append({
                'query': sentences[0],
                'positive': '. '.join(sentences[1:])
            })

for row in dora_sample:
    instruction = row.get('instruction', row.get('query', ''))
    response = row.get('response', row.get('answer', ''))
    
    if instruction and response:
        training_data.append({
            'query': instruction,
            'positive': response
        })

if len(training_data) == 0:
    exit(1)

train_dataset = ContrastiveDataset(training_data, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
num_training_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

model.train()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        query_inputs = {k: v.to(device) for k, v in batch['query'].items()}
        pos_inputs = {k: v.to(device) for k, v in batch['positive'].items()}
        neg_inputs = {k: v.to(device) for k, v in batch['negative'].items()}
        
        query_emb = get_embeddings(model, query_inputs)
        pos_emb = get_embeddings(model, pos_inputs)
        neg_emb = get_embeddings(model, neg_inputs)
        
        loss = contrastive_loss(query_emb, pos_emb, neg_emb)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()

model.save_pretrained("./dora_finetuned_model")

model.eval()
docs = []
for row in sample[:10]:
    if "tokens" in row:
        text = temp_tokenizer.decode(row["tokens"], skip_special_tokens=True)
    else:
        text = row.get("text", "")
    
    if text:
        docs.append(text)

if not docs:
    docs = ["Sample document 1", "Sample document 2", "Sample document 3"]

embeddings_list = []
with torch.no_grad():
    for doc in docs:
        inputs = tokenizer(doc, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        embedding = get_embeddings(model, inputs)
        embeddings_list.append(embedding.cpu().numpy())

if embeddings_list:
    embeddings_np = np.vstack(embeddings_list)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        embeddings_file = output_path.replace('.json', '_embeddings.npy')
        np.save(embeddings_file, embeddings_np)
        
        with open(output_path, 'w') as f:
            data = {
                'texts': docs,
                'embedding_shape': embeddings_np.shape,
                'embedding_file': embeddings_file,
                'model_used': f"{model_name}_dora_finetuned",
                'training_samples': len(training_data),
                'epochs': num_epochs
            }
            json.dump(data, f, indent=2)
    
    if docs:
        query_inputs = tokenizer(test_query, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
        
        with torch.no_grad():
            query_embedding = get_embeddings(model, query_inputs).cpu().numpy()
        
        similarities = np.dot(embeddings_np, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:search_top_k]

        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                'rank': i + 1,
                'text': docs[idx],
                'similarity': float(similarities[idx])
            })