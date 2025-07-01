import torch 
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.optim import AdamW
from datasets import load_dataset
import json
from tqdm import tqdm
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType

dataset = load_dataset("nvidia/ClimbLab", split="train", streaming=True)
dora_dataset = load_dataset("data4elm/ELMB-FunctionCalling", split="train", streaming=True)
tokenizer = AutoTokenizer.from_pretrained('nvidia/NV-Retriever-v1', trust_remote_code=True)



config = AutoConfig.from_pretrained("nvidia/NV-Retriever-v1", trust_remote_code=True)
config._attn_implementation = "eager"

model = AutoModel.from_pretrained(
    "nvidia/NV-Retriever-v1",
    config=config,
    trust_remote_code=True,
    torch_dtype="auto"
)

dora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    use_dora=True,
)

model = get_peft_model(model, dora_config)
model.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
batch_size = 4
num_epochs = 3

def prepare_batch(examples):
    texts = []     
    
    for ex in examples:         
        parts = []
        
        if "question" in ex and ex["question"]:
            parts.append(f"Question: {ex['question']}")
        
        if "correct_answer_content" in ex and ex["correct_answer_content"]:
            parts.append(f"Answer: {ex['correct_answer_content']}")
        if "ability" in ex and ex["ability"]:
            parts.append(f"Ability: {ex['ability']}")
        
        if parts:
            text_content = " ".join(parts)
            texts.append(f"passage: {text_content}")
        else:
            print(f"Warning: No suitable content found in example: {list(ex.keys())}")
    
    # print(f"Prepared {len(texts)} texts from {len(examples)} examples")
    return texts

dora_dataset_size = 24200
num_batches = dora_dataset_size // batch_size

for epoch in range(num_epochs):
    shuffled_dataset = dora_dataset.shuffle(buffer_size=10000, seed=epoch)
    batch_iterator = shuffled_dataset.iter(batch_size=batch_size)
    progress_bar = tqdm(batch_iterator, total=num_batches, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_dict in progress_bar:
        batch_examples = [dict(zip(batch_dict, t)) for t in zip(*batch_dict.values())]
        texts = prepare_batch(batch_examples)
        
        if not texts:
            print(f"Skipping batch because no valid texts were prepared.")
            continue
        
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        optimizer.zero_grad()
        
        outputs = model(**inputs)
        
        if hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state[:, 0]
        elif isinstance(outputs, torch.Tensor):
            embeddings = outputs
        elif isinstance(outputs, tuple):
            embeddings = outputs[0]
        else:
            raise TypeError(f"Unsupported model output type: {type(outputs)}")

        if embeddings.dim() == 3 and embeddings.shape[1] > 1:
             embeddings = embeddings[:, 0]
        
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
            
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)
        labels = torch.arange(similarity_matrix.size(0)).to(device)
        
        loss = F.cross_entropy(similarity_matrix * 20, labels)
        
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())

model.save_pretrained("./nv_retriever_dora_finetuned")

def tokens_to_text(tokens):
    return tokenizer.decode(tokens, skip_special_tokens=True)

texts = []
max_samples = 10
for i, example in enumerate(dataset):
    if i >= max_samples: 
        break
        
    if "tokens" in example:
        text = tokens_to_text(example["tokens"])
    elif "text" in example:
        text = example["text"]
    else:
        continue
    
    texts.append(f"passage: {text}")

model.eval()
embeddings = []

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_emb = model(**batch)
        
        if hasattr(batch_emb, 'last_hidden_state'):
            embeddings_tensor = batch_emb.last_hidden_state[:, 0]
        elif isinstance(batch_emb, torch.Tensor):
            embeddings_tensor = batch_emb
        elif isinstance(batch_emb, tuple):
            embeddings_tensor = batch_emb[0]
        else:
            raise TypeError(f"Unsupported model output type: {type(batch_emb)}")

        if embeddings_tensor.dim() == 3 and embeddings_tensor.shape[1] > 1:
            embeddings_tensor = embeddings_tensor[:, 0]

        if embeddings_tensor.dim() == 1:
            embeddings_tensor = embeddings_tensor.unsqueeze(0)
        
        embeddings.append(embeddings_tensor.cpu())

embeddings = torch.cat(embeddings, dim=0)
embeddings_list = embeddings.tolist()

with open("climblab_nv_retriever_dora_embeddings.jsonl", "w") as f:
    for i, embedding in enumerate(embeddings_list):
        json.dump({"id": i, "embedding": embedding}, f)
        f.write("\n")