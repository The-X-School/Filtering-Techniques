import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked semaphore objects")
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
import torch
import re
from math import log
from tqdm import tqdm

gpt2tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=True)

def detokenize(token_ids, tokenizer):
    """Detokenizes a list of token IDs into a cleaned string."""
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()

def reward(dataset, model_name, epoch, epsilon):
    # dataset is list of lists 
    print("Step 1: Loading model and tokenizer...")
    device = torch.device("cpu")
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model and tokenizer loaded successfully.")
    # detokenize, will be a list of strings
    print("Step 2: Detokenizing text...")
    dataset = [detokenize(doc, gpt2tokenizer) for doc in dataset]
    # tokenize
    print("Step 3: Tokenizing text...")
    encodings = tokenizer(dataset, truncation=True, padding=True, return_tensors="pt")
    # train the model
    print("Step 4: Training Model...")
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __getitem__(self, idx):
            item = {key: tensor[idx] for key, tensor in self.encodings.items()}
            item['labels'] = item['input_ids'].clone()
            return item
        def __len__(self):
            return self.encodings["input_ids"].shape[0]
        
    train_dataset = SimpleDataset(encodings)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epoch,
        per_device_train_batch_size=2,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    # download validation data
    print("Step 5: Downloading validation data...")
    val_ds = load_dataset("allenai/openbookqa", "additional")
    print("Step 6: Reformatting data...")
    val_ds = list(val_ds['test']) + list(val_ds['train']) + list(val_ds['validation'])
    questions = []
    for q in val_ds:
        cur_question = "Please answer with one of A, B, C, or D! Question: "
        cur_question += q['question_stem'] + '. '
        cur_question += "Fact: " + q['fact1'] + '. '
        for txt, lbl in zip(q['choices']['text'], q['choices']['label']):
            cur_question += lbl + ": " + txt + ", "
        questions.append((cur_question, q['answerKey']))
    # get the models accuracy
    print("Step 7: Getting model accuracy...")
    acc = 0
    model.eval()
    for i, question in enumerate(tqdm(questions, desc="Getting score...", total=len(questions))):
        input = tokenizer(question[0], return_tensors="pt").to(model.device)
        input_len = input.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **input,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Slice off the prompt
        gen_ids = outputs[0][input_len:]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Extract A/B/C/D from the output
        match = re.search(r"\b([ABCD])\b", answer.upper())
        pred = match.group(1) if match else "?"
        if pred == question[1]:
            acc += 1
            
    acc /= len(questions)
    # "normalize" the accuracy to reward smaller train sets
    print("Step 8: Normalizing accuracy...")
    reward_score = acc / (1 + log(len(dataset) * epsilon))
    return reward_score

def main():
    total_sample = 10
    print("Step 0.1: Loading dataset...")
    full_dataset = load_dataset("nvidia/ClimbLab", split="train", streaming=True)

    # Convert to list of token lists
    print("Step 0.2: Collecting sample...")
    dataset = []
    for i, item in enumerate(tqdm(full_dataset, desc="Collecting sample", total=total_sample)):
        if i >= total_sample:
            break
        dataset.append(item)

    print("Step 0.3: Reformatting sample...")
    for i, item in enumerate(tqdm(dataset, desc="Reformatting sample")):
        dataset[i] = list(item["tokens"])
    
    print(f"Loaded {len(dataset)} documents")
    print(dataset[0])

    print(reward(dataset, "distilbert/distilgpt2", 1, 0.1))

main()
