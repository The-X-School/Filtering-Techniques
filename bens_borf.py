from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
import torch

gpt2tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=True)

def detokenize(token_ids: list[int], tokenizer: AutoTokenizer) -> str:
    """Detokenizes a list of token IDs into a cleaned string."""
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()

def reward(dataset, model_name, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset is list of lists
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # detokenize, will be a list of strings
    dataset = [detokenize(doc, gpt2tokenizer) for doc in dataset]
    # tokenize
    encodings = tokenizer(dataset, truncation=True, padding=True, return_tensors="pt")
    # train the model
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
        per_device_train_batch_size=8,
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
    val_ds = load_dataset("allenai/openbookqa", "additional")
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
    acc = 0
    model.eval()
    for question in questions:
        input = tokenizer(question[0], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **input,
                max_new_tokens=10,
                do_sample=False,  # deterministic output
                pad_token_id=tokenizer.eos_token_id
            )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ans_strip = answer.strip().upper()
        pred = ans_strip[0] if ans_strip else "?"
        if pred == question[1]:
            acc += 1
    acc /= len(questions)
    # "normalize" the accuracy to reward smaller train sets
    reward_score = acc / (1 + 0.01 * len(dataset))
    return reward_score
