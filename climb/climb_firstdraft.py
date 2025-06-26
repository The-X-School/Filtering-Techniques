import random
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import (
    GPT2TokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

"""
/Users/Benj/miniconda3/envs/climb310/bin/python climb.py
"""

class MixtureLossPredictor(nn.Module):
    def __init__(self, clusters):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(clusters, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

def collate_fn(batch):
    # batch is list of lists of tokens
    input_ids = [torch.tensor(x) for x in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=50257)  # GPT-2 pad token
    attention_mask = (input_ids_padded != 50257).long()
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": input_ids_padded,
    }

def chunk_tokens(tokens, chunk_size=512):
    return [tokens[i : min(i + chunk_size, len(tokens))] for i in range(0, len(tokens), chunk_size)]

def test_mixture(mixture, dataset, epochs, datasetSize):
    ds = []
    for i in range(len(mixture)):
        amt = min(int(datasetSize * mixture[i]), len(dataset[i]))
        ds.extend(random.sample(dataset[i], amt))

    # Chunk all token lists into fixed size chunks
    chunked_ds = []
    for tokens in ds:
        chunks = chunk_tokens(tokens, chunk_size=512)
        chunked_ds.extend(chunks)  # now chunked_ds is list of token lists (chunks of text)

    # Train-test split on chunks
    train_chunks, val_chunks = train_test_split(chunked_ds, test_size=0.1)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=256,
        n_layer=4,
        n_head=4,
        n_positions=512,
        pad_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)  # train from scratch

    # Create HF datasets directly from list of token lists
    train_dataset = Dataset.from_dict({"input_ids": train_chunks})
    val_dataset = Dataset.from_dict({"input_ids": val_chunks})

    training_args = TrainingArguments(
        output_dir="./tmp",
        evaluation_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=10,
        logging_dir="./logs",
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    return mixture, eval_result["eval_loss"]

def find_best_mixture_via_predictor(predictor, clusters, steps=500, lr=0.1):
    """
    Use gradient descent to find the mixture that minimizes predicted loss.
    The mixture is constrained to lie on the probability simplex (sum = 1, all ≥ 0).
    """
    predictor.eval()

    # Initialize with random valid mixture (on simplex)
    mix = torch.rand(clusters, requires_grad=True)
    mix = mix / mix.sum()  # normalize
    mix = mix.requires_grad_(True)

    optimizer = torch.optim.Adam([mix], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()

        norm_mix = torch.clamp(mix, min=1e-6)
        norm_mix = norm_mix / norm_mix.sum()  # stay on the simplex

        pred_loss = predictor(norm_mix.unsqueeze(0))  # (1, clusters) -> (1, 1)
        pred_loss.backward()
        optimizer.step()

        # Project back to simplex (optional)
        with torch.no_grad():
            mix.data = torch.clamp(mix.data, min=1e-6)
            mix.data /= mix.data.sum()

    final_mix = mix.detach().cpu().numpy()
    final_loss = predictor(mix.unsqueeze(0)).item()
    return final_mix, final_loss

def climb(dataset, iterations, samples, smallmodelepochs, epochs, clusters, datasetSize, variance, topk):
    """ 
    give the dataset, the number of iterations for generate more data,
    the amount of samples to test per iteration, the number of epochs to train each small model,
    the amount of clusters in the dataset, and the size of the dataset you want to train the small models with
    """

    data = [] # collected data to train predictor
    ds = [] # previous tested mixtures
    for _ in range(samples):
        mixture = np.random.rand(clusters)
        mixture /= np.sum(mixture)
        ds.append(mixture)
    
    for i in range(samples):
        data.append(test_mixture(ds[i], dataset, smallmodelepochs, datasetSize))

    predictor = MixtureLossPredictor(clusters)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(predictor.parameters(), lr=1e-3)

    X = torch.tensor([x[0] for x in data], dtype=torch.float32)  # mixtures
    y = torch.tensor([x[1] for x in data], dtype=torch.float32).unsqueeze(1)  # losses

    for epoch in range(epochs):
        predictor.train()
        optimizer.zero_grad()
        preds = predictor(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

    for i in range(iterations):
        bestMix = data[0][0]
        lowestLoss = data[0][1]
        for j in range(len(data)):
            if data[j][1] < lowestLoss:
                bestMix = data[j][0]
                lowestLoss = data[j][1]

        new_data = []
        for _ in range(samples):
            new_mix = np.random.normal(loc=bestMix, scale=variance)
            new_mix = np.clip(new_mix, 1e-6, None)  # prevent negatives
            new_mix /= np.sum(new_mix)
            new_data.append(new_mix)

        # Predict loss for each candidate
        predictor.eval()
        with torch.no_grad():
            candidates_X = torch.tensor(new_data, dtype=torch.float32)
            pred_losses = predictor(candidates_X).squeeze().numpy()

        # Select top-k predicted mixtures (here we use all)
        ranked = sorted(zip(new_data, pred_losses), key=lambda x: x[1])[: topk]
        for mix, _ in ranked:
            loss = test_mixture(mix, dataset, smallmodelepochs, datasetSize)[1]
            data.append((mix, loss))

        # Retrain predictor with updated data
        X = torch.tensor([x[0] for x in data], dtype=torch.float32)
        y = torch.tensor([x[1] for x in data], dtype=torch.float32).unsqueeze(1)

        for epoch in range(epochs):
            predictor.train()
            optimizer.zero_grad()
            preds = predictor(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
    
    best_mix, pred_loss = find_best_mixture_via_predictor(predictor, clusters)
    return best_mix, pred_loss, predictor

    
