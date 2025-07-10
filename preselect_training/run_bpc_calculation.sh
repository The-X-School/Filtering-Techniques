#!/bin/bash

models=("data4elm/Llama-400M-12L" "TinyLlama/TinyLlama-1.1B-Chat-v1.0" "huggyllama/llama-7b")
parts=(0)

for model in "${models[@]}"; do
  for part in "${parts[@]}"; do
    echo "Running $model part $part"
    python -u /workspace/Filtering-Techniques/PreSelect/data_processing/bpc/main.py \
      --model_name $model \
      --block_size 512 \
      --stride 512 \
      --batch_size 1 \
      --part $part \
      --cluster stage2_10k_preselect
  done
done