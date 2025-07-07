#!/bin/bash

models=("Llama-400M-12L" "llama-7b" "Llama-13B")
parts=(0 1 2 3)

for model in "${models[@]}"; do
  for part in "${parts[@]}"; do
    echo "Running $model part $part"
    python -u .py \
      --task_name your_task \
      --model_name $model \
      --block_size 1900 \
      --stride 512 \
      --batch_size 4 \
      --part $part \
      --cluster your_cluster
  done
done