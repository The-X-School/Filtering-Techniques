#!/bin/bash

# --- Configuration ---
initial_dataset_path="data/filtered_output/embeddings.jsonl"
curriculum_data_dir="/home/ubuntu/curriculum_data"
num_curriculum_stages=5

# --- Setup ---
project_dir=$(cd "$(dirname $0)"; pwd)
cd "${project_dir}" # Change to project root

mkdir -p ${curriculum_data_dir}

# --- Prepare Curriculum Data ---
echo "--- Preparing curriculum data... ---"
python prepare_curriculum_data.py "${initial_dataset_path}" "${curriculum_data_dir}" "${num_curriculum_stages}"
if [ $? -ne 0 ]; then
    echo "Error: Curriculum data preparation failed."
    exit 1
fi
echo "--- Curriculum data preparation complete. ---"
