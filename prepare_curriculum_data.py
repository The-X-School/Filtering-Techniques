import os
import sys
import json
import random

def prepare_curriculum_data(initial_dataset_path, curriculum_data_dir, num_curriculum_stages):
    print(f"Initial dataset path: {initial_dataset_path}")
    print(f"Curriculum data directory: {curriculum_data_dir}")
    print(f"Number of curriculum stages: {num_curriculum_stages}")

    os.makedirs(curriculum_data_dir, exist_ok=True)

    if not os.path.exists(initial_dataset_path):
        print(f"Error: Initial dataset '{initial_dataset_path}' not found.")
        sys.exit(1)

    with open(initial_dataset_path, 'r') as f:
        lines = f.readlines()

    random.shuffle(lines)

    chunk_size = len(lines) // num_curriculum_stages
    for i in range(num_curriculum_stages):
        stage_file_path = os.path.join(curriculum_data_dir, f"stage_{i}.jsonl")
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size if i < num_curriculum_stages - 1 else len(lines)
        stage_lines = lines[start_index:end_index]
        with open(stage_file_path, 'w') as f:
            f.writelines(stage_lines)
        print(f"Created {stage_file_path} with {len(stage_lines)} examples.")

    print("Curriculum data preparation complete.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python prepare_curriculum_data.py <initial_dataset_path> <curriculum_data_dir> <num_curriculum_stages>")
        sys.exit(1)

    initial_dataset_path = sys.argv[1]
    curriculum_data_dir = sys.argv[2]
    num_curriculum_stages = int(sys.argv[3])

    prepare_curriculum_data(initial_dataset_path, curriculum_data_dir, num_curriculum_stages)