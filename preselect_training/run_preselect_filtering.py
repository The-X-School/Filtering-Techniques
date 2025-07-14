import subprocess
import os
import sys
import re
import argparse
from pathlib import Path

# install packages before running this
# pip install datatrove datasets orjson fasteners fasttext-numpy2-wheel regex multiprocess dill

def run_command(command, check=True):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    if check and result.returncode != 0:
        print(f"Command failed: {command}")
        sys.exit(1)

def find_latest_jsonl_file(directory):
    pattern = re.compile(r"(\d{5})\.jsonl$")
    files = [f for f in os.listdir(directory) if pattern.match(f)]
    if not files:
        raise FileNotFoundError(f"No jsonl files found in {directory}")
    latest = max(files, key=lambda f: int(pattern.match(f).group(1)))
    return os.path.join(directory, latest)

def main():
    parser = argparse.ArgumentParser(description="Run the full PreSelect filtering pipeline.")
    parser.add_argument("--input_path", required=True, help="Path to the input data.jsonl file")
    parser.add_argument("--model_path", default="PreSelect-classifier.bin", help="Path to FastText model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classifier threshold")
    parser.add_argument("--output_dir", default="Data-Filtering-Challenge/data/default_dir", help="Directory to save final output")

    args = parser.parse_args()

    preselect_script_dir = "Filtering-Techniques/preselect_training"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Run preselect_training.py
    run_command(
        f"python {preselect_script_dir}/preselect_training.py "
        f"--input_path={args.input_path} "
        f"--output_path={args.output_dir} "
        f"--model_path={args.model_path} "
        f"--threshold={args.threshold}"
    )

    # Step 2: Locate the generated .jsonl file
    original_jsonl = find_latest_jsonl_file(args.output_dir)
    print(f"Found preselect output file: {original_jsonl}")

    # Step 3: Format the preselect output
    formatted_file = os.path.join(args.output_dir, "formatted_preselect.jsonl")
    run_command(
        f"python {preselect_script_dir}/format_preselect.py "
        f"{original_jsonl} {formatted_file}"
    )

    os.remove(original_jsonl)
    print(f"Deleted original file: {original_jsonl}")

    # Step 4: Convert to LMFlow format
    run_command(
        f"python Data-Filtering-Challenge/format_data.py {formatted_file}"
    )

    print(f"All steps completed successfully. Final file is at: {formatted_file}")

if __name__ == "__main__":
    main()