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
    parser.add_argument("--steps", type=str, default="1,2,3,4", help="Comma-separated list of steps to run (default is 1,2,3,4)")

    args = parser.parse_args()

    preselect_script_dir = "Filtering-Techniques/preselect_training"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Run preselect_training.py
    if 1 in steps:
        run_command(
            f"python {preselect_script_dir}/preselect_training.py "
            f"--input_path={args.input_path} "
            f"--output_path={args.output_dir} "
            f"--model_path={args.model_path} "
            f"--threshold={args.threshold}"
        )
        original_jsonl = find_latest_jsonl_file(args.output_dir)
        print(f"Found preselect output file: {original_jsonl}")

    # Step 2: Sort the preselect output using sort_preselect.py
    if 2 in steps:
        sorted_file = os.path.join(args.output_dir, "sorted_preselect.jsonl")
        run_command(f"python {preselect_script_dir}/sort_preselect.py {original_jsonl} {sorted_file}")
        os.remove(original_jsonl)
        print(f"Deleted original files : {original_jsonl}")

    # Step 3: Format the preselect output
    if 3 in steps:
        formatted_file = os.path.join(args.output_dir, "formatted_preselect.jsonl")
        run_command(
            f"python {preselect_script_dir}/format_preselect.py "
            f"{sorted_file} {formatted_file}"
        )
        os.remove(sorted_file)
        print(f"Deleted original files : {sorted_file}")

    # Step 4: Convert to LMFlow format
    if 4 in steps:
        run_command(
            f"python Data-Filtering-Challenge/format_data.py {formatted_file}"
        )

        print(f"All steps completed successfully. Final file is at: {formatted_file}")

if __name__ == "__main__":
    main()