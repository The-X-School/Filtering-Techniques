# to install fasttext:
# export CXXFLAGS="-stdlib=libc++ -I$(xcrun --show-sdk-path)/usr/include/c++/v1" && pip install fasttext

# other packages to install: 
# pip install datasets
# pip install datatrove
# pip install orjson
# pip install fasteners fasttext-numpy2-wheel

# to run training: 
# python preselect_training.py --input_path=climblab_sample.jsonl --output_path=filtered_output

from datasets import load_dataset
from itertools import islice
import os
import json
from huggingface_hub import hf_hub_download, try_to_load_from_cache
import fasttext

# Load the ClimbLab dataset (default split)
# use text dataset instead of tokenized dataset
print("Loading ClimbLab dataset...")
dataset = load_dataset("OptimalScale/ClimbLab", split="train", streaming=True)

# sample = dataset.select(range(100)) doesn't work with streaming dataset
sample = list(islice(dataset, 1000))

output_jsonl = "climblab_sample.jsonl"
with open(output_jsonl, "w", encoding="utf-8") as f:
    for ex in sample:
        if 'text' in ex:
            f.write(json.dumps({"text": ex["text"]}) + "\n")

print("Checking for cached model...")
# Check if model is already cached
cached_path = try_to_load_from_cache("hkust-nlp/preselect-fasttext-classifier", "PreSelect-classifier.bin")
if cached_path is not None:
    print(f"Using cached model at: {cached_path}")
    model_path = cached_path
else:
    print("Downloading model (this may take a while the first time)...")
    model_path = hf_hub_download(repo_id="hkust-nlp/preselect-fasttext-classifier", filename="PreSelect-classifier.bin")
    print(f"Model downloaded to: {model_path}")

print("Loading fasttext model...")
model = fasttext.load_model(model_path)

print("Processing samples...")
for ex in sample:
    if 'text' in ex:
        text = ex["text"].replace('\n', ' ').strip()
        score = model.predict(text)
        print(score)

print(hf_hub_download("hkust-nlp/preselect-fasttext-classifier", "PreSelect-classifier.bin"))