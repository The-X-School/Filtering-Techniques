import argparse
import json
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
from huggingface_hub import login

# 🔐 Paste your token between the quotes (keep it private!)
HF_TOKEN = "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"
login(token=HF_TOKEN)

def detokenize_climblab(output_path: str, output_jsonl: bool = False, max_samples: int = None, print_samples: int = 0):
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load NVIDIA ClimbLab dataset (streaming for efficiency)
    dataset = load_dataset("nvidia/ClimbLab", split="train", streaming=True)

    results = []
    count = 0
    printed = 0
    
    if output_jsonl:
        out_file = open(output_path, "w", encoding="utf-8")
    else:
        out_file = None

    for sample in tqdm(dataset, desc="Detokenizing ClimbLab"):
        tokens = sample.get("tokens")
        if tokens is None:
            continue
        text = tokenizer.decode(tokens)
        entry = {"text": text}
        if output_jsonl:
            out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        else:
            results.append(entry)
        count += 1
        # Print sample if requested
        if printed < print_samples:
            print(f"Sample {count}: {text}\n{'-'*40}")
            printed += 1
        if max_samples and count >= max_samples:
            break

    if not output_jsonl:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        out_file.close()
    print(f"✅ Detokenized {count} samples. Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Detokenize OptimalScale ClimbLab dataset to text format.")
    parser.add_argument("--output", type=str, default="climblab_detokenized.json", help="Output file path")
    parser.add_argument("--jsonl", action="store_true", help="Output as JSONL instead of a single JSON file")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--print_samples", type=int, default=0, help="Print the first N detokenized samples")
    args = parser.parse_args()

    detokenize_climblab(args.output, args.jsonl, args.max_samples, args.print_samples)


if __name__ == "__main__":
    main() 