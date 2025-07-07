import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== CONFIG ==========

FILTER_MODEL = "data4elm/Llama-400M-12L"
INPUT_JSONL = "climblab_sample.jsonl"
OUTPUT_JSONL = "superfiltered.jsonl"
KEEP_RATIO = 0.05  # top 5%

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== LOAD MODEL ==========

tokenizer = AutoTokenizer.from_pretrained(FILTER_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(FILTER_MODEL).to(device).eval()

# ========== HELPER FUNCTIONS ==========

def score_logprob(prefix: str, response: str):
    inputs = tokenizer(prefix + response, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
    logprob = -outputs.loss.item() * inputs.input_ids.size(1)
    return logprob

def compute_ifd(instr: str, resp: str):
    eos = tokenizer.eos_token or ""
    full_resp = resp + eos
    logp_with = score_logprob(instr + eos, full_resp)
    logp_wo   = score_logprob("", full_resp)
    return logp_with - logp_wo

# ========== LOAD JSONL DATASET ==========

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    instances = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(instances)} instances from {INPUT_JSONL}")

# ========== PREPROCESS ==========

def split_instruction_response(text):
    parts = text.split(". ", 1)
    if len(parts) == 2:
        instr, resp = parts[0] + ".", parts[1]
    else:
        instr, resp = "Instruction:", text
    return instr.strip(), resp.strip()

processed = []
for item in instances:
    instr, resp = split_instruction_response(item["text"])
    processed.append({
        "instruction": instr,
        "response": resp,
        "raw_text": item["text"]
    })

# ========== APPLY FILTERING ==========

scored = []
print("Scoring samples using IFD (may take time)...")

for ex in tqdm(processed, desc="Computing IFD"):
    try:
        score = compute_ifd(ex["instruction"], ex["response"])
    except Exception as e:
        print(f"Error scoring example: {e}")
        score = float("-inf")
    scored.append((score, ex))

scored.sort(key=lambda s: s[0], reverse=True)
k = max(1, int(len(scored) * KEEP_RATIO))
filtered = [ex for (_, ex) in scored[:k]]
scores = [s for (s, _) in scored]

# ========== WRITE JSONL OUTPUT ==========

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
    for ex in filtered:
        f_out.write(json.dumps({"text": ex["raw_text"]}, ensure_ascii=False) + "\n")

print(f"\n✅ Wrote {len(filtered)} filtered samples to {OUTPUT_JSONL}")

# ========== PRINT STATS ==========

print("\n📊 Filtering Statistics:")
print(f" - Total samples: {len(instances)}")
print(f" - Kept samples: {len(filtered)} ({KEEP_RATIO*100:.1f}%)")
print(f" - Score range: min={min(scores):.2f}, avg={sum(scores)/len(scores):.2f}, max={max(scores):.2f}")
print(f" - Top kept scores: {[round(s,2) for s in scores[:min(10, len(scores))]]}")
