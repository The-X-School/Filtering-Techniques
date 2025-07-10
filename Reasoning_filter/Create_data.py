from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value, Sequence
import re

def parse_mathqa_options(options_string):
    """Parse MathQA options string into a list"""
    if not options_string:
        return []
    
    # MathQA options are typically formatted as "a ) 1 , b ) 2 , c ) 3 , d ) 4 , e ) 5"
    options = []
    parts = options_string.split(' , ')
    for part in parts:
        part = part.strip()
        if ') ' in part:
            # Extract the option content after the letter and )
            option_content = part.split(') ', 1)[1] if ') ' in part else part
            options.append(option_content.strip())
    
    return options

def get_choice_content_from_letter(choices, letter):
    """Get the actual content of a choice from its letter (A, B, C, D, E)"""
    if not choices or not letter:
        return ""
    
    letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    
    if letter in letter_to_index and letter_to_index[letter] < len(choices):
        return choices[letter_to_index[letter]]
    
    return ""

def transform_mathqa(example):
    """Transform MathQA example to unified format"""
    choices = parse_mathqa_options(example["options"])
    correct_content = get_choice_content_from_letter(choices, example["correct"])
    
    return {
        "problem": example["Problem"],
        "choices": choices,
        "correct_answer_content": correct_content,
        "answer": example["correct"],  # The letter (A, B, C, D, E)
        "solution": example["Rationale"],
        "source": "mathqa"
    }

def transform_aqua_rat(example):
    """Transform AQUA-RAT example to unified format"""
    choices = example["options"]
    correct_content = get_choice_content_from_letter(choices, example["correct"])
  
    return {
        "problem": example["question"],
        "choices": choices,
        "correct_answer_content": correct_content,
        "answer": example["correct"],  # The letter (A, B, C, D, E)
        "solution": example["rationale"],
        "source": "aqua_rat"
    }

def transform_mmlu(example):
    """Transform MMLU example to unified format"""
    choices = example["choices"]
    correct_content = choices[example["answer"]] if example["answer"] < len(choices) else ""
    answer_letter = chr(ord('A') + example["answer"]) if example["answer"] < 5 else "A"
    
    return {
        "problem": example["question"],
        "choices": choices,
        "correct_answer_content": correct_content,
        "answer": answer_letter,
        "solution": f"The correct answer is {answer_letter}: {correct_content}",
        "source": "mmlu"
    }

def transform_logiqa(example):
    """Transform LogiQA example to unified format"""
    choices = example["options"]
    correct_content = choices[example["correct_option"]] if example["correct_option"] < len(choices) else ""
    answer_letter = chr(ord('A') + example["correct_option"]) if example["correct_option"] < 5 else "A"
    
    # Combine context and query
    problem_text = f"Context: {example['context']}\n\nQuestion: {example['query']}"
    
    return {
        "problem": problem_text,
        "choices": choices,
        "correct_answer_content": correct_content,
        "answer": answer_letter,
        "solution": f"The correct answer is {answer_letter}: {correct_content}",
        "source": "logiqa"
    }

def transform_arc_challenge(example):
    """Transform ARC-Challenge example to unified format"""
    choices = example["choices"]["text"]
    correct_idx = example["choices"]["label"].index(example["answerKey"])
    correct_content = choices[correct_idx] if correct_idx < len(choices) else ""
    
    return {
        "problem": example["question"],
        "choices": choices,
        "correct_answer_content": correct_content,
        "answer": example["answerKey"],
        "solution": f"The correct answer is {example['answerKey']}: {correct_content}",
        "source": "arc_challenge"
    }

def transform_reclor(example):
    """Transform ReClor example to unified format"""
    choices = example["answers"]
    correct_content = choices[example["label"]] if example["label"] < len(choices) else ""
    answer_letter = chr(ord('A') + example["label"]) if example["label"] < 5 else "A"
    
    # Combine context and question
    problem_text = f"Context: {example['context']}\n\nQuestion: {example['question']}"
    
    return {
        "problem": problem_text,
        "choices": choices,
        "correct_answer_content": correct_content,
        "answer": answer_letter,
        "solution": f"The correct answer is {answer_letter}: {correct_content}",
        "source": "reclor"
    }

def transform_agieval(example):
    """Transform AGIEval example to unified format"""
    choices = example["options"]
    correct_content = choices[example["answer"]] if example["answer"] < len(choices) else ""
    answer_letter = chr(ord('A') + example["answer"]) if example["answer"] < 5 else "A"
    
    return {
        "problem": example["question"],
        "choices": choices,
        "correct_answer_content": correct_content,
        "answer": answer_letter,
        "solution": f"The correct answer is {answer_letter}: {correct_content}",
        "source": "agieval"
    }

# Define the unified features schema
unified_features = Features({
    "problem": Value("string"),           # The main problem statement (without choices)
    "choices": Sequence(Value("string")), # List of multiple choice options (empty for open-ended)
    "correct_answer_content": Value("string"),  # The actual content of the correct answer
    "answer": Value("string"),            # For MC: the letter (A,B,C,D,E), for open-ended: the final answer
    "solution": Value("string"),          # Step-by-step solution/explanation
    "source": Value("string")             # The source dataset
})

# Load the datasets
print("Loading datasets...")
mathqa = load_dataset("allenai/math_qa", trust_remote_code=True)
aqua_rat = load_dataset("deepmind/aqua_rat", "raw", trust_remote_code=True)
mmlu = load_dataset("cais/mmlu", "all", trust_remote_code=True)
logiqa = load_dataset("lucasmccabe/logiqa", trust_remote_code=True)
arc_challenge = load_dataset("ai2_arc", "ARC-Challenge", trust_remote_code=True)
# reclor = load_dataset("reclor", split="train", trust_remote_code=True)  # Requires manual download
# agieval = load_dataset("agieval", "all", split="train", trust_remote_code=True)  # May have issues

print(f"MathQA: {sum(len(split) for split in mathqa.values())} examples")
print(f"AQUA-RAT: {sum(len(split) for split in aqua_rat.values())} examples")
print(f"MMLU: {sum(len(split) for split in mmlu.values())} examples")
print(f"LogiQA: {sum(len(split) for split in logiqa.values())} examples")
print(f"ARC-Challenge: {sum(len(split) for split in arc_challenge.values())} examples")
# print(f"ReClor: {len(reclor)} examples")
# print(f"AGIEval: {len(agieval)} examples")

# Transform datasets to unified format
print("\nTransforming datasets to unified format...")

# Transform MathQA with explicit features
mathqa_datasets = []
for split_name, split_dataset in mathqa.items():
    if len(split_dataset) > 0:
        transformed_split = split_dataset.map(
            transform_mathqa,
            remove_columns=split_dataset.column_names,
            features=unified_features
        )
        mathqa_datasets.append(transformed_split)
mathqa_transformed = concatenate_datasets(mathqa_datasets)
print(f"MathQA transformed: {len(mathqa_transformed)} examples")

# Transform AQUA-RAT with explicit features
aqua_rat_datasets = []
for split_name, split_dataset in aqua_rat.items():
    if len(split_dataset) > 0:
        transformed_split = split_dataset.map(
            transform_aqua_rat,
            remove_columns=split_dataset.column_names,
            features=unified_features
        )
        aqua_rat_datasets.append(transformed_split)
aqua_rat_transformed = concatenate_datasets(aqua_rat_datasets)
print(f"AQUA-RAT transformed: {len(aqua_rat_transformed)} examples")

# Transform MMLU with explicit features
# MMLU is a DatasetDict, so we need to iterate through its splits (e.g., 'test', 'dev', etc.)
mmlu_datasets = []
for split_name, split_dataset in mmlu.items():
    if len(split_dataset) > 0:
        transformed_split = split_dataset.map(
            transform_mmlu,
            remove_columns=split_dataset.column_names,
            features=unified_features
        )
        mmlu_datasets.append(transformed_split)

# Concatenate all MMLU splits
mmlu_transformed = concatenate_datasets(mmlu_datasets)
print(f"MMLU transformed: {len(mmlu_transformed)} examples")

# Transform LogiQA with explicit features
logiqa_datasets = []
for split_name, split_dataset in logiqa.items():
    if len(split_dataset) > 0:
        transformed_split = split_dataset.map(
            transform_logiqa,
            remove_columns=split_dataset.column_names,
            features=unified_features
        )
        logiqa_datasets.append(transformed_split)
logiqa_transformed = concatenate_datasets(logiqa_datasets)
print(f"LogiQA transformed: {len(logiqa_transformed)} examples")

# Transform ARC-Challenge with explicit features
arc_challenge_datasets = []
for split_name, split_dataset in arc_challenge.items():
    if len(split_dataset) > 0:
        transformed_split = split_dataset.map(
            transform_arc_challenge,
            remove_columns=split_dataset.column_names,
            features=unified_features
        )
        arc_challenge_datasets.append(transformed_split)
arc_challenge_transformed = concatenate_datasets(arc_challenge_datasets)
print(f"ARC-Challenge transformed: {len(arc_challenge_transformed)} examples")

# Transform ReClor with explicit features
# reclor_transformed = reclor.map(
#     transform_reclor, 
#     remove_columns=reclor.column_names,
#     features=unified_features
# )
# print(f"ReClor transformed: {len(reclor_transformed)} examples")

# Transform AGIEval with explicit features
# agieval_transformed = agieval.map(
#     transform_agieval, 
#     remove_columns=agieval.column_names,
#     features=unified_features
# )
# print(f"AGIEval transformed: {len(agieval_transformed)} examples")

# Verify features are aligned
print("\nVerifying features alignment...")
print(f"mathqa features: {mathqa_transformed.features}")
print(f"aqua_rat features: {aqua_rat_transformed.features}")
print(f"mmlu features: {mmlu_transformed.features}")
print(f"logiqa features: {logiqa_transformed.features}")
print(f"arc_challenge features: {arc_challenge_transformed.features}")

# Create merged dataset
print("\nMerging datasets...")
merged_dataset = concatenate_datasets([
    mathqa_transformed,
    aqua_rat_transformed,
    mmlu_transformed,
    logiqa_transformed,
    arc_challenge_transformed
])
print(f"Merged dataset: {len(merged_dataset)} examples")

# Print sample examples
print("\nSample examples from merged dataset:")
print("="*50)

for i in range(min(3, len(merged_dataset))):
    example = merged_dataset[i]
    print(f"\nExample {i+1} (from {example['source']}):")
    print(f"Problem: {example['problem'][:200]}...")
    print(f"Choices: {example['choices']}")
    print(f"Correct Answer Content: {example['correct_answer_content']}")
    print(f"Answer: {example['answer']}")
    print(f"Solution: {example['solution'][:200]}...")
    print("-" * 30)

# Save the merged dataset
print(f"\nSaving merged dataset...")
merged_dataset.save_to_disk("Reasoning_filter/merged_math_dataset")
print("Dataset saved to 'merged_math_dataset'")

# Optional: Save as JSON for easier inspection
print("Saving as JSON...")
merged_dataset.to_json("Reasoning_filter/merged_math_dataset.json")
print("Dataset saved to 'merged_math_dataset.json'")

# Print final statistics
print(f"\nFinal dataset statistics:")
print(f"Total examples: {len(merged_dataset)}")

# Count examples by source
source_counts = {}
for example in merged_dataset:
    source = example['source']
    source_counts[source] = source_counts.get(source, 0) + 1

for source, count in source_counts.items():
    print(f"{source} examples: {count}")

# Print schema
print(f"\nDataset schema:")
print(f"- problem: string (the main problem statement without choices)")
print(f"- choices: list of strings (multiple choice options, empty for open-ended questions)")
print(f"- correct_answer_content: string (the actual content of the correct answer)")
print(f"- answer: string (for MC: the letter A/B/C/D/E, for open-ended: the final answer)")
print(f"- solution: string (full step-by-step solution/explanation)")
print(f"- source: string (original dataset: mathqa, aqua_rat, mmlu, logiqa, arc_challenge, reclor, or agieval)")

# Print distribution of question types
print(f"\nQuestion type distribution:")
mc_count = sum(1 for example in merged_dataset if len(example['choices']) > 0)
open_ended_count = len(merged_dataset) - mc_count
print(f"Multiple choice questions: {mc_count}")
print(f"Open-ended questions: {open_ended_count}")