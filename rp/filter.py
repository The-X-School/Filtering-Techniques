from huggingface_hub import hf_hub_download
import fasttext
import json
from transformers import AutoTokenizer, GPT2Tokenizer
from tqdm import tqdm
from datasets import load_dataset
from math import sqrt

gpt2tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=True)
"""
def detokenize(token_ids: list[int], tokenizer: AutoTokenizer) ->str:
    Detokenizes a list of token IDs into a cleaned string.
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
"""
def fasttext_predictive_strength(dataset: list[str], topk: float):
    """
    Filters a dataset based on FastText model predictions, sorting by a specific
    predictive strength heuristic (prob_1 descending, then prob_0 ascending).
    dataset: A list of documents, where each document is a list of token IDs.
    topk: The fraction (0.0 to 1.0) of documents with highest predictive strength to keep.
    """
    print("Loading model...")
    try:
        model_filename = "PreSelect-classifier.bin"
        model_path = hf_hub_download("hkust-nlp/preselect-fasttext-classifier", model_filename)
        preselect = fasttext.load_model(model_path)
        print(f"Loaded FastText model from {model_path}")
        model_path = "trained_model.bin"
        rp = fasttext.load_model(model_path)
    except Exception as e:
        print(f"Error loading FastText model from Hugging Face: {e}")
        print("Please ensure the model filename is correct and you have network access.")
        return [] # Return empty list if model cannot be loaded

    predictive_strength_results = []
    roleplay_strength = []
    
    print("Calculating predictive strength...")
    for idx, doc in enumerate(tqdm(dataset, desc="Calculating Predictive Strength")):
        text = doc.replace('\n', ' ')

        # Predict with preselect model
        labels, probs = preselect.predict(text)
        labels = [l.replace('__label__', '') for l in labels]
        # probs = probs.tolist()

        # Convert labels to int and compute weighted average score for predictive strength
        labels_int = list(map(int, labels))
        predictive_score = sum(((2 * l - 1) * p) for l, p in zip(labels_int, probs))

        predictive_strength_results.append({
            'original_tokens': doc,
            'predicted_labels': labels,
            'predicted_probs': probs,
            'score': predictive_score
        })

        # Predict with roleplay model
        labels, probs = rp.predict(text)
        labels = [l.replace('__label__', '') for l in labels]
        # probs = probs.tolist()

        # Convert roleplay labels to float, normalize to [0,1], then weighted average with probabilities
        labels_float = list(map(float, labels))
        normalized_labels = [ (l - 1) / 4 for l in labels_float ]  # maps 1-5 to 0-1
        # soften certainty by sqrt of prob
        roleplay_score = sum(nl * sqrt(p) for nl, p in zip(normalized_labels, probs))

        roleplay_strength.append({
            'original_tokens': doc,
            'predicted_labels': labels,
            'predicted_probs': probs,
            'score': roleplay_score
        })

    alpha = 0.6
    beta = 0.4

    combined_scores = []

    for ps, roleplay, doc in zip(predictive_strength_results, roleplay_strength, dataset):
        cur = alpha * ps['score'] + beta * roleplay['score']
        cur = (cur) / (alpha + beta)

        combined_scores.append((cur, doc))

    total_documents_to_keep = int(len(dataset) * min(max(topk, 0.0), 1.0))
    
    combined_scores.sort(reverse=True)

    filtered_output = []
    for score_val, doc in combined_scores[:total_documents_to_keep]:
        filtered_output.append({
            'score': score_val,
            'doc': doc
        })

    return filtered_output

def main():
    samples_to_process = 10000 # Number of documents to sample from the dataset
    topk_fraction = 0.01 # Percentage of top documents to keep

    print("Step 1: Loading dataset (nvidia/ClimbLab)...")
    dataset_stream = load_dataset("OptimalScale/ClimbLab", split="train", streaming=True)

    print(f"Step 2: Sampling {samples_to_process} documents...")
    sample_documents = []
    # Collect the specified number of documents from the streaming dataset
    for i, item in enumerate(tqdm(dataset_stream, desc=f"Collecting {samples_to_process} documents", total=samples_to_process)):
        if i >= samples_to_process:
            break
        sample_documents.append(item["text"])

    print(f"Collected {len(sample_documents)} documents.")

    print("Step 3: Running FastText Predictive Strength filtering...")
    filtered_data_with_details = fasttext_predictive_strength(sample_documents, topk=topk_fraction)

    print(f"\nFiltering completed: Reduced from {len(sample_documents)} to {len(filtered_data_with_details)} documents.")

    print(f"\nStep 4: Saving filtered output to 'final_filtered.jsonl'...")
    output_filename = "final_filtered.jsonl"
    with open(output_filename, 'w', encoding='utf-8') as f:
        for item_data in filtered_data_with_details:
            f.write(json.dumps(item_data) + '\n')

    print(f"Done! Filtered data saved to `{output_filename}`.")
    print("\n--- FastText Filtering Process Finished ---")
    print(filtered_data_with_details)

if __name__ == "__main__":
    main()
