from huggingface_hub import hf_hub_download
import fasttext
import json
from transformers import AutoTokenizer, GPT2Tokenizer
from tqdm import tqdm
from datasets import load_dataset

gpt2tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=True)

def detokenize(token_ids: list[int], tokenizer: AutoTokenizer) -> str:
    """Detokenizes a list of token IDs into a cleaned string."""
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

def fasttext_predictive_strength(dataset: list[list[int]], topk: float):
    """
    Filters a dataset based on FastText model predictions, sorting by a specific
    predictive strength heuristic (prob_1 descending, then prob_0 ascending).
    dataset: A list of documents, where each document is a list of token IDs.
    topk: The fraction (0.0 to 1.0) of documents with highest predictive strength to keep.
    """
    print("Loading model...")
    try:
        model_filename = "PreSelect-classifier.bin" # Common filename for FastText models
        model_path = hf_hub_download("hkust-nlp/preselect-fasttext-classifier", model_filename)
        model = fasttext.load_model(model_path)
        print(f"Loaded FastText model from {model_path}")
    except Exception as e:
        print(f"Error loading FastText model from Hugging Face: {e}")
        print("Please ensure the model filename is correct and you have network access.")
        return [] # Return empty list if model cannot be loaded

    predictive_strength_results = []
    
    print("Calculating predictive strength...")
    for idx, doc_tokens in enumerate(tqdm(dataset, desc="Calculating Predictive Strength")):
        # Handle empty input token lists
        if not doc_tokens:
            predictive_strength_results.append({
                'original_tokens': doc_tokens,
                'predicted_labels': [],
                'predicted_probs': [],
                '_sort_key_values': (-0.0, 0.0)
            })
            continue

        # Detokenize the document (which is a list of tokens)
        text = detokenize(doc_tokens, gpt2tokenizer)
        
        # If detokenized text is empty after stripping (e.g., only special tokens)
        if not text.strip():
            predictive_strength_results.append({
                'original_tokens': doc_tokens,
                'predicted_labels': [],
                'predicted_probs': [],
                '_sort_key_values': (-0.0, 0.0)
            })
            continue

        # Predict labels and probabilities using the loaded FastText model
        text = text.replace('\n', ' ')
        labels, probs = model.predict(text)
        
        label_prob_map = {l.replace('__label__', ''): p for l, p in zip(labels, probs)}
        prob_1 = label_prob_map.get('1', 0.0) 
        prob_0 = label_prob_map.get('0', 0.0) 
        
        current_sort_key = (-prob_1, prob_0)

        predictive_strength_results.append({
            'original_tokens': doc_tokens,
            'predicted_labels': [l.replace('__label__', '') for l in labels],
            'predicted_probs': probs.tolist(),
            '_sort_key_values': current_sort_key  
        })

    predictive_strength_results.sort(key=lambda item: item['_sort_key_values'])

    total_documents_to_keep = int(len(dataset) * min(max(topk, 0.0), 1.0))
    
    filtered_output = []
    for item_data in predictive_strength_results[:total_documents_to_keep]:
        filtered_output.append({
            'predicted_labels': item_data['predicted_labels'],
            'predicted_probs': item_data['predicted_probs'],
            'original_tokens': item_data['original_tokens']
        })
    
    return filtered_output

def main():
    samples_to_process = 10000 # Number of documents to sample from the dataset
    topk_fraction = 0.01 # Percentage of top documents to keep

    print("Step 1: Loading dataset (nvidia/ClimbLab)...")
    dataset_stream = load_dataset("nvidia/ClimbLab", split="train", streaming=True)

    print(f"Step 2: Sampling {samples_to_process} documents...")
    sample_documents = []
    # Collect the specified number of documents from the streaming dataset
    for i, item in enumerate(tqdm(dataset_stream, desc=f"Collecting {samples_to_process} documents", total=samples_to_process)):
        if i >= samples_to_process:
            break
        sample_documents.append(list(item["tokens"]))

    print(f"Collected {len(sample_documents)} documents.")

    print("Step 3: Running FastText Predictive Strength filtering...")
    filtered_data_with_details = fasttext_predictive_strength(sample_documents, topk=topk_fraction)

    print(f"\nFiltering completed: Reduced from {len(sample_documents)} to {len(filtered_data_with_details)} documents.")

    print(f"\nStep 4: Saving filtered output to 'filtered.jsonl'...")
    output_filename = "pretrained_fasttext.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        for item_data in filtered_data_with_details:
            f.write(json.dumps(item_data) + '\n')

    print(f"Done! Filtered data saved to `{output_filename}`.")
    print("\n--- FastText Filtering Process Finished ---")
    print(filtered_data_with_details)

if __name__ == "__main__":
    main()
