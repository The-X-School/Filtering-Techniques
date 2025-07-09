from datasets import load_from_disk
from huggingface_hub import login
import argparse

def upload_dataset_to_hub(token):
    # Log in to Hugging Face Hub
    try:
        login(token=token)
        print("Successfully logged in to Hugging Face Hub.")
    except Exception as e:
        print(f"An error occurred during login: {e}")
        return

    # Load the dataset from disk
    print("Loading dataset from disk...")
    try:
        dataset = load_from_disk("/Users/maki2030147/Desktop/Desktop - Maki2030147/Filtering-Techniques/Reasoning_filter/merged_math_dataset")
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Define the repository name
    repo_name = "MakiLS/merged_math_dataset"
    
    # Push the dataset to the Hub
    print(f"Pushing dataset to {repo_name}...")
    try:
        dataset.push_to_hub(repo_name)
        print("Dataset successfully uploaded to the Hugging Face Hub.")
        print(f"You can view it at: https://huggingface.co/datasets/{repo_name}")
    except Exception as e:
        print(f"Failed to push dataset to the Hub: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a dataset to the Hugging Face Hub.")
    parser.add_argument("token", type=str, help="Your Hugging Face API token.")
    args = parser.parse_args()
    upload_dataset_to_hub(args.token)