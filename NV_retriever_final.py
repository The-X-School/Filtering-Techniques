import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from contextlib import nullcontext
from typing import List, Optional, Dict, Union

# Assuming these are imported/defined elsewhere in your actual project
# Placeholder imports and definitions for context
# You will need to replace these with your actual imports and configurations
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PretrainedConfig
import numpy as np
import os
import json

def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load a JSONL dataset from the specified file path.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        List[Dict]: List of dictionaries, each representing a data sample
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line {line_num} in {file_path}: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []
    
    print(f"Loaded {len(data)} samples from {file_path}")
    return data



# Placeholder for your specific configurations
class NVEmbedConfig(PretrainedConfig):
    model_type = "nvembed"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class LatentAttentionConfig(PretrainedConfig):
    model_type = "latent_attention"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BidirectionalMistralConfig(PretrainedConfig):
    model_type = "bidirectional_mistral"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# Placeholder for input_transform_func and NVEmbedFeatures
# You would typically have these defined in your project's utilities
def input_transform_func(tokenizer, batch_dict, always_add_eos, max_length, instruction):
    """
    Placeholder for your input transformation function.
    This function should tokenize the input texts and return a dictionary
    of tensors suitable for model input.
    """
    texts = [f"{instruction}{text}" if instruction else text for text in batch_dict["input_texts"]]
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding='longest', # Ensure padding for batching
        max_length=max_length,
        return_tensors='pt'
    )
    return tokenized_inputs

# Placeholder for NVEmbedFeatures (if it's a specific dataclass)
# If not a dataclass, it's just a dictionary.
class NVEmbedFeatures(dict):
    pass

# --- Start of modeling_nvembed.py content (simulated) ---

# Assuming these are your model definitions from modeling_nvembed.py
# (Adjust class names and inheritance if they differ in your actual file)

class LatentAttentionModel(nn.Module):
    # Placeholder for your LatentAttentionModel implementation
    # This is a dummy implementation to allow the code to run
    def __init__(self, config):
        super().__init__()
        # Example: a simple pooling layer
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.config = config

    def forward(self, last_hidden_state, pool_mask):
        # Dummy pooling: take the mean of the masked tokens
        if pool_mask is not None:
            masked_hidden_state = last_hidden_state * pool_mask.unsqueeze(-1)
            # Sum and divide by the number of unmasked tokens
            # Avoid division by zero if all tokens are masked
            sum_masked = masked_hidden_state.sum(dim=1)
            num_unmasked = pool_mask.sum(dim=1).unsqueeze(-1)
            embeds = sum_masked / (num_unmasked + 1e-9) # Add small epsilon to prevent div by zero
        else:
            # If no pool_mask, just take mean of all tokens
            embeds = last_hidden_state.mean(dim=1)
        return self.linear(embeds) # Apply a linear layer


class BidirectionalMistralModel(PreTrainedModel):
    config_class = BidirectionalMistralConfig
    # Placeholder for your BidirectionalMistralModel implementation
    # This is a dummy implementation to allow the code to run
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # Simulate a simple decoder stack
        self.decoder = nn.ModuleDict({
            "block": nn.ModuleList([
                nn.Linear(config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)
            ])
        })
        # Assuming a simple embedding layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = False, # Ensure use_cache is handled
        output_attentions: Optional[bool] = False, # Pass this through
        output_hidden_states: Optional[bool] = False, # Pass this through
        return_dict: bool = True,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")
        elif input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None # Initialize next_decoder_cache for the loop

        # Dummy causal mask (for demonstration, real Mistral would be more complex)
        causal_mask = None
        if attention_mask is not None:
            seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            causal_mask = attention_mask.unsqueeze(1).unsqueeze(1) # Simple attention mask
            # For a true causal mask, it would be a triangular matrix

        for idx, decoder_layer in enumerate(self.decoder.block):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Dummy layer_outputs - in a real model, this would be from a transformer layer
            # The structure of layer_outputs depends on output_attentions and use_cache
            # For this dummy, we'll simulate it based on parameters
            current_layer_outputs = [decoder_layer(hidden_states)] # Always has hidden_states at index 0

            if use_cache:
                # Simulate a dummy past_key_value (e.g., a tensor of zeros)
                dummy_past_key_value = torch.zeros(1, 1, 1, 1, device=hidden_states.device, dtype=hidden_states.dtype)
                current_layer_outputs.append(dummy_past_key_value)

            if output_attentions:
                # Simulate a dummy attention output
                dummy_attention = torch.zeros(1, 1, 1, 1, device=hidden_states.device, dtype=hidden_states.dtype)
                current_layer_outputs.append(dummy_attention)
                if all_self_attentions is not None:
                    all_self_attentions += (dummy_attention,)

            layer_outputs = tuple(current_layer_outputs) # Convert to tuple

            hidden_states = layer_outputs[0] # Update hidden_states for next layer

            # --- MODIFICATION FOR next_decoder_cache HANDLING (from previous steps) ---
            # This block replaces the problematic line: next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            current_next_decoder_cache = None
            if use_cache:
                if output_attentions and len(layer_outputs) > 2:
                    current_next_decoder_cache = layer_outputs[2] # Attentions are at index 2 if present
                elif not output_attentions and len(layer_outputs) > 1:
                    current_next_decoder_cache = layer_outputs[1] # Cache is at index 1 if no attentions

            # Accumulate caches if needed (for the overall model output)
            if use_cache:
                if next_decoder_cache == (): # First iteration, or empty tuple
                    next_decoder_cache = (current_next_decoder_cache,) if current_next_decoder_cache is not None else None
                elif next_decoder_cache is not None:
                    if current_next_decoder_cache is not None:
                        next_decoder_cache += (current_next_decoder_cache,)
                    else:
                        next_decoder_cache += (None,) # Keep tuple length consistent if a layer doesn't return cache
            # --- END OF MODIFICATION ---

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Line 191 in your traceback:
        # This line is now within BidirectionalMistralModel, where next_decoder_cache is handled.
        # We need to ensure next_decoder_cache is not None before calling .to_legacy_cache()
        # This is the fix for AttributeError: 'NoneType' object has no attribute 'to_legacy_cache'
        final_next_cache = None
        if next_decoder_cache is not None: # Check if next_decoder_cache is not None
             # Assuming use_legacy_cache is a config attribute or passed in
             use_legacy_cache = getattr(self.config, 'use_legacy_cache', False) # Default to False
             if use_legacy_cache:
                 # This method might not exist on a simple tensor,
                 # it's typical for a PastKeyValues object from Hugging Face
                 # For this dummy, we'll just assign it if it's not None
                 final_next_cache = next_decoder_cache.to_legacy_cache() if hasattr(next_decoder_cache, 'to_legacy_cache') else next_decoder_cache
             else:
                 final_next_cache = next_decoder_cache
        # End of fix for AttributeError

        if not return_dict:
            return (hidden_states, final_next_cache, all_hidden_states, all_self_attentions)

        # Return a dictionary similar to Hugging Face BaseModelOutputWithPastAndCrossAttentions
        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=final_next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class NVEmbedModel(PreTrainedModel):
    config_class = NVEmbedConfig
    base_model_prefix = "embedding_model"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # Assuming embedding_model is a BidirectionalMistralModel
        self.embedding_model = BidirectionalMistralModel(config.embedding_model_config)
        self.latent_attention_model = LatentAttentionModel(config.latent_attention_config)
        self.tokenizer = None # Will be set during model loading/initialization
        self.padding_side = "right" # Default, adjust if needed
        self.is_mask_instruction = True # Default, adjust if needed

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        # This is a placeholder; real models have more complex initialization
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def prepare_kwargs_from_batch(self, batch_dict, instruction_lens, device):
        """
        Prepares keyword arguments for the model's forward pass.
        This is a placeholder and should align with your actual implementation.
        """
        features = NVEmbedFeatures()
        features["input_ids"] = batch_dict["input_ids"].to(device)
        features["attention_mask"] = batch_dict["attention_mask"].to(device)
        # You might need to generate pool_mask here if it's not in batch_dict
        # For simplicity, we'll assume it's not needed for basic operation or is handled internally
        features["pool_mask"] = None # Placeholder, adjust if needed
        return features

    @torch.no_grad()
    def encode(self, prompts: List[str], instruction: str="", max_length: int=4096, **kwargs):
        if self.padding_side == "right" and self.is_mask_instruction == True and len(instruction) > 0:
            instruction_lens = len(self.tokenizer.tokenize(instruction))
        else:
            instruction_lens = 0
        
        device = next(self.embedding_model.parameters()).device
        batch_dict = input_transform_func(self.tokenizer,
                                          {"input_texts": [prompt for prompt in prompts]},
                                          always_add_eos=True,
                                          max_length=max_length,
                                          instruction=instruction)

        features: NVEmbedFeatures = self.prepare_kwargs_from_batch(batch_dict, instruction_lens, device=device)
        return self(**features)["sentence_embeddings"].squeeze(1)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                pool_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: bool = True):
        if inputs_embeds is not None:
            raise NotImplementedError("NVEmbedModel does not support 'inputs_embeds'. Please use 'input_ids' instead.")
        
        autocast_ctx = torch.autocast if torch.cuda.is_available() else nullcontext
        with autocast_ctx("cuda"):
            ## decoder only layer
            outputs = self.embedding_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,       # Set to False due to SDPA warning
                output_hidden_states=False,    # Set to False to save memory
                return_dict=True,
            )
            ## latent attention layer
            embeds = self.latent_attention_model(
                outputs.last_hidden_state,
                pool_mask,
            )
        if not return_dict:
            return (embeds,)
        return {"sentence_embeddings": embeds}

# AutoModel Register (assuming these are part of your setup)
# AutoModel.register(NVEmbedConfig, NVEmbedModel)
# AutoModel.register(LatentAttentionConfig, LatentAttentionModel)
# AutoModel.register(BidirectionalMistralConfig, BidirectionalMistralModel)

# Register for auto class (assuming these are part of your setup)
# NVEmbedModel.register_for_auto_class("AutoModel")
# LatentAttentionModel.register_for_auto_class("AutoModel")
# BidirectionalMistralModel.register_for_auto_class("AutoModel")

# --- End of modeling_nvembed.py content (simulated) ---


# --- Start of NV_retriever.py content ---

# This function is designed to handle batching internally
def get_embeddings(model, inputs: Dict[str, torch.Tensor], batch_size: int = 32):
    """
    Generates embeddings for a batch of inputs, processing them in smaller sub-batches
    to manage memory usage.

    Args:
        model: The embedding model (e.g., NVEmbedModel instance).
        inputs (Dict): A dictionary containing input_ids, attention_mask, etc.
                       These should be PyTorch tensors already on the correct device.
        batch_size (int): The maximum number of samples to process at once.
                          Reduce this value if you encounter 'Killed' errors.

    Returns:
        torch.Tensor: The concatenated sentence embeddings for all inputs.
    """
    all_embeddings = []
    
    # Assuming 'input_ids' is the primary key to determine the number of samples
    num_samples = inputs['input_ids'].shape[0]
    device = next(model.parameters()).device # Get model's current device

    for i in range(0, num_samples, batch_size):
        batch_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                # Ensure batch is on the correct device
                batch_inputs[key] = value[i:i + batch_size].to(device)
            else:
                batch_inputs[key] = value # Keep other types as is

        try:
            # Ensure only relevant keys are passed to the model's forward method
            # based on NVEmbedModel's forward signature
            filtered_batch_inputs = {
                k: v for k, v in batch_inputs.items()
                if k in ['input_ids', 'attention_mask', 'pool_mask']
            }
            
            outputs = model(**filtered_batch_inputs)

            # Assuming the output structure is {'sentence_embeddings': embeds}
            if isinstance(outputs, dict) and 'sentence_embeddings' in outputs:
                batch_emb = outputs['sentence_embeddings']
            else:
                # Fallback if output structure is different (e.g., a tuple)
                batch_emb = outputs[0] # Assuming embeddings are the first element

            all_embeddings.append(batch_emb.cpu()) # Move to CPU to free GPU memory
            
            # Explicitly clear CUDA cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f"Error processing batch {i} to {i+batch_size}: {e}")
            # Consider more sophisticated error handling or logging
            break # Stop processing if an error occurs

    # Concatenate all embeddings from CPU
    if all_embeddings:
        return torch.cat(all_embeddings, dim=0)
    else:
        # Return an empty tensor with the correct embedding dimension if no embeddings were generated
        # You might need to get the embedding dimension from model.config.hidden_size or similar
        # For now, returning a 0-dim tensor. Adjust if you know the exact dim.
        return torch.empty(0, 768) # Placeholder: 768 is a common hidden size


# --- Main script execution (example usage) ---

if __name__ == "__main__":
    # --- Placeholder for your actual setup ---
    # You need to replace these with your actual model loading, tokenizer, data, etc.
    # Example:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_name = "your_model_path_or_name"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name) # This would load your NVEmbedModel if registered correctly
    # model.to(device)

    # Dummy setup for demonstration
    class DummyConfig(PretrainedConfig): # Inherit from PretrainedConfig
        def __init__(self, **kwargs):
            super().__init__(**kwargs) # Call parent constructor
            self.hidden_size = 768
            self.num_hidden_layers = 2
            self.vocab_size = 30522
            self.initializer_range = 0.02
            # Ensure these are instances of their respective config classes
            self.embedding_model_config = BidirectionalMistralConfig(
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                vocab_size=self.vocab_size,
                initializer_range=self.initializer_range
            )
            self.latent_attention_config = LatentAttentionConfig(
                hidden_size=self.hidden_size # LatentAttentionConfig might need hidden_size
            )
            self.use_legacy_cache = False # Important for BidirectionalMistralModel

    config = DummyConfig()
    
    # Manually instantiate models for demonstration since AutoModel.register might not be active in this context
    model = NVEmbedModel(config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2") # Use GPT-2 tokenizer for ClimbLab dataset
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    temp_tokenizer = tokenizer # Assuming temp_tokenizer is the same or similar
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = 512
    # --- MODIFICATION: Specify a directory for output_path ---
    output_path = "./artifacts/embeddings_output.json" 
    # --- END OF MODIFICATION ---
    model_name = "NV-Embed-v2"
    num_epochs = 3
    test_query = "What is the capital of France?"
    search_top_k = 3

    # --- Dataset Configuration ---
    # Set your JSONL dataset path here
    dataset_path = "data/climblab_processed_clusters.jsonl"  # Set this to your dataset file path, e.g., "data/your_dataset.jsonl"
    text_field = "detokenized_text"  # The field name containing the text in your dataset
    
    # Load JSONL dataset or use dummy data
    if dataset_path and os.path.exists(dataset_path):
        print(f"Loading JSONL dataset from: {dataset_path}")
        sample = load_jsonl(dataset_path)
    else:
        print("No dataset path provided or file not found. Using dummy data for demonstration.")
        sample = [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "Artificial intelligence is transforming industries worldwide."},
            {"text": "Paris is known for its Eiffel Tower and delicious pastries."},
            {"text": "Machine learning is a subset of AI."},
            {"text": "Dogs are loyal companions."},
            {"text": "Deep learning models require vast amounts of data."},
            {"text": "The Seine river flows through Paris."},
            {"text": "Computers are becoming increasingly powerful."},
            {"text": "France is a country in Western Europe."},
            {"text": "Neural networks are inspired by the human brain."},
        ]
    
    training_data = sample  # Use the loaded dataset as training data
    # --- End of Dataset Configuration ---


    print("Generating embeddings with fine-tuned model...")
    model.eval() # Set model to evaluation mode

    docs = []
    # Process all sample texts to get the documents
    for row in sample:
        if "tokens" in row:
            docs.append(temp_tokenizer.decode(row["tokens"], skip_special_tokens=True))
        else:
            # Use the specified text field, with fallback to "text" and then empty string
            text_content = row.get(text_field, row.get("text", ""))
            docs.append(text_content)

    # --- Batching for document embeddings ---
    all_doc_inputs = tokenizer(
        docs,
        truncation=True,
        padding='longest', # Pad to the longest sequence in the batch
        max_length=max_length,
        return_tensors='pt'
    )

    # Move all inputs to the device once
    all_doc_inputs = {k: v.to(device) for k, v in all_doc_inputs.items()}

    embeddings_list = []
    with torch.no_grad():
        # Call get_embeddings once with all tokenized documents
        # The get_embeddings function will handle internal sub-batching.
        # Adjust inference_batch_size based on your GPU memory.
        inference_batch_size = 4 # Start with a small batch size, e.g., 4, 8, 16
        
        embeddings_tensor = get_embeddings(model, all_doc_inputs, batch_size=inference_batch_size)
        
        # Move the final embeddings to CPU and convert to numpy
        embeddings_np = embeddings_tensor.cpu().numpy()

    print(f"Generated embeddings with shape: {embeddings_np.shape}")

    if output_path:
        print(f"Saving artifacts to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        embeddings_file = output_path.replace('.json', '_embeddings.npy')
        np.save(embeddings_file, embeddings_np)
        
        with open(output_path, 'w') as f:
            data = {
                'texts': docs,
                'embedding_shape': embeddings_np.shape,
                'embedding_file': embeddings_file,
                'model_used': f"{model_name}_dora_finetuned",
                'training_samples': len(training_data),
                'epochs': num_epochs
            }
            json.dump(data, f, indent=2)
        print(f"Embeddings saved to {embeddings_file}")
        print(f"Metadata saved to {output_path}")
        
    if docs:
        print(f"\n--- Searching with DoRA fine-tuned model: '{test_query}' ---")
        query_inputs = tokenizer(test_query, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
        
        with torch.no_grad():
            # Use the same get_embeddings function for the query,
            # ensuring it handles the batching (even for a single query, it's consistent)
            query_embedding = get_embeddings(model, query_inputs, batch_size=1).cpu().numpy()
            
        similarities = np.dot(embeddings_np, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:search_top_k]

        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                'rank': i + 1,
                'text': docs[idx],
                'similarity': float(similarities[idx])
            })

        for result in results:
            print(f"\nRank {result['rank']}: Similarity={result['similarity']:.3f}")
            print(f"Preview: {result['text'][:200]}...")