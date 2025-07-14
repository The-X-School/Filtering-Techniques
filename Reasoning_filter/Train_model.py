import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# For multi-GPU training, launch this script with `accelerate launch Train_model.py`
# after running `accelerate config`.

# To push the model to Hugging Face Hub, ensure you are logged in.
# You can do this by:
# 1. Running `huggingface-cli login` in your terminal before executing this script.
# 2. Setting the HF_TOKEN environment variable: `export HF_TOKEN="your_token_here"`
# login() # Removed to avoid interactive prompt during script execution

# Load the dataset and inspect its structure
logger.info("Loading and inspecting dataset...")
dataset = load_dataset("MakiLS/merged_math_dataset", split="train")
logger.info(f"Loaded {len(dataset)} samples from the dataset.")

# Inspect the dataset structure
logger.info(f"Dataset columns: {dataset.column_names}")
logger.info(f"Dataset features: {dataset.features}")

# Show a few examples to understand the format
for i in range(min(3, len(dataset))):
    logger.info(f"Sample {i}: {dataset[i]}")

# Model and tokenizer names
model_name = "meta-llama/Llama-3.2-1B"

# Load model with A100 optimizations
logger.info(f"Loading model: {model_name}")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )
    logger.info("Using Flash Attention 2 - optimized for A100.")
except (ImportError, torch.cuda.OutOfMemoryError) as e:
    logger.warning(f"Flash Attention 2 not available or failed ({e}), falling back to 'sdpa'.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True
    )

# Compilation will be handled by Trainer via training_args.torch_compile=True

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad token if it's not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Pad token set to EOS token.")

# --- Define LoRA configuration optimized for A100 ---
lora_config = LoraConfig(
    r=64,  # Increased rank for better performance with A100 memory
    lora_alpha=128,  # Scaled proportionally
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,  # Reduced dropout for faster convergence
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    use_dora=True,  # DoRA works well with A100
    use_rslora=True,  # Rank-Stabilized LoRA for better training dynamics
)

# Apply LoRA to the model
logger.info("Applying LoRA configuration...")
model = get_peft_model(model, lora_config)

# Enable gradient computation for LoRA parameters
# get_peft_model already sets base model params to requires_grad=False and LoRA params to requires_grad=True
# So, explicit loops are not needed and might interfere.

# Enable input gradients for gradient checkpointing compatibility
model.enable_input_require_grads()

model.print_trainable_parameters()
logger.info(f"Model is in training mode: {model.training}")
# Ensure the model is in training mode after PEFT model creation
model.train()

# --- Improved data preprocessing function ---
def preprocess_function(examples):
    """
    Improved preprocessing function that handles various dataset formats.
    This function now includes comprehensive debugging and format detection.
    """
    # Get available columns
    available_columns = list(examples.keys())
    logger.info(f"Available columns: {available_columns}")
    
    # Show first example for debugging
    if len(examples[available_columns[0]]) > 0:
        sample_example = {col: examples[col][0] for col in available_columns}
        logger.info(f"First example structure: {sample_example}")
    
    # Check if this is the MakiLS/merged_math_dataset format
    if 'problem' in available_columns and 'solution' in available_columns and 'choices' in available_columns:
        logger.info("Detected MakiLS/merged_math_dataset format with multiple choice questions")
        # Handle the specific format of this dataset
        texts = []
        for i in range(len(examples['problem'])):
            problem = str(examples['problem'][i]).strip()
            choices = examples['choices'][i] if examples['choices'][i] else []
            correct_answer = str(examples['correct_answer_content'][i]).strip() if examples['correct_answer_content'][i] else ""
            solution = str(examples['solution'][i]).strip()
            
            # Skip empty examples
            if not problem or not solution:
                continue
            
            # Format the choices as a numbered list
            choices_text = ""
            if choices:
                choices_text = "\n".join([f"{chr(97 + idx)}) {choice}" for idx, choice in enumerate(choices)])
                choices_text = f"\n\nChoices:\n{choices_text}"
            
            # Create the instruction with problem and choices
            instruction = f"Solve this math problem step by step:\n\n{problem}{choices_text}"
            
            # Create the output with the correct answer and solution
            if correct_answer:
                output = f"The correct answer is: {correct_answer}\n\nSolution:\n{solution}"
            else:
                output = f"Solution:\n{solution}"
            
            # Format as instruction-following conversation
            formatted_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>{tokenizer.eos_token}"
            texts.append(formatted_text)
        
        # Filter out empty texts
        texts = [text for text in texts if text.strip()]
        
        if not texts:
            logger.warning("No valid texts found after preprocessing!")
            return {"input_ids": [], "attention_mask": [], "labels": []}
        
        logger.info(f"Successfully processed {len(texts)} examples")
        logger.info(f"Example formatted text: {texts[0][:300]}...")
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=1024,
            return_tensors=None
        )
        
        # Set labels equal to input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Define common field mappings for other dataset formats
    field_mappings = [
        # Format 1: Standard instruction-tuning format
        {'instruction': 'instruction', 'output': 'output'},
        {'instruction': 'input', 'output': 'output'},
        {'instruction': 'prompt', 'output': 'response'},
        {'instruction': 'question', 'output': 'answer'},
        {'instruction': 'problem', 'output': 'solution'},
        {'instruction': 'query', 'output': 'answer'},
        {'instruction': 'text', 'output': 'target'},
        
        # Format 2: Conversation format
        {'instruction': 'human', 'output': 'gpt'},
        {'instruction': 'user', 'output': 'assistant'},
        
        # Format 3: Math-specific formats
        {'instruction': 'problem', 'output': 'answer'},
        {'instruction': 'question', 'output': 'solution'},
        {'instruction': 'math_problem', 'output': 'math_solution'},
        
        # Format 4: Single text field (needs parsing)
        {'text_field': 'text'},
        {'text_field': 'content'},
        {'text_field': 'conversation'},
    ]
    
    # Find matching field mapping
    instruction_field = None
    output_field = None
    text_field = None
    
    for mapping in field_mappings:
        if 'text_field' in mapping:
            # Handle single text field format
            if mapping['text_field'] in available_columns:
                text_field = mapping['text_field']
                logger.info(f"Using single text field: '{text_field}'")
                break
        else:
            # Handle instruction-output format
            if mapping['instruction'] in available_columns and mapping['output'] in available_columns:
                instruction_field = mapping['instruction']
                output_field = mapping['output']
                logger.info(f"Using instruction-output format: '{instruction_field}' -> '{output_field}'")
                break
    
    # Process based on detected format
    if instruction_field and output_field:
        # Standard instruction-output format
        texts = []
        for i in range(len(examples[instruction_field])):
            instruction = str(examples[instruction_field][i]).strip()
            output = str(examples[output_field][i]).strip()
            
            # Skip empty examples
            if not instruction or not output:
                continue
            
            # Format as instruction-following conversation
            formatted_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>{tokenizer.eos_token}"
            texts.append(formatted_text)
            
    elif text_field:
        # Single text field format - assume it's already formatted or contains conversations
        texts = []
        for i in range(len(examples[text_field])):
            text = str(examples[text_field][i]).strip()
            if not text:
                continue
            
            # Check if text already contains conversation markers
            if '<|im_start|>' in text or any(marker in text for marker in ['Human:', 'Assistant:', 'User:', 'AI:']):
                # Text is already formatted or contains conversation markers
                if not text.endswith(tokenizer.eos_token):
                    text += tokenizer.eos_token
                texts.append(text)
            else:
                # Assume it's a single response and format it
                formatted_text = f"<|im_start|>assistant\n{text}<|im_end|>{tokenizer.eos_token}"
                texts.append(formatted_text)
    
    else:
        # Fallback: try to infer from available columns
        logger.error(f"Could not detect format from columns: {available_columns}")
        logger.error("Please manually specify the correct field mapping in the preprocess_function.")
        
        # Show all available data for the first few examples to help debug
        for i in range(min(2, len(examples[available_columns[0]]))):
            example_data = {col: examples[col][i] for col in available_columns}
            logger.error(f"Example {i}: {example_data}")
        
        raise ValueError(f"Could not detect appropriate format from columns: {available_columns}")
    
    # Filter out empty texts
    texts = [text for text in texts if text.strip()]
    
    if not texts:
        logger.warning("No valid texts found after preprocessing!")
        return {"input_ids": [], "attention_mask": [], "labels": []}
    
    logger.info(f"Successfully processed {len(texts)} examples")
    logger.info(f"Example formatted text: {texts[0][:200]}...")
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=2048,
        return_tensors=None
    )
    
    # Set labels equal to input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Test the preprocessing function on a small sample first
logger.info("Testing preprocessing function on a small sample...")
try:
    # Test on first 5 examples - convert to the format expected by the function
    test_indices = list(range(min(5, len(dataset))))
    test_sample = {col: [dataset[i][col] for i in test_indices] for col in dataset.column_names}
    test_result = preprocess_function(test_sample)
    logger.info("Preprocessing test successful!")
    logger.info(f"Test result keys: {test_result.keys()}")
    logger.info(f"Number of tokenized examples: {len(test_result['input_ids'])}")
except Exception as e:
    logger.error(f"Preprocessing test failed: {e}")
    logger.error("Please check the dataset format and modify the preprocess_function accordingly.")
    raise

# Preprocess the full dataset
logger.info("Preprocessing full dataset...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing dataset",
    num_proc=4  # Use multiple processes for faster preprocessing
)

# Split dataset for training and validation
train_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_data = train_dataset['train']
eval_data = train_dataset['test']

logger.info(f"Training samples: {len(train_data)}")
logger.info(f"Validation samples: {len(eval_data)}")

# --- Data collator optimized for A100 ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=64  # Optimized for A100 tensor cores
)

# --- Training arguments optimized for A100 ---
# --- Training arguments optimized for A100 ---
training_args = TrainingArguments(
    output_dir="./llama-3.2-1b-math-lora",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    weight_decay=0.01,
    logging_steps=5,
    logging_strategy="steps",
    eval_strategy="steps",  # Changed from evaluation_strategy to eval_strategy
    eval_steps=50,
    save_steps=200,
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    bf16=True,
    tf32=True,
    torch_compile=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    push_to_hub=True,
    report_to="tensorboard",
    save_total_limit=2,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    dataloader_drop_last=True,
    max_grad_norm=1.0,
    fp16_full_eval=False,
    include_inputs_for_metrics=False,
    skip_memory_metrics=True,
    dataloader_persistent_workers=True,
    label_names=["labels"], # Explicitly tell Trainer which column contains labels
)
# --- Initialize trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Verify model is ready for training
logger.info("Verifying model setup...")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Trainable parameters: {trainable_params:,}")
logger.info(f"Total parameters: {total_params:,}")
logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

# Check if any parameters require gradients
has_trainable_params = any(p.requires_grad for p in model.parameters())
logger.info(f"Model has trainable parameters: {has_trainable_params}")

if not has_trainable_params:
    logger.error("No trainable parameters found! Check LoRA configuration.")
    raise RuntimeError("No trainable parameters found!")

# --- Start training with A100 optimizations ---
logger.info("Starting training optimized for A100...")

# Set optimal CUDA settings for A100
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Enable optimized attention for A100
torch.backends.cuda.enable_flash_sdp(True)

trainer.train()

# --- Save the final model ---
logger.info("Saving final model...")
trainer.save_model()
tokenizer.save_pretrained(training_args.output_dir)

# --- Push to Hub ---
logger.info("Pushing model to Hub...")
trainer.push_to_hub()

logger.info("Training completed successfully!")

# --- Optional: Test the model ---
def test_model():
    logger.info("Testing the trained model...")
    
    # Load the trained model
    from peft import PeftModel
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(base_model, training_args.output_dir)
    
    # Test prompt
    test_prompt = "<|im_start|>user\nSolve: 2x + 5 = 13<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Model response: {response}")

if __name__ == "__main__":
    # Uncomment the line below to test the model after training
    # test_model()
    pass
