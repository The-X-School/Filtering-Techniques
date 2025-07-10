from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from huggingface_hub import login
from peft import LoraConfig, get_peft_model

login()

dataset = load_dataset("MakiLS/merged_math_dataset")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    use_dora=True,
)

model = get_peft_model(model, dora_config)
model.train()

