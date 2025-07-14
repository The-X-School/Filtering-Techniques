import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

def main():
    parser = argparse.ArgumentParser(description="Finetune a causal language model with DeepSpeed and LoRA/DoRA.")

    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pre-trained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file (JSONL format).")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model checkpoints and logs.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="The initial learning rate for AdamW.")
    parser.add_argument("--block_size", type=int, default=1024, help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=24, help="Batch size per GPU for training.")
    parser.add_argument("--use_dora", type=int, default=0, help="Whether to use DoRA (1) or LoRA (0).")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension (r).")
    parser.add_argument("--lora_target_modules", type=str, default="embed_tokens,q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head", help="Comma-separated list of module names to apply LoRA/DoRA to.")
    parser.add_argument("--save_aggregated_lora", type=int, default=0, help="Whether to save the aggregated LoRA model (1) or not (0).")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config file.")
    parser.add_argument("--bf16", action="store_true", help="Whether to use bf16 (mixed) precision instead of fp16.")
    parser.add_argument("--run_name", type=str, default="finetune_model", help="A descriptive name for the training run.")
    parser.add_argument("--validation_split_percentage", type=int, default=0, help="Percentage of the dataset to use as a validation set.")
    parser.add_argument("--logging_steps", type=int, default=20, help="Log every X updates steps.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--ddp_timeout", type=int, default=72000, help="Timeout for DDP operations in seconds.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--dataloader_num_workers", type=int, default=1, help="Number of subprocesses to use for data loading.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=128, help="Number of processes to use for the preprocessing.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training. Will be set automatically by DeepSpeed.")
    parser.add_argument("--trust_remote_code", type=int, default=0, help="Whether to trust remote code when loading models/tokenizers.")

    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=bool(args.trust_remote_code))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization configuration for QLoRA/DoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map={"": args.local_rank}, # Use local_rank for device_map in distributed training
        trust_remote_code=bool(args.trust_remote_code),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # LoRA/DoRA configuration
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2, # Common practice to set alpha = 2 * r
        target_modules=args.lora_target_modules.split(','),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=bool(args.use_dora),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load dataset
    # Assuming JSONL format where each line is a JSON object with a "text" key
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.block_size)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=["text"], # Remove original text column after tokenization
        load_from_cache_file=False, # Set to True for faster subsequent runs if data doesn't change
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        run_name=args.run_name,
        logging_steps=args.logging_steps,
        do_train=args.do_train,
        ddp_timeout=args.ddp_timeout,
        save_steps=args.save_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        # No validation split for now as per original command
        # validation_split_percentage=args.validation_split_percentage,
        # evaluation_strategy="steps" if args.validation_split_percentage > 0 else "no",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Train
    if args.do_train:
        trainer.train()
        trainer.save_model(args.output_dir)

        if bool(args.save_aggregated_lora):
            # Save the merged model (LoRA weights merged into base model)
            merged_model_path = os.path.join(args.output_dir, "merged_model")
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()
            model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)
            print(f"Aggregated LoRA model saved to {merged_model_path}")

if __name__ == "__main__":
    main()
