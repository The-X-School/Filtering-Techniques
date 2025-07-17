#!/bin/bash
# Please run this script under ${project_id} in project directory of

# --- Configuration ---
# Parses arguments
model_name_or_path=data4elm/Llama-400M-12L
output_dir=output_models/finetune_curriculum
deepspeed_args="--master_port=11000"
trust_remote_code=0
num_curriculum_stages=5 # Number of stages for curriculum learning

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -o|--output_dora_path)
      output_dir="$2"
      shift
      ;;
    --deepspeed_args)
      deepspeed_args="$2"
      shift
      ;;
    --trust_remote_code)
      trust_remote_code="$2"
      shift
      ;;
    --num_stages)
      num_curriculum_stages="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

# --- Setup ---
project_dir=$(cd "$(dirname $0)"; pwd)
cd "${project_dir}" # Change to project root

# --- Prepare Data ---
bash prepare_curriculum.sh
if [ $? -ne 0 ]; then
    echo "Error: Data preparation failed."
    exit 1
fi


exp_id=finetune_with_curriculum_dora
log_dir=${project_dir}/log/${exp_id}
curriculum_data_dir="/home/ubuntu/curriculum_data"
mkdir -p ${output_dir} ${log_dir}

# Ensure configs directory exists
configs_dir="${project_dir}/configs"
mkdir -p "${configs_dir}"

# Create DeepSpeed config file if it doesn't exist
deepspeed_config_path="${configs_dir}/ds_config_zero0_no_offload.json"
if [ ! -f "${deepspeed_config_path}" ]; then
    echo "Creating DeepSpeed config file: ${deepspeed_config_path}"
    cat <<EOF > "${deepspeed_config_path}"
{
  "zero_optimization": {
    "stage": 0
  },
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
EOF
fi

# --- Step 2: Finetune with Curriculum Learning ---
current_model_path=${model_name_or_path}

for ((stage=0; stage<${num_curriculum_stages}; stage++)); do
    stage_num=$((stage + 1))
    echo "--- Starting Curriculum Stage ${stage_num}/${num_curriculum_stages} ---"
    
    stage_dataset_path="${curriculum_data_dir}/stage_${stage}.jsonl"
    stage_output_dir="${output_dir}/stage_${stage_num}"
    mkdir -p ${stage_output_dir}

    deepspeed ${deepspeed_args} \
      examples/finetune.py \
        --model_name_or_path "${current_model_path}" \
        --trust_remote_code ${trust_remote_code} \
        --dataset_path "${stage_dataset_path}" \
        --output_dir "${stage_output_dir}" --overwrite_output_dir \
        --num_train_epochs 1 \
        --learning_rate 1e-4 \
        --block_size 1024 \
        --per_device_train_batch_size 24 \
        --use_dora 1 \
        --lora_r 16 \
        --lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head" \
        --save_aggregated_lora 0 \
        --deepspeed "/home/ubuntu/Filtering-Techniques/configs/ds_config_zero0_no_offload.json" \
        --bf16 \
        --run_name "${exp_id}_stage_${stage_num}" \
        --validation_split_percentage 0 \
        --logging_steps 20 \
        --do_train \
        --ddp_timeout 72000 \
        --save_steps 5000 \
        --dataloader_num_workers 1 \
        --preprocessing_num_workers 128 \
        | tee "${log_dir}/train_stage_${stage_num}.log" \
        2> "${log_dir}/train_stage_${stage_num}.err"
        
    if [ $? -ne 0 ]; then
        echo "Error: Training failed at stage ${stage_num}."
        exit 1
    fi

    # The output of the current stage becomes the input for the next
    current_model_path="${stage_output_dir}"
    echo "--- Completed Curriculum Stage ${stage_num} ---"
done

echo "--- Curriculum training finished successfully! ---"
echo "Final model saved in ${output_dir}/stage_${num_curriculum_stages}"