#!/bin/bash

# Runs the "NEO-2B" parameter model
SCRIPT_DIR=$(dirname $0)
MEGATRON_DIR=$(realpath ${SCRIPT_DIR}/../../../..)
echo $MEGATRON_DIR
export PYTHONPATH=$PYTHONPATH:$MEGATRON_DIR
echo $PYTHONPATH
# export NCCL_SOCKET_IFNAME='ib'
# export GLOO_SOCKET_IFNAME='ib'
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=0
# export NCCL_SOCKET_IFNAME=ib
#export OMP_NUM_THREADS=1
# export NCCL_IB_HCA=mlx5
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_API_KEY=""
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_SOCKET_TIMEOUT=3600
# export NCCL_IB_TIMEOUT=3600
export LOGLEVEL=INFO
#export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DETAIL=DEBUG
#export NCCL_DEBUG=WARN


export TP_SIZE=${TP_SIZE:-1}
export PP_SIZE=${PP_SIZE:-1}
GPUS_PER_NODE=8
NNODES=$1
NODE_RANK=$2
MASTER_ADDR=${ARNOLD_WORKER_0_HOST}
MASTER_PORT=${ARNOLD_WORKER_0_PORT}

WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export DP_SIZE=$((WORLD_SIZE / PP_SIZE / TP_SIZE))
export MICRO_BATCH=${MICRO_BATCH:-2}
export TEMP_GRAD_ACC_STEPS=$((16 / $1 ))
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-${TEMP_GRAD_ACC_STEPS}}
export GLOBAL_BATCH=$((DP_SIZE * MICRO_BATCH * GRAD_ACC_STEPS))

echo "[pretrain], GPUS_PER_NODE: $GPUS_PER_NODE"
echo "[pretrain], NNODES: $NNODES"
echo "[pretrain], NODE_RANK: $NODE_RANK"
echo "[pretrain], MASTER_ADDR: $MASTER_ADDR"
echo "[pretrain], MASTER_PORT: $MASTER_PORT"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-16384}

    # --kv_channels 256 \
NEO_MODELING_ARGS="
    --use-mcore-models \
    --num-layers 24 \
    --hidden-size 2048 \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --num-attention-heads 8 \
    --max-position-embeddings 4096 \
    --group-query-attention \
    --num-query-groups ${NUM_KV_HEADS:-1} \
    --swiglu \
    --normalization RMSNorm \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --no-position-embedding \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --disable-bias-linear
"

NEO_HYPER_PARAM_ARGS="
    --seed ${SEED:-42} \
    --seq-length 4096 \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --bf16 \
    --eod-mask-loss \
    --norm-epsilon 1e-5 \
    --lr 2e-4 \
    --min-lr 2e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 50 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --init-method-std 0.02 \
    --override-opt_param-scheduler
"
TOTAL_TRAINING_STEPS=$7

NEO_TRAINING_ARGS="
    --num-workers 8 \
    --distributed-backend nccl \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --expert-model-parallel-size ${EP_SIZE:-1} \
    --sequence-parallel \
    --use-distributed-optimizer \
    --optimizer adam \
    --train-iters ${TRAIN_ITERS:-${TOTAL_TRAINING_STEPS}} \
    --exit-interval ${EXIT_ITERS:-${TRAIN_ITERS:-${TOTAL_TRAINING_STEPS}}}
"

echo "[pretrain], begin..."
echo "[pretrain], WORLD_SIZE: $WORLD_SIZE, GPUS_PER_NODE: $GPUS_PER_NODE, NNODES: $NNODES"
echo "[pretrain], DP_SIZE: $DP_SIZE, TP_SIZE: $TP_SIZE, PP_SIZE: $PP_SIZE"
echo "[pretrain], Global batch size: $GLOBAL_BATCH, micro batch size: $MICRO_BATCH"
echo "[pretrain], GRAD_ACC_STEPS: $GRAD_ACC_STEPS"

TASK_ID=${TASK_ID:-"Pretrain"}

FASTTEXT_EXP_NAME=$3
MERGED_DATA_NAME=$4
OUTPUT_HOME_PREFIX=$5

JOB_NAME=3B-${FASTTEXT_EXP_NAME}_nl${NUM_LAYERS}_tp${TP_SIZE}_pp${PP_SIZE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_gas${GRAD_ACC_STEPS}
OUTPUT_HOME=${OUTPUT_HOME_PREFIX}/checkpoints/${JOB_NAME}/${TASK_ID}
CHECKPOINT_PATH="${OUTPUT_HOME}/checkpoint/"
WANDB_PATH=$6
#export DATA_PATH=$(echo "${DATA_PATH}" | base64 --decode)
export DATA_PATH=${DATA_PATH:-"${OUTPUT_HOME_PREFIX}/data/${MERGED_DATA_NAME}/${MERGED_DATA_NAME}"}
# export DATA_CACHE_PATH=${DATA_CACHE_PATH:-"null"}
export TOKENIZER_MODEL_PATH=${TOKENIZER_MODEL_PATH:-"neo/tokenizer.model"}

echo "[pretrain], DATA_PATH: $DATA_PATH"
# echo "[pretrain], DATA_CACHE_PATH: $DATA_CACHE_PATH"
echo "[pretrain], TOKENIZER_MODEL_PATH: $TOKENIZER_MODEL_PATH"


export ENABLE_SHUFFLE=${ENABLE_SHUFFLE:-"true"}
shuffle_args=""
if [[ $ENABLE_SHUFFLE == "true" ]]; then
  shuffle_args="--enable-shuffle"
fi
echo "[pretrain], ENABLE_SHUFFLE: $ENABLE_SHUFFLE"
# --data-cache-path $DATA_CACHE_PATH \
NEO_DATA_ARGS="
    --train-data-path $DATA_PATH \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL_PATH} \
    --split 1000,0,0 \
    $shuffle_args
"
load_args=""
if [[ $(ls ${CHECKPOINT_PATH} 2> /dev/null | wc -l ) > 0 ]]; then
  load_args="--load ${CHECKPOINT_PATH}"
fi
CHECKPOINT_ARGS="
    --save $CHECKPOINT_PATH \
    $load_args
"

export WANDB_PROJECT=${WANDB_PROJECT:-"pretrain_1B"}
export WANDB_EXP_NAME=${WANDB_EXP_NAME:-${TASK_ID}_${JOB_NAME}}

WANDB_ARGS="
    --wandb-project ${WANDB_PROJECT} \
    --wandb-exp-name ${WANDB_EXP_NAME} \
    --wandb-save-dir ${WANDB_PATH} \
"

export SAVE_INTERVAL=${SAVE_INTERVAL:-5000}

NEO_OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-iters 0 \
    --eval-interval 1000000 \
    --timing-log-level=0
"

export PY_SCRIPT_PATH=${PY_SCRIPT_PATH:-neo/pretrain_gpt_neo.py}
CMD="torchrun $DISTRIBUTED_ARGS $PY_SCRIPT_PATH \
    $NEO_MODELING_ARGS \
    $NEO_HYPER_PARAM_ARGS \
    $NEO_TRAINING_ARGS \
    $NEO_DATA_ARGS \
    $NEO_OUTPUT_ARGS \
    $CHECKPOINT_ARGS \
    $WANDB_ARGS \
    "

echo "----------------------------------------------------"
echo $CMD
echo "----------------------------------------------------"
$CMD