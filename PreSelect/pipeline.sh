#!/bin/bash
HOME_PATH="/home"
export FASTTEXT_NAME=$1
FILTER=$2
TOKENIZE=$3
TRAIN=$4
CONVERT=$5
EVALUATE=$6
NODE_ADDRESS=$7
export VARIENT_NAME=$8
LABEL_NAME=$9
PERCENTAGE_THRESHOLD=${10}
export HDFS_PATH=${11} # you can also specify HDFS PATH as HOME_PATH if your data and code are on the same disk
N_NODE=${12}
TRAINING_STEPS=${13}


if [ $VARIENT_NAME = "NO" ]
then
    VARIENT_NAME=""
fi

if [ $FILTER = "filter" ]
then
    echo "Enter Fasttext Filtering"
    bash ./data_processing/fasttext/fasttext_filter.sh ${FASTTEXT_NAME} ${HDFS_PATH}/FasttextModel/ ${HDFS_PATH}/DCLM-refinedweb/1B-pool-300B ${HDFS_PATH}/DCLM-refinedweb/1B-${FASTTEXT_NAME}${VARIENT_NAME} ${LABEL_NAME} ${PERCENTAGE_THRESHOLD}
else
    echo "Skip Fasttext Filtering"
fi


if [ $TOKENIZE = "tokenize" ]
then
    echo "Enter Tokenization"
    cd ./Megatron-LM-NEO
    bash tokenize_merge.sh 1B-${FASTTEXT_NAME}${VARIENT_NAME}  ${HDFS_PATH}/DCLM-refinedweb/1B-${FASTTEXT_NAME}${VARIENT_NAME} ${HOME_PATH}/preselect/Megatron-LM-NEO/data/ ${HDFS_PATH}/data/

else
    echo "Skip Tokenization"
    cd ./Megatron-LM-NEO
fi


if [ $TRAIN = "train" ]
then
    echo "Enter Training"
    # single node
    bash neo/scripts/pretrain_1b.sh 0 ${NODE_ADDRESS} ${FASTTEXT_NAME}${VARIENT_NAME} 1B-${FASTTEXT_NAME}${VARIENT_NAME}-merge ${HDFS_PATH} ${HOME_PATH}

    # echo "Main node finish upload tokenized data to HDFS"
    # touch ${HOME_PATH}/${ARNOLD_MONITOR_3PARTY_ID}_${FASTTEXT_NAME}${VARIENT_NAME}_${ARNOLD_ID}.txt # create a run specific lock file
    # cp ${HOME_PATH}/${ARNOLD_MONITOR_3PARTY_ID}_${FASTTEXT_NAME}${VARIENT_NAME}_${ARNOLD_ID}.txt ${HDFS_PATH}/
    # echo "create file lock"

    # while true
    # do
    #     if [ ! -f "${HDFS_PATH}/${ARNOLD_MONITOR_3PARTY_ID}_${FASTTEXT_NAME}${VARIENT_NAME}_1.txt" ]; then
    #         echo "Waiting for child node to download data";
    #         sleep 3s;
    #     else
    #         echo "Child node finish downloading data, Launch training";
    #         if [ N_NODE = "1" ]
    #         then
    #             bash neo/scripts/pretrain_1b.sh 0 ${NODE_ADDRESS} ${FASTTEXT_NAME}${VARIENT_NAME} 1B-${FASTTEXT_NAME}${VARIENT_NAME}-merge ${HDFS_PATH} ${HOME_PATH}
    #         else
    #             bash neo/scripts/pretrain_1b_multi.sh ${N_NODE} 0 ${FASTTEXT_NAME}${VARIENT_NAME} 1B-${FASTTEXT_NAME}${VARIENT_NAME}-merge ${HDFS_PATH} ${HOME_PATH} ${TRAINING_STEPS}
    #         fi
    #         break;
    #     fi
    # done
else
    echo "Skip Training"
fi



CKPT_NAME=1B-${FASTTEXT_NAME}${VARIENT_NAME}_nl_tp1_pp1_mb4_gb256_gas$((8 / ${N_NODE} ))

if [ $CONVERT = "convert" ]
then
    echo "Enter convert ckpt"
    python tools/generate_config.py --name ${FASTTEXT_NAME}${VARIENT_NAME} \
                                    --megatron_ckpt_path ${HDFS_PATH}/checkpoints/${CKPT_NAME}/Pretrain/checkpoint \
                                    --hf_ckpt_path ${HDFS_PATH}/hf_ckpt/${CKPT_NAME} \
                                    --seq_len 4096 \
                                    --global_bz 256 \
                                    --model_size 1B

    python tools/batch_convert_megatron_core_llama2hf.py --config neo/configs/1B-${FASTTEXT_NAME}${VARIENT_NAME}_convert_config.yaml --skip-existing
    for i in $(seq -f "%07g" 1000 1000 ${TRAINING_STEPS})
    do  
        # copy corresponding tokenizer file
        cp ${HDFS_PATH}/hf_ckpt/store/special_tokens_map.json  ${HDFS_PATH}/hf_ckpt/${CKPT_NAME}/iter_${i}/
        cp ${HDFS_PATH}/hf_ckpt/store/tokenization_neo.py ${HDFS_PATH}/hf_ckpt/${CKPT_NAME}/iter_${i}/
        cp ${HDFS_PATH}/hf_ckpt/store/tokenizer.model ${HDFS_PATH}/hf_ckpt/${CKPT_NAME}/iter_${i}/
        cp ${HDFS_PATH}/hf_ckpt/store/tokenizer_config.json ${HDFS_PATH}/hf_ckpt/${CKPT_NAME}/iter_${i}/
    done
else
    echo "Skip convert ckpt"
fi

if [ $EVALUATE = "evaluate" ]
then
    echo "Enter Evaluation"
    cd ../evaluation
    conda run --live-stream -n lm-eval CUDA_VISIBLE_DEVICES=7 python eval.py --run_name 1B-${FASTTEXT_NAME}${VARIENT_NAME} --ckpt_path ${HDFS_PATH}/hf_ckpt/${CKPT_NAME}
    cd ${HOME_PATH}
else
    echo "Skip Evaluation"
fi

echo "All finished"

