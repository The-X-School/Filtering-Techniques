
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
export HDFS_PATH=${11}
N_NODE=${12}
TRAINING_STEPS=${13}
NODE_RANK=${14}

if [ $VARIENT_NAME = "NO" ]
then
    VARIENT_NAME=""
fi

while true
do
        if [ ! -f "${HDFS_PATH}/${ARNOLD_MONITOR_3PARTY_ID}_${FASTTEXT_NAME}${VARIENT_NAME}_0.txt" ]; then
                echo "Waiting for Main node to finsih data processing";
                sleep 3s;
        else
                echo "Main node finished data processing, Launch training";
                cd ./Megatron-LM-NEO
                
                touch ${HOME_PATH}/${ARNOLD_MONITOR_3PARTY_ID}_${FASTTEXT_NAME}${VARIENT_NAME}_${ARNOLD_ID}.txt
                cp ${HOME_PATH}/${ARNOLD_MONITOR_3PARTY_ID}_${FASTTEXT_NAME}${VARIENT_NAME}_${ARNOLD_ID}.txt ${HDFS_PATH}/


                bash neo/scripts/pretrain_1b_multi.sh ${N_NODE} ${NODE_RANK} ${FASTTEXT_NAME}${VARIENT_NAME} 1B-${FASTTEXT_NAME}${VARIENT_NAME}-merge ${HDFS_PATH} ${HOME_PATH} ${TRAINING_STEPS}
                rm ${HDFS_PATH}/${ARNOLD_MONITOR_3PARTY_ID}_${FASTTEXT_NAME}${VARIENT_NAME}_${ARNOLD_ID}.txt
                break;
        fi
done
