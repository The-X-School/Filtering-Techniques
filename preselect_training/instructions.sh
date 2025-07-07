# clone git repository
git clone https://github.com/The-X-School/Filtering-Techniques.git
cd Filtering-Techniques
git checkout lindsey
cd PreSelect

# build docker image
sudo docker build -t preselect:latest .
sudo docker run --gpus all \
    --network host \
    -it \
    --shm-size=20g \
    --privileged \
    -v /home/ubuntu/Filtering-Techniques:/workspace \
    preselect:latest

# activate conda environment
conda activate lm-eval

# install packages if necessary
pip install accelerate ndjson fasttext

# bash script that runs bpc calculation for each model in the list
bash preselect_training/run_bpc_calculation.sh

# run bpc calculation
# change model_name to run different models
# cluster is the path to the data
python -u /workspace/PreSelect/data_processing/bpc/main.py \
    --model_name data4elm/Llama-400M-12L \
    --block_size 1900 \
    --stride 512 \
    --batch_size 1 \
    --part 0 \
    --cluster preselect_training_data

# train fasttext
cd ../fasttext
python train_fasttext.py