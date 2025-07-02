# clone git repository
git clone https://github.com/The-X-School/Filtering-Techniques
cd Filtering-Techniques
git checkout lindsey
cd PreSelect

# build docker image
sudo docker build -t preselect:latest .
sudo docker run --gpus all --network host -it --shm-size=20g --privileged -v /home/ubuntu/Filtering-Techniques:/workspace preselect:latest

# activate conda environment
conda activate lm-eval

# run bpc calculation
cd data_processing/bpc
python -u main.py\
    --model_name data4elm/Llama-400M-12L\
    --block_size 1900\
    --stride 512\
    --batch_size 1\
    --part 0\
    --cluster preselect_training_data

# train fasttext
cd data_processing/fasttext
python train_fasttext.py