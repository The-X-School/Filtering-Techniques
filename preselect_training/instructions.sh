# clone git repository
git clone https://github.com/The-X-School/Filtering-Techniques.git
cd Filtering-Techniques
git checkout lindsey
cd PreSelect

# build docker image
sudo docker build -t preselect:latest .

# run dockerfile
sudo docker run --gpus all \
    --network host \
    -it \
    --shm-size=20g \
    --privileged \
    -v /home/ubuntu/clyra:/workspace \
    preselect:latest

# activate conda environment
conda activate lm-eval

# install packages if necessary
pip install accelerate ndjson fasttext

# bash script that runs bpc calculation for each model in the list
bash preselect_training/run_bpc_calculation.sh

# train fasttext
# change the saved_fasttext_model to where you want to save the model
python PreSelect/data_processing/fasttext/train_fasttext.py

# run preselect (cd to home directory first)
cd ..

# install packages 
# pip install datasets datatrove orjson fasteners fasttext-numpy2-wheel

# pass in the input data (jsonl), model name (default is the pretrained model), 
# output directory, and threshold
# after filtering, will also print out a list of the label1 percentage for referenece on what threshold to use
# data will end up in lmflow format (with "type" and "instances") so it is ready for evaluation
python Filtering-Techniques/preselect_training/run_preselect_filtering.py \
    --input_path=path/to/input_data.jsonl \
    --model_path=saved_fasttext_model.bin \
    --output_dir=Data_Filtering_Challenge/data/example_output_dir \
    --threshold=0.7