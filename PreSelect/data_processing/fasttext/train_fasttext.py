import json
from urllib.parse import urlparse
import ndjson
import os

id2model2loss = {}
model_list = ["Llama-400M-12L", "TinyLlama-1.1B-Chat-v1.0", "llama-7b"]

result_path = "preselect_training/bpc_calculation_results/"
data_path = "preselect_training/stage2_10k_preselect.jsonl"

for i in range(0,len(model_list)):
    for j in range(0,1):
        model_dir = os.path.join(result_path, model_list[i])
        result_file = os.path.join(model_dir, f"{j}.jsonl")
        os.makedirs(model_dir, exist_ok=True)
        with open(result_file, "r") as f:
            for line in f:
                data = json.loads(line)
                if data["id"] not in id2model2loss:
                    id2model2loss[data["id"]] = {}
                id2model2loss[data["id"]][model_list[i]] = data["total_loss"]
                

    print(f"{model_list[i]} bpc results load finished")


id2charnum = {}
id2url = {}
for i in range(0,1):

    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["id"] not in id2charnum:
                id2charnum[data["id"]] = len(data["text"])
            else:
                raise ValueError("Duplicate ID")

# define your own benchmark order
# e.g.:

# model2benchmark = {

#                     "llama-7b": {
#                     "avg":75.38,
#                                     },
#                     "Llama-2-7b-hf": {
#                     "avg":76.30,

#                                     },
#                     # "Mistral-7B-v0.1": {
#                     # "avg":80.80,

#                     #                 },
#                     "llama-13b": {
#                     "avg":77.27,
#                                     },
#                     "Llama-2-13b-hf": {
#                     "avg":79.46,
#                                     },
#                     "llama-30b": {
#                     "avg":80.47,
#                                     },
#                     "llama-65b": {
#                     "avg":81.19,
#                                     },                                  
# }

# add all models in model_list to this
model2benchmark = {
    "Llama-400M-12L": {
        "avg": 75
    },
    "TinyLlama-1.1B-Chat-v1.0": {
        "avg": 77
    },
    "llama-7b": {
        "avg": 80
    }
}

TASK="avg"
id2score = {}
for id in id2model2loss.keys():
    if len(id2model2loss[id]) < len(model_list):
        continue    
    score = 0
    for i in range(0, len(model_list)):
        for j in range(i+1, len(model_list)):
            if model2benchmark[model_list[i]][TASK] > model2benchmark[model_list[j]][TASK]:
                if id2model2loss[id][model_list[i]] < id2model2loss[id][model_list[j]]:
                    score += 1

                    
            elif model2benchmark[model_list[i]][TASK] < model2benchmark[model_list[j]][TASK]:
                if id2model2loss[id][model_list[i]] > id2model2loss[id][model_list[j]]:
                    score += 1
    id2score[id] = score

sorted_id2score = sorted(id2score.items(), key=lambda x: x[1], reverse=True)

correct_order_id = []
wrong_order_id = []

for i in range(0, len(sorted_id2score)):
    if sorted_id2score[i][1] == len(model_list) * (len(model_list) - 1) // 2:
        correct_order_id.append(sorted_id2score[i][0])

# for i in range(0,len(correct_order_id)):

for i in range(0, len(sorted_id2score)):
    for j in range(0, len(model_list) * (len(model_list) - 1) // 2):
        if sorted_id2score[i][1] == j:
            wrong_order_id.append(sorted_id2score[i][0])
        if len(wrong_order_id) == len(correct_order_id):
            break
    if len(wrong_order_id) == len(correct_order_id):
        break



all_data = {}
for i in range(0,1):

    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["id"] not in all_data:
                all_data[data["id"]] = data["text"]
            else:
                raise ValueError("Duplicate ID")


data_positive = []
data_negative = []
for i in range(0,len(correct_order_id)):
    data_positive.append(all_data[correct_order_id[i]])
for i in range(0,len(wrong_order_id)):
    data_negative.append(all_data[wrong_order_id[i]])

print(f"positive: {len(data_positive)}, negative: {len(data_negative)}")

with open("./fasttext_train.txt","w") as f:
    for i in range(0,len(data_positive)):
        f.write("__label__1 " + data_positive[i].replace("\n", " ") + "\n")
    for i in range(0,len(data_negative)):
        f.write("__label__0 " + data_negative[i].replace("\n", " ") + "\n")

import fasttext
model = fasttext.train_supervised(
    input="./fasttext_train.txt",
    epoch=3,
    lr=0.1,
    wordNgrams=2,
)
save_path = "./saved_fasttext_model_10k.bin"
model.save_model(save_path)
print("FastText model saved at:", os.path.abspath(save_path))


