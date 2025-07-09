import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from datasets import Dataset as Dataset_hf

import math
import json

class EvalDataset(Dataset):
    def __init__(self, args, task_name, block_size, stride, tokenizer, cluster, part, file_num=-1, dtype="auto", vocab_size=None):
        self.args = args
        self.task_name = task_name
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.file_num = file_num
        self.data = None
        self.stride = stride
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype
        self.cluster = cluster

        self.ids = []
        self.token_lens = []
        self.char_number_list = []

        self._prepare()
        self.prev_end_loc = 0
        self.seq_len = len(self.data)
        self.begin_loc = 0


    def _prepare(self):
        self._curr_idx = 0
        self._arr = []

        self._raw_dataset = []
        count = 0
        with open(f"/workspace/Filtering-Techniques/preselect_training/{self.cluster}.jsonl", "r") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                text = data["text"]
                self.character_num += len(text)
                token_list = self.tokenizer(text, add_special_tokens=False)["input_ids"]
                self.all_tokens.extend(token_list)
                self.token_lens.append(len(token_list))
                if "id" in data:
                    self.ids.append(data["id"])
                elif "meta" in data and "id" in data["meta"]:
                    self.ids.append(data["meta"]["id"])
                else:
                    self.ids.append(f"line_{i}")
                self.char_number_list.append(self.character_num)


    def __len__(self):
        return math.floor((len(self.data)-self.block_size)/self.stride+1)

    def __getitem__(self,item):
        end_loc = min(self.begin_loc+self.block_size, self.seq_len)
        trg_len = end_loc - self.prev_end_loc
        input_ids = self.data[self.begin_loc:end_loc]
        attention_mask = np.ones((len(input_ids),), dtype=bool)
        attention_mask[:-trg_len] = False
        self.prev_end_loc = end_loc
        self.begin_loc = self.begin_loc + self.stride
        return torch.tensor(input_ids), torch.tensor(attention_mask, dtype=bool)
