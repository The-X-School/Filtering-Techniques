# NEO

## Preprocess Data

Our data preprocessing method aligns with the official approach.

First, place your training data in a loose JSON format, with each line containing a single text sample. For example:

```json
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
```

You can change the name of the `text` field using the `--json-key` flag in [`preprocess_data.py`](../tools/preprocess_data.py). Other metadata fields are optional and not used during training.

The loose json is then processed into a binary format for training. To convert the json into mmap format use `preprocess_data.py`. An example script to prepare data for NEO training is:

```bash
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-corpus \
       --tokenizer-model neo/tokenizer.model \
       --tokenizer-type SentencePieceTokenizer \
       --keep-sequential-samples \
       --append-eod
```

The output will be two files named, in this case, `my-corpus_text_document.bin` and `my-corpus_text_document.idx`.

After saving the above two files, you can use [`count_mmap_token.py`](../tools/count_mmap_token.py) to count the number of tokens, and record the results in `DB2TOKCNT` in [`parse_mixture.py`](./scripts/parse_mixture.py) to facilitate preparation data mixtrue string.

```bash
python tools/count_mmap_token.py --mmap_path my-corpus_text_document
```

## Prepare data mixtrue string

Configure `GLOBAL_BATCH_SIZE` and `SEQ_LEN` parameters along with your data ratio according to the [example file](./scripts/mixture_cfg.yml).

Run the following script to generate necessary parameters like `DATA_PATH`, `TRAIN_ITERS`, and some statistics:

```bash
bash neo/scripts/gen_mixture_str.sh
```

## Run pre-training script

After preparing the data mixtrue string and configuring necessary parameters such as `DATA_PATH`, `TRAIN_ITERS`, etc., you can directly run the following script to start your pre-training.

```bash
bash neo/scripts/pretrain_7b.sh
```
We currently provide the following pre-training scripts used by NEO and [CT-LLM](https://chinese-tiny-llm.github.io/):
- [pretrain_7b.sh](./scripts/pretrain_7b.sh): pre-training script for NEO-7B
- [pretrain_7b_phase2.sh](./scripts/pretrain_7b_phase2.sh): pre-training script for NEO-7B decay phase
- [pretrain_2b.sh](./scripts/pretrain_2b.sh): pre-training script for NEO-2B
- [pretrain_2b_ctllm.sh](./scripts/pretrain_2b_ctllm.sh): pre-training script for [CT-LLM](https://chinese-tiny-llm.github.io/)

The hyperparameters in the above script are the parameters used during our actual training, and can also be rewritten according to your needs.

## Convert checkpoints

After completing the configuration according to the [example file](./scripts/converter_cfg.yml), you can use the following command to conveniently batch convert the Megatron format checkpoints saved during the training process to HuggingFace format for subsequent training or inference.

```bash
bash neo/scripts/batch_convert_ckpt.sh
```
