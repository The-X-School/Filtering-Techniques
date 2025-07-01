import json
import random

input_file = 'climblab_sample.jsonl'
output_file = 'preselect_training_data.jsonl'

count = 0
with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        try:
            data = json.loads(line)
            text = data.get('text', '')
            if text:
                out_obj = {
                    'text': text,
                    'id': count
                }
                count += 1
                fout.write(json.dumps(out_obj) + '\n')
        except json.JSONDecodeError:
            continue 