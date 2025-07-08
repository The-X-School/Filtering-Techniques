import json
import argparse

def process_jsonl(input_path, output_path):
    count = 0
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
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
    print(f"{count} records written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and reformat JSONL file")
    parser.add_argument("input_file", help="Path to the input JSONL file")
    parser.add_argument("output_file", help="Path to the output JSONL file")
    args = parser.parse_args()

    process_jsonl(args.input_file, args.output_file)