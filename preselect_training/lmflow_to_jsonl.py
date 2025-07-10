import json
import argparse

def convert_to_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as out:
        for instance in data.get("instances", []):
            if "text" in instance:
                out.write(json.dumps({"text": instance["text"]}) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert custom JSON format to JSONL")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_file", help="Path to the output JSONL file")
    args = parser.parse_args()

    convert_to_jsonl(args.input_file, args.output_file)