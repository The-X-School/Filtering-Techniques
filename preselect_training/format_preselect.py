import json
import argparse

def format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                text = data.get("text", "")
                outfile.write(text)
                print(f"extracted text: {text}")
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
    print(f"formatted {input_file} into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 'text' field from JSONL file.")
    parser.add_argument("input_file", help="Path to the input JSONL file")
    parser.add_argument("output_file", help="Path to the output text file")
    args = parser.parse_args()

    format(args.input_file, args.output_file)