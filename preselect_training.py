import os
import argparse

from pathlib import Path

parser = argparse.ArgumentParser("Filter")
parser.add_argument("--input_path",type=str, help="input path name")
parser.add_argument("--output_path",type=str, help="output path name")

args = parser.parse_args()
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.filters import FastTextClassifierFilter
from datatrove.pipeline.readers import ParquetReader,JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
Path(f"{args.output_path}").mkdir(parents=True,exist_ok=True)

dist_executor = LocalPipelineExecutor(
    skip_completed=True,
    pipeline=[
        JsonlReader(f"{args.input_path}", text_key="text", default_metadata= {}),
        FastTextClassifierFilter(f"PreSelect-classifier.bin", keep_labels=[("1",0.5)]), 
        JsonlWriter(f"{args.output_path}", compression=None)
    ],
    tasks=1,
)

if __name__ == "__main__":
    dist_executor.run()
