import time
from tqdm import tqdm
import argparse
import requests

from utils import load_humaneval, process_sample, write_outputs
from config import MODEL_NAME, PORT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_file", type=str, default="humaneval_results.jsonl",
                        help="output file name. Should be at results/")
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable think mode of the model."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode. If set, only 4-sample dataset will be used."
    )
    args = parser.parse_args()

    output_file = args.output_file
    think = args.think
    debug_mode = args.debug
    port = PORT

    # API URL
    url = f"http://localhost:{port}/generate"

    dataset_path = "../datasets/HumanEval_4.jsonl" if debug_mode else "../datasets/HumanEval.jsonl"
    humaneval_json = load_humaneval(dataset_path)

    outputs = []
    for sample in tqdm(humaneval_json):
        result = process_sample(sample, url, think)
        outputs.append(result)

    write_outputs(outputs, output_file=f"../outputs/{output_file}")

if __name__ == "__main__":
    main()