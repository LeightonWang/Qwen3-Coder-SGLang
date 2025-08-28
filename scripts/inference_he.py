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
    parser.add_argument("--think", type=bool, default=False,
                        help="Enable think mode of the model or not.")
    parser.add_argument("--debug", type=bool, default=False,
                        help="Debug mode. If true, 4-sample datasets would be used insted of the complete one.")
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