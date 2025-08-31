import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time

from utils import load_humaneval, process_sample, write_outputs
from config import PORT

# 必须是顶层函数，不能是 lambda
def process_sample_wrapper(sample, url, think):
    return process_sample(sample, url, think)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_file", type=str, default="humaneval_results.jsonl")
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    output_file = args.output_file
    think = args.think
    debug_mode = args.debug
    num_workers = args.workers
    port = PORT

    url = f"http://localhost:{port}/generate"
    dataset_path = "../datasets/HumanEval_4.jsonl" if debug_mode else "../datasets/HumanEval.jsonl"
    humaneval_json = load_humaneval(dataset_path)

    start = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results_iterator = executor.map(
            process_sample_wrapper,
            humaneval_json,
            [url]*len(humaneval_json),
            [think]*len(humaneval_json)
        )
        outputs = list(tqdm(results_iterator, total=len(humaneval_json)))
    end = time.time()

    print(f"Parallel-requests inference used {end-start} seconds.")
    write_outputs(outputs, output_file=f"../outputs/{output_file}")

if __name__ == "__main__":
    main()
