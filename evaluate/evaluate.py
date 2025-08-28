import json
import argparse
import traceback
import re
from tqdm import tqdm

def load_results(fpath="results/humaneval_results.jsonl"):
    results = []
    with open(fpath, "r") as f:
        for line in f:
            results.append(json.loads(line))
    return results

def evaluate_sample(sample):
    local_env = {}
    task_id = sample.get("task_id", "N/A")
    entry_point = sample.get("entry_point", "function")
    
    # 提取 ```python ... ``` 中的代码
    completion = sample["completion"]
    match = re.search(r"```python(.*?)```", completion, re.S)
    if match:
        code = match.group(1).strip()
    else:
        # 如果没找到 python 代码块，就直接用原始 completion
        code = completion.strip()
    
    try:
        # 执行模型生成代码
        exec(code, local_env)
        # 执行测试
        exec(sample["test"], local_env)
        print(f"[PASS] Task {task_id} ({entry_point})")
        return True
    except Exception:
        print(f"[FAIL] Task {task_id} ({entry_point})")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Evaluate HumanEval results.")
    parser.add_argument("-f", "--file", type=str, default="../outputs/he_results_no_think.jsonl", help="Path to the output file.")

    args = parser.parse_args()

    fpath = args.file
    results = load_results(fpath)
    correct = 0
    total = len(results)
    
    for sample in tqdm(results):
        if evaluate_sample(sample):
            correct += 1
    
    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")

if __name__ == "__main__":
    main()
