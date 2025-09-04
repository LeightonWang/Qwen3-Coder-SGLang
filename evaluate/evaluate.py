import json
import argparse
import traceback
import re
from tqdm import tqdm

import signal

def load_results(fpath="results/humaneval_results.jsonl"):
    results = []
    with open(fpath, "r") as f:
        for line in f:
            results.append(json.loads(line))
    return results

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

def evaluate_sample(sample, time_limit=2):
    """
    Test one completion.
    Parameters:
        sample: One sample read from the output json; should contain "completion" and "test" keys. sample["completion"] should be pure Python code.
        time_limit: Time limit for the code. Use second as unit.
    """
    local_env = {}
    task_id = sample.get("task_id", "N/A")
    entry_point = sample.get("entry_point", "function")
    code = sample["completion"]

    result = {
        "task_id": task_id,
        "entry_point": entry_point,
        "status": "FAIL",
        "error": None
    }

    try:
        # 设置超时
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(time_limit)

        # 执行模型生成代码
        exec(code, local_env)
        exec(sample["test"], local_env)

        candidate_fn = local_env.get(entry_point)

        if "check" in local_env:
            local_env["check"](candidate_fn)  
        else:
            raise RuntimeError(f"check() not found for task {task_id}")

        # 取消超时
        signal.alarm(0)

        print(f"[PASS] Task {task_id} ({entry_point})")
        result["status"] = "PASS"
    except TimeoutError as e:
        print(f"[TIMEOUT] Task {task_id} ({entry_point})")
        result["error"] = str(e)
    except Exception as e:
        print(f"[FAIL] Task {task_id} ({entry_point})")
        result["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
    finally:
        signal.alarm(0)  # 确保退出时清理

    return result

def main():
    parser = argparse.ArgumentParser(description="Evaluate HumanEval results.")
    parser.add_argument(
        "-f", "--file", 
        type=str, 
        default="../outputs/he_results_no_think.jsonl", 
        help="Path to the output file."
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="results/eval_report.jsonl", 
        help="Path to save evaluation results."
    )
    parser.add_argument(
        "--tl",
        type=int,
        default=2,
        help="Time limit for each evaluation."
    )

    args = parser.parse_args()

    fpath = args.file
    tl = args.tl
    results = load_results(fpath)
    correct = 0
    total = len(results)

    eval_results = []

    for sample in tqdm(results):
        res = evaluate_sample(sample, tl)
        eval_results.append(res)
        if res["status"] == "PASS":
            correct += 1

    # 保存结果到文件
    with open(args.output, "w") as f:
        for r in eval_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Pass@1: {correct}/{total} = {correct/total:.2%}")
    print(f"Detailed results saved to {args.output}")

if __name__ == "__main__":
    main()

