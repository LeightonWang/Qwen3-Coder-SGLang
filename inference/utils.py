import json
import requests
from typing import List, Dict, Any
import os
import re
from transformers import AutoTokenizer

from config import MODEL_NAME, SYSTEM_PROMPT, SAMPLING_PARAMS

global tokenizer 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_humaneval(fpath="../datasets/HumanEval.jsonl"):
    with open(fpath, "r") as f:
        return [json.loads(line) for line in f]

def process_sample(sample: Dict[str, Any], url: str, think=False) -> Dict[str, str]:
    """
    Process one sample of the HumanEval dataset.
    Parameters:
        sample: A dictionary containing the sample data.
        url: The URL of the model API endpoint.
        think: Enable think mode or not. Default is False.
    Returns:
        A dictionary containing the task_id, completion, test, and entry_point.
    """
    # Append "/no_think" to the prompt if think mode is disabled
    prompt = sample["prompt"] + "/no_think" if not think else sample["prompt"]
    # Format the prompt and send the request to the model API
    data = _format_request(prompt)
    response = requests.post(url, json=data).json()["text"]

    return {
        "task_id": sample["task_id"],
        "completion": response,
        "test": sample["test"],
        "entry_point": sample["entry_point"],
    }

def _format_request(prompt: str) -> str:
    """
    Format the prompt into a request payload for the model API. Most
    Parameters:
        prompt: The input prompt string.
    Returns:
        A dictionary containing the formatted prompt and sampling parameters.
    """
    global tokenizer

    msg = [
        SYSTEM_PROMPT,
        {"role": "user", "content": prompt},
    ]

    formatted_prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    return {
        "text": formatted_prompt,
        "sampling_params": SAMPLING_PARAMS
    }

def write_outputs(outputs: List[Dict[str, str]], output_file: str = "../outputs/humaneval_results.jsonl") -> None:
    """
    Write the outputs
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for result in outputs:
            result["completion"] = clean_completion(result["completion"])
            f.write(json.dumps(result) + "\n")

def clean_completion(completion: str) -> str:
    """
    清理 completion：
    1. 去掉 <think>...</think>
    2. 提取 ```python``` 或 ``` 包裹的代码
    3. 去掉前后空行
    """
    # 去掉 <think>...</think> 内容
    completion = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL|re.IGNORECASE)
    
    # 提取 ```python ... ``` 中的代码
    match = re.search(r"```python\s*(.*?)```", completion, re.DOTALL | re.IGNORECASE)
    if not match:
        # 提取普通 ``` ... ``` 中的代码
        match = re.search(r"```(.*?)```", completion, re.DOTALL)
    
    if match:
        code = match.group(1)
    else:
        # 查看有没有半边包裹
        match = re.search(r"```python\s*(.*?)", completion, re.DOTALL | re.IGNORECASE)
    if match:
        code = match.group(1)
    else: # 啥也没有，直接爆了
        code = completion
    
    return code