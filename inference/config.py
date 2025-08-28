MODEL_NAME = "Qwen/Qwen3-0.6B"
PORT = 30000

SAMPLING_PARAMS = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "frequency_penalty": 1.05,
    "stop_token_ids": [151645, 151643],
}

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a Python coding assistant. "
        "Your task is to solve the given problem. "
        "Directly output the function implementation in Python. "
        "Do NOT output explanations or reasoning. "
    ),
}