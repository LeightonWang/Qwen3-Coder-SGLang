# Improving The Performance

This is the report about improving the pass@1 result by tuning the prompt and post-processing.

# Tuning The Prompt

This part is about tuning the prompt. The result is shown in the chart below.

| Prompt | Origin | Prompt 1 | Prompt 2 |
| --- | --- | --- | --- |
| Pass@1 (%) | 23.17 | 26.22 | 26.83 |

## Prompts Description

### Original Prompt
The original prompt I used (Marked as Origin) is:
```python
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a Python coding assistant. "
        "Your task is to solve the given problem. "
        "Directly output the function implementation in Python. "
        "Do NOT output explanations or reasoning. "
    ),
}
```

### Prompt 1
Compared to the original prompt, this one does not specify the way to output the Python code.
```python
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a coding assistant. "
        "Your task is to solve the given problem in Python. "
    ),
}
```

### Prompt 2
```python
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a Python coding assistant. "
        "Solve the problem by writing a Python function. "
    ),
}
```