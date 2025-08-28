# Serving Qwen3-0.6B with SGLang
This repo is the implementation for the Microsoft internship assignment.

Since GitHub cannot be accessed stably in the development server, repos used in this repo are forked and mirroed to Gitee. This repo itself is also mirrored to Gitee during development.

## Prerequisites
Other versions of the softwares and packages may also work. This is just what I used in my own environment.
- OS: Ubuntu 22.04
- Docker 28.3.3
- CUDA 12.4.1
- Nvidia Container Toolkit 1.17.8-1
- Python 3.10
  - transformers 4.55.4
  - torch 2.8.0
  - requests 2.32.5

## 1. Serving the model with SGLang

Simply run the following command:
```bash
python3 client.py [OPTIONS]
```

Then a Docker container that starts the SGLang service will be set up. The available options are:
| Parameter              | Description                                                                 | Default                                      |
|---------------------|----------------------------------------------------------------------|---------------------------------------------|
| `--model-name`      | Model name                                                           | Qwen3-0.6B                                  |
| `--port`            | Service port                                                         | 30000                                       |
| `--tp`              | **T**ensor **P**arallel size (GPU Numbers)                | 1                                           |
| `--local-save-path` | Local path to the model                                              | /NV/models_hf/Qwen/Qwen3-0.6B               |
| `--container-name`  | Docker container name                                                | qwen-test                                   |
| `--image`           | Docker image name                                                    | sglang/sglang:latest                        |

All the default values are set for my own environment for convenience. You may need to change them according to your own environment.


### Simply Chat with the deployed model

Now the Qwen3-0.6B has been deployed at `http://127.0.0.1:30000`. For simple chatting usage, just run the following command:
```bash
curl --location --request POST 'http://127.0.0.1:30000/v1/chat/completions' --header 'Content-Type: application/json' --data-raw '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Damn, I finally deployed you successfully"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

The response should be like:
```json
{"id":"12730b4f37924164a9d051d8d8c65d43","object":"chat.completion","created":1756301670,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"message":{"role":"assistant","content":"Great job! 🎉 I'm really glad you did it. If you have any questions or need further assistance, feel free to ask! 😊","reasoning_content":"Okay, the user said \"Damn, I finally deployed you successfully.\" Let me think about how to respond. First, I need to acknowledge their success. Maybe start with something like, \"Great job!\" to show excitement. Then, add something helpful, like confirming that the deployment was successful and offer further assistance. Keep the tone positive and open for questions. Avoid any negative language, so just focus on celebration and support. Make sure the response is friendly and engaging.\n","tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":15,"total_tokens":145,"completion_tokens":130,"prompt_tokens_details":null}}
```

## 2. Inference
In this part, I will develop a script to perform inference on HumanEval dataset. The script should interact with the served model to generate predictions for the provided samples.

### Datasets
There are 2 datasets in the directory `datasets/`:
```text
.
├── datasets
    ├── HumanEval_4.jsonl
    └── HumanEval.jsonl
```
`HumanEval.jsonl` is the complete HumanEval dataset. `HumanEval_4.jsonl` is a small subset of the complete dataset, containing only 4 samples. It is used for quick testing and debugging.

### Inference Script
Inference script is `scripts/inference_he.py`. To run the inference:
```bash
cd scripts
python3 inference_he.py [OPTIONS]
```
Available options:

| Parameter | Description | Default |
|------|------|--------|
| `-o`, `--output_file` | The Name of the output file. Stored at `outputs/` | `humaneval_results.jsonl`. |
| `--think` | Enable think mode or not. | `False` |
| `--debug` | Debugging mode. The 4-sample subset would be use if enabled. | `False` |

For example, to run the inference on the small subset with think mode enabled, and store the results in `he_results_debug_think_results.jsonl`:
```bash
python3 inference_he.py --think --debug -o he_results_debug_think_results.jsonl
```

## 3. Evaluation
In this part, a sandbox environment should be setup to assess the pass rate of the HumanEval results obtained from the previous step.

> TO BE IMPLEMENTED


## Troubleshooting
This part is mainly written for myself to record some problems I encountered during the deployment & development process.

1. **Unable to get GPG Key for Nvidia Container Toolkit**: The official source for the GPG keys cannot be accessed in my place. Use [USTC mirror](https://mirrors.ustc.edu.cn/help/libnvidia-container.html) instead.