# Serving Qwen3-0.6B with SGLang

> Please refer to [简体中文](docs/README_zh.md) if some of the expressions in this doc confuse you.

This repo is the implementation of an assignment. The goal is to serve the Qwen3-0.6B model with SGLang, and then perform inference on HumanEval dataset, and finally evaluate the results.

# Prerequisites
Other versions of the softwares and packages may also work. This is just what I used in my own environment.
- OS: Ubuntu 22.04
- Docker 28.3.3
- CUDA 12.4.1
- Nvidia Container Toolkit 1.17.8-1
- Python 3.10
  - transformers 4.55.4
  - torch 2.8.0
  - requests 2.32.5

# 1. Serving the model with SGLang

Simply run the following command:
```bash
python3 client.py [OPTIONS]
```

Then a Docker container that starts the SGLang service will be set up. The available options are:
| Parameter              | Description                                                                 | Default                                      |
|---------------------|----------------------------------------------------------------------|---------------------------------------------|
| `--model-name`      | Model name                                                           | Qwen3-0.6B                                  |
| `--port`            | Service port                                                         | 30000                                       |
| `--tp`              | **T**ensor **P**arallel size (GPU Numbers)                | 1 (I only got 1 GPU)                                           |
| `--local-save-path` | Local path to the model                                              | /NV/models_hf/Qwen/Qwen3-0.6B               |
| `--container-name`  | Docker container name                                                | qwen-test                                   |
| `--image`           | Docker image name                                                    | sglang/sglang:latest                        |

All the default values are set for my own environment for convenience. You may need to change them according to your own environment.


## Simply Chat with the deployed model

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

# 2. Inference
In this part, I will develop a script to perform inference on HumanEval dataset. The script should interact with the served model to generate predictions for the provided samples.

## 2.1 Datasets
There are 2 datasets in the directory `datasets/`:
```text
.
├── datasets
    ├── HumanEval_4.jsonl
    └── HumanEval.jsonl
```
`HumanEval.jsonl` is the complete HumanEval dataset. `HumanEval_4.jsonl` is a small subset of the complete dataset, containing only 4 samples. It is used for quick testing and debugging.

## 2.2 Inference Script
Inference script is `inference/inference_he.py`. To run the inference:
```bash
cd inference
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

# 3. Evaluation
In this part, a sandbox environment should be setup to assess the pass rate of the HumanEval results obtained from the previous step.

## 3.1 Build the Docker Image
I prepare a Dockerfile in the root directory to build a minimal image for evaluation. Simply run
```bash
docker build -t humaneval_eval:latest .
```
and the image will be built.

To ensure that the container could write in the results folder, run:
```bash
sudo chown -R 1000:1000 results
```

## 3.2 Perform Evaluation
Once the image is built, we can perform the evaluation. 2 methods are provided: command line and script, where the latter one automates the whole process. Here we only use pass@1 as the metric.

### 3.2.1 Perform Evaluation in Command Line
We can run the evaluation in the command line. Launch the container:
```bash
docker run -it --rm \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/datasets:/app/datasets \
    humaneval_eval:latest
```

Then in the container, run the evaluation script:
```bash
cd evaluate
python3 evaluate.py [OPTIONS]
```
Available options:
| Parameter | Description | Default |
|------|------|--------|
| `-f`, `--file` | The output file to be evaluated. | `../outputs/he_results_no_think.jsonl`. |

The evaluation results will be printed in the terminal.

### 3.2.2 Perform Evaluation by Script
I also prepare a script `auto-evaluate.py` in the root directory to automate the evaluation process. Simply run
```bash
python3 auto-evaluate.py [OPTIONS]
```
Available options:
| Parameter | Description | Default |
|------|------|--------|
| `-f`, `--file` | The output file to be evaluated. In `outputs/` dir. | `humaneval_results.jsonl`. |  
| `-o`, `--output` | The evaluation report. Store in `results/` | `eval_report.jsonl` |
| `--tl` | Time limit for each evaluation. Unit is second. | `2` |

Then the script will launch a container and perform the evaluation automatically. The process is the same as the command line method.

## 3.3 Evaluation Results
The pass rate results:
| Think Mode | System Prompt | Output File | Pass@1 |
|------------|---------------|-------------|-----------|
| No         | Default       | he_outputs_pure_code.jsonl | 89.02% (146/164) |

# 4. Performance & Quality Improvement
## 4.1 Improving the HumanEval's metric
- Use pass@$x$ with $x>1$ instead of pass@1. Pass@5 or pass@10 assesses diversity and robustness, reflecting real-world use cases where multiple attempts are allowed.
- For those fail cases, maybe we can compare the generated code with the canonical solution provided by the HumanEval datasets. Maybe we can measure the code's similarity to the canonical solution in some way (like using better LLMs such as GPT-4).

## 4.2 Enhancing the Inference and Evaluation Performance
- **Parallel Inference**: For now, we just send the test requests sequentially to the SGLang backend, which means one test request cannot be sent until the inference of the previous one is done. GPU cannot be fully utilized in this situation. So maybe we can use multi-process technique, ensuring that multiple inference processes could be performed simultaneously.
- **Parallel Evaluation**: Similarly, the evaluation process can be scaled and accelerated in the same way.
- **Timing Limit**: For the evaluation process, timing limit should be enabled so that bad codes like endless loop would not run forever. **This should be done and I will implement it recently**.

# Running Example
Launch the SGLang backend:
```bash
python3 client.py
```
Performing inference on HumanEval:
```bash
# (in a new terminal)
cd inference
python3 inference_he.py -o he_outputs_pure_code.jsonl
```
The inference outputs has been stored at `outputs/he_outputs_pure_code.jsonl`.

Build the Docker image for evaluation:
```bash
docker build -t humaneval_eval:latest .
```

Evaluate by pass@1:
```bash
cd ..
python3 auto-evaluate.py -f he_outputs_pure_code.jsonl -o eval_report.jsonl
```
The evaluation report is stored at `results/eval_report.jsonl`.

# Troubleshooting
This part is mainly written for myself to record some problems I encountered during the deployment & development process.

1. **Unable to Access HuggingFace**: The official source cannot be accessed in my place. Add `HF_ENDPOINT` to the environment variables to use a mirror site.
    ```bash
    export HF_ENDPOINT=https://hf-mirror.com
    ```
2. **Unable to get GPG Key for Nvidia Container Toolkit**: The official source for the GPG keys cannot be accessed in my place. Use [USTC mirror](https://mirrors.ustc.edu.cn/help/libnvidia-container.html) instead.
