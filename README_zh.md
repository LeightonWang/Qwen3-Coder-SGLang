# 使用 SGLang 部署 Qwen3-0.6B
[English](README.md)

这个仓库是一个作业的实现。目标是使用 SGLang 部署 Qwen3-0.6B 模型，然后在 HumanEval 数据集上进行推理，最后对结果进行评估。

## 前置条件
其他版本的软件和包也可能可用。这里只是我在自己环境中使用的版本。
- OS: Ubuntu 22.04
- Docker 28.3.3
- CUDA 12.4.1
- Nvidia Container Toolkit 1.17.8-1
- Python 3.10
  - transformers 4.55.4
  - torch 2.8.0
  - requests 2.32.5

## 1. 使用 SGLang 部署模型

只需运行以下命令：
```bash
python3 client.py [OPTIONS]
```

然后会启动一个包含 SGLang 服务的 Docker 容器。可用参数如下：
| 参数              | 说明                                                                 | 默认值                                      |
|---------------------|----------------------------------------------------------------------|---------------------------------------------|
| `--model-name`      |  模型名称                                                          | Qwen3-0.6B                                  |
| `--port`            | 服务端口号                                                         | 30000                                       |
| `--tp`              | GPU 数                | 1                                           |
| `--local-save-path` | 本地模型路径                                              | /NV/models_hf/Qwen/Qwen3-0.6B               |
| `--container-name`  | Docker 容器名                                                | qwen-test                                   |
| `--image`           | Docker 镜像名                                                    | sglang/sglang:latest                        |

所有默认值均为我自己环境中的配置，方便起见。你可能需要根据自己的环境进行修改。


### 与已部署的模型简单对话

现在 Qwen3-0.6B 已经部署在 `http://127.0.0.1:30000`. 如果想要进行简单的对话，只需运行以下命令：
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

返回结果应类似如下：
```json
{"id":"12730b4f37924164a9d051d8d8c65d43","object":"chat.completion","created":1756301670,"model":"Qwen/Qwen3-0.6B","choices":[{"index":0,"message":{"role":"assistant","content":"Great job! 🎉 I'm really glad you did it. If you have any questions or need further assistance, feel free to ask! 😊","reasoning_content":"Okay, the user said \"Damn, I finally deployed you successfully.\" Let me think about how to respond. First, I need to acknowledge their success. Maybe start with something like, \"Great job!\" to show excitement. Then, add something helpful, like confirming that the deployment was successful and offer further assistance. Keep the tone positive and open for questions. Avoid any negative language, so just focus on celebration and support. Make sure the response is friendly and engaging.\n","tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":15,"total_tokens":145,"completion_tokens":130,"prompt_tokens_details":null}}
```

## 2. 推理
在这一部分，我将开发一个脚本在 HumanEval 数据集上执行推理。该脚本应与已部署的模型交互，为提供的样本生成预测。

### 2.1 数据集
在目录中有两个数据集 `datasets/`:
```text
.
├── datasets
    ├── HumanEval_4.jsonl
    └── HumanEval.jsonl
```
`HumanEval.jsonl` 是完整的 HumanEval 数据集。 `HumanEval_4.jsonl` 是完整数据集的一个小子集，仅包含 4 个样本，用于快速测试和调试。

### 2.2 推理脚本
推理脚本是 `inference/inference_he.py`. 运行推理：
```bash
cd inference
python3 inference_he.py [OPTIONS]
```
可用选项：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-o`, `--output_file` | 输出文件名。存储在 `outputs/` | `humaneval_results.jsonl`. |
| `--think` | 是否启用 think 模式。 | `False` |
| `--debug` | 调试模式。如果启用，将使用 4 个样本子集。 | `False` |

例如，要在小子集上运行推理并启用 think 模式，并将结果存储到 `he_results_debug_think_results.jsonl`:
```bash
python3 inference_he.py --think --debug -o he_results_debug_think_results.jsonl
```

## 3. 评估
这一部分，需要搭建一个沙箱环境来评估前一步得到的 HumanEval 结果的通过率。

### 3.1 构建 Docker 镜像
我在根目录准备了一个 Dockerfile，用于构建最小化的评估镜像。只需运行：
```bash
docker build -t humaneval_eval:latest .
```
镜像将被构建完成。

### 3.2 执行评估
镜像构建完成后，我们可以执行评估。提供了两种方法：命令行和脚本，其中脚本会自动化整个流程。

#### 3.2.1 在命令行中执行评估
我们可以在命令行中运行评估。启动容器：
```bash
docker run -it --rm \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/datasets:/app/datasets \
    humaneval_eval:latest
```

然后在容器中运行评估脚本：
```bash
cd evaluate
python3 evaluate.py [OPTIONS]
```
可用选项：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-f`, `--file` | The output file to be evaluated. | `../outputs/he_results_no_think.jsonl`. |

评估结果将会打印在终端中。

#### 3.2.2 通过脚本执行评估
我还准备了一个脚本 `auto-evaluate.py` 在根目录中自动化执行评估过程。只需运行：
```bash
python3 auto-evaluate.py [OPTIONS]
```
可用选项：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-f`, `--file` | 待评估文件，在 `outputs/` 目录下 | `humaneval_results.jsonl`. |  
| `-o`, `--output` | 评估报告，存储在`results/` 目录下 | eval_report.jsonl |

接下来脚本会自动启动一个容器并执行评估。过程与命令行方法一致。

### 3.3 评估结果
通过率结果如下：
| Think Mode | System Prompt | Output File | Pass Rate |
|------------|---------------|-------------|-----------|
| No         | 默认值       | he_outputs_pure_code.jsonl | 89.02% (146/164) |

# 运行示例
启动 SGLang 后端
```bash
python3 client.py
```
在 HumanEval 数据集上进行推理：
```bash
# (in a new terminal)
cd inference
python3 inference_he.py -o he_outputs_pure_code.jsonl
```
推理结果已存储在 `outputs/he_outputs_pure_code.jsonl`.

使用 pass@1 指标进行评估：
```bash
cd ..
python3 auto-evaluate.py -f he_outputs_pure_code.jsonl -o eval_report.jsonl
```
评估报告存储在 `results/eval_report.jsonl`。

## 故障排查
这一部分主要是为自己记录在部署和开发过程中遇到的一些问题。

1. **无法访问 HuggingFace**：我这里无法访问官方源。添加 `HF_ENDPOINT` 到环境变量以使用镜像站。
    ```bash
    export HF_ENDPOINT=https://hf-mirror.com
    ```
2. **无法获取 Nvidia Container Toolkit 的 GPG 密钥**：我这里无法访问官方源。使用[科大源](https://mirrors.ustc.edu.cn/help/libnvidia-container.html) 。