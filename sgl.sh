# 定义模型名称。
MODEL_NAME="Qwen3-0.6B"

# 定义服务运行时监听的端口号。可以根据实际需求进行调整，默认使用30000端口
PORT="30000"

# 定义使用的GPU数量。这取决于实例上可用的GPU数量，可以通过nvidia-smi -L命令查询
TENSOR_PARALLEL_SIZE="1"

# 设置本地存储路径
LOCAL_SAVE_PATH="/NV/models_hf/Qwen/Qwen3-0.6B"

sudo docker run -t -d --name="qwen-test"  --ipc=host \
--cap-add=SYS_PTRACE --network=host --gpus all \
--privileged --ulimit memlock=-1 --ulimit stack=67108864 \
-v ${LOCAL_SAVE_PATH}:${LOCAL_SAVE_PATH} \
egs-registry.cn-hangzhou.cr.aliyuncs.com/egs/sglang:0.4.6.post1-pytorch2.6-cu124-20250429 \
/bin/bash -c "python3 -m sglang.launch_server \
--model-path ${LOCAL_SAVE_PATH} \
--port ${PORT} --tp ${TENSOR_PARALLEL_SIZE} \
--host 0.0.0.0 \
--reasoning-parser qwen3 --enable-torch-compile"