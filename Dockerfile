# 使用 Python 3.10 slim 镜像
FROM docker.xuanyuan.me/python:3.10-slim
# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app

# 配置 pip 使用阿里云镜像源
RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ && \
    pip install tqdm -i https://mirrors.aliyun.com/pypi/simple/

# 创建并切换到非 root 用户
RUN useradd -m -s /bin/bash appuser
USER appuser

# 设置环境变量，确保 --user 安装的包可用
ENV PATH=/home/appuser/.local/bin:$PATH

# 默认命令
CMD ["bash"]
