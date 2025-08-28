# The Python script to launch SGLang client
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Launch SGLang Client with Docker")

    parser.add_argument("--model-name", type=str, default="Qwen3-0.6B",
                        help="模型名称 (默认: Qwen3-0.6B)")
    parser.add_argument("--port", type=str, default="30000",
                        help="服务运行端口号 (默认: 30000)")
    parser.add_argument("--tp", type=str, default="1",
                        help="Tensor 并行大小 (默认: 1)")
    parser.add_argument("--local-save-path", type=str, default="/NV/models_hf/Qwen/Qwen3-0.6B",
                        help="模型本地路径")
    parser.add_argument("--container-name", type=str, default="qwen-test",
                        help="Docker 容器名称 (默认: qwen-test)")
    parser.add_argument("--image", type=str, 
                        default="egs-registry.cn-hangzhou.cr.aliyuncs.com/egs/sglang:0.4.6.post1-pytorch2.6-cu124-20250429",
                        help="Docker 镜像 (默认: sglang 0.4.6)")

    args = parser.parse_args()

    # 拼接 docker 命令
    docker_cmd = [
        "sudo", "docker", "run", "-t",
        f"--name={args.container_name}",
        "--ipc=host",
        "--cap-add=SYS_PTRACE",
        "--network=host",
        "--gpus", "all",
        "--privileged",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "-v", f"{args.local_save_path}:{args.local_save_path}",
        args.image,
        "/bin/bash", "-c",
        (
            f"python3 -m sglang.launch_server "
            f"--model-path {args.local_save_path} "
            f"--port {args.port} "
            f"--tp {args.tp} "
            f"--host 0.0.0.0 "
            f"--reasoning-parser qwen3 "
            f"--enable-torch-compile"
        )
    ]

    print("运行的命令为：")
    print(" ".join(docker_cmd))

    # 执行命令
    subprocess.run(docker_cmd)

if __name__ == "__main__":
    main()
