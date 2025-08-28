import subprocess
import argparse
import os

def run_docker(file_to_eval: str):
    """
    启动容器并运行 evaluation
    """
    # 挂载目录
    current_dir = os.getcwd()
    results_dir = os.path.join(current_dir, "results")
    datasets_dir = os.path.join(current_dir, "datasets")

    docker_cmd = [
        "docker", "run", "--rm", "-i", "-t",
        "-v", f"{results_dir}:/app/results",
        "-v", f"{datasets_dir}:/app/datasets",
        "humaneval_eval:latest",
        "bash", "-c",
        f"cd evaluate && python3 evaluate.py --file {file_to_eval}"
    ]

    print("🚀 Running docker command:\n", " ".join(docker_cmd))
    subprocess.run(docker_cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HumanEval evaluation inside Docker.")
    parser.add_argument(
        "-f", "--file",
        default="../outputs/he_results_no_think.jsonl",
        help="The output file to be evaluated (default: ../outputs/he_results_no_think.jsonl)"
    )
    args = parser.parse_args()

    run_docker(args.file)
