#!/usr/bin/env python3
"""extract_metrics.py — 从 benchmark 结果文件提取指标，输出结构化 JSON

用法:
    python3 extract_metrics.py \
        --fd-result <FD_RESULT.txt> \
        --sg-result <SG_RESULT.txt> \
        --model-path <MODEL_PATH> \
        --fd-config '{"gpu":"H800","tp":1,"concurrency":32}' \
        --sg-config '{"gpu":"H800","tp":1,"concurrency":32}' \
        --output <metrics.json>
"""

import argparse
import json
import os
import re
import subprocess
import sys


def parse_benchmark_result(filepath):
    """解析 benchmark_serving.py 的输出文件，提取所有指标"""
    metrics = {}
    if not os.path.isfile(filepath):
        print(f"[WARN] 结果文件不存在: {filepath}", file=sys.stderr)
        return metrics

    with open(filepath, "r") as f:
        content = f.read()

    patterns = {
        "successful_requests": r"Successful requests:\s+([\d.]+)",
        "benchmark_duration": r"Benchmark duration \(s\):\s+([\d.]+)",
        "total_input_tokens": r"Total input tokens:\s+([\d.]+)",
        "total_generated_tokens": r"Total generated tokens:\s+([\d.]+)",
        "request_throughput": r"Request throughput \(req/s\):\s+([\d.]+)",
        "output_token_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
        "total_token_throughput": r"Total Token throughput \(tok/s\):\s+([\d.]+)",
        "mean_ttft": r"Mean TTFT \(ms\):\s+([\d.]+)",
        "median_ttft": r"Median TTFT \(ms\):\s+([\d.]+)",
        "p80_ttft": r"P80 TTFT \(ms\):\s+([\d.]+)",
        "p95_ttft": r"P95 TTFT \(ms\):\s+([\d.]+)",
        "p99_ttft": r"P99 TTFT \(ms\):\s+([\d.]+)",
        "mean_tpot": r"Mean TPOT \(ms\):\s+([\d.]+)",
        "median_tpot": r"Median TPOT \(ms\):\s+([\d.]+)",
        "p80_tpot": r"P80 TPOT \(ms\):\s+([\d.]+)",
        "p95_tpot": r"P95 TPOT \(ms\):\s+([\d.]+)",
        "p99_tpot": r"P99 TPOT \(ms\):\s+([\d.]+)",
        "mean_itl": r"Mean ITL \(ms\):\s+([\d.]+)",
        "median_itl": r"Median ITL \(ms\):\s+([\d.]+)",
        "p80_itl": r"P80 ITL \(ms\):\s+([\d.]+)",
        "p95_itl": r"P95 ITL \(ms\):\s+([\d.]+)",
        "p99_itl": r"P99 ITL \(ms\):\s+([\d.]+)",
        "mean_e2el": r"Mean E2EL \(ms\):\s+([\d.]+)",
        "median_e2el": r"Median E2EL \(ms\):\s+([\d.]+)",
        "p80_e2el": r"P80 E2EL \(ms\):\s+([\d.]+)",
        "p95_e2el": r"P95 E2EL \(ms\):\s+([\d.]+)",
        "p99_e2el": r"P99 E2EL \(ms\):\s+([\d.]+)",
        "mean_decode": r"Mean Decode \(tok/s\):\s+([\d.]+)",
        "median_decode": r"Median Decode \(tok/s\):\s+([\d.]+)",
        "p80_decode": r"P80 Decode \(tok/s\):\s+([\d.]+)",
        "p95_decode": r"P95 Decode \(tok/s\):\s+([\d.]+)",
        "p99_decode": r"P99 Decode \(tok/s\):\s+([\d.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))

    return metrics


def get_model_info(model_path):
    """从模型目录读取配置信息"""
    info = {
        "name": os.path.basename(model_path),
        "path": model_path,
        "model_type": "unknown",
        "hidden_size": 0,
        "num_layers": 0,
        "n_routed_experts": 0,
        "n_shared_experts": 0,
        "num_experts_per_tok": 0,
        "size_gb": 0,
    }

    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        info["model_type"] = config.get("model_type", "unknown")
        info["hidden_size"] = config.get("hidden_size", 0)
        info["num_layers"] = config.get("num_hidden_layers", 0)
        info["n_routed_experts"] = config.get("n_routed_experts", 0)
        info["n_shared_experts"] = config.get("n_shared_experts", 0)
        info["num_experts_per_tok"] = config.get("num_experts_per_tok", 0)
        info["vocab_size"] = config.get("vocab_size", 0)

    # 获取模型大小
    try:
        result = subprocess.run(["du", "-sb", model_path], capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            size_bytes = int(result.stdout.split()[0])
            info["size_gb"] = round(size_bytes / (1024**3), 1)
    except Exception:
        pass

    return info


def compute_comparison(fd_metrics, sg_metrics):
    """计算对比指标（差异百分比、胜出方）"""
    comparison = {}

    # 吞吐类指标：越高越好
    higher_is_better = {
        "total_token_throughput",
        "output_token_throughput",
        "request_throughput",
        "mean_decode",
        "median_decode",
        "p80_decode",
        "p95_decode",
        "p99_decode",
    }

    # 延迟类指标：越低越好
    lower_is_better = {
        "mean_ttft",
        "median_ttft",
        "p80_ttft",
        "p95_ttft",
        "p99_ttft",
        "mean_tpot",
        "median_tpot",
        "p80_tpot",
        "p95_tpot",
        "p99_tpot",
        "mean_itl",
        "median_itl",
        "p80_itl",
        "p95_itl",
        "p99_itl",
        "mean_e2el",
        "median_e2el",
        "p80_e2el",
        "p95_e2el",
        "p99_e2el",
        "benchmark_duration",
    }

    all_keys = set(fd_metrics.keys()) | set(sg_metrics.keys())

    for key in sorted(all_keys):
        fd_val = fd_metrics.get(key)
        sg_val = sg_metrics.get(key)

        if fd_val is None or sg_val is None:
            continue

        entry = {"fd": fd_val, "sg": sg_val}

        # 计算差异百分比 (FD 相对于 SG)
        if sg_val != 0:
            diff_pct = round((fd_val - sg_val) / sg_val * 100, 2)
        else:
            diff_pct = 0
        entry["diff_pct"] = diff_pct

        # 判断胜出方
        if key in higher_is_better:
            entry["winner"] = "fd" if fd_val > sg_val else "sg"
        elif key in lower_is_better:
            entry["winner"] = "fd" if fd_val < sg_val else "sg"
        else:
            entry["winner"] = "tie"

        comparison[key] = entry

    return comparison


def main():
    parser = argparse.ArgumentParser(description="从 benchmark 结果提取指标并生成对比 JSON")
    parser.add_argument("--fd-result", required=True, help="FastDeploy 结果文件路径")
    parser.add_argument("--sg-result", required=True, help="SGLang 结果文件路径")
    parser.add_argument("--model-path", required=True, help="模型权重目录路径")
    parser.add_argument("--fd-config", default="{}", help="FD 部署配置 JSON 字符串")
    parser.add_argument("--sg-config", default="{}", help="SG 部署配置 JSON 字符串")
    parser.add_argument("--output", default="metrics.json", help="输出 JSON 路径")
    args = parser.parse_args()

    print(f"[INFO] 解析 FD 结果: {args.fd_result}")
    fd_metrics = parse_benchmark_result(args.fd_result)
    print(f"[INFO] 解析 SG 结果: {args.sg_result}")
    sg_metrics = parse_benchmark_result(args.sg_result)

    print(f"[INFO] 读取模型信息: {args.model_path}")
    model_info = get_model_info(args.model_path)

    print("[INFO] 计算对比指标...")
    comparison = compute_comparison(fd_metrics, sg_metrics)

    # 解析部署配置
    fd_config = json.loads(args.fd_config) if args.fd_config else {}
    sg_config = json.loads(args.sg_config) if args.sg_config else {}

    output = {
        "model": model_info,
        "config": {
            "fd": fd_config,
            "sg": sg_config,
        },
        "raw_metrics": {
            "fd": fd_metrics,
            "sg": sg_metrics,
        },
        "comparison": comparison,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[INFO] 指标已写入: {args.output}")

    # 打印摘要
    key_metrics = [
        "total_token_throughput",
        "output_token_throughput",
        "mean_ttft",
        "mean_tpot",
        "mean_itl",
        "mean_e2el",
        "mean_decode",
        "benchmark_duration",
    ]
    print("\n========== 核心指标摘要 ==========")
    print(f"{'Metric':<30} {'FD':>12} {'SG':>12} {'Diff%':>8} {'Winner':>8}")
    print("-" * 72)
    for key in key_metrics:
        if key in comparison:
            c = comparison[key]
            print(f"{key:<30} {c['fd']:>12.2f} {c['sg']:>12.2f} {c['diff_pct']:>+7.1f}% {c['winner']:>8}")
    print("=" * 72)


if __name__ == "__main__":
    main()
