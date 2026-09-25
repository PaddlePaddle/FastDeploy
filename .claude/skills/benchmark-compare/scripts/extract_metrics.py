#!/usr/bin/env python3
"""extract_metrics.py — 从 benchmark 结果文件提取指标，输出结构化 JSON

支持框架: fd (FastDeploy) / sg (SGLang) / vllm (vLLM)
任意框架结果均可缺省，缺省的不参与对比。

用法:
    python3 extract_metrics.py \
        --fd-result <FD_RESULT.txt> \
        --sg-result <SG_RESULT.txt> \
        --vllm-result <VLLM_RESULT.txt> \
        --model-path <MODEL_PATH> \
        --fd-config '{"gpu":"H800","tp":1,"concurrency":32}' \
        --sg-config '{"gpu":"H800","tp":1,"concurrency":32}' \
        --vllm-config '{"gpu":"H800","tp":1,"concurrency":32}' \
        --output <metrics.json>
"""

import argparse
import json
import os
import re
import subprocess
import sys

# 支持的框架列表
FRAMEWORKS = ("fd", "sg", "vllm")


def parse_benchmark_result(filepath):
    """解析 benchmark_serving.py 的输出文件，提取所有指标"""
    metrics = {}
    if not filepath or not os.path.isfile(filepath):
        if filepath:
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


# 吞吐类指标：越高越好
HIGHER_IS_BETTER = {
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
LOWER_IS_BETTER = {
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


def compute_comparison(all_metrics, baseline="sg"):
    """计算多框架对比指标。

    all_metrics: {"fd": {...}, "sg": {...}, "vllm": {...}}（任意 key 可为空 dict）
    baseline:    用于计算 diff_pct 的基准框架（默认 SGLang）

    返回:
    {
      metric_key: {
        "fd": ..., "sg": ..., "vllm": ...,
        "diff_pct": {"fd": ..., "vllm": ...},   # 相对 baseline
        "winner": "fd" | "sg" | "vllm" | "tie"
      }
    }
    """
    comparison = {}

    # 只比较实际有数据的框架
    active = [fw for fw in FRAMEWORKS if all_metrics.get(fw)]
    if not active:
        return comparison

    # 收集所有指标 key
    all_keys = set()
    for fw in active:
        all_keys |= set(all_metrics[fw].keys())

    for key in sorted(all_keys):
        entry = {}
        per_fw_val = {}
        for fw in active:
            val = all_metrics[fw].get(key)
            if val is None:
                continue
            entry[fw] = val
            per_fw_val[fw] = val

        if len(per_fw_val) < 2:
            # 单框架数据，无法对比但仍记录
            comparison[key] = entry
            continue

        # 计算相对 baseline 的差异百分比
        diff_pct = {}
        base_val = per_fw_val.get(baseline)
        for fw, val in per_fw_val.items():
            if fw == baseline or base_val is None:
                continue
            if base_val != 0:
                diff_pct[fw] = round((val - base_val) / base_val * 100, 2)
            else:
                diff_pct[fw] = 0
        if diff_pct:
            entry["diff_pct"] = diff_pct

        # 判断胜出方
        if key in HIGHER_IS_BETTER:
            entry["winner"] = max(per_fw_val, key=per_fw_val.get)
        elif key in LOWER_IS_BETTER:
            entry["winner"] = min(per_fw_val, key=per_fw_val.get)
        else:
            entry["winner"] = "tie"

        comparison[key] = entry

    return comparison


def main():
    parser = argparse.ArgumentParser(description="从 benchmark 结果提取指标并生成对比 JSON")
    parser.add_argument("--fd-result", default=None, help="FastDeploy 结果文件路径")
    parser.add_argument("--sg-result", default=None, help="SGLang 结果文件路径")
    parser.add_argument("--vllm-result", default=None, help="vLLM 结果文件路径")
    parser.add_argument("--model-path", required=True, help="模型权重目录路径")
    parser.add_argument("--fd-config", default="{}", help="FD 部署配置 JSON 字符串")
    parser.add_argument("--sg-config", default="{}", help="SG 部署配置 JSON 字符串")
    parser.add_argument("--vllm-config", default="{}", help="vLLM 部署配置 JSON 字符串")
    parser.add_argument(
        "--baseline", default="sg", choices=FRAMEWORKS, help="对比基准框架（计算 diff_pct 用），默认 sg"
    )
    parser.add_argument("--output", default="metrics.json", help="输出 JSON 路径")
    args = parser.parse_args()

    # 至少需要一份结果
    if not any([args.fd_result, args.sg_result, args.vllm_result]):
        parser.error("至少需要提供 --fd-result / --sg-result / --vllm-result 中的一个")

    result_paths = {
        "fd": args.fd_result,
        "sg": args.sg_result,
        "vllm": args.vllm_result,
    }
    config_strs = {
        "fd": args.fd_config,
        "sg": args.sg_config,
        "vllm": args.vllm_config,
    }
    framework_display = {"fd": "FastDeploy", "sg": "SGLang", "vllm": "vLLM"}

    all_metrics = {}
    for fw in FRAMEWORKS:
        path = result_paths[fw]
        if path:
            print(f"[INFO] 解析 {framework_display[fw]} 结果: {path}")
            all_metrics[fw] = parse_benchmark_result(path)
        else:
            all_metrics[fw] = {}

    print(f"[INFO] 读取模型信息: {args.model_path}")
    model_info = get_model_info(args.model_path)

    print(f"[INFO] 计算对比指标 (baseline={args.baseline})...")
    comparison = compute_comparison(all_metrics, baseline=args.baseline)

    # 解析部署配置
    configs = {}
    for fw in FRAMEWORKS:
        try:
            configs[fw] = json.loads(config_strs[fw]) if config_strs[fw] else {}
        except json.JSONDecodeError as e:
            print(f"[WARN] 解析 --{fw}-config 失败: {e}", file=sys.stderr)
            configs[fw] = {}

    output = {
        "model": model_info,
        "config": configs,
        "raw_metrics": all_metrics,
        "comparison": comparison,
        "baseline": args.baseline,
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
    active = [fw for fw in FRAMEWORKS if all_metrics.get(fw)]
    if not active:
        print("[WARN] 没有任何有效的结果数据")
        return

    print("\n========== 核心指标摘要 ==========")
    header = f"{'Metric':<30}"
    for fw in active:
        header += f" {framework_display[fw]:>12}"
    header += f" {'Winner':>10}"
    print(header)
    print("-" * len(header))
    for key in key_metrics:
        if key not in comparison:
            continue
        c = comparison[key]
        line = f"{key:<30}"
        for fw in active:
            val = c.get(fw)
            line += f" {val:>12.2f}" if isinstance(val, (int, float)) else f" {'-':>12}"
        line += f" {c.get('winner', '-'):>10}"
        print(line)
    print("=" * len(header))


if __name__ == "__main__":
    main()
