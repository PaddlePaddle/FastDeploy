"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(file_path: str) -> list[dict]:
    records = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


_THROUGHPUT_KEYS = ["request_throughput", "output_throughput", "total_throughput"]


def detect_metrics(records: list[dict]) -> list[str]:
    """Detect metric names from JSONL records (throughput scalars + dict-valued keys)."""
    throughput_found = set()
    dict_metrics = []

    for rec in records:
        for k in _THROUGHPUT_KEYS:
            if k in rec:
                throughput_found.add(k)
        if not dict_metrics:
            for k, v in rec.items():
                if isinstance(v, dict):
                    dict_metrics.append(k)
        if len(throughput_found) == len(_THROUGHPUT_KEYS) and dict_metrics:
            break

    metrics = [k for k in _THROUGHPUT_KEYS if k in throughput_found]
    metrics.extend(dict_metrics)
    return metrics


def plot_metric(records: list[dict], metric: str, output_path: str):
    """Plot a single metric for a chunk of records."""
    stat_keys = []
    for rec in records:
        val = rec.get(metric)
        if isinstance(val, dict):
            stat_keys = list(val.keys())
            break

    if not stat_keys:
        values = [rec.get(metric) for rec in records if rec.get(metric) is not None]
        if not values:
            return
        x = np.arange(len(values))
        plt.figure(figsize=(12, 6))
        plt.plot(x, values, linewidth=2, label=metric)
        plt.xlabel("Request Index")
        plt.ylabel(metric)
        plt.title(f"{metric} over requests")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")
        return

    plt.figure(figsize=(12, 6))

    for key in stat_keys:
        values = []
        for rec in records:
            val = rec.get(metric)
            if isinstance(val, dict) and key in val:
                values.append(val[key])
            else:
                values.append(None)

        valid_x = [i for i, v in enumerate(values) if v is not None]
        valid_v = [v for v in values if v is not None]

        if not valid_v:
            continue

        linestyle = "-" if key == "mean" else "--"
        lw = 2 if key == "mean" else 1.5
        plt.plot(valid_x, valid_v, linewidth=lw, linestyle=linestyle, label=key)

    plt.xlabel("Request Index")
    unit = ""
    if metric.endswith("_ms"):
        unit = " (ms)"
    elif metric in ("s_decode",):
        unit = " (tok/s)"
    elif metric == "request_throughput":
        unit = " (req/s)"
    elif metric in ("output_throughput", "total_throughput"):
        unit = " (tok/s)"
    plt.ylabel(f"{metric}{unit}")
    plt.title(f"{metric} over requests")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark metrics from JSONL file. "
        "Metrics and window_size are read from the data automatically. "
        "Records are grouped into window-sized chunks for per-step analysis."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to benchmark_metrics.jsonl. Defaults to $FD_LOG_DIR/benchmark_metrics.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output PNG files. Defaults to $FD_LOG_DIR/benchmark_plots/",
    )
    args = parser.parse_args()

    log_dir = os.environ.get("FD_LOG_DIR", "./log")

    if args.file:
        file_path = args.file
    else:
        file_path = os.path.join(log_dir, "benchmark_metrics.jsonl")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir if args.output_dir else os.path.join(log_dir, "benchmark_plots")

    records = load_jsonl(file_path)
    if not records:
        print("No data in file.", file=sys.stderr)
        sys.exit(1)

    window_size = records[0].get("window_size", 0)
    metrics = detect_metrics(records)

    if not metrics:
        print("No plottable metrics found in data.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} records from {file_path}")
    print(f"Window size: {window_size or 'all'}")
    print(f"Metrics: {', '.join(metrics)}")

    if window_size > 0:
        num_chunks = (len(records) + window_size - 1) // window_size
        chunks = []
        for i in range(num_chunks):
            start = i * window_size
            end = min(start + window_size, len(records))
            chunks.append(records[start:end])
        print(f"Split into {num_chunks} step(s) of size {window_size}")
    else:
        chunks = [records]

    os.makedirs(output_dir, exist_ok=True)

    for step_idx, chunk in enumerate(chunks):
        if len(chunks) > 1:
            step_dir = os.path.join(output_dir, f"step_{step_idx + 1}")
            os.makedirs(step_dir, exist_ok=True)
        else:
            step_dir = output_dir

        for metric in metrics:
            output_path = os.path.join(step_dir, f"{metric}.png")
            plot_metric(chunk, metric, output_path)


if __name__ == "__main__":
    main()
