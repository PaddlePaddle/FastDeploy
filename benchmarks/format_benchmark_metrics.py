"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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


def read_last_line(file_path: str) -> str:
    with open(file_path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        if file_size == 0:
            return ""
        pos = file_size - 1
        while pos > 0:
            f.seek(pos)
            char = f.read(1)
            if char == b"\n" and pos < file_size - 1:
                break
            pos -= 1
        if pos == 0:
            f.seek(0)
        return f.read().decode("utf-8").strip()


def print_stat_block(data: dict, key: str, metric_name: str, header: str, is_time: bool = True):
    stats = data.get(key)
    if not stats:
        return
    suffix = "(ms)" if is_time else ""
    if key == "decode_speed":
        suffix = "(tok/s)"

    print("{s:{c}^{n}}".format(s=header, n=50, c="-"))
    print("{:<40} {:<10.2f}".format(f"Mean {metric_name} {suffix}:", stats["mean"]))
    print("{:<40} {:<10.2f}".format(f"Median {metric_name} {suffix}:", stats["median"]))

    for k, v in stats.items():
        if k.startswith("p"):
            label = k.upper()
            print("{:<40} {:<10.2f}".format(f"{label} {metric_name} {suffix}:", v))


def main():
    parser = argparse.ArgumentParser(description="Read and display benchmark metrics from JSONL file.")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to benchmark_metrics.jsonl. Defaults to $FD_LOG_DIR/benchmark_metrics.jsonl",
    )
    args = parser.parse_args()

    if args.file:
        file_path = args.file
    else:
        log_dir = os.environ.get("FD_LOG_DIR", "./log")
        file_path = os.path.join(log_dir, "benchmark_metrics.jsonl")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    last_line = read_last_line(file_path)
    if not last_line:
        print("No data in file.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(last_line)

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Timestamp:", data.get("timestamp", "N/A")))
    print("{:<40} {:<10}".format("Window size:", data.get("window_size", 0) or "all"))
    print("{:<40} {:<10}".format("Completed requests:", data.get("completed", 0)))
    print("{:<40} {:<10}".format("Total input tokens:", data.get("total_input_tokens", 0)))
    print("{:<40} {:<10}".format("Total output tokens:", data.get("total_output_tokens", 0)))

    if "request_throughput" in data:
        print("{:<40} {:<10.3f}".format("Request throughput (req/s):", data["request_throughput"]))
    if "output_throughput" in data:
        print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", data["output_throughput"]))
    if "total_throughput" in data:
        print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", data["total_throughput"]))

    print_stat_block(data, "s_decode", "Decode", "解码速度(tok/s)", is_time=False)
    print_stat_block(data, "ttft_ms", "TTFT", "Time to First Token")
    print_stat_block(data, "s_ttft_ms", "S_TTFT", "Infer Time to First Token")
    print_stat_block(data, "tpot_ms", "TPOT", "Time per Output Token (excl. 1st token)")
    print_stat_block(data, "itl_ms", "S_ITL", "Infer Inter-token Latency")
    print_stat_block(data, "e2el_ms", "E2EL", "End-to-end Latency")
    print_stat_block(data, "s_e2el_ms", "S_E2EL", "Infer End-to-end Latency")
    print_stat_block(data, "input_len", "Cached Tokens", "Cached Tokens", is_time=False)
    print_stat_block(data, "s_input_len", "Input Length", "Infer Input Length", is_time=False)
    print_stat_block(data, "output_len", "Output Length", "Output Length", is_time=False)

    print("=" * 50)


if __name__ == "__main__":
    main()
