[简体中文](zh/benchmark.md)

# Benchmark

FastDeploy extends the [vLLM benchmark](https://github.com/vllm-project/vllm/blob/main/benchmarks/) script with additional metrics, enabling more detailed performance benchmarking for FastDeploy.

## Benchmark Dataset

The following dataset is sourced from open-source data (original data from [HuggingFace Datasets](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json)):

| Dataset | Description |
| :------ | :---------- |
| https://fastdeploy.bj.bcebos.com/eb_query/filtered_sharedgpt_2000_input_1136_output_200_fd.json | Open-source dataset |

## How to Run

```
cd FastDeploy/benchmarks
python -m pip install -r requirements.txt

# Start service
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-0.3B-Base-Paddle \
       --port 8188 \
       --tensor-parallel-size 1 \
       --max-model-len 8192

# Run benchmark
python benchmark_serving.py \
  --backend openai-chat \
  --model baidu/ERNIE-4.5-0.3B-Base-Paddle \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port 8188 \
  --dataset-name EBChat \
  --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --save-result
```

## In-Process Benchmark Metrics Logger

FastDeploy provides a built-in performance monitoring module that runs inside the inference process. It collects per-request timing data and computes rolling statistics aligned with `benchmark_serving.py`, writing results to a JSONL file for real-time monitoring and post-hoc analysis.

### Enable

Add `--benchmark-metrics-config` with a JSON string to the service startup command:

```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-0.3B-Base-Paddle \
       --benchmark-metrics-config '{"enable": true}'
```

### Configuration Parameters

| Parameter | Type | Default | Description |
| :-------- | :--- | :------ | :---------- |
| `enable` | bool | `false` | Whether to enable the benchmark metrics logger. Must be set to `true` to activate. |
| `window_size` | int | `0` | Number of recent requests to aggregate. `0` = cumulative (all requests since start). |
| `window_mode` | str | `"sliding"` | Window aggregation mode. `"sliding"` = sliding window (keeps last N records, oldest automatically dropped). `"tumbling"` = tumbling window (clears and restarts after every N records). |
| `percentiles` | str | `"50,90,95,99"` | Comma-separated percentile values to compute. |
| `metrics` | str | `"all"` | Comma-separated metric names to report, or `"all"` for all metrics. |

### Available Metrics

Metrics are aligned with `benchmark_serving.py --percentile-metrics`:

| Metric Name | Description | Unit |
| :---------- | :---------- | :--- |
| `ttft` | Time to First Token (client arrival → first token) | ms |
| `s_ttft` | Server TTFT (inference start → first token) | ms |
| `tpot` | Time per Output Token (excluding first token) | ms |
| `s_itl` | Infer Inter-token Latency | ms |
| `e2el` | End-to-end Latency (client arrival → last token) | ms |
| `s_e2el` | Server E2EL (inference start → last token) | ms |
| `s_decode` | Decode speed (excluding first token) | tok/s |
| `input_len` | Prefix cache hit token count ("Cached Tokens") | tokens |
| `s_input_len` | Infer input length (total prompt tokens) | tokens |
| `output_len` | Output token length per request | tokens |

In addition, the following throughput metrics are always computed (not user-selectable) when there are 2+ records:

| Metric | Description | Unit |
| :----- | :---------- | :--- |
| `request_throughput` | Request throughput | req/s |
| `output_throughput` | Output token throughput | tok/s |
| `total_throughput` | Total token throughput (input + output) | tok/s |

### Window Modes

**Sliding Window** (`"sliding"`, default):

The window keeps the most recent N records. When a new record arrives and the window is full, the oldest record is automatically dropped. Each output line reflects the statistics of the latest N requests.

```bash
--benchmark-metrics-config '{"enable": true, "window_size": 64, "window_mode": "sliding"}'
```

**Tumbling Window** (`"tumbling"`):

The window accumulates records up to N, then clears and starts fresh. Each output line still reflects the current window's accumulated statistics, but the window resets at every boundary. This is useful for RL training scenarios where each step has a fixed batch size and you want per-step independent analysis.

```bash
--benchmark-metrics-config '{"enable": true, "window_size": 64, "window_mode": "tumbling"}'
```

**No Window** (`window_size: 0`):

All completed requests are accumulated. Statistics reflect the entire lifetime of the service.

```bash
--benchmark-metrics-config '{"enable": true, "window_size": 0}'
```

### Output

Results are written to `{FD_LOG_DIR}/benchmark_metrics.jsonl` (default: `./log/benchmark_metrics.jsonl`). Each line is a JSON object representing the window statistics at the time of a request completion.

Example output line:

```json
{
  "timestamp": "2026-05-14T10:30:05.123",
  "window_size": 64,
  "window_mode": "sliding",
  "completed": 64,
  "total_input_tokens": 8192,
  "total_output_tokens": 16384,
  "request_throughput": 5.2,
  "output_throughput": 1250.0,
  "total_throughput": 2500.0,
  "ttft_ms": {"mean": 45.0, "median": 42.1, "p50": 42.1, "p90": 68.5, "p95": 82.3, "p99": 120.5},
  "s_decode": {"mean": 67.3, "median": 67.5, "p50": 67.5, "p90": 70.1, "p95": 71.2, "p99": 73.0}
}
```
