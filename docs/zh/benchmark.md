[English](../benchmark.md)

# Benchmark

FastDeploy基于[vLLM benchmark](https://github.com/vllm-project/vllm/blob/main/benchmarks/)脚本，增加了部分统计信息，可用于benchmark FastDeploy更详细的性能指标。

## 测试数据集

以下数据集来源于开源数据集(源数据来源于[HuggingFace Datasets](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json))

| 数据集                                                       | 说明       |
| :----------------------------------------------------------- | :--------- |
| https://fastdeploy.bj.bcebos.com/eb_query/filtered_sharedgpt_2000_input_1136_output_200_fd.json | 开源数据集 |

## 测试方式

```
cd FastDeploy/benchmarks
python -m pip install -r requirements.txt

# 启动服务
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-0.3B-Base-Paddle \
       --port 8188 \
       --tensor-parallel-size 1 \
       --max-model-len 8192

# 压测服务
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

## 进程内性能监控（Benchmark Metrics Logger）

FastDeploy 提供了内置的进程内性能监控模块，在推理进程内部运行，复用已有的请求时间戳数据，每个请求完成时计算滚动统计并写入 JSONL 文件，可用于实时监控和事后分析。

### 启用方式

在服务启动命令中添加 `--benchmark-metrics-config` 参数，传入 JSON 配置字符串：

```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-0.3B-Base-Paddle \
       --benchmark-metrics-config '{"enable": true}'
```

### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :----- | :--- |
| `enable` | bool | `false` | 是否启用性能监控。必须设置为 `true` 才会激活。 |
| `window_size` | int | `0` | 统计窗口大小。`0` = 累计模式（统计所有请求）；`>0` = 统计最近 N 个请求。 |
| `window_mode` | str | `"sliding"` | 窗口聚合模式。`"sliding"` = 滑动窗口（保持最近 N 条，旧记录自动淘汰）；`"tumbling"` = 翻滚窗口（满 N 条后清空重新累积）。 |
| `percentiles` | str | `"50,90,95,99"` | 要计算的分位值，逗号分隔。 |
| `metrics` | str | `"all"` | 要统计的指标子集，逗号分隔，或 `"all"` 表示全部指标。 |

### 可用指标

指标与 `benchmark_serving.py --percentile-metrics` 对齐：

| 指标名称 | 说明 | 单位 |
| :------- | :--- | :--- |
| `ttft` | 首 Token 时延（客户端到达 → 首 Token） | ms |
| `s_ttft` | 服务端首 Token 时延（推理开始 → 首 Token） | ms |
| `tpot` | 每 Token 输出时延（不含首 Token） | ms |
| `s_itl` | 推理 Token 间时延 | ms |
| `e2el` | 端到端时延（客户端到达 → 最后一个 Token） | ms |
| `s_e2el` | 服务端端到端时延（推理开始 → 最后一个 Token） | ms |
| `s_decode` | 解码速度（不含首 Token） | tok/s |
| `input_len` | 前缀缓存命中 Token 数（"Cached Tokens"） | tokens |
| `s_input_len` | 推理输入长度（总 prompt token 数） | tokens |
| `output_len` | 输出 Token 长度 | tokens |

此外，以下吞吐量指标在有 2 个以上请求完成时自动计算（不受 `metrics` 参数控制）：

| 指标 | 说明 | 单位 |
| :--- | :--- | :--- |
| `request_throughput` | 请求吞吐量 | req/s |
| `output_throughput` | 输出 Token 吞吐量 | tok/s |
| `total_throughput` | 总 Token 吞吐量（输入 + 输出） | tok/s |

### 窗口模式

**滑动窗口**（`"sliding"`，默认）：

窗口始终保持最近 N 条记录。当新记录到达且窗口已满时，最旧的记录自动淘汰。每行输出反映最近 N 个请求的统计值。

```bash
--benchmark-metrics-config '{"enable": true, "window_size": 64, "window_mode": "sliding"}'
```

**翻滚窗口**（`"tumbling"`）：

窗口累积到 N 条后清空重新开始。每行输出反映当前窗口已累积请求的统计值，窗口在边界处重置。适用于 RL 训练场景，每个 step 有固定 batch size，需要逐 step 独立分析。

```bash
--benchmark-metrics-config '{"enable": true, "window_size": 64, "window_mode": "tumbling"}'
```

**无窗口**（`window_size: 0`）：

所有已完成请求持续累积，统计值反映服务启动以来的全量数据。

```bash
--benchmark-metrics-config '{"enable": true, "window_size": 0}'
```

### 输出说明

结果写入 `{FD_LOG_DIR}/benchmark_metrics.jsonl`（默认路径：`./log/benchmark_metrics.jsonl`）。每行为一个 JSON 对象，表示某个请求完成时刻窗口内的统计快照。

输出示例：

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

读取最后一行即可获取当前最新的性能快照：

```bash
tail -1 log/benchmark_metrics.jsonl | python -m json.tool
```
