# Thinking Budget Logits Processor

## 概述

`ThinkingBudgetLogitsProcessor` 用于限制 `<think> ... </think>` 区间的生成长度。当预算达到阈值时，会强制生成换行符 token，再强制生成 `</think>`，从而结束思考段。

## 适用场景

- 模型会输出 `<think>`/`</think>` 的思考标记。
- 需要对思考段做硬限制，但不希望改变采样策略。

## 工作原理

1. **CPU 侧预计算（DataProcessor）**：当请求中包含 `thinking_budget`，会基于 prompt 的 token ids 计算是否已进入思考段、是否已结束，以及已有的思考长度。
2. **每步更新**：解码过程中跟踪 `last_token_id` 与 `tokens_after_start`。
3. **预算约束**：达到预算后，依次强制换行符与思考结束 token。

## 前置要求

- 模型需提供有效的 `think_start_id`、`think_end_id`、`line_break_id`（来自 `ModelConfig`）。
- 若任意 id 无效，处理器会禁用，`thinking_budget` 不生效。

## 请求参数

- `thinking_budget`（int，启用所需）：`<think>` 之后允许的最大 token 数。
- `think_stop_sentence`（string，可选）：CPU 侧会将该字符串编码为 token ids，并在预算边界附近强制输出。

## 算子级限制 vs LogitsProcessor

FastDeploy 当前有两种思考长度控制方式：

- **算子级限制**（`enable_thinking=true` + `reasoning_max_tokens`）：
  - 由内置后处理算子完成。
  - 高并发下开销更低、吞吐更稳定。
  - 适合“只限制思考长度”的简单场景。
- **`ThinkingBudgetLogitsProcessor`**（`logits_processors_args.thinking_budget`）：
  - 由每步 Python 侧 logits 处理实现。
  - 支持更灵活的行为，例如 `think_stop_sentence`（在结束前插入自定义话术）。
  - 相比算子级限制，在高并发下通常有更高开销。

可按以下原则选择：

- 仅需限制思考长度：优先用 `reasoning_max_tokens`。
- 需要更灵活控制（如插入自定义话术）：使用 `ThinkingBudgetLogitsProcessor`。

## 建议实践

当前实现中，`reasoning_max_tokens` 与 `thinking_budget` 不是互斥关系。
同一请求如果同时配置，两套约束都可能生效，谁先触发就先结束思考段。

- **只用算子级限制**：这是请求级配置。仅在请求中设置 `enable_thinking=true` + `reasoning_max_tokens`，不要传 `thinking_budget`。
- **只用 LogitsProcessor**（尤其要用 `think_stop_sentence`）：这是“服务启动 + 请求参数”两级配置。服务启动时必须加 `--logits-processors ThinkingBudgetLogitsProcessor`，并在请求里通过 `logits_processors_args` 传 `thinking_budget`（以及可选的 `think_stop_sentence`）；同时不要设置 `reasoning_max_tokens`。
- 如果业务要求“必须完整插入自定义话术”，不建议与算子级限制同时开启，否则可能被算子级提前截断。

## 在线使用

### 1. 启动服务

```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --port 8180 \
  --metrics-port 8181 \
  --engine-worker-queue-port 8182 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --logits-processors ThinkingBudgetLogitsProcessor
```

### 2. 发送请求

```bash
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好！"}],
    "max_completion_tokens": 30,
    "logits_processors_args": {
      "thinking_budget": 20,
      "think_stop_sentence": "思考已达上限，开始回复"
    }
  }'
```

如果某个请求不需要思考限制，直接省略 `thinking_budget` 即可。

### 3. 仅使用算子级思考长度限制（不启用 logits processor）

```bash
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好！"}],
    "max_completion_tokens": 512,
    "enable_thinking": true,
    "reasoning_max_tokens": 200
  }'
```

## 离线使用

```python
from fastdeploy import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    engine_worker_queue_port=8282,
    cache_queue_port=8383,
    logits_processors=["ThinkingBudgetLogitsProcessor"],
)

sampling_params = SamplingParams(
    max_tokens=512,
    logits_processors_args={"thinking_budget": 20, "think_stop_sentence": "思考已达上限，开始回复"},
)

outputs = llm.chat([{"role": "user", "content": "将李白的静夜思改为现代诗"}], sampling_params)
print(outputs[0].outputs.text)
```

## 性能说明

该处理器会在每个 decode step 执行 `update_state` 与 `apply`。如果仅需要硬性的思考长度限制且更关注吞吐，建议优先使用算子级思考长度控制方案。
