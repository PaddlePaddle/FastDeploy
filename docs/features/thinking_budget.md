# Thinking Budget Logits Processor

## Overview

`ThinkingBudgetLogitsProcessor` limits the number of tokens generated inside the `<think> ... </think>`
segment. When the budget is reached, it terminates thinking by forcing `</think>`. If
`think_stop_sentence` is configured, it forces the custom sentence first and then `</think>`.

## When to Use

- Models that emit `<think>`/`</think>` tokens for reasoning.
- You need a hard cap on thinking length without changing sampling logic.

## How It Works

1. **Request-side precompute (DataProcessor)**: when a request includes `thinking_budget`, the prompt token ids are scanned to determine whether thinking has started, whether it already ended, and how many tokens are already inside the thinking section.
2. **Per-step update**: during decoding, the processor tracks `last_token_id` and `tokens_after_start`.
3. **Budget enforcement**: once the budget is reached, it forces `</think>` directly. If `think_stop_sentence`
   is configured, it forces that sentence first and then `</think>`.

## Requirements

- The model must provide valid token ids for `think_start_id` and `think_end_id` (via `ModelConfig`).
- If either of these ids is invalid, the processor is disabled and `thinking_budget` will not take effect.

## Request Parameters

- `thinking_budget` (int, required to enable): maximum number of decode-time tokens after `<think>` before forced
  termination.
- `think_stop_sentence` (string, optional): a literal custom sentence that will be tokenized on the request side
  and enforced near the budget boundary.

## Operator-Level vs LogitsProcessor

FastDeploy has two ways to limit thinking length:

- **Operator-level limit** (`enable_thinking=true` + `reasoning_max_tokens`):
  - Implemented in built-in post-processing kernels.
  - Lower overhead and better throughput under high concurrency.
  - Best for simple "cap the thinking length" use cases.
- **`ThinkingBudgetLogitsProcessor`** (`logits_processors_args.thinking_budget`):
  - Implemented in per-step Python logits processing.
  - Supports flexible controls, such as `think_stop_sentence` (custom inserted sentence before ending thinking).
  - Higher runtime overhead under high concurrency compared with operator-level limit.

In short:

- If you only need a hard cap on thinking length, prefer `reasoning_max_tokens`.
- If you need custom behavior (for example, inserting a custom sentence before `</think>`), use
  `ThinkingBudgetLogitsProcessor`.

## Practical guidance

`reasoning_max_tokens` and `thinking_budget` are not mutually exclusive in current implementation.
If both are configured for the same request, both constraints can take effect, and whichever triggers first will end the thinking phase.

- To use **operator-level-only** behavior: this is request-level config only. Set
  `enable_thinking=true` and `reasoning_max_tokens` in request, and do not set `thinking_budget`.
- To use **logits-processor-only** behavior (especially with `think_stop_sentence`): this requires
  service-level + request-level config. Start service with `--logits-processors ThinkingBudgetLogitsProcessor`,
  and set `thinking_budget` (and optional `think_stop_sentence`) in `logits_processors_args`; leave
  `reasoning_max_tokens` unset.
- `thinking_budget` itself does not require `enable_thinking=true`.
- If an ERNIE chat template already appends `<think>` in the prompt, `thinking_budget` should still take effect; it
  does not require the model to emit another `<think>` during decoding.
- Avoid enabling both for strict custom sentence insertion requirements, because operator-level
  termination may cut the custom sentence path earlier.

## Online Usage

### 1. Start service

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

### 2. Send request

```bash
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_completion_tokens": 30,
    "logits_processors_args": {
      "thinking_budget": 20,
      "think_stop_sentence": "Thinking limit reached, now replying."
    }
  }'
```

If you do not need thinking control for a request, simply omit `thinking_budget`.

### 3. Operator-level thinking cap only (no logits processor)

```bash
curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_completion_tokens": 512,
    "enable_thinking": true,
    "reasoning_max_tokens": 200
  }'
```

## Offline Usage

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
    logits_processors_args={"thinking_budget": 20, "think_stop_sentence": "Thinking limit reached, now replying."},
)

outputs = llm.chat([{"role": "user", "content": "Hello, who are u?"}], sampling_params)
print(outputs[0].outputs.text)
```

## Performance Note

This processor runs `update_state` and `apply` on every decode step. If you only need a hard
thinking-length cap and care most about throughput, consider the operator-level reasoning-length
controls instead of per-step logits processing.
