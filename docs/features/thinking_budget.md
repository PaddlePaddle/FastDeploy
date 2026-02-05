# Thinking Budget Logits Processor

## Overview

`ThinkingBudgetLogitsProcessor` limits the number of tokens generated inside the `<think> ... </think>` segment. When the budget is reached, it forces a line break token and then the `</think>` token to terminate the thinking section.

## When to Use

- Models that emit `<think>`/`</think>` tokens for reasoning.
- You need a hard cap on thinking length without changing sampling logic.

## How It Works

1. **CPU precompute (DataProcessor)**: when a request includes `thinking_budget`, the prompt token ids are scanned to determine whether thinking has started, whether it already ended, and how many tokens are already inside the thinking section.
2. **Per-step update**: during decoding, the processor tracks `last_token_id` and `tokens_after_start`.
3. **Budget enforcement**: once the budget is reached, it forces a line break and then the thinking end token.

## Requirements

- The model must provide valid token ids for `think_start_id`, `think_end_id`, and `line_break_id` (via `ModelConfig`).
- If any of these ids are invalid, the processor is disabled and `thinking_budget` will not take effect.

## Request Parameters

- `thinking_budget` (int, required to enable): maximum number of tokens after `<think>` before forced termination.
- `think_stop_sentence` (string, optional): a stop sentence that will be tokenized on the CPU side and enforced near the budget boundary.

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

This processor runs `update_state` and `apply` on every decode step. If you only need a hard thinking-length cap and care most about throughput, consider the operator-level reasoning-length controls instead of per-step logits processing.
