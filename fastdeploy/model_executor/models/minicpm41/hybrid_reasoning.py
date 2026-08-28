# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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


from fastdeploy.model_executor.logits_processor.thinking_budget import (
    ThinkingBudgetLogitsProcessor,
)

_DEFAULT_REASONING_TOKENS = "thinking"
_DEFAULT_MAX_THINKING_LENGTH = 512


def _encode_text(tokenizer, text: str) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    elif isinstance(token_ids, dict):
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if len(token_ids) == 1 and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def build_minicpm41_thinking_token_sequences(tokenizer) -> dict:
    """Build MiniCPM4.1 marker variants from its real tokenizer."""

    def continuation_ids(prefix: str, suffix: str) -> list[int]:
        prefix_ids = _encode_text(tokenizer, prefix)
        combined_ids = _encode_text(tokenizer, prefix + suffix)
        if combined_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(f"MiniCPM4.1 tokenizer cannot isolate {suffix!r} after {prefix!r}")
        continuation = combined_ids[len(prefix_ids) :]
        if not continuation:
            raise ValueError(f"MiniCPM4.1 tokenizer produced an empty marker for {suffix!r}")
        return continuation

    def unique(sequences: list[list[int]]) -> list[list[int]]:
        return [list(sequence) for sequence in dict.fromkeys(tuple(sequence) for sequence in sequences)]

    start_sequences = unique([_encode_text(tokenizer, "<think>"), continuation_ids("x", "<think>")])
    end_sequences = unique([_encode_text(tokenizer, "</think>"), continuation_ids("x", "</think>")])
    if any(not sequence for sequence in start_sequences + end_sequences):
        raise ValueError("MiniCPM4.1 tokenizer produced an empty thinking marker")
    return {
        "start": start_sequences,
        "end": end_sequences,
        "forced_end": continuation_ids("x", "\n</think>\n"),
    }


def _config_value(fd_config, name, default):
    if hasattr(fd_config, name):
        return getattr(fd_config, name)
    model_config = getattr(fd_config, "model_config", None)
    if model_config is not None and hasattr(model_config, name):
        return getattr(model_config, name)
    return default


def _flat_scalar(value) -> int:
    if hasattr(value, "numpy"):
        return int(value.numpy().reshape(-1)[0])
    if hasattr(value, "tolist"):
        result = value.tolist()
        return int(result[0] if isinstance(result, list) else result)
    return int(value)


class HybridReasoningMode(ThinkingBudgetLogitsProcessor):
    """MiniCPM4.1 hybrid reasoning mode.

    Reuses the generic ThinkingBudgetLogitsProcessor state machine; the
    MiniCPM4.1-specific part is only the budget source: an explicit
    `thinking_budget` request argument wins, then the request's
    `reasoning_max_tokens` (share_inputs["max_think_lens"]), and finally the
    model default `max_thinking_length`. Requests with `enable_thinking=False`
    get no budget and are never forced.
    """

    def __init__(self, fd_config):
        super().__init__(fd_config)
        if not self._enabled:
            raise ValueError("MiniCPM4.1 requires valid single-token markers or think_token_sequences")
        self.reasoning_tokens = _config_value(fd_config, "reasoning_tokens", _DEFAULT_REASONING_TOKENS)
        max_thinking_length = _config_value(fd_config, "max_thinking_length", _DEFAULT_MAX_THINKING_LENGTH)
        if (
            isinstance(max_thinking_length, bool)
            or not isinstance(max_thinking_length, int)
            or max_thinking_length <= 0
        ):
            raise ValueError("MiniCPM4.1 max_thinking_length must be a positive integer")
        self.max_thinking_length = max_thinking_length

    def _resolve_thinking_budget(self, logit_proc_args, slot_id, share_inputs):
        enable_thinking = share_inputs["enable_thinking"]
        if not bool(_flat_scalar(enable_thinking[slot_id])):
            return None
        budget = super()._resolve_thinking_budget(logit_proc_args, slot_id, share_inputs)
        if budget is not None:
            return budget
        max_think_lens = share_inputs["max_think_lens"]
        configured = _flat_scalar(max_think_lens[slot_id])
        if configured == 0 or configured < -1:
            raise ValueError("MiniCPM4.1 reasoning_max_tokens must be a positive integer or -1")
        if configured > 0:
            return configured
        return self.max_thinking_length
