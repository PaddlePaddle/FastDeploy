"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from collections import Counter, deque

from fastdeploy.output.fallback.base import (
    OutputFallbackContext,
    OutputFallbackStrategy,
    StreamFallbackDecision,
)
from fastdeploy.output.fallback.manager import OutputFallbackManager


class RepeatTruncateWindow:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.tokens = deque()
        self.token_freq = Counter()

    def add_tokens(self, token_ids: list[int]) -> None:
        for token_id in token_ids:
            if len(self.tokens) == self.max_size:
                old_token_id = self.tokens.popleft()
                self.token_freq[old_token_id] -= 1
                if self.token_freq[old_token_id] == 0:
                    del self.token_freq[old_token_id]
            self.tokens.append(token_id)
            self.token_freq[token_id] += 1

    def get_all_tokens(self) -> list[int]:
        return list(self.tokens)

    def get_unique_token_count(self) -> int:
        return len(self.token_freq)

    def is_full(self) -> bool:
        return len(self.tokens) == self.max_size


@OutputFallbackManager.register("repeat-truncate")
class RepeatTruncateFallbackStrategy(OutputFallbackStrategy):
    name = "repeat-truncate"
    default_free_len = 500
    default_window_len = 150
    default_max_cycle_len = 15

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.free_len = int(self.config.get("free_len", self.default_free_len))
        self.window_len = int(self.config.get("window_len", self.default_window_len))
        self.max_cycle_len = int(self.config.get("max_cycle_len", self.default_max_cycle_len))

    def should_apply(self, text: str, context: OutputFallbackContext) -> bool:
        return False

    def apply(self, text: str, context: OutputFallbackContext) -> str:
        return text

    def on_delta(self, delta_text: str, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        token_ids = context.output.get("token_ids") or []
        if not token_ids:
            return StreamFallbackDecision(action="send", text=delta_text)

        window = state.get("repeat_truncate_window")
        if window is None:
            window = RepeatTruncateWindow(self.window_len)
            state["repeat_truncate_window"] = window

        window.add_tokens(token_ids)
        output_tokens = state.get("output_tokens", 0) + len(token_ids)
        state["output_tokens"] = output_tokens

        if output_tokens < self.free_len or not window.is_full():
            return StreamFallbackDecision(action="send", text=delta_text)

        if window.get_unique_token_count() > self.max_cycle_len:
            return StreamFallbackDecision(action="send", text=delta_text)

        if self._detect_prefix_coverage(window):
            return StreamFallbackDecision(action="truncate", text=delta_text)

        return StreamFallbackDecision(action="send", text=delta_text)

    def _detect_prefix_coverage(self, window: RepeatTruncateWindow) -> bool:
        tokens = window.get_all_tokens()
        token_len = len(tokens)
        unique_token_count = window.get_unique_token_count()
        if unique_token_count == 1:
            return True

        for prefix_len in range(unique_token_count, self.max_cycle_len + 1):
            if prefix_len * 2 > token_len:
                break
            prefix = tokens[:prefix_len]
            if self._prefix_covers_tokens(tokens, prefix):
                return True
        return False

    def _prefix_covers_tokens(self, tokens: list[int], prefix: list[int]) -> bool:
        prefix_len = len(prefix)
        for start in range(0, len(tokens), prefix_len):
            part = tokens[start : start + prefix_len]
            if part != prefix[: len(part)]:
                return False
        return True
