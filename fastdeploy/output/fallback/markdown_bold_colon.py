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

import re

from fastdeploy.output.fallback.base import (
    OutputFallbackContext,
    OutputFallbackStrategy,
    StreamFallbackDecision,
)
from fastdeploy.output.fallback.manager import OutputFallbackManager


@OutputFallbackManager.register("markdown-bold-colon")
class MarkdownBoldColonFallbackStrategy(OutputFallbackStrategy):
    name = "markdown-bold-colon"
    bold = "**"
    pattern = re.compile(r"(\*\*.*?)([:：])?(\*\*)")
    colon_and_bold_pattern = re.compile(r"([:：])?(\*\*)")

    def should_apply(self, text: str, context: OutputFallbackContext) -> bool:
        return bool(self.pattern.search(text))

    def apply(self, text: str, context: OutputFallbackContext) -> str:
        return self.pattern.sub(r"\1\3\2", text)

    def on_delta(self, delta_text: str, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        if not delta_text:
            return StreamFallbackDecision(action="send", text=delta_text)
        text = self._handle_stream_text(delta_text, state)
        if not text:
            return StreamFallbackDecision(action="hold")
        return StreamFallbackDecision(action="send", text=text)

    def on_finish(self, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        cached_text = state.pop("gfm_cache", "")
        state["bold_prefix_exist"] = False
        if not cached_text:
            return StreamFallbackDecision(action="flush")
        return StreamFallbackDecision(action="flush", text=cached_text)

    def _handle_stream_text(self, text: str, state: dict) -> str:
        parts = []
        while text:
            if not state.get("bold_prefix_exist", False):
                bold_index = text.find(self.bold)
                if bold_index < 0:
                    parts.append(text)
                    break
                bold_end = bold_index + len(self.bold)
                parts.append(text[:bold_end])
                text = text[bold_end:]
                state["bold_prefix_exist"] = True
                state["gfm_cache"] = ""
                continue

            bold_index = text.find(self.bold)
            if bold_index >= 0:
                prefix_with_bold = text[: bold_index + len(self.bold)]
                suffix = text[bold_index + len(self.bold) :]
                state["bold_prefix_exist"] = False
                cached_text = state.get("gfm_cache", "")
                if cached_text:
                    prefix_without_bold = text[:bold_index]
                    if prefix_without_bold:
                        new_prefix = self._swap_colon_before_bold(prefix_with_bold)
                        parts.append(cached_text + new_prefix)
                    else:
                        new_text = cached_text + text
                        new_bold_index = new_text.find(self.bold)
                        new_prefix = self._swap_colon_before_bold(new_text[: new_bold_index + len(self.bold)])
                        parts.append(new_prefix)
                        suffix = new_text[new_bold_index + len(self.bold) :]
                    state["gfm_cache"] = ""
                else:
                    new_prefix = self._swap_colon_before_bold(prefix_with_bold)
                    parts.append(new_prefix)
                text = suffix
                continue

            if text.endswith((":", "：")):
                cached_text = state.get("gfm_cache", "")
                if cached_text:
                    parts.append(cached_text)
                state["gfm_cache"] = text
                break

            cached_text = state.get("gfm_cache", "")
            if cached_text:
                parts.append(cached_text)
                state["gfm_cache"] = ""
            parts.append(text)
            break
        return "".join(parts)

    def _swap_colon_before_bold(self, text: str) -> str:
        return self.colon_and_bold_pattern.sub(r"\2\1", text)
