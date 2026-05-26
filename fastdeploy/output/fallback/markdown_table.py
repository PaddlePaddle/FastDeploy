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


@OutputFallbackManager.register("markdown-table")
class MarkdownTableFallbackStrategy(OutputFallbackStrategy):
    name = "markdown-table"

    start_pattern = re.compile(r"(?m)^ *\|")
    first_row_pattern = re.compile(r"(?m)^ *\|.*?\| *\n*")
    second_row_like_pattern = re.compile(r"^ *\|[:\- |]*$")
    second_row_pattern = re.compile(r"^ *\|([:\- |]*)\|?$")
    match_two_row_with_newline_pattern = re.compile(r"(?m)^ *\|.*?\| *\n *\|([:\- ]*\|)+ *\n")
    match_two_row_pattern = re.compile(r"(?m)^ *\|.*?\| *\n *\|([:\- ]*\|)+")
    table_replace_pattern = re.compile(r"(\| *: *: *\|)|(\| *: *\|)|(\| *\|)")

    def should_apply(self, text: str, context: OutputFallbackContext) -> bool:
        return bool(self.match_two_row_pattern.search(text))

    def apply(self, text: str, context: OutputFallbackContext) -> str:
        return self.match_two_row_pattern.sub(self._replace_table_match, text)

    def on_delta(self, delta_text: str, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        text = self._handle_stream_text(delta_text, state)
        if not text:
            return StreamFallbackDecision(action="hold")
        return StreamFallbackDecision(action="send", text=text)

    def on_finish(self, context: OutputFallbackContext, state: dict) -> StreamFallbackDecision:
        cached_text = state.get("md_table_cache", "")
        self._reset_state(state)
        if not cached_text:
            return StreamFallbackDecision(action="flush")
        return StreamFallbackDecision(action="flush", text=cached_text)

    def _handle_stream_text(self, text: str, state: dict) -> str:
        if not state.get("md_table_cache_start", False):
            if "|" not in text or not self.start_pattern.search(text):
                return text
            state["md_table_cache_start"] = True
            state["md_table_cache"] = text
            state["md_table_cache_start_idx"] = text.find("|")
            state["md_table_cache_first_row"] = False
        else:
            state["md_table_cache"] = state.get("md_table_cache", "") + text

        cached_text = state["md_table_cache"]
        start_idx = state.get("md_table_cache_start_idx", 0)
        right = cached_text[start_idx:]

        if not state.get("md_table_cache_first_row", False):
            if "\n" not in right:
                return ""
            if not self.first_row_pattern.search(right) or not self._is_first_row(right):
                self._reset_state(state)
                return cached_text
            state["md_table_cache_first_row"] = True

        if self._first_two_row_like(right):
            if self.match_two_row_with_newline_pattern.search(right):
                prefix = cached_text[:start_idx]
                match = self.match_two_row_pattern.search(right)
                if not match:
                    self._reset_state(state)
                    return cached_text

                target = right[: match.end()]
                post = right[match.end() :]
                if target.endswith("\n"):
                    target = target[:-1]
                    post = "\n" + post
                fixed_target = self._fix_table_rows(target)
                self._reset_state(state)
                return prefix + fixed_target + post
            return ""

        self._reset_state(state)
        return cached_text

    def _replace_table_match(self, match: re.Match) -> str:
        return self._fix_table_rows(match.group(0))

    def _fix_table_rows(self, text: str) -> str:
        split = text.split("\n")
        if len(split) != 2:
            return text

        first, second = split
        if not self._is_first_row(first):
            return text

        second = self._normalize_second_row(second)
        col = first.count("|") - 1
        second_col = second.count("|") - 1
        diff = col - second_col
        if diff > 0:
            second += "-|" * diff
        elif diff < 0:
            indexes = [index for index, char in enumerate(second) if char == "|"]
            if len(indexes) >= col + 1:
                second = second[: indexes[col] + 1]
        return first + "\n" + second

    def _normalize_second_row(self, text: str) -> str:
        for _ in range(20):
            fixed = self.table_replace_pattern.sub("|-|", text)
            if fixed == text:
                break
            text = fixed
        return text

    def _is_first_row(self, text: str) -> bool:
        return any(cell.strip() for cell in text.split("|"))

    def _first_two_row_like(self, text: str) -> bool:
        split = text.split("\n")
        if len(split) < 2:
            return False

        first_row_match = bool(self.first_row_pattern.search(split[0]))
        second_row_match = bool(self.second_row_pattern.search(split[1]))
        second_row_like = bool(self.second_row_like_pattern.search(split[1]))
        if len(split) == 2:
            return first_row_match and (not split[1].strip() or second_row_like)
        return first_row_match and second_row_match

    def _reset_state(self, state: dict) -> None:
        state["md_table_cache"] = ""
        state["md_table_cache_start"] = False
        state["md_table_cache_start_idx"] = 0
        state["md_table_cache_first_row"] = False
