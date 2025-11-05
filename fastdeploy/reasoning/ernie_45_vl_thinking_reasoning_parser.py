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

from collections.abc import Sequence
from typing import Optional, Union

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from fastdeploy.reasoning import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module("ernie-45-vl-thinking")
class Ernie45VLThinkingReasoningParser(ReasoningParser):
    """
    Reasoning parser for ernie_vl model.

    The ernie_vl model uses ...</think>... tokens to denote reasoning text
    within its output. The model provides a strict switch to disable reasoning
    output via the 'enable_thinking=False' parameter. This parser extracts the
    reasoning content enclosed by <think> and </think> tokens from the model's
    output.
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.think_end_token = "</think>"
        self.tool_begin_token = "<tool_call>"

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser " "constructor during construction."
            )

        self.think_end_token_id = self.vocab.get(self.think_end_token)
        self.tool_begin_token_id = self.vocab.get(self.tool_begin_token)
        if self.tool_begin_token_id is None:
            self.tool_begin_token_id = -1

        if self.think_end_token_id is None:
            raise RuntimeError("Test reasoning parser could not locate think end tokens in the tokenizer!")

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self.think_end_token_id in input_ids

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        """
        Extract reasoning content from a delta message.
        Handles streaming output where previous + delta = current.
        Uses token IDs for faster processing.
        For text abc</think>xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content
        """
        if self.think_end_token not in current_text:
            return DeltaMessage(reasoning_content=delta_text)
        # Skip single special tokens
        if len(delta_token_ids) == 1 and delta_token_ids[0] == self.think_end_token_id:
            return None
        if self._is_with_tool(current_text=current_text, current_token_ids=current_token_ids):
            if self.think_end_token in delta_text:
                think_begin = delta_text.find(self.think_end_token)
                reasoning_content = delta_text[:think_begin]
                return DeltaMessage(reasoning_content=reasoning_content)
            return None
        if self.think_end_token in delta_text:
            reasoning_content, _, content = delta_text.partition(self.think_end_token)
            striped_content = content.strip("\n")
            if len(striped_content) == 0:
                return DeltaMessage(reasoning_content=reasoning_content) if reasoning_content else None
            return (
                DeltaMessage(reasoning_content=reasoning_content, content=content)
                if reasoning_content
                else DeltaMessage(content=content)
            )
        think_end = current_text.find(self.think_end_token) + len(self.think_end_token)
        suffix = current_text[think_end:]
        striped_suffix = suffix.strip("\n")
        if len(striped_suffix) == 0:
            return None
        return DeltaMessage(content=delta_text)

    def extract_reasoning_content(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from the model output.

        For text abc</think>xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # Check if the model output contains the </think> tokens.
        if self.think_end_token not in model_output:
            return model_output, ""
        reasoning_content, _, content = model_output.partition(self.think_end_token)
        if self.tool_begin_token in content:
            prefix, _, _ = content.partition(self.tool_begin_token)
            prefix_strip = prefix.lstrip("\n")
            if len(prefix_strip) > 0:
                return reasoning_content, content
            return reasoning_content, ""
        return reasoning_content, content

    def _is_with_tool(self, current_text: str, current_token_ids: Sequence[int]) -> bool:
        think_end_index = current_text.find(self.think_end_token)
        think_end = think_end_index + len(self.think_end_token)
        middle_str = current_text[think_end:]
        if self.tool_begin_token_id in current_token_ids:
            prefix, _, _ = middle_str.partition(self.tool_begin_token)
            striped_prefix = prefix.strip("\n")
            if len(striped_prefix) > 0:
                return False
            return True
        return False
