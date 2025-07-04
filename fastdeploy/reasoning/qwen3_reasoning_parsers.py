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

from fastdeploy.entrypoints.openai.protocol import (ChatCompletionRequest,
                                                    DeltaMessage)
from fastdeploy.reasoning import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module("qwen3")
class Qwen3ReasoningParser(ReasoningParser):
    """
    Reasoning parser for ernir_vl model.

    The ernie_vl model uses ...</think>... tokens to denote reasoning text
    within its output. The model provides a strict switch to disable reasoning
    output via the 'enable_thinking=False' parameter. This parser extracts the
    reasoning content enclosed by <think> and </think> tokens from the model's
    output.
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

        self.think_start_token_id = self.vocab.get(self.think_start_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)
        if self.think_end_token_id is None:
            raise RuntimeError(
                "Qwen3  reasoning parser could not locate think end "
                "tokens in the tokenizer!")

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
        # Skip single special tokens
        if len(delta_token_ids) == 1 and (delta_token_ids[0] in [
                self.think_start_token_id, self.think_end_token_id
        ]):
            return "", ""

        if self.think_start_token_id in previous_token_ids:
            if self.think_end_token_id in delta_token_ids:
                # <think> in previous, </think> in delta,
                # extract reasoning content
                end_index = delta_text.find(self.think_end_token)
                reasoning_content = delta_text[:end_index]
                content = delta_text[end_index + len(self.think_end_token):]
                content = content if content else None
                return reasoning_content, content
            elif self.think_end_token_id in previous_token_ids:
                # <think> in previous, </think> in previous,
                # reasoning content continues
                return "", delta_text
            else:
                # <think> in previous, no </think> in previous or delta,
                # reasoning content continues
                return delta_text, ""
        elif self.think_start_token_id in delta_token_ids:
            if self.think_end_token_id in delta_token_ids:
                # <think> in delta, </think> in delta, extract reasoning content
                start_index = delta_text.find(self.think_start_token)
                end_index = delta_text.find(self.think_end_token)
                reasoning_content = delta_text[start_index +
                                               len(self.think_start_token
                                                   ):end_index]
                content = delta_text[end_index + len(self.think_end_token):]
                content = content if content else None
                return reasoning_content, content
            else:
                # <think> in delta, no </think> in delta,
                # reasoning content continues
                return delta_text, ""
        else:
            # thinking is disabled, just content
            return "", delta_text

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from the model output.

        支持两种格式:
        1. <think>abc</think>xyz - 标准格式
        2. abc</think>xyz - 缺少起始标签的格式

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # 检查是否包含结束标签
        if self.think_end_token not in model_output:
            return None, model_output

        # 检查是否有起始标签
        if self.think_start_token in model_output:
            # 标准格式：<think>content</think>answer
            start_idx = model_output.find(self.think_start_token)
            end_idx = model_output.find(self.think_end_token)

            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                reasoning_content = model_output[start_idx +
                                                 len(self.think_start_token
                                                     ):end_idx].strip()
                final_content = model_output[end_idx +
                                             len(self.think_end_token):].strip(
                                             )
                final_content = final_content if final_content else None

                return reasoning_content, final_content
        else:
            # 缺少起始标签的格式：content</think>answer
            parts = model_output.split(self.think_end_token, 1)

            if len(parts) == 2:
                reasoning_content = parts[0].strip()
                final_content = parts[1].strip() if parts[1].strip() else None
                return reasoning_content, final_content

        return None, model_output
