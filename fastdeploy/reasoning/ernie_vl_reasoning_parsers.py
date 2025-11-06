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


@ReasoningParserManager.register_module("ernie-45-vl")
class ErnieVLReasoningParser(ReasoningParser):
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
        self.think_end_token = "</think>"

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser " "constructor during construction."
            )

        self.think_end_token_id = self.vocab.get(self.think_end_token)
        if self.think_end_token_id is None:
            raise RuntimeError("Ernie VL reasoning parser could not locate think end " "tokens in the tokenizer!")

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
        # Skip single special tokens
        if len(delta_token_ids) == 1 and delta_token_ids[0] == self.think_end_token_id:
            return None
        if self.think_end_token_id in delta_token_ids:
            end_index = delta_text.find(self.think_end_token)
            reasoning_content = delta_text[:end_index]
            content = delta_text[end_index + len(self.think_end_token) :]
            return DeltaMessage(reasoning_content=reasoning_content, content=content)
        elif self.think_end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text)
        else:
            return DeltaMessage(reasoning_content=delta_text)

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
            return "", model_output
        reasoning_content, _, content = model_output.partition(self.think_end_token)

        final_content = content or ""
        return reasoning_content, final_content
