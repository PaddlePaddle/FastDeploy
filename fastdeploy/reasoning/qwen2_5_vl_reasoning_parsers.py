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

from collections.abc import Sequence
from typing import Optional, Union

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from fastdeploy.reasoning import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module("qwen-25-vl")
class Qwen25VLReasoningParser(ReasoningParser):
    """
    Reasoning parser for Qwen2.5-VL model.

    The Qwen2.5-VL model uses <|im_start|>thinking<|im_end|> tokens to denote reasoning text
    within its output. This parser extracts the reasoning content enclosed by these tokens.
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.thinking_start_token = "<|im_start|>thinking<|im_end|>"
        self.thinking_end_token = "<|im_start|>assistant<|im_end|>"

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser constructor."
            )

        self.thinking_start_token_id = self.vocab.get(self.thinking_start_token)
        self.thinking_end_token_id = self.vocab.get(self.thinking_end_token)
        if self.thinking_end_token_id is None:
            raise RuntimeError("Qwen2.5-VL reasoning parser could not locate thinking tokens in the tokenizer!")

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """
        Check if the reasoning content ends in the input_ids.
        """
        return self.thinking_end_token_id in input_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.
        """
        if self.thinking_end_token_id not in input_ids:
            return input_ids
            
        end_idx = input_ids.index(self.thinking_end_token_id)
        return input_ids[end_idx + 1:]

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
        Extract reasoning content from streaming output.
        """
        if len(delta_token_ids) == 1 and delta_token_ids[0] in [self.thinking_start_token_id, self.thinking_end_token_id]:
            return "", ""

        if self.thinking_end_token_id in delta_token_ids:
            end_index = delta_text.find(self.thinking_end_token)
            reasoning_content = delta_text[:end_index]
            content = delta_text[end_index + len(self.thinking_end_token):]
            return reasoning_content, content
        elif self.thinking_end_token_id in previous_token_ids:
            return "", delta_text
        elif self.thinking_start_token_id in previous_token_ids:
            return delta_text, ""
        elif self.thinking_start_token_id in delta_token_ids:
            start_index = delta_text.find(self.thinking_start_token)
            reasoning_content = delta_text[start_index + len(self.thinking_start_token):]
            content = ""
            return reasoning_content, content
        else:
            return delta_text, ""

    def extract_reasoning_content(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from the complete model output.
        """
        if self.thinking_end_token not in model_output:
            return None, model_output

        if self.thinking_start_token in model_output:
            parts = model_output.split(self.thinking_start_token, 1)
            if len(parts) == 2:
                reasoning_content = parts[1].split(self.thinking_end_token, 1)[0]
                content = parts[1].split(self.thinking_end_token, 1)[1] if len(parts[1].split(self.thinking_end_token, 1)) > 1 else ""
                return reasoning_content.strip(), content.strip() or None

        parts = model_output.split(self.thinking_end_token, 1)
        if len(parts) == 2:
            reasoning_content = parts[0].strip()
            content = parts[1].strip() if parts[1].strip() else None
            return reasoning_content, content

        return None, model_output