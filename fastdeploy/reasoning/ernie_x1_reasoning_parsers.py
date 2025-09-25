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
from typing import Tuple, Union

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from fastdeploy.reasoning import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module("ernie_x1")
class ErnieX1ReasoningParser(ReasoningParser):
    """
    Reasoning parser for ernie_x1 model with stricter boundary checking.

    This implementation follows the user's proposed approach:
    1. For thinking content: waits for \n then checks for </think> tag
    2. For response content: checks for <response> tag first, then waits for \n
    3. Handles newlines in content more precisely
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)

        # 定义所有需要检查的token
        token_definitions = {
            "think_start_token": "<think>",
            "think_end_token": "</think>",
            "response_start_token": "<response>",
            "response_end_token": "</response>",
            "tool_call_start_token": "<tool_call>",
            "tool_call_end_token": "</tool_call>",
        }

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ReasoningParser constructor.")

        missing_tokens = []
        for name, token_value in token_definitions.items():
            setattr(self, name, token_value)
            token_id = self.vocab.get(token_value)
            setattr(self, f"{name}_id", token_id)
            if token_id is None:
                missing_tokens.append(f"{name.replace('_', ' ')} token")

        if missing_tokens:
            raise RuntimeError(
                f"Could not find the following token ids in tokenizer vocabulary: {', '.join(missing_tokens)}"
            )

        self.token_status_mapping = {
            self.think_start_token_id: "think_start",
            self.think_end_token_id: "think_end",
            self.response_start_token_id: "response_start",
            self.response_end_token_id: "response_end",
            self.tool_call_start_token_id: "tool_call_start",
            self.tool_call_end_token_id: "tool_call_end",
        }

    def find_last_special_token(self, prompt_token_ids: list[int]) -> int:
        for i in range(len(prompt_token_ids) - 1, -1, -1):
            if prompt_token_ids[i] in [
                self.think_end_token_id,
                self.think_start_token_id,
                self.response_start_token_id,
                self.response_end_token_id,
                self.tool_call_start_token_id,
                self.tool_call_end_token_id,
            ]:
                return prompt_token_ids[i]
        return -1

    def get_model_status(self, prompt_token_ids: list[int]):
        special_token_id = self.find_last_special_token(prompt_token_ids)

        if special_token_id == -1:
            return "think_start"

        return self.token_status_mapping[special_token_id]

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self.tool_call_start_token_id in input_ids

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        model_status: str,
    ) -> Union[DeltaMessage, None]:

        if len(delta_token_ids) == 1 and delta_token_ids[0] in [
            self.think_end_token_id,
            self.response_start_token_id,
            self.response_end_token_id,
        ]:
            return None

        if model_status == "think_start":
            if self.think_end_token_id not in current_token_ids:
                return DeltaMessage(reasoning_content=delta_text)
            else:
                if (
                    self.response_start_token_id in current_token_ids
                    and self.response_end_token_id not in current_token_ids
                ):
                    return DeltaMessage(content=delta_text)
        elif model_status == "think_end":
            if (
                self.response_start_token_id in current_token_ids
                and self.response_end_token_id not in current_token_ids
            ):
                return DeltaMessage(content=delta_text)
        elif model_status == "response_start":
            if self.response_end_token_id not in current_token_ids:
                return DeltaMessage(content=delta_text)

        return None

    def extract_reasoning_content(
        self, model_output: str, request: ChatCompletionRequest, model_status: str
    ) -> Tuple[str, str]:
        """
        Optimized batch version of the enhanced parser.
        Preserves newlines in both reasoning and response content,
        only removing the single newline before closing tags.
        """
        reasoning_content = ""
        response_content = ""

        if model_status == "think_start":
            think_end_pos = model_output.find(self.think_end_token)
            if think_end_pos != -1:
                reasoning_content = model_output[:think_end_pos]
                remaining = model_output[think_end_pos + len(self.think_end_token) :].lstrip("\n")

                # Determine if remaining content is a response or tool call
                if remaining.startswith(self.response_start_token):
                    response_start_len = len(self.response_start_token)
                    response_content = self._extract_response_content(remaining[response_start_len:])
                elif remaining.startswith(self.tool_call_start_token):
                    pass  # No response content
            else:
                reasoning_content = model_output

        elif model_status == "think_end":
            remaining = model_output.lstrip("\n")
            if remaining.startswith(self.response_start_token):
                response_start_len = len(self.response_start_token)
                response_content = self._extract_response_content(remaining[response_start_len:])

        elif model_status == "response_start":
            response_content = self._extract_response_content(model_output)

        return reasoning_content, response_content

    def _extract_response_content(self, remaining: str) -> str:
        """
        Extracts response content, ensuring that the last newline before
        the </response> tag is removed.
        """
        response_end_pos = remaining.find(self.response_end_token)
        if response_end_pos != -1:
            return remaining[:response_end_pos]
        return remaining
