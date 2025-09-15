# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
#
from collections.abc import Sequence
from typing import Tuple, Union

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from fastdeploy.reasoning import ReasoningParser, ReasoningParserManager

#
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
        self.think_end_token = "</think>"
        self.response_start_token = "<response>"
        self.response_end_token = "</response>"
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"

        if not self.model_tokenizer:
            raise ValueError("The model tokenizer must be passed to the ReasoningParser constructor.")

        self.think_end_token_id = self.vocab.get("</think>")
        if self.think_end_token_id is None:
            raise RuntimeError("Could not find think end token id in tokenizer vocabulary")
        self.tool_call_start_token_id = self.vocab.get("<tool_call>")

        # The number of \n
        self._pending_newlines = 0

    def _flush_newlines(self, text: str, reasoning: bool) -> DeltaMessage:
        """Output the accumulated \n"""
        pending = "\n" * self._pending_newlines
        self._pending_newlines = 0
        if reasoning:
            return DeltaMessage(reasoning_content=pending + text)
        else:
            return DeltaMessage(content=pending + text)

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Union[DeltaMessage, None]:
        if len(delta_token_ids) == 1 and delta_token_ids[0] == self.think_end_token_id:
            return None

        # ----------- Reasoning phase -----------
        if self.think_end_token not in previous_text:
            if delta_text.startswith(self.think_end_token):
                self._pending_newlines = 0
                return None

            if delta_text == "\n":
                self._pending_newlines += 1
                return None

            if self._pending_newlines > 0:
                return self._flush_newlines(delta_text, reasoning=True)

            return DeltaMessage(reasoning_content=delta_text)

        # ----------- After reasoning: response/tool_call phase -----------
        after_think = current_text[current_text.find(self.think_end_token) + len(self.think_end_token) :]
        after_think = after_think.lstrip("\n")

        if after_think.startswith(self.tool_call_start_token):
            self._pending_newlines = 0
            return None

        if after_think.startswith(self.response_start_token):
            if delta_text == self.response_start_token:
                self._pending_newlines = 0
                return None

            if delta_text == "\n" and previous_text.endswith(self.response_start_token):
                self._pending_newlines = 0
                return None

            if delta_text == self.response_end_token:
                self._pending_newlines = 0
                return None

            if delta_text == "\n":
                self._pending_newlines += 1
                return None

            if self._pending_newlines > 0:
                return self._flush_newlines(delta_text, reasoning=False)

            return DeltaMessage(content=delta_text)

        self._pending_newlines = 0
        return None

    def extract_reasoning_content(self, model_output: str, request: ChatCompletionRequest) -> Tuple[str, str]:
        """
        Batch version of the enhanced parser.
        Modified to preserve newlines in both reasoning and response content,
        only removing the single newline before closing tags.
        """
        reasoning_content = ""
        response_content = ""

        think_end_pos = model_output.find(self.think_end_token)
        if think_end_pos != -1:
            # Extract thinking content - only remove the last newline before </think>
            reasoning_content = model_output[:think_end_pos]
            if think_end_pos > 0 and reasoning_content[-1] == "\n":
                reasoning_content = reasoning_content[:-1]

            remaining = model_output[think_end_pos + len(self.think_end_token) :]

            # Skip newlines after </think>
            remaining = remaining.lstrip("\n")

            # Check for response or tool_call
            if remaining.startswith(self.response_start_token):
                response_pos = len(self.response_start_token)
                remaining = remaining[response_pos:].lstrip("\n")
                response_end_pos = remaining.find(self.response_end_token)
                if response_end_pos != -1:
                    # Only strip the last newline before </response>, not all
                    if response_end_pos > 0 and remaining[response_end_pos - 1] == "\n":
                        response_content = remaining[: response_end_pos - 1]
                    else:
                        response_content = remaining[:response_end_pos]
                else:
                    # If no </response> found, return the rest as response content
                    response_content = remaining
            elif remaining.startswith(self.tool_call_start_token):
                pass  # No response content
        else:
            # No thinking content found, return the whole input as reasoning
            reasoning_content = model_output
            response_content = ""
        return reasoning_content, response_content
