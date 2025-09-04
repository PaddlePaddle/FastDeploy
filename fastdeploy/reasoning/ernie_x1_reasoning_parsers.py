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
    ) -> Union[DeltaMessage, None]:
        """
        Streaming parsing method implemented based on user requirements:
        1. Initial content is treated as reasoning content
        2. When encountering newline, check for </think> tag
        3. After reasoning phase, determine if it's <response> or <tool_call>
        4. For <response> content, process line breaks and termination tags
        """
        if len(delta_token_ids) == 1 and delta_token_ids[0] == self.think_end_token_id:
            return None
        # Thinking content processing
        if not previous_text.endswith(self.think_end_token) and self.think_end_token not in previous_text:
            # If thinking not ended, check for think end token
            if delta_text == "\n":
                return None
            # If previous is newline and current is </think>, end reasoning
            elif previous_text.endswith("\n") and delta_text.startswith(self.think_end_token):
                return None
            # If directly encountering </think>, end reasoning
            elif delta_text.startswith(self.think_end_token):
                return None
            # Otherwise continue returning reasoning content
            return DeltaMessage(reasoning_content=delta_text)

        # After reasoning phase, check for tool_call or response
        remaining_text = previous_text + delta_text
        after_think = remaining_text[remaining_text.find(self.think_end_token) + len(self.think_end_token) :]
        after_think = after_think.lstrip("\n")  # Skip newlines after think tag

        # Handle tool_call case
        if after_think.startswith(self.tool_call_start_token):
            return None

        # Handle response case
        if after_think.startswith(self.response_start_token):
            # Don't return immediately when encountering <response> tag
            if delta_text == self.response_start_token:
                return None
            # Also don't return immediately for newline after <response>
            elif delta_text == "\n" and previous_text.endswith(self.response_start_token):
                return None
            # Process newlines in response content
            if delta_text == "\n":
                return None
            # If previous is newline and current is </response>, end response
            elif previous_text.endswith("\n") and delta_text == self.response_end_token:
                return None
            # If directly encountering </response>, end response
            elif delta_text == self.response_end_token:
                return None
            # In other cases return actual content
            else:
                return DeltaMessage(content=delta_text)

        # Default case returns nothing
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
