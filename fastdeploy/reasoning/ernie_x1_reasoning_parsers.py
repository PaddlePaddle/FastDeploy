# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from collections.abc import Sequence
from typing import Tuple, Union

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from fastdeploy.reasoning import ReasoningParser, ReasoningParserManager

from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from fastdeploy.reasoning import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module("ernie_x1")
class ErnieX1ReasoningParser(ReasoningParser):
    """
    Reasoning parser for ernie_x1 model with stricter boundary checking.

    This implementation handles streaming in three stages:
    1. Thinking content (<think>...</think>):
       - Cache newlines until it is clear they are not the trailing ones before </think>.
       - Drop the last newline immediately before </think>.
    2. Response content (<response>...</response>):
       - Ignore the first newline right after <response>.
       - Cache newlines and flush them when next token is normal text.
       - Drop the newline immediately before </response>.
    3. Tool call content (<tool_call>...</tool_call>):
       - Ignored in this parser.
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
            raise RuntimeError("Could not find </think> token id in tokenizer vocabulary")
        self.tool_call_start_token_id = self.vocab.get("<tool_call>")

        # Cached count of newlines waiting to be flushed
        self._pending_newlines = 0

    def _encode_tokens(self, text: str) -> List[int]:
        """Convert text into token ids using model tokenizer."""
        tokens = self.model_tokenizer.encode(text, add_special_tokens=False)
        # If returned object is BatchEncoding, extract input_ids
        if hasattr(tokens, "input_ids"):
            return tokens.input_ids
        return list(tokens)

    def _flush_newlines(
        self, text: str, reasoning: bool, delta_token_ids: Optional[List[int]] = None, keep_last: bool = False
    ) -> DeltaMessage:
        """
        Flush cached newlines along with the current delta text.

        Args:
            text (str): Current text to send.
            reasoning (bool): Whether this is reasoning content or normal response.
            delta_token_ids (Optional[List[int]]): Token ids of the current text.
            keep_last (bool): If True, keep the last pending newline (used when the next token
                              is a closing tag, so the last newline immediately before the tag is dropped).

        Returns:
            DeltaMessage: The combined message containing flushed newlines and current text.
        """
        pending_count = self._pending_newlines
        if keep_last and pending_count > 0:
            # Only flush all but the last newline
            pending_count -= 1

        pending_text = "\n" * pending_count
        print("到达转化为tokens前")
        pending_token_ids = self._encode_tokens(pending_text) if pending_count > 0 else []

        self._pending_newlines = 0

        combined_text = pending_text + text
        print("到达拼接tokens前")
        combined_token_ids = pending_token_ids + (delta_token_ids or [])

        if reasoning:
            return DeltaMessage(
                reasoning_content=combined_text,
                completion_token_ids=combined_token_ids,
            )
        else:
            return DeltaMessage(
                content=combined_text,
                completion_token_ids=combined_token_ids,
            )

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
        Streaming parser for reasoning, response, and tool call stages.

        Key logic:
        1. Thinking (<think>...</think>):
           - Cache \n and only flush them when next token is not immediately </think>.
           - Drop the \n immediately before </think>.
        2. Response (<response>...</response>):
           - Ignore first \n after <response>.
           - Cache \n in middle content.
           - Drop the \n immediately before </response>.
        3. Tool call (<tool_call>...</tool_call>) is ignored.
        """
        # ----------- Reasoning phase -----------
        if self.think_end_token not in previous_text:
            # If </think> token is reached, end reasoning phase
            if delta_token_ids and delta_token_ids[-1] == self.think_end_token_id:
                self._pending_newlines = 0
                return None

            if delta_text.startswith(self.think_end_token):
                self._pending_newlines = 0
                return None

            if delta_text == "\n":
                self._pending_newlines += 1
                return None

            if self._pending_newlines > 0:
                return self._flush_newlines(delta_text, reasoning=True, delta_token_ids=delta_token_ids)

            return DeltaMessage(reasoning_content=delta_text, completion_token_ids=delta_token_ids)

        # ----------- After reasoning: response/tool_call phase -----------
        after_think = current_text[current_text.find(self.think_end_token) + len(self.think_end_token) :]
        after_think = after_think.lstrip("\n")

        # Tool call phase is ignored
        if after_think.startswith(self.tool_call_start_token):
            self._pending_newlines = 0
            return None

        # Response content phase
        if after_think.startswith(self.response_start_token):
            # Skip the <response> token itself
            if delta_text == self.response_start_token:
                self._pending_newlines = 0
                return None

            # Ignore first newline immediately after <response>
            if delta_text == "\n" and previous_text.endswith(self.response_start_token):
                self._pending_newlines = 0
                return None

            # If closing response, flush cached newlines except the last one before </response>
            if delta_text == self.response_end_token:
                if self._pending_newlines > 0:
                    return self._flush_newlines("", reasoning=False, keep_last=True)
                self._pending_newlines = 0
                return None

            # Cache newlines in middle of response
            if delta_text == "\n":
                self._pending_newlines += 1
                return None

            # Flush any cached newlines before current text
            if self._pending_newlines > 0:
                return self._flush_newlines(delta_text, reasoning=False, delta_token_ids=delta_token_ids)

            return DeltaMessage(content=delta_text, completion_token_ids=delta_token_ids)

        self._pending_newlines = 0
        return None

    def extract_reasoning_content(self, model_output: str, request: ChatCompletionRequest) -> Tuple[str, str]:
        """
        Batch parser preserving newlines in reasoning and response content.
        Only removes the single newline immediately before closing tags.
        """
        reasoning_content = ""
        response_content = ""

        think_end_pos = model_output.find(self.think_end_token)
        if think_end_pos != -1:
            # Extract thinking content - remove only the last newline before </think>
            reasoning_content = model_output[:think_end_pos]
            if think_end_pos > 0 and reasoning_content[-1] == "\n":
                reasoning_content = reasoning_content[:-1]

            remaining = model_output[think_end_pos + len(self.think_end_token) :]
            remaining = remaining.lstrip("\n")

            # Check for response or tool_call
            if remaining.startswith(self.response_start_token):
                response_pos = len(self.response_start_token)
                remaining = remaining[response_pos:].lstrip("\n")
                response_end_pos = remaining.find(self.response_end_token)
                if response_end_pos != -1:
                    # Remove only the newline immediately before </response>
                    if response_end_pos > 0 and remaining[response_end_pos - 1] == "\n":
                        response_content = remaining[: response_end_pos - 1]
                    else:
                        response_content = remaining[:response_end_pos]
                else:
                    # If no </response> found, return rest as response content
                    response_content = remaining
            elif remaining.startswith(self.tool_call_start_token):
                pass  # No response content
        else:
            # No thinking content found
            reasoning_content = model_output
            response_content = ""
        return reasoning_content, response_content
