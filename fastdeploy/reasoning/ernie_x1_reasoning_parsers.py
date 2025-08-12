# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
#
from collections.abc import Sequence
from typing import Tuple

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest
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

        self.think_end_token_id = self.model_tokenizer.vocab.get("\n</think>\n\n")
        if self.think_end_token_id is None:
            raise RuntimeError("Could not find think end token id in tokenizer vocabulary")

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> tuple[str, str]:
        """
        # v4.5.1 流式提取思考内容和回复内容.
        """
        # 合并文本处理
        text = previous_text + delta_text

        # 处理思考内容
        think_end_pos = text.find("</think>")
        reasoning_content = ""
        if think_end_pos != -1:
            # 检查是否是完整的结束标记(\n</think>\n\n)
            is_complete_end = (
                think_end_pos > 0
                and text[think_end_pos - 1] == "\n"
                and think_end_pos + len("</think>") < len(text)
                and text[think_end_pos + len("</think>")] == "\n"
            )
            print("is_complete_end", is_complete_end)

            # 提取思考内容
            content_end = think_end_pos - 1 if is_complete_end else think_end_pos
            reasoning_content = text[:content_end]
            print("思考内容", reasoning_content)
            remaining_text = text[think_end_pos + len("</think>") :]
            # 去除remaining_text前的所有换行符
            while remaining_text.startswith("\n"):
                remaining_text = remaining_text[1:]
            print("去除前缀remaining_text", remaining_text)

            # 检查response或tool_call
            if remaining_text.startswith("<tool_call>"):
                return reasoning_content, ""

            if remaining_text.startswith("<response>"):
                response_text = remaining_text[len("<response>") :]
                # 处理response内容
                if response_text.startswith("\n"):
                    response_text = response_text[1:]  # 跳过开始的\n

                # 查找response结束
                response_end_pos = response_text.find("</response>")
                if response_end_pos != -1:
                    # 检查结束标记前是否有\n
                    if response_end_pos > 0 and response_text[response_end_pos - 1] == "\n":
                        content = response_text[: response_end_pos - 1]
                    else:
                        content = response_text[:response_end_pos]
                    return reasoning_content, content
                else:
                    # 流式输出中还未收到完整response
                    return reasoning_content, response_text

        # 默认处理：如果之前已检测到think_end，则只输出delta_text作为content
        if self.think_end_token_id in previous_token_ids:
            return "", delta_text

        # 否则将delta_text作为reasoning_content
        return delta_text, ""

    def extract_reasoning_content(self, model_output: str, request: ChatCompletionRequest) -> Tuple[str, str]:
        """
        # v4.5.1 非流式提取思考内容和回复内容.
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
