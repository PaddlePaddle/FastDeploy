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

        self.think_end_token_id = self.vocab.get("</think>")
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
        根据用户需求实现的流式解析方法:
        1. 初始内容都视为思考内容
        2. 当遇到\n时检查后续是否是</think>
        3. 思考结束后检查是<response>还是<tool_call>
        4. 对于<response>内容，处理换行和结束标记
        """
        # 如果还在思考阶段
        if not previous_text.endswith(self.think_end_token):
            # 如果遇到\n后接</think>或直接遇到</think>，思考结束
            if (previous_text.endswith("\n") and delta_text == self.think_end_token) or (
                not previous_text.endswith("\n") and delta_text == self.think_end_token
            ):
                return "", ""
            # 否则继续返回思考内容
            return delta_text, ""

        # 思考结束后检查是tool_call还是response
        remaining_text = previous_text + delta_text
        after_think = remaining_text[remaining_text.find(self.think_end_token) + len(self.think_end_token) :]

        # 跳过think后的换行
        after_think = after_think.lstrip("\n")

        # 处理tool_call情况
        if after_think.startswith(self.tool_call_start_token):
            return "", ""

        # 处理response情况
        if after_think.startswith(self.response_start_token):
            response_content = after_think[len(self.response_start_token) :]
            # 跳过response后的换行
            response_content = response_content.lstrip("\n")

            # 检查response是否结束
            if response_content.endswith(self.response_end_token):
                return "", ""

            # 返回response内容(使用delta_text确保流式输出)
            return "", delta_text

        # 默认情况不返回内容
        return "", ""

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


import unittest
from unittest.mock import MagicMock


class TestErnieX1ReasoningParser(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MagicMock()
        self.tokenizer.vocab = {
            "\n</think>\n\n": 1001,
            "<response>\n": 1002,
            "\n</response>\n": 1003,
            "<tool_call>\n": 1004,
            "\n</tool_call>\n": 1005,
        }
        self.parser = ErnieX1ReasoningParser(self.tokenizer)

    def test_streaming_with_think_and_response(self):
        # 测试标准情况：\n</think>\n\n<response>\ncontent\n</response>\n
        prev_text = "thinking"
        delta_text = "\n</think>\n\n<response>\nanswer\n</response>\n"
        result = self.parser.extract_reasoning_content_streaming(prev_text, "", delta_text, [], [], [])
        self.assertEqual(result, ("thinking", "answer"))

    def test_streaming_with_think_and_tool_call(self):
        # 测试tool_call情况
        prev_text = "thinking"
        delta_text = "\n</think>\n\n<tool_call>\ndetails\n</tool_call>\n"
        result = self.parser.extract_reasoning_content_streaming(prev_text, "", delta_text, [], [], [])
        self.assertEqual(result, ("thinking", ""))

    def test_streaming_with_think_no_newline(self):
        # 测试没有前置换行的情况
        prev_text = "thinking"
        delta_text = "</think>\n\n<response>answer</response>\n"
        result = self.parser.extract_reasoning_content_streaming(prev_text, "", delta_text, [], [], [])
        self.assertEqual(result, ("thinking", "answer"))

    def test_streaming_response_without_leading_newline(self):
        # 测试response内容没有前置换行
        prev_text = "thinking\n</think>\n\n"
        delta_text = "<response>answer\n</response>\n"
        result = self.parser.extract_reasoning_content_streaming(prev_text, "", delta_text, [1001], [], [])
        self.assertEqual(result, ("thinking", "answer"))

    def test_streaming_response_with_middle_newline(self):
        # 测试response内容中间的换行符
        prev_text = "thinking\n</think>\n\n<response>\n"
        delta_text = "line1\nline2\n</response>\n"
        result = self.parser.extract_reasoning_content_streaming(prev_text, "", delta_text, [1001], [], [])
        self.assertEqual(result, ("thinking", "line1\nline2"))

    def test_streaming_partial_response(self):
        # 测试不完整的response流式输出
        prev_text = "thinking\n</think>\n\n<response>\n"
        delta_text = "partial answer"
        result = self.parser.extract_reasoning_content_streaming(prev_text, "", delta_text, [1001], [], [])
        self.assertEqual(result, ("thinking", "partial answer"))


if __name__ == "__main__":
    unittest.main()
