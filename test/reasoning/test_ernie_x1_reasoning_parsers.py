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

import unittest
from unittest.mock import MagicMock

from fastdeploy.reasoning.ernie_x1_reasoning_parsers import ErnieX1ReasoningParser


class TestErnieX1ReasoningParser(unittest.TestCase):
    """Test cases for ErnieX1ReasoningParser"""

    class MockTokenizer:
        def encode(self, text, *args, **kwargs):
            return [ord(c) for c in text]

    class MockRequest:
        pass

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
        self.request = self.MockRequest()

    def test_basic_xml_parsing(self):
        """测试基础XML解析"""
        test_input = "思考内容\n</think>\n\n<response>\n回复内容\n</response>"
        reasoning, response = self.parser.extract_reasoning_content(test_input, self.request)
        self.assertEqual(reasoning, "思考内容")
        self.assertEqual(response, "回复内容")

    def test_newline_handling(self):
        """测试换行符处理"""
        # 测试前导和尾部换行符
        test_input = "\n\n思考内容\n</think>\n\n<response>\n回复内容\n\n</response>\n"
        reasoning, response = self.parser.extract_reasoning_content(test_input, self.request)
        self.assertEqual(reasoning, "\n\n思考内容")
        self.assertEqual(response, "回复内容\n")

        # 测试无</think>的截断情况
        test_input = "\n\n思考内容\n<response>\n回复内容\n</response>"
        reasoning, response = self.parser.extract_reasoning_content(test_input, self.request)
        self.assertEqual(reasoning, "\n\n思考内容\n<response>\n回复内容\n</response>")
        self.assertEqual(response, "")

    def test_tool_call_parsing(self):
        """测试工具调用解析"""
        # 测试标准工具调用
        test_input1 = '思考内容\n</think>\n<tool_call>\n{"name":"tool","arguments":"args"}\n</tool_call>'
        reasoning1, response1 = self.parser.extract_reasoning_content(test_input1, self.request)
        self.assertEqual(reasoning1, "思考内容")
        self.assertEqual(response1, "")

        # 测试空工具调用
        test_input2 = "</think>\n<tool_call>\n{}\n</tool_call>"
        reasoning2, response2 = self.parser.extract_reasoning_content(test_input2, self.request)
        self.assertEqual(reasoning2, "")
        self.assertEqual(response2, "")

    def test_empty_content(self):
        """测试空内容处理"""
        # 测试空输入
        test_input = ""
        reasoning, response = self.parser.extract_reasoning_content(test_input, self.request)
        self.assertEqual(reasoning, "")
        self.assertEqual(response, "")

        # 测试只有标签
        test_input = "</think>\n\n<response></response>"
        reasoning, response = self.parser.extract_reasoning_content(test_input, self.request)
        self.assertEqual(reasoning, "")
        self.assertEqual(response, "")

        # 测试只有思考内容
        test_input = "只有思考内容"
        reasoning, response = self.parser.extract_reasoning_content(test_input, self.request)
        self.assertEqual(reasoning, "只有思考内容")
        self.assertEqual(response, "")

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
