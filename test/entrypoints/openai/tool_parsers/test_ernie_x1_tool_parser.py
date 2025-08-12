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
# 当前只有非流式测试代码

import unittest
from unittest.mock import MagicMock

from fastdeploy.entrypoints.openai.protocol import ExtractedToolCallInformation
from fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser import (
    ErnieX1ToolParser,
)


class TestErnieX1ToolParser(unittest.TestCase):
    """Test cases for ErnieX1ToolParser"""

    def setUp(self):
        self.tokenizer = MagicMock()
        self.parser = ErnieX1ToolParser(self.tokenizer)

    def test_extract_tool_calls_normal(self):
        """测试标准工具调用提取"""
        model_output = """\n</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}\n</tool_call>\n"""

        result = self.parser.extract_tool_calls(model_output, MagicMock())
        self.assertIsInstance(result, ExtractedToolCallInformation)
        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertEqual(
            result.tool_calls[0].function.arguments, '{"location": "San Francisco, CA", "unit": "fahrenheit"}'
        )
        self.assertEqual(result.content, "")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertEqual(
            result.tool_calls[0].function.arguments, '{"location": "San Francisco, CA", "unit": "fahrenheit"}'
        )
        self.assertEqual(result.content, "")

    def test_extract_tool_calls_multiple(self):
        """测试多个工具调用提取"""
        model_output = """\n</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"location": "New York", "unit": "celsius"}}\n</tool_call>\n<tool_call>\n{"name": "search", "arguments": {"query": "AI news"}}\n</tool_call>\n"""

        result = self.parser.extract_tool_calls(model_output, MagicMock())
        self.assertIsInstance(result, ExtractedToolCallInformation)
        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertEqual(result.tool_calls[1].function.name, "search")
        self.assertEqual(result.content, "")

    def test_extract_tool_calls_complex_args(self):
        """测试复杂参数的工具调用提取"""
        model_output = """\n</think>\n\n<tool_call>\n{"name": "analyze", "arguments": {"data": [1,2,3], "options": {"threshold": 0.5, "enabled": true}}}\n</tool_call>\n"""

        result = self.parser.extract_tool_calls(model_output, MagicMock())
        self.assertIsInstance(result, ExtractedToolCallInformation)
        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "analyze")
        self.assertIn('"threshold": 0.5', result.tool_calls[0].function.arguments)

    def test_extract_tool_calls_missing_fields(self):
        """测试缺少字段的工具调用提取"""
        model_output = """\n</think>\n\n<tool_call>\n{"arguments": {"location": "Tokyo"}}\n</tool_call>\n"""
        result = self.parser.extract_tool_calls(model_output, MagicMock())
        self.assertIsInstance(result, ExtractedToolCallInformation)
        self.assertFalse(result.tools_called)
        self.assertEqual(result.content, model_output)

    def test_extract_tool_calls_unmatched_tags(self):
        """测试不匹配标签的工具调用提取"""
        model_output = """\n</think>\n\n<tool_call>\n{"name": "test", "arguments": {}}\n</wrong_tag>\n"""

        result = self.parser.extract_tool_calls(model_output, MagicMock())
        self.assertIsInstance(result, ExtractedToolCallInformation)
        self.assertFalse(result.tools_called)
        self.assertEqual(result.content, model_output)

    def test_extract_tool_calls_empty_content(self):
        """测试空内容的工具调用提取"""
        model_output = """\n</think>\n\n<tool_call>\n\n</tool_call>\n"""
        result = self.parser.extract_tool_calls(model_output, MagicMock())
        self.assertIsInstance(result, ExtractedToolCallInformation)
        self.assertFalse(result.tools_called)
        self.assertEqual(result.content, model_output)

    def test_extract_tool_calls_partial_arguments(self):
        """测试参数不完整的工具调用"""
        model_output = """<tool_call>\n{"name": "get_weather", "arguments": {"loc"""
        result = self.parser.extract_tool_calls(model_output, MagicMock())
        self.assertIsInstance(result, ExtractedToolCallInformation)
        self.assertFalse(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")

    def test_extract_tool_calls_partial_name(self):
        """测试名称不完整的工具调用"""
        model_output = """<tool_call>\n{"nam"""
        result = self.parser.extract_tool_calls(model_output, MagicMock())
        self.assertIsInstance(result, ExtractedToolCallInformation)
        self.assertFalse(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "")

    def test_extract_tool_calls_incomplete_json(self):
        """测试JSON结构不完整的工具调用"""
        model_output = """<tool_call>\n{"""
        result = self.parser.extract_tool_calls(model_output, MagicMock())
        self.assertIsInstance(result, ExtractedToolCallInformation)
        self.assertFalse(result.tools_called)

    def test_extract_tool_calls_mixed_complete(self):
        """测试混合完整和不完整工具调用"""
        model_output = """<tool_call>\n{"name": "complete_tool", "arguments": {"key": "value"}}\n</tool_call>\n<tool_call>\n{"name": "partial_tool", "arguments": {"key": "val"""
        result = self.parser.extract_tool_calls(model_output, MagicMock())
        self.assertIsInstance(result, ExtractedToolCallInformation)
        self.assertFalse(result.tools_called)  # 只要有一个不完整就返回False
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0].function.name, "complete_tool")
        self.assertEqual(result.tool_calls[1].function.name, "partial_tool")


if __name__ == "__main__":
    unittest.main()
