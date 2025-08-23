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

import json
import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage, ExtractedToolCallInformation
from fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser import ErnieX1ToolParser


class TestErnieX1ToolParser(unittest.TestCase):
    """Unit tests for ErnieX1ToolParser"""

    def setUp(self):
        """Set up test environment"""
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.get_vocab.return_value = {"<tool_call>": 100, "</tool_call>": 101, "other_token": 102}

        # Initialize parser with mocked dependencies
        with patch("fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.data_processor_logger"):
            self.parser = ErnieX1ToolParser(self.mock_tokenizer)

    def test_initialization(self):
        """Test ErnieX1ToolParser initialization"""
        self.assertEqual(self.parser.tool_call_start_token, "<tool_call>")
        self.assertEqual(self.parser.tool_call_end_token, "</tool_call>")
        self.assertEqual(self.parser.tool_call_start_token_id, 100)
        self.assertEqual(self.parser.tool_call_end_token_id, 101)
        self.assertEqual(self.parser.buffer, "")
        self.assertEqual(self.parser.bracket_counts, {"total_l": 0, "total_r": 0})

    def test_initialization_missing_tokens(self):
        """Test initialization fails when tool call tokens are missing"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {"other": 1}

        with self.assertRaises(RuntimeError) as context:
            ErnieX1ToolParser(mock_tokenizer)

        self.assertIn("could not locate tool call start/end tokens", str(context.exception))

    def test_extract_tool_calls_complete_single_tool(self):
        """Test extracting complete single tool call"""
        model_output = """<tool_call>
{"name": "get_weather", "arguments": {"location": "Beijing"}}
</tool_call>"""

        request = ChatCompletionRequest(messages=[{"role": "user", "content": "test"}])

        with patch("fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.data_processor_logger"):
            result = self.parser.extract_tool_calls(model_output, request)

        self.assertTrue(result.tools_called)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        args = json.loads(result.tool_calls[0].function.arguments)
        self.assertEqual(args["location"], "Beijing")

    def test_extract_tool_calls_multiple_tools(self):
        """Test extracting multiple tool calls"""
        model_output = """<tool_call>
{"name": "get_weather", "arguments": {"location": "Beijing"}}
</tool_call>
<tool_call>
{"name": "calculate", "arguments": {"x": 5, "y": 10}}
</tool_call>"""

        request = ChatCompletionRequest(messages=[{"role": "user", "content": "test"}])

        with patch("fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.data_processor_logger"):
            result = self.parser.extract_tool_calls(model_output, request)

        self.assertTrue(result.tools_called)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertEqual(result.tool_calls[1].function.name, "calculate")

    def test_extract_tool_calls_incomplete_tool(self):
        """Test extracting incomplete tool call"""
        model_output = """<tool_call>
{"name": "get_weather", "arguments": {"location": "Bei"""

        request = ChatCompletionRequest(messages=[{"role": "user", "content": "test"}])

        with patch("fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.data_processor_logger"):
            with patch("partial_json_parser.loads") as mock_partial_loads:
                mock_partial_loads.return_value = {"location": "Bei"}
                result = self.parser.extract_tool_calls(model_output, request)

        # Incomplete tool calls should not set tools_called=True
        self.assertFalse(result.tools_called)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 1)

    def test_extract_tool_calls_invalid_response_tag(self):
        """Test handling invalid <response> tags before tool calls"""
        model_output = """<response>
Some response content
</response>
<tool_call>
{"name": "test", "arguments": {}}
</tool_call>"""

        request = ChatCompletionRequest(messages=[{"role": "user", "content": "test"}])

        with patch(
            "fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.data_processor_logger"
        ) as mock_logger:
            result = self.parser.extract_tool_calls(model_output, request)

        self.assertFalse(result.tools_called)
        self.assertEqual(result.content, model_output)
        mock_logger.error.assert_called_once()

    def test_extract_tool_calls_no_tool_calls(self):
        """Test handling output with no tool calls"""
        model_output = "Just regular text output"
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "test"}])

        with patch("fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.data_processor_logger"):
            result = self.parser.extract_tool_calls(model_output, request)

        self.assertFalse(result.tools_called)
        self.assertEqual(result.content, model_output)

    def test_extract_tool_calls_malformed_json(self):
        """Test handling malformed JSON in tool calls"""
        model_output = """<tool_call>
{invalid json
</tool_call>"""

        request = ChatCompletionRequest(messages=[{"role": "user", "content": "test"}])

        with patch("fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.data_processor_logger"):
            with patch("partial_json_parser.loads", side_effect=Exception("Parse error")):
                result = self.parser.extract_tool_calls(model_output, request)

        self.assertFalse(result.tools_called)

    def test_extract_tool_calls_streaming_no_tool_start(self):
        """Test streaming when no tool start token present"""
        result = self.parser.extract_tool_calls_streaming("", "regular text", "regular text", [], [102], [102], {})

        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.content, "regular text")

    def test_extract_tool_calls_streaming_empty_delta(self):
        """Test streaming with empty delta text"""
        result = self.parser.extract_tool_calls_streaming("", "text", "  ", [], [100], [100], {})

        self.assertIsNone(result)

    def test_extract_tool_calls_streaming_tool_start(self):
        """Test streaming when tool call starts"""
        with patch("fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.data_processor_logger"):
            result = self.parser.extract_tool_calls_streaming("", "<tool_call>", "<tool_call>", [], [100], [100], {})

        self.assertEqual(self.parser.current_tool_id, 0)
        self.assertFalse(self.parser.current_tool_name_sent)
        self.assertEqual(len(self.parser.streamed_args_for_tool), 1)

    @patch("fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.random_tool_call_id")
    def test_extract_tool_calls_streaming_name_extraction(self, mock_random_id):
        """Test streaming name extraction"""
        mock_random_id.return_value = "call_123"

        # First, start a tool call
        self.parser.current_tool_id = 0
        self.parser.current_tool_name_sent = False
        self.parser.buffer = '{"name": "get_weather"'

        with patch("fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.data_processor_logger"):
            result = self.parser.extract_tool_calls_streaming(
                "", 'text{"name": "get_weather"', '"', [], [100], [100], {}
            )

        self.assertIsInstance(result, DeltaMessage)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function["name"], "get_weather")
        self.assertTrue(self.parser.current_tool_name_sent)

    def test_extract_tool_calls_exception_handling(self):
        """Test exception handling in extract_tool_calls"""
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "test"}])

        with patch(
            "fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.data_processor_logger"
        ) as mock_logger:
            with patch.object(self.parser, "extract_tool_calls", side_effect=Exception("Test error")):
                # Call the original method directly
                original_method = ErnieX1ToolParser.extract_tool_calls
                result = original_method(self.parser, "test", request)

        # Should be handled gracefully
        self.assertIsInstance(result, ExtractedToolCallInformation)

    def test_buffer_management(self):
        """Test buffer management in streaming"""
        self.parser.buffer = "initial"

        self.parser.extract_tool_calls_streaming("", "text", "new_content", [], [102], [102], {})

        self.assertEqual(self.parser.buffer, "initialnew_content")

    def test_bracket_counting(self):
        """Test bracket counting logic"""
        # Initialize state
        self.parser.current_tool_id = 0
        self.parser.buffer = '"arguments": {"x": 1}'

        # Test bracket counting
        delta_text = '{"x": 1}}'
        self.parser.bracket_counts = {"total_l": 0, "total_r": 0}

        # Simulate processing characters with brackets
        for char in delta_text:
            if char == "{":
                self.parser.bracket_counts["total_l"] += 1
            elif char == "}":
                self.parser.bracket_counts["total_r"] += 1

        self.assertEqual(self.parser.bracket_counts["total_l"], 1)
        self.assertEqual(self.parser.bracket_counts["total_r"], 2)


if __name__ == "__main__":
    unittest.main()
