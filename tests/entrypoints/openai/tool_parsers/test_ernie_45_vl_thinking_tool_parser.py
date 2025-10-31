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

import unittest
from unittest.mock import patch

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from fastdeploy.entrypoints.openai.tool_parsers.ernie_45_vl_thinking_tool_parser import (
    Ernie45VLThinkingToolParser,
)


class DummyTokenizer:
    """Dummy tokenizer with minimal vocab for testing"""

    def __init__(self):
        self.vocab = {"<tool_call>": 1, "</tool_call>": 2}


class TestErnie45VLThinkingToolParser(unittest.TestCase):
    def setUp(self):
        class DummyTokenizer:
            def __init__(self):
                self.vocab = {"<tool_call>": 1, "</tool_call>": 2}

            def get_vocab(self):
                return self.vocab

        self.tokenizer = DummyTokenizer()
        self.parser = Ernie45VLThinkingToolParser(tokenizer=self.tokenizer)
        self.dummy_request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])

    # ---------------- Batch extraction tests ----------------

    def test_extract_tool_calls_complete(self):
        """Test normal extraction of complete tool_call JSON"""
        output = '<tool_call>{"name": "get_weather", "arguments": {"location": "北京"}}</tool_call>'
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertTrue(result.tools_called)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")

    def test_extract_tool_calls_partial_arguments(self):
        """Test partial extraction when arguments incomplete"""
        output = '<tool_call>{"name": "get_weather", "arguments": {"location": "北"</tool_call>'
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertFalse(result.tools_called)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")

    def test_extract_tool_calls_no_toolcall(self):
        """Test when no tool_call tags are present"""
        output = "no tool call here"
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertFalse(result.tools_called)

    def test_extract_tool_calls_invalid_json(self):
        """Test tool_call with badly formatted JSON triggers fallback parser"""
        output = '<tool_call>"name": "get_weather", "arguments": {</tool_call>'
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertFalse(result.tools_called)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")

    def test_extract_tool_calls_exception(self):
        """Force exception to cover error branch"""
        with patch(
            "fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser.json.loads", side_effect=Exception("boom")
        ):
            output = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
            result = self.parser.extract_tool_calls(output, self.dummy_request)
            self.assertFalse(result.tools_called)

    def test_extract_tool_calls_illegal(self):
        output = '</think>abc<tool_call>{"name": "get_weather", "arguments": {"location": "北京"}}</tool_call>'
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertFalse(result.tools_called)
        self.assertEqual(
            result.content,
            '</think>abc<tool_call>{"name": "get_weather", "arguments": {"location": "北京"}}</tool_call>',
        )
        output = 'abc<tool_call>{"name": "get_weather", "arguments": {"location": "北京"}}</tool_call>'
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertFalse(result.tools_called)
        self.assertEqual(
            result.content, 'abc<tool_call>{"name": "get_weather", "arguments": {"location": "北京"}}</tool_call>'
        )

    # ---------------- Streaming extraction tests ----------------

    def test_streaming_no_toolcall(self):
        """Streaming extraction returns normal DeltaMessage when no <tool_call>"""
        result = self.parser.extract_tool_calls_streaming(
            "", "abc", "abc", [], [], [], self.dummy_request.model_dump()
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertIsNone(result.tool_calls)
        self.assertEqual(result.content, "abc")

    def test_streaming_skip_empty_chunk(self):
        """Streaming extraction skips empty chunks"""
        result = self.parser.extract_tool_calls_streaming(
            "", "<tool_call>", "   ", [], [1], [1], self.dummy_request.model_dump()
        )
        self.assertIsNone(result)

    def test_streaming_new_toolcall_and_name(self):
        """Streaming extraction detects new toolcall and extracts name"""
        delta = self.parser.extract_tool_calls_streaming(
            "", "<tool_call>", '<tool_call>{"name": "get_weather"', [], [1], [1], self.dummy_request.model_dump()
        )
        self.assertIsNotNone(delta)
        self.assertEqual(delta.tool_calls[0].function.name, "get_weather")

    def test_streaming_partial_arguments(self):
        """Streaming extraction yields partial arguments deltas"""
        text = '"arguments": {"location":'
        delta = self.parser.extract_tool_calls_streaming(
            "", "<tool_call>" + text, text, [], [1], [1], self.dummy_request.model_dump()
        )
        self.assertIsInstance(delta, DeltaMessage)
        self.assertIn("arguments", delta.tool_calls[0].function.arguments)

    def test_streaming_complete_arguments_and_end(self):
        """Streaming extraction completes arguments with brackets matched and closes tool_call"""
        text = '"arguments": {"location": "北京"}}'
        delta = self.parser.extract_tool_calls_streaming(
            "", "<tool_call>" + text, "<tool_call>" + text, [], [1], [1], self.dummy_request.model_dump()
        )
        self.assertIsInstance(delta, DeltaMessage)
        # Also simulate closing tag
        end_delta = self.parser.extract_tool_calls_streaming(
            "<tool_call>" + text,
            "<tool_call>" + text + "</tool_call>",
            "</tool_call>",
            [1],
            [1, 2],
            [2],
            self.dummy_request.model_dump(),
        )
        self.assertIsNone(end_delta)

    def test_streaming_no_tool_illegal(self):
        result = self.parser.extract_tool_calls_streaming(
            "", "abc<tool_call>", "abc<tool_call>", [], [], [], self.dummy_request.model_dump()
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertIsNone(result.tool_calls)
        self.assertEqual(result.content, "abc<tool_call>")
        result = self.parser.extract_tool_calls_streaming(
            "", "</think>abc<tool_call>", "</think>abc<tool_call>", [], [], [], self.dummy_request.model_dump()
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertIsNone(result.tool_calls)
        self.assertEqual(result.content, "</think>abc<tool_call>")

    def test_streaming_tool_with_reasoning(self):
        delta = self.parser.extract_tool_calls_streaming(
            "",
            '</think><tool_call>{"name": "get_weather"',
            '</think><tool_call>{"name": "get_weather"',
            [],
            [1],
            [1],
            self.dummy_request.model_dump(),
        )
        self.assertIsNotNone(delta)
        self.assertEqual(delta.tool_calls[0].function.name, "get_weather")
        delta = self.parser.extract_tool_calls_streaming(
            "",
            '</think>\n\n<tool_call>{"name": "get_weather"',
            '</think>\n\n<tool_call>{"name": "get_weather"',
            [],
            [1],
            [1],
            self.dummy_request.model_dump(),
        )
        self.assertIsNotNone(delta)
        self.assertEqual(delta.tool_calls[0].function.name, "get_weather")


if __name__ == "__main__":
    unittest.main()
