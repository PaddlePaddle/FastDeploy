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

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from fastdeploy.entrypoints.openai.tool_parsers.ernie_x1_tool_parser import (
    ErnieX1ToolParser,
)


class DummyTokenizer:
    """Dummy tokenizer with vocab containing tool_call tokens"""

    def __init__(self):
        self.vocab = {"<tool_call>": 1, "</tool_call>": 2}

    def get_vocab(self):
        return self.vocab


class TestErnieX1ToolParser(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()
        self.parser = ErnieX1ToolParser(tokenizer=self.tokenizer)
        self.dummy_request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])

    # ---------------- Batch extraction tests ----------------

    def test_extract_tool_calls_complete(self):
        """Test normal extraction of complete tool_call JSON"""
        output = '<tool_call>{"name": "get_weather", "arguments": {"location": "Beijing"}}</tool_call>'
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertTrue(result.tools_called)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")

    def test_extract_tool_calls_no_toolcall(self):
        """Test when no tool_call tags are present"""
        output = "no tool call here"
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertFalse(result.tools_called)

    def test_extract_tool_calls_exception(self):
        """Completely broken JSON triggers the exception branch"""
        output = "<tool_call>not json at all{{{</tool_call>"
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        self.assertFalse(result.tools_called)

    def test_extract_tool_calls_partial_json_parser_failure(self):
        """Test partial_json_parser failure path for arguments (L165-166).
        json.loads fails on malformed JSON, partial_json_parser.loads also fails on deeply broken args.
        Partial result has _is_partial=True so tools_called=False, but tool_calls is populated."""
        # Malformed JSON: valid name but arguments is a bare invalid token
        # that breaks both json.loads and partial_json_parser
        output = '<tool_call>{"name": "test", "arguments": @@@INVALID@@@}</tool_call>'
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        # _is_partial=True → tools_called=False, but tool_calls list is populated
        self.assertFalse(result.tools_called)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(result.tool_calls[0].function.name, "test")
        # arguments=None → converted to {} → serialized as "{}"
        self.assertEqual(result.tool_calls[0].function.arguments, "{}")

    def test_partial_json_parser_exception_triggers_debug_log(self):
        """Malformed JSON + partial_json_parser failure exercises L165-166 exactly."""
        # Unclosed string in arguments breaks both json.loads and partial_json_parser
        output = '<tool_call>{"name": "my_tool", "arguments": {"key": "unterminated}</tool_call>'
        result = self.parser.extract_tool_calls(output, self.dummy_request)
        # Partial parse → tools_called=False but tool_calls has entries
        self.assertFalse(result.tools_called)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(result.tool_calls[0].function.name, "my_tool")

    # ---------------- Streaming extraction tests ----------------

    def test_streaming_no_toolcall(self):
        """Streaming extraction returns normal DeltaMessage when no toolcall tag"""
        result = self.parser.extract_tool_calls_streaming(
            "", "abc", "abc", [], [], [], self.dummy_request.model_dump()
        )
        self.assertIsInstance(result, DeltaMessage)
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
        text = '"arguments": {"location": "Beijing"}}'
        delta = self.parser.extract_tool_calls_streaming(
            "", "<tool_call>" + text, text, [], [1], [1], self.dummy_request.model_dump()
        )
        self.assertIsInstance(delta, DeltaMessage)
        # Also simulate closing tag
        end_delta = self.parser.extract_tool_calls_streaming(
            "", "</tool_call>", "</tool_call>", [], [2], [2], self.dummy_request.model_dump()
        )
        self.assertIsNotNone(end_delta)
        self.assertEqual(end_delta.content, "</tool_call>")


if __name__ == "__main__":
    unittest.main()
