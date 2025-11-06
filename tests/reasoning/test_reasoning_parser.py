"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
from fastdeploy.reasoning import ReasoningParser, ReasoningParserManager
from fastdeploy.reasoning.ernie_45_vl_thinking_reasoning_parser import (
    Ernie45VLThinkingReasoningParser,
)
from fastdeploy.reasoning.ernie_vl_reasoning_parsers import ErnieVLReasoningParser
from fastdeploy.reasoning.ernie_x1_reasoning_parsers import ErnieX1ReasoningParser


class DummyTokenizer:
    """Minimal tokenizer with vocab for testing."""

    def __init__(self):
        self.vocab = {
            "</think>": 100,
            "<tool_call>": 101,
            "</tool_call>": 102,
            "<response>": 103,
            "</response>": 104,
        }

    def get_vocab(self):
        """Return vocab dict for testing."""
        return self.vocab


class TestReasoningParser(ReasoningParser):
    def is_reasoning_end(self, input_ids):
        """
        Return True to simulate end of reasoning content.
        """
        return True

    def extract_content_ids(self, input_ids):
        """
        Return input_ids directly for testing.
        """
        return input_ids

    def extract_reasoning_content(self, model_output, request):
        """
        Used for testing non-streaming extraction.
        """
        return model_output, model_output

    def extract_reasoning_content_streaming(
        self, previous_text, current_text, delta_text, previous_token_ids, current_token_ids, delta_token_ids
    ):
        """
        Return None for streaming extraction; minimal implementation for testing.
        """
        return None


class TestReasoningParserManager(unittest.TestCase):
    """
    Unit tests for ReasoningParserManager functionality.
    """

    def setUp(self):
        """
        Save original registry to restore after each test.
        """
        self.original_parsers = ReasoningParserManager.reasoning_parsers.copy()

    def tearDown(self):
        """
        Restore original registry to avoid test pollution.
        """
        ReasoningParserManager.reasoning_parsers = self.original_parsers.copy()

    def test_register_and_get_parser(self):
        """
        Test that a parser can be registered and retrieved successfully.
        Verifies normal registration and retrieval functionality.
        """
        ReasoningParserManager.register_module(module=TestReasoningParser, name="test-parser", force=True)
        parser_cls = ReasoningParserManager.get_reasoning_parser("test_parser")
        self.assertIs(parser_cls, TestReasoningParser)

    def test_register_duplicate_without_force_raises(self):
        """
        Test that registering a parser with an existing name without force raises KeyError.
        Ensures duplicate registrations are handled correctly.
        """
        ReasoningParserManager.register_module(module=TestReasoningParser, name="test_parser2", force=True)
        with self.assertRaises(KeyError):
            ReasoningParserManager.register_module(module=TestReasoningParser, name="test_parser2", force=False)

    def test_register_non_subclass_raises(self):
        """
        Test that registering a class not inheriting from ReasoningParser raises TypeError.
        Ensures type safety for registered modules.
        """

        class NotParser:
            pass

        with self.assertRaises(TypeError):
            ReasoningParserManager.register_module(module=NotParser, name="not_parser")

    def test_get_unregistered_parser_raises(self):
        """
        Test that retrieving a parser that was not registered raises KeyError.
        Ensures get_reasoning_parser handles unknown names correctly.
        """
        with self.assertRaises(KeyError):
            ReasoningParserManager.get_reasoning_parser("nonexistent_parser")


class TestErnieX1ReasoningParser(unittest.TestCase):
    def setUp(self):
        self.parser = ErnieX1ReasoningParser(DummyTokenizer())
        self.request = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "test message"}])
        self.tokenizer = DummyTokenizer()

    # ---- Streaming parsing ----
    def test_streaming_thinking_content(self):
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="a",
            delta_text="a",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[200],
        )
        self.assertEqual(msg.reasoning_content, "a")

    def test_streaming_thinking_newline_preserved(self):
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="abc",
            current_text="abc\n",
            delta_text="\n",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[201],
        )
        self.assertEqual(msg.reasoning_content, "\n")

    def test_streaming_thinking_end_tag(self):
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="abc",
            current_text="abc</think>",
            delta_text="</think>",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[self.parser.think_end_token_id],
        )
        self.assertIsNone(msg)

    def test_streaming_response_content(self):
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="</think><response>",
            current_text="</think><response>h",
            delta_text="h",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[202],
        )
        self.assertEqual(msg.content, "h")

    def test_streaming_response_newline_preserved(self):
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="</think><response>hi",
            current_text="</think><response>hi\n",
            delta_text="\n",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[203],
        )
        self.assertEqual(msg.content, "\n")

    def test_streaming_response_ignore_tags(self):
        self.assertIsNone(
            self.parser.extract_reasoning_content_streaming(
                previous_text="</think>",
                current_text="</think><response>",
                delta_text="<response>",
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[self.parser.vocab["<response>"]],
            )
        )

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="</think><response>",
            current_text="</think><response>\n",
            delta_text="\n",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[204],
        )
        self.assertIsInstance(msg, DeltaMessage)
        self.assertEqual(msg.content, "\n")

        self.assertIsNone(
            self.parser.extract_reasoning_content_streaming(
                previous_text="</think><response>\n",
                current_text="</think><response>\n</response>",
                delta_text="</response>",
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[self.parser.vocab["</response>"]],
            )
        )

    def test_streaming_tool_call(self):
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="</think>",
            current_text="</think><tool_call>",
            delta_text="<tool_call>",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[self.parser.vocab["<tool_call>"]],
        )
        self.assertIsNone(msg)

    # ---- Batch parsing ----
    def test_batch_reasoning_and_response(self):
        text = "abc\n</think>\n<response>hello\nworld</response>"
        reasoning, response = self.parser.extract_reasoning_content(text, self.request)
        self.assertEqual(reasoning, "abc\n")
        self.assertEqual(response, "hello\nworld")

    def test_batch_reasoning_and_tool_call(self):
        text = "abc</think><tool_call>call_here"
        reasoning, response = self.parser.extract_reasoning_content(text, self.request)
        self.assertEqual(reasoning, "abc")
        self.assertEqual(response, "")

    def test_batch_no_thinking_tag(self):
        text = "no_thinking_here"
        reasoning, response = self.parser.extract_reasoning_content(text, self.request)
        self.assertEqual(reasoning, "no_thinking_here")
        self.assertEqual(response, "")

    def test_batch_response_without_end_tag(self):
        text = "abc</think><response>partial response"
        reasoning, response = self.parser.extract_reasoning_content(text, self.request)
        self.assertEqual(reasoning, "abc")
        self.assertEqual(response, "partial response")

    def test_batch_preserve_all_newlines(self):
        text = "abc\n</think>\n<response>line1\nline2\n</response>"
        reasoning, response = self.parser.extract_reasoning_content(text, self.request)
        self.assertEqual(reasoning, "abc\n")
        self.assertEqual(response, "line1\nline2\n")


class TestErnie45VLThinkingReasoningParser(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()
        self.parser = Ernie45VLThinkingReasoningParser(tokenizer=self.tokenizer)
        self.test_request = ChatCompletionRequest(
            model="ernie-test", messages=[{"role": "user", "content": "test prompt"}]
        )

    def test_streaming_non_reasoning(self):
        result = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="a",
            delta_text="a",
            previous_token_ids=[],
            current_token_ids=[200],
            delta_token_ids=[200],
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.reasoning_content, "a")
        self.assertIsNone(result.content)

    def test_streaming_with_reasoning(self):
        result = self.parser.extract_reasoning_content_streaming(
            previous_text="ab",
            current_text="ab</think>",
            delta_text="</think>",
            previous_token_ids=[200, 201],
            current_token_ids=[200, 201, 100],
            delta_token_ids=[100],
        )
        self.assertIsNone(result)

    def test_streaming_with_reasoning_and_content(self):
        result = self.parser.extract_reasoning_content_streaming(
            previous_text="ab",
            current_text="ab</think>\n\ncd",
            delta_text="</think>\n\ncd",
            previous_token_ids=[200, 201],
            current_token_ids=[200, 201, 100, 300, 400],
            delta_token_ids=[100, 300, 400],
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertIsNone(result.reasoning_content)
        self.assertEqual(result.content, "\n\ncd")

    def test_streaming_with_reasoning_new_line(self):
        result = self.parser.extract_reasoning_content_streaming(
            previous_text="abc",
            current_text="abc</think>\n\n",
            delta_text="</think>\n\n",
            previous_token_ids=[200, 201, 202],
            current_token_ids=[200, 201, 202, 100],
            delta_token_ids=[100],
        )
        self.assertIsNone(result)

    def test_streaming_with_reasoning_and_tool(self):
        result = self.parser.extract_reasoning_content_streaming(
            previous_text="abc",
            current_text="abc</think>\n\n<tool_call>",
            delta_text="</think>\n\n<tool_call>",
            previous_token_ids=[200, 201, 202],
            current_token_ids=[200, 201, 202, 100, 200, 101],
            delta_token_ids=[100, 200, 101],
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.reasoning_content, "")

    def test_streaming_with_reasoning_and_illegal_tool(self):
        result = self.parser.extract_reasoning_content_streaming(
            previous_text="abc</think>",
            current_text="abc</think>\n\nhello<tool_call>",
            delta_text="\n\nhello<tool_call>",
            previous_token_ids=[200, 201, 202],
            current_token_ids=[200, 201, 202, 100, 200, 101],
            delta_token_ids=[109, 200, 101],
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.content, "\n\nhello<tool_call>")

    def test_streaming_with_reasoning_no_tool(self):
        result = self.parser.extract_reasoning_content_streaming(
            previous_text="abc",
            current_text="abchello</think>\nworld",
            delta_text="hello</think>\nworld",
            previous_token_ids=[200, 201, 202],
            current_token_ids=[200, 201, 202, 100, 200, 110],
            delta_token_ids=[100, 200, 110],
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.reasoning_content, "hello")
        self.assertEqual(result.content, "\nworld")

    def test_streaming_reasoning_previous_no_tool(self):
        result = self.parser.extract_reasoning_content_streaming(
            previous_text="</think>",
            current_text="</think>\nhello",
            delta_text="\nhello",
            previous_token_ids=[100],
            current_token_ids=[100, 110, 111],
            delta_token_ids=[110, 111],
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertIsNone(result.reasoning_content)
        self.assertEqual(result.content, "\nhello")

    def test_streaming_no_reasoning_previous_tool(self):
        result = self.parser.extract_reasoning_content_streaming(
            previous_text="<tool_call>",
            current_text="<tool_call>hello",
            delta_text="hello",
            previous_token_ids=[101],
            current_token_ids=[101, 110],
            delta_token_ids=[110],
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.reasoning_content, "hello")

    def test_batch_no_think_end(self):
        reasoning, content = self.parser.extract_reasoning_content(
            model_output="direct response", request=self.test_request
        )
        self.assertEqual(reasoning, "direct response")
        self.assertEqual(content, "")

    def test_batch_no_think_end_with_tool(self):
        reasoning, content = self.parser.extract_reasoning_content(
            model_output="direct response<tool_call>abc", request=self.test_request
        )
        self.assertEqual(reasoning, "direct response<tool_call>abc")
        self.assertEqual(content, "")

    def test_batch_think_end_normal_content(self):
        reasoning, content = self.parser.extract_reasoning_content(
            model_output="reasoning</think>\nresponse", request=self.test_request
        )
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(content, "\nresponse")

    def test_batch_think_end_with_tool(self):
        reasoning, content = self.parser.extract_reasoning_content(
            model_output="reasoning</think>\n<tool_call>tool params</tool_call>", request=self.test_request
        )
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(content, "")

    def test_batch_think_end_with_illegal_tool(self):
        reasoning, content = self.parser.extract_reasoning_content(
            model_output="reasoning</think>\nABC\n<tool_call>tool params</tool_call>", request=self.test_request
        )
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(content, "\nABC\n<tool_call>tool params</tool_call>")

    def test_batch_think_end_content_with_newline(self):
        reasoning, content = self.parser.extract_reasoning_content(
            model_output="reasoning</think>\n\n  actual response", request=self.test_request
        )
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(content, "\n\n  actual response")


class TestErnieVLReasoningParser(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()
        self.parser = ErnieVLReasoningParser(tokenizer=self.tokenizer)
        self.test_request = ChatCompletionRequest(
            model="ernie-test", messages=[{"role": "user", "content": "test prompt"}]
        )

    def test_extract_reasoning_content_stream(self):
        result = self.parser.extract_reasoning_content_streaming(
            previous_text="abc",
            current_text="abc</think>xyz",
            delta_text="</think>xyz",
            previous_token_ids=[200, 201, 202],
            current_token_ids=[200, 201, 202, 100, 110, 120, 130],
            delta_token_ids=[100, 110, 120, 130],
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.reasoning_content, "")
        self.assertEqual(result.content, "xyz")

    def test_extract_reasoning_content_stream_think_in_previous(self):
        result = self.parser.extract_reasoning_content_streaming(
            previous_text="abc</think>",
            current_text="abc</think>xyz",
            delta_text="xyz",
            previous_token_ids=[200, 201, 202, 100],
            current_token_ids=[200, 201, 202, 100, 110, 120, 130],
            delta_token_ids=[110, 120, 130],
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertIsNone(result.reasoning_content)
        self.assertEqual(result.content, "xyz")

    def test_extract_reasoning_content_stream_no_think_token(self):
        result = self.parser.extract_reasoning_content_streaming(
            previous_text="abc",
            current_text="abcxyz",
            delta_text="xyz",
            previous_token_ids=[200, 201, 202],
            current_token_ids=[200, 201, 202, 110, 120, 130],
            delta_token_ids=[110, 120, 130],
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertIsNone(result.content)
        self.assertEqual(result.reasoning_content, "xyz")

    def test_extract_reasoning_content(self):
        reasoning, content = self.parser.extract_reasoning_content(
            model_output="reasoning</think>\nactual response", request=self.test_request
        )
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(content, "\nactual response")


if __name__ == "__main__":
    unittest.main()
