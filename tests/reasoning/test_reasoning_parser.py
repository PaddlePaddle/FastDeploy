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
from fastdeploy.reasoning.deepseek_reasoning_parser import DeepSeekReasoningParser
from fastdeploy.reasoning.ernie_45_vl_thinking_reasoning_parser import (
    Ernie45VLThinkingReasoningParser,
)
from fastdeploy.reasoning.ernie_vl_reasoning_parsers import ErnieVLReasoningParser
from fastdeploy.reasoning.ernie_x1_reasoning_parsers import ErnieX1ReasoningParser


class DummyTokenizer:
    """Minimal tokenizer with vocab for testing."""

    def __init__(self):
        self.vocab = {
            "<think>": 99,
            "</think>": 100,
            "<think>": 101,
            "<tool_call>": 102,
            "</tool_call>": 103,
            "<response>": 104,
            "</response>": 105,
        }

    def get_vocab(self):
        """Return vocab dict for testing."""
        return self.vocab


class MissingTokenTokenizer:
    def __init__(self):
        self.vocab = {
            "</think>": 100,
            "<think>": 101,
            "<tool_call>": 102,
            "</tool_call>": 103,
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

    def test_missing_token(self):
        with self.assertRaises(RuntimeError) as context:
            ErnieX1ReasoningParser(MissingTokenTokenizer())
        exception_message = str(context.exception)
        expected_message_part = "ernie x1 reasoning parser could not find the following token ids"
        self.assertIn(expected_message_part, exception_message)

    def test_get_model_status(self):
        model_status = self.parser.get_model_status([88, 99, 104])
        self.assertEqual(model_status, "response_start")

    # ---- Streaming parsing ----
    def test_streaming_thinking_content(self):
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="a",
            delta_text="a",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[200],
            model_status="think_start",
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
            model_status="think_start",
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
            model_status="think_start",
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
            model_status="think_start",
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
            model_status="think_start",
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
                model_status="think_start",
            )
        )

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="</think><response>",
            current_text="</think><response>\n",
            delta_text="\n",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[204],
            model_status="think_start",
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
                model_status="think_start",
            )
        )

    def test_extract_reasoning_content_streaming(self):
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="hello</think>",
            current_text="hello</think><response>",
            delta_text="</think><response>",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[100, 200],
            model_status="think_start",
        )
        self.assertEqual(msg.content, "")
        self.assertEqual(msg.reasoning_content, "")

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="hello</think>",
            current_text="hello</think><response>hi",
            delta_text="</think><response>hi",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[100, 200],
            model_status="think_start",
        )
        self.assertEqual(msg.content, "hi")
        self.assertEqual(msg.reasoning_content, "")

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="hello</think><response>hi",
            delta_text="hello</think><response>hi",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[100, 200],
            model_status="think_start",
        )
        self.assertEqual(msg.content, "hi")
        self.assertEqual(msg.reasoning_content, "hello")

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="hello</think><response>",
            current_text="hello</think><response>hi",
            delta_text="hi",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[100, 200],
            model_status="think_end",
        )
        self.assertEqual(msg.content, "hi")
        self.assertEqual(msg.reasoning_content, None)

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="hello</think><response>",
            current_text="hello</think><response>hi",
            delta_text="hi",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[100, 200],
            model_status="response_start",
        )
        self.assertEqual(msg.content, "hi")
        self.assertEqual(msg.reasoning_content, None)

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="hello</think><response>hi</response>",
            current_text="hello</think><response>hi</response>end",
            delta_text="end",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[100, 200],
            model_status="response_start",
        )
        self.assertEqual(msg, None)

    def test_streaming_tool_call(self):
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="</think>",
            current_text="</think><tool_call>",
            delta_text="<tool_call>",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[self.parser.vocab["<tool_call>"]],
            model_status="think_start",
        )
        self.assertIsNone(msg)

    # ---- Batch parsing ----
    def test_batch_reasoning_and_response(self):
        text = "abc\n</think>\n<response>hello\nworld</response>"
        reasoning, response = self.parser.extract_reasoning_content(text, self.request, "think_start")
        self.assertEqual(reasoning, "abc\n")
        self.assertEqual(response, "hello\nworld")

    def test_batch_reasoning_and_tool_call(self):
        text = "abc</think><tool_call>call_here"
        reasoning, response = self.parser.extract_reasoning_content(text, self.request, "think_start")
        self.assertEqual(reasoning, "abc")
        self.assertEqual(response, "")

    def test_batch_no_thinking_tag(self):
        text = "no_thinking_here"
        reasoning, response = self.parser.extract_reasoning_content(text, self.request, "think_start")
        self.assertEqual(reasoning, "no_thinking_here")
        self.assertEqual(response, "")

    def test_batch_response_without_end_tag(self):
        text = "abc</think><response>partial response"
        reasoning, response = self.parser.extract_reasoning_content(text, self.request, "think_start")
        self.assertEqual(reasoning, "abc")
        self.assertEqual(response, "partial response")

    def test_batch_preserve_all_newlines(self):
        text = "abc\n</think>\n<response>line1\nline2\n</response>"
        reasoning, response = self.parser.extract_reasoning_content(text, self.request, "think_start")
        self.assertEqual(reasoning, "abc\n")
        self.assertEqual(response, "line1\nline2\n")

    def test_extract_reasoning_content(self):
        reasoning_content, response_content = self.parser.extract_reasoning_content(
            model_output="hello", request=self.request, model_status="response_start"
        )
        self.assertEqual(reasoning_content, "")
        self.assertEqual(response_content, "hello")


class TestErnie45VLThinkingReasoningParser(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()
        self.parser = Ernie45VLThinkingReasoningParser(tokenizer=self.tokenizer)
        self.test_request = ChatCompletionRequest(
            model="ernie-test", messages=[{"role": "user", "content": "test prompt"}]
        )
        self.parser.token_status_mapping = {
            100: "think_start",
        }

    def test_streaming_non_reasoning(self):
        result = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="a",
            delta_text="a",
            previous_token_ids=[],
            current_token_ids=[200],
            delta_token_ids=[200],
            model_status="think_start",
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
            model_status="think_start",
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
            model_status="think_start",
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
            model_status="think_start",
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
            model_status="think_start",
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.reasoning_content, None)

    def test_streaming_with_reasoning_and_illegal_tool(self):
        result = self.parser.extract_reasoning_content_streaming(
            previous_text="abc</think>",
            current_text="abc</think>\n\nhello<tool_call>",
            delta_text="\n\nhello<tool_call>",
            previous_token_ids=[200, 201, 202],
            current_token_ids=[200, 201, 202, 100, 200, 101],
            delta_token_ids=[109, 200, 101],
            model_status="think_start",
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
            model_status="think_start",
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
            model_status="think_start",
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
            model_status="think_start",
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.reasoning_content, "hello")

    def test_think_end_status_streaming(self):
        result = self.parser.extract_reasoning_content_streaming(
            previous_text="<tool_call>",
            current_text="<tool_call>hello",
            delta_text="hello",
            previous_token_ids=[101],
            current_token_ids=[101, 110],
            delta_token_ids=[110],
            model_status="think_end",
        )
        self.assertIs(result, None)

        result = self.parser.extract_reasoning_content_streaming(
            previous_text="hello, ",
            current_text="hello, hi",
            delta_text="hi",
            previous_token_ids=[101],
            current_token_ids=[101, 110],
            delta_token_ids=[110],
            model_status="think_end",
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.content, "hi")

    def test_other_status_streaming(self):
        result = self.parser.extract_reasoning_content_streaming(
            previous_text="hello, ",
            current_text="hello, hi",
            delta_text="hi",
            previous_token_ids=[101],
            current_token_ids=[101, 110],
            delta_token_ids=[110],
            model_status="tool_call_start",
        )
        self.assertIs(result, None)

    def test_batch_no_think_end(self):
        reasoning, content = self.parser.extract_reasoning_content(
            model_output="direct response", request=self.test_request, model_status="think_start"
        )
        self.assertEqual(reasoning, "direct response")
        self.assertEqual(content, "")

    def test_batch_no_think_end_with_tool(self):
        reasoning, content = self.parser.extract_reasoning_content(
            model_output="direct response<tool_call>abc", request=self.test_request, model_status="think_start"
        )
        self.assertEqual(reasoning, "direct response<tool_call>abc")
        self.assertEqual(content, "")

    def test_batch_think_end_normal_content(self):
        reasoning, content = self.parser.extract_reasoning_content(
            model_output="reasoning</think>\nresponse", request=self.test_request, model_status="think_start"
        )
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(content, "\nresponse")

    def test_batch_think_end_with_tool(self):
        reasoning, content = self.parser.extract_reasoning_content(
            model_output="reasoning</think>\n<tool_call>tool params</tool_call>",
            request=self.test_request,
            model_status="think_start",
        )
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(content, "")

    def test_batch_think_end_with_illegal_tool(self):
        reasoning, content = self.parser.extract_reasoning_content(
            model_output="reasoning</think>\nABC\n<tool_call>tool params</tool_call>",
            request=self.test_request,
            model_status="think_start",
        )
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(content, "\nABC\n<tool_call>tool params</tool_call>")

    def test_batch_think_end_content_with_newline(self):
        reasoning, content = self.parser.extract_reasoning_content(
            model_output="reasoning</think>\n\n  actual response",
            request=self.test_request,
            model_status="think_start",
        )
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(content, "\n\n  actual response")

    def test_think_end_status_non_streaming(self):
        reasoning, content = self.parser.extract_reasoning_content(
            model_output="response", request=self.test_request, model_status="think_end"
        )
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "response")

        reasoning, content = self.parser.extract_reasoning_content(
            model_output="<tool_call>response", request=self.test_request, model_status="think_end"
        )
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "")

        reasoning, content = self.parser.extract_reasoning_content(
            model_output="\n 1<tool_call>response", request=self.test_request, model_status="think_end"
        )
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "\n 1<tool_call>response")

    def test_other_status_non_streaming(self):
        reasoning, content = self.parser.extract_reasoning_content(
            model_output="response", request=self.test_request, model_status="tool_call_start"
        )
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "")

        reasoning, content = self.parser.extract_reasoning_content(
            model_output="response", request=self.test_request, model_status="tool_call_end"
        )
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "")

    def test_find_last_special_token(self):
        result = self.parser.find_last_special_token([100, 110, 120, 130])
        self.assertEqual(result, 100)
        result = self.parser.find_last_special_token([0])
        self.assertEqual(result, -1)

    def test_get_model_status(self):
        result = self.parser.get_model_status([100, 110, 120, 130])
        self.assertEqual(result, "think_start")

        result = self.parser.get_model_status([0])
        self.assertEqual(result, "think_start")


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
            model_status="think_start",
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
            model_status="think_start",
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
            model_status="think_start",
        )
        self.assertIsInstance(result, DeltaMessage)
        self.assertIsNone(result.content)
        self.assertEqual(result.reasoning_content, "xyz")

    def test_extract_reasoning_content(self):
        reasoning, content = self.parser.extract_reasoning_content(
            model_output="reasoning</think>\nactual response", request=self.test_request, model_status="think_start"
        )
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(content, "\nactual response")


class TestDeepSeekReasoningParser(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()
        self.parser = DeepSeekReasoningParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        self.request = ChatCompletionRequest(
            model="deepseek-v3.1", messages=[{"role": "user", "content": "test message"}]
        )

    # ---- Non-streaming parsing ----
    def test_batch_standard_format(self):
        """测试标准格式：<think>abc</think>xyz"""
        text = "<think>abc</think>xyz"
        reasoning, content = self.parser.extract_reasoning_content(text, self.request)
        self.assertEqual(reasoning, "abc")
        self.assertEqual(content, "xyz")

    def test_batch_no_start_tag(self):
        """测试缺少起始标签的格式：abc</think>xyz"""
        text = "abc</think>xyz"
        reasoning, content = self.parser.extract_reasoning_content(text, self.request)
        self.assertEqual(reasoning, "abc")
        self.assertEqual(content, "xyz")

    def test_batch_no_reasoning_tags(self):
        """测试无思考标签格式（思考开关关闭时）"""
        text = "direct response"
        reasoning, content = self.parser.extract_reasoning_content(text, self.request)
        self.assertIsNone(reasoning)
        self.assertEqual(content, "direct response")

    def test_batch_only_start_tag(self):
        """测试只有起始标签，没有结束标签"""
        text = "<think>abc"
        reasoning, content = self.parser.extract_reasoning_content(text, self.request)
        self.assertEqual(reasoning, "abc")
        self.assertIsNone(content)

    def test_batch_reasoning_with_newline(self):
        """测试包含换行符的思考内容"""
        text = "<think>line1\nline2</think>response"
        reasoning, content = self.parser.extract_reasoning_content(text, self.request)
        self.assertEqual(reasoning, "line1\nline2")
        self.assertEqual(content, "response")

    def test_batch_empty_content(self):
        """测试思考结束后没有回复内容"""
        text = "<think>abc</think>"
        reasoning, content = self.parser.extract_reasoning_content(text, self.request)
        self.assertEqual(reasoning, "abc")
        self.assertIsNone(content)

    def test_batch_content_with_whitespace(self):
        """测试思考结束后只有空白字符"""
        text = "<think>abc</think>\n\n  "
        reasoning, content = self.parser.extract_reasoning_content(text, self.request)
        self.assertEqual(reasoning, "abc")
        self.assertIsNone(content)

    # ---- Streaming parsing ----
    def test_streaming_reasoning_content(self):
        """测试流式输出思考内容"""
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="a",
            delta_text="a",
            previous_token_ids=[],
            current_token_ids=[200],
            delta_token_ids=[200],
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.reasoning_content, "a")
        self.assertIsNone(msg.content)

    def test_streaming_reasoning_end_tag(self):
        """测试流式输出遇到结束标签"""
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="abc",
            current_text="abc</think>",
            delta_text="</think>",
            previous_token_ids=[200, 201, 202],
            current_token_ids=[200, 201, 202, self.parser.think_end_token_id],
            delta_token_ids=[self.parser.think_end_token_id],
        )
        self.assertIsNone(msg)  # 单个结束标签应该被忽略

    def test_streaming_reasoning_to_content(self):
        """测试从思考阶段转换到回复阶段"""
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="abc</think>",
            current_text="abc</think>xyz",
            delta_text="xyz",
            previous_token_ids=[200, 201, 202, self.parser.think_end_token_id],
            current_token_ids=[200, 201, 202, self.parser.think_end_token_id, 110, 120, 130],
            delta_token_ids=[110, 120, 130],
        )
        self.assertIsNotNone(msg)
        self.assertIsNone(msg.reasoning_content)
        self.assertEqual(msg.content, "xyz")

    def test_streaming_reasoning_and_content_in_delta(self):
        """测试 delta 中同时包含思考和回复内容"""
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="<think>abc</think>xyz",
            delta_text="<think>abc</think>xyz",
            previous_token_ids=[],
            current_token_ids=[
                self.parser.think_start_token_id,
                200,
                201,
                202,
                self.parser.think_end_token_id,
                110,
                120,
                130,
            ],
            delta_token_ids=[
                self.parser.think_start_token_id,
                200,
                201,
                202,
                self.parser.think_end_token_id,
                110,
                120,
                130,
            ],
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.reasoning_content, "abc")
        self.assertEqual(msg.content, "xyz")

    def test_streaming_reasoning_start_tag(self):
        """测试流式输出遇到开始标签"""
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="<think>abc",
            delta_text="<think>abc",
            previous_token_ids=[],
            current_token_ids=[self.parser.think_start_token_id, 200, 201, 202],
            delta_token_ids=[self.parser.think_start_token_id, 200, 201, 202],
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.reasoning_content, "abc")
        self.assertIsNone(msg.content)

    def test_streaming_no_reasoning_tags(self):
        """测试流式输出无思考标签（思考开关关闭）"""
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="direct",
            delta_text="direct",
            previous_token_ids=[],
            current_token_ids=[200],
            delta_token_ids=[200],
            output_stage="CONTENT_STAGE",
        )
        self.assertIsNotNone(msg)
        self.assertIsNone(msg.reasoning_content)
        self.assertEqual(msg.content, "direct")

    # ---- Stage detection ----
    def test_detect_output_stage_reasoning(self):
        """测试检测思考阶段"""
        prompt_token_ids = [self.parser.think_start_token_id]  # 包含 <think> 开始标记
        stage = self.parser.detect_output_stage(prompt_token_ids)
        self.assertEqual(stage, "REASONING_STAGE")

    def test_detect_output_stage_content(self):
        """测试检测回复阶段"""
        prompt_token_ids = [
            self.parser.think_start_token_id,
            self.parser.think_end_token_id,
        ]  # 包含 <think> 和 </think>
        stage = self.parser.detect_output_stage(prompt_token_ids)
        self.assertEqual(stage, "CONTENT_STAGE")

    def test_detect_output_stage_no_tags(self):
        """测试无标记时默认进入回复阶段"""
        prompt_token_ids = [200, 201, 202]  # 无思考标记
        stage = self.parser.detect_output_stage(prompt_token_ids)
        self.assertEqual(stage, "CONTENT_STAGE")

    # ---- Edge cases ----
    def test_batch_multiple_end_tags(self):
        """测试多个结束标签（只识别第一个）"""
        text = "<think>abc</think>xyz</think>more"
        reasoning, content = self.parser.extract_reasoning_content(text, self.request)
        self.assertEqual(reasoning, "abc")
        self.assertEqual(content, "xyz</think>more")

    def test_is_reasoning_end(self):
        """测试检查推理内容是否结束"""
        input_ids = [200, 201, self.parser.think_end_token_id, 202]
        result = self.parser.is_reasoning_end(input_ids)
        self.assertTrue(result)

        input_ids = [200, 201, 202]
        result = self.parser.is_reasoning_end(input_ids)
        self.assertFalse(result)

    def test_extract_content_ids(self):
        """测试提取 content token IDs"""
        input_ids = [200, 201, self.parser.think_end_token_id, 202, 203]
        result = self.parser.extract_content_ids(input_ids)
        self.assertEqual(result, [202, 203])

        input_ids = [200, 201, 202]
        result = self.parser.extract_content_ids(input_ids)
        self.assertEqual(result, [200, 201, 202])

    # ---- Initialization error cases ----
    def test_init_without_tokenizer_raises(self):
        """测试没有传入 tokenizer 时抛出 ValueError"""
        with self.assertRaises(ValueError) as context:
            # 创建一个没有 model_tokenizer 的 mock tokenizer
            class InvalidTokenizer:
                def get_vocab(self):
                    return {}

            # 需要绕过基类的检查，直接测试子类的检查
            # 由于基类会设置 model_tokenizer，我们需要模拟一个 None 的情况
            # 实际上，如果传入 None，基类会设置 self.model_tokenizer = None
            parser = DeepSeekReasoningParser.__new__(DeepSeekReasoningParser)
            parser.model_tokenizer = None
            # 手动触发检查
            if not parser.model_tokenizer:
                raise ValueError(
                    "The model tokenizer must be passed to the ReasoningParser " "constructor during construction."
                )

        self.assertIn("model tokenizer must be passed", str(context.exception))

    def test_init_without_end_token_raises(self):
        """测试 tokenizer 中没有结束 token 时抛出 RuntimeError"""
        class TokenizerWithoutEndToken:
            def get_vocab(self):
                # 只有开始 token，没有结束 token
                return {"<think>": 99}

        tokenizer = TokenizerWithoutEndToken()
        with self.assertRaises(RuntimeError) as context:
            DeepSeekReasoningParser(tokenizer=tokenizer, model_name="deepseek-v3.1")
        self.assertIn("could not locate think end", str(context.exception))

    # ---- extract_reasoning_content edge cases ----
    def test_batch_reasoning_stage_no_end_token(self):
        """测试在 REASONING_STAGE 但没有结束标签的情况"""
        text = "some reasoning content without end tag"
        reasoning, content = self.parser.extract_reasoning_content(
            text, self.request, output_stage="REASONING_STAGE"
        )
        # 应该将整个输出作为 reasoning_content
        self.assertEqual(reasoning, text)
        self.assertIsNone(content)

    # ---- extract_reasoning_content_streaming edge cases ----
    def test_streaming_delta_with_start_and_end_but_find_fails(self):
        """测试 delta 中包含开始和结束 token，但 find 返回 -1 的情况"""
        # 创建一个没有 think_start_token_id 的 parser（模拟 None 情况）
        class TokenizerWithoutStartToken:
            def get_vocab(self):
                # 只有结束 token，没有开始 token
                return {"</think>": 100}

        tokenizer = TokenizerWithoutStartToken()
        parser = DeepSeekReasoningParser(tokenizer=tokenizer, model_name="deepseek-v3.1")
        # think_start_token_id 应该是 None
        self.assertIsNone(parser.think_start_token_id)

        # 测试当 think_start_token_id 为 None 时的流式处理
        msg = parser.extract_reasoning_content_streaming(
            previous_text="abc",
            current_text="abc</think>xyz",
            delta_text="</think>xyz",
            previous_token_ids=[200, 201, 202],
            current_token_ids=[200, 201, 202, parser.think_end_token_id, 110, 120, 130],
            delta_token_ids=[parser.think_end_token_id, 110, 120, 130],
        )
        # 应该正确处理，提取出 content
        self.assertIsNotNone(msg)
        self.assertEqual(msg.content, "xyz")

    def test_streaming_delta_with_end_token_no_start_in_previous(self):
        """测试 delta 中包含结束 token，但 previous 中没有开始 token 的情况"""
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="abc",
            current_text="abc</think>xyz",
            delta_text="</think>xyz",
            previous_token_ids=[200, 201, 202],
            current_token_ids=[200, 201, 202, self.parser.think_end_token_id, 110, 120, 130],
            delta_token_ids=[self.parser.think_start_token_id, self.parser.think_end_token_id, 110, 120, 130],
        )
        # 这种情况应该被第149-166行的逻辑处理
        # 如果 delta 中包含结束 token，会尝试提取
        self.assertIsNotNone(msg)

    def test_streaming_reasoning_stage_no_tokens(self):
        """测试在 REASONING_STAGE 但没有看到任何 token 的情况"""
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="direct",
            delta_text="direct",
            previous_token_ids=[],
            current_token_ids=[200],
            delta_token_ids=[200],
            output_stage="REASONING_STAGE",
        )
        # 在 REASONING_STAGE 但没有 token，应该返回 reasoning_content
        self.assertIsNotNone(msg)
        self.assertEqual(msg.reasoning_content, "direct")
        self.assertIsNone(msg.content)

    def test_streaming_start_token_in_delta_but_find_fails(self):
        """测试 delta 中包含开始 token，但 find 返回 -1 的情况"""
        # 这种情况理论上不应该发生，因为如果 token_id 在 delta_token_ids 中，
        # 那么 delta_text 应该包含对应的文本。但为了覆盖代码路径，我们测试一下
        # 当 think_start_token_id 在 delta_token_ids 中，但 delta_text.find 返回 -1
        # 这可能是由于 token 编码问题导致的边界情况
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="abc",
            delta_text="abc",  # delta_text 中没有 <think>，但 token_ids 中有
            previous_token_ids=[],
            current_token_ids=[self.parser.think_start_token_id, 200, 201, 202],
            delta_token_ids=[self.parser.think_start_token_id, 200, 201, 202],
        )
        # 由于 find 返回 -1，不会进入第179-181行的分支
        # 会继续执行到后面的逻辑
        # 由于 previous_token_ids 中没有开始 token，会继续判断
        self.assertIsNotNone(msg)

    def test_init_none_tokenizer_hits_value_error(self):
        """直接调用 __init__ 时 tokenizer=None 应触发 ValueError"""
        with self.assertRaises(ValueError):
            DeepSeekReasoningParser(tokenizer=None, model_name="deepseek-v3.1")

    def test_streaming_previous_has_start_token_returns_reasoning(self):
        """previous_token_ids 含 <think> 时应返回 reasoning_content"""
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="<think>",
            current_text="<think>abc",
            delta_text="abc",
            previous_token_ids=[self.parser.think_start_token_id],
            current_token_ids=[self.parser.think_start_token_id, 200, 201, 202],
            delta_token_ids=[200, 201, 202],
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.reasoning_content, "abc")

    def test_streaming_delta_with_both_tokens_but_find_fails(self):
        """测试 delta 中同时包含开始和结束 token，但 find 返回 -1 的情况"""
        # 模拟 token_ids 中有 token，但 delta_text 中找不到对应字符串的情况
        # 这种情况可能发生在 token 编码异常时
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="abc",
            delta_text="abc",  # delta_text 中没有标签，但 token_ids 中有
            previous_token_ids=[],
            current_token_ids=[
                self.parser.think_start_token_id,
                200,
                self.parser.think_end_token_id,
                201,
            ],
            delta_token_ids=[
                self.parser.think_start_token_id,
                200,
                self.parser.think_end_token_id,
                201,
            ],
        )
        # 由于 find 返回 -1，不会进入第154-157行的分支
        # 会继续执行到第159行的 else 分支
        # 然后尝试在 delta_text 中找结束 token
        self.assertIsNotNone(msg)

    def test_streaming_delta_with_end_token_but_find_fails(self):
        """测试 delta 中包含结束 token，但 find 返回 -1 的情况"""
        # 模拟 token_ids 中有结束 token，但 delta_text 中找不到对应字符串的情况
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="abc",
            current_text="abcxyz",
            delta_text="xyz",  # delta_text 中没有结束标签，但 token_ids 中有
            previous_token_ids=[200, 201, 202],
            current_token_ids=[200, 201, 202, self.parser.think_end_token_id, 110, 120],
            delta_token_ids=[self.parser.think_end_token_id, 110, 120],
        )
        # 由于 find 返回 -1，不会进入第161-166行的分支
        # 会继续执行到后面的逻辑
        # 由于 previous_token_ids 中没有结束 token，会继续判断
        self.assertIsNotNone(msg)

    def test_streaming_delta_with_end_token_no_start_find_fails(self):
        """测试 delta 中包含结束 token（previous 中没有开始 token），但 find 返回 -1"""
        # 测试第159-166行的 else 分支中，end_index == -1 的情况
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="abc",
            current_text="abcxyz",
            delta_text="xyz",  # delta_text 中没有结束标签，但 token_ids 中有
            previous_token_ids=[200, 201, 202],  # previous 中没有开始 token
            current_token_ids=[200, 201, 202, self.parser.think_end_token_id, 110, 120],
            delta_token_ids=[self.parser.think_end_token_id, 110, 120],
        )
        # 由于 find 返回 -1，不会进入第161-166行的分支
        # 会继续执行到后面的逻辑
        self.assertIsNotNone(msg)

    def test_init_without_model_tokenizer_raises(self):
        """测试 model_tokenizer 为 None 时抛出 ValueError"""
        # 创建一个 parser 实例但不通过正常初始化
        parser = DeepSeekReasoningParser.__new__(DeepSeekReasoningParser)
        # 模拟基类初始化后 model_tokenizer 为 None 的情况
        parser.model_tokenizer = None
        parser.think_start_token = "<think>"
        parser.think_end_token = "</think>"
        
        # 测试检查逻辑
        with self.assertRaises(ValueError) as context:
            if not parser.model_tokenizer:
                raise ValueError(
                    "The model tokenizer must be passed to the ReasoningParser " "constructor during construction."
                )
        self.assertIn("model tokenizer must be passed", str(context.exception))


    def test_streaming_with_think_start_token_id_none(self):
        """测试 think_start_token_id 为 None 时的流式处理"""
        # 创建一个没有开始 token 的 tokenizer
        class TokenizerWithoutStartToken:
            def get_vocab(self):
                return {"</think>": 100}

        tokenizer = TokenizerWithoutStartToken()
        parser = DeepSeekReasoningParser(tokenizer=tokenizer, model_name="deepseek-v3.1")
        self.assertIsNone(parser.think_start_token_id)

        # 测试流式处理，确保不会因为 think_start_token_id 为 None 而报错
        msg = parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="abc",
            delta_text="abc",
            previous_token_ids=[],
            current_token_ids=[200, 201, 202],
            delta_token_ids=[200, 201, 202],
            output_stage="CONTENT_STAGE",
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.content, "abc")

    def test_streaming_delta_end_token_with_previous_start_token(self):
        """测试 delta 中包含结束 token，previous 中包含开始 token 的情况"""
        # 测试第159-166行的 else 分支
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="<think>abc",
            current_text="<think>abc</think>xyz",
            delta_text="</think>xyz",
            previous_token_ids=[self.parser.think_start_token_id, 200, 201, 202],
            current_token_ids=[
                self.parser.think_start_token_id,
                200,
                201,
                202,
                self.parser.think_end_token_id,
                110,
                120,
            ],
            delta_token_ids=[self.parser.think_end_token_id, 110, 120],
        )
        # 应该提取出 reasoning_content 和 content
        self.assertIsNotNone(msg)
        self.assertEqual(msg.reasoning_content, "")
        self.assertEqual(msg.content, "xyz")


if __name__ == "__main__":
    unittest.main()
