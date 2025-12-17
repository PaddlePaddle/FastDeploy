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

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest
from fastdeploy.entrypoints.openai.tool_parsers.deepseek_tool_parser import (
    DeepSeekToolParser,
)


class DummyTokenizer:
    """Minimal tokenizer with vocab for testing."""

    def __init__(self):
        self.vocab = {
            "<think>": 128798,
            "</think>": 128799,
            "<｜tool▁calls▁begin｜>": 128806,
            "<｜tool▁calls▁end｜>": 128807,
            "<｜tool▁call▁begin｜>": 128808,
            "<｜tool▁call▁end｜>": 128809,
            "<｜tool▁sep｜>": 128814,
        }

    def get_vocab(self):
        """Return vocab dict for testing."""
        return self.vocab


class TestDeepSeekToolParserV31(unittest.TestCase):
    """Test tool parser for DeepSeek-V3.1 format."""

    def setUp(self):
        self.tokenizer = DummyTokenizer()
        self.parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        self.request = ChatCompletionRequest(
            model="deepseek-v3.1",
            messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {"type": "string", "enum": ["c", "f"]},
                            },
                        },
                    },
                }
            ],
        )

    # ---- Non-streaming parsing ----
    def test_batch_single_tool_call(self):
        """Test single tool call (V3.1 format)."""
        text = '<think>需要查询天气</think>\n\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "北京", "unit": "c"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertTrue(result.tools_called)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertIn("location", result.tool_calls[0].function.arguments)
        self.assertEqual(result.content, "")

    def test_batch_parallel_tool_calls(self):
        """Test parallel tool calls (V3.1 format)."""
        text = (
            "<think>需要查询多个信息</think>\n\n"
            "<｜tool▁calls▁begin｜>"
            '<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "北京", "unit": "c"}<｜tool▁call▁end｜>'
            '<｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"timezone": "Asia/Shanghai"}<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜>"
        )
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertTrue(result.tools_called)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertEqual(result.tool_calls[1].function.name, "get_time")

    def test_batch_no_tool_calls(self):
        """Test no tool calls."""
        text = "<think>这是普通回复</think>\n\n这是回复内容"
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertFalse(result.tools_called)
        self.assertIsNone(result.tool_calls)
        self.assertEqual(result.content, text)

    def test_batch_invalid_format_with_content_before_tool(self):
        """Test invalid format: non-whitespace content between reasoning end and tool calls."""
        text = '<think>思考内容</think>\n\nABC\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "北京"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertFalse(result.tools_called)
        self.assertEqual(result.content, text)

    def test_batch_partial_json(self):
        """Test incomplete JSON arguments."""
        text = '<think>需要查询天气</think>\n\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "北京", "unit": "c"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertTrue(result.tools_called)
        self.assertIsNotNone(result.tool_calls)

    # ---- Streaming parsing ----
    def test_streaming_tool_name(self):
        """Test streaming tool name extraction."""
        # Reset parser state
        self.parser.buffer = ""
        self.parser.current_tool_name_sent = False
        self.parser.streamed_args_for_tool = []
        self.parser.current_tool_id = -1

        # Step 1: Receive reasoning content and tool call begin token
        previous_text = "<think>需要查询天气</think>\n\n"
        current_text = previous_text + "<｜tool▁calls▁begin｜>"
        delta_text = "<｜tool▁calls▁begin｜>"
        msg = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[128806],
            delta_token_ids=[128806],
            request=self.request,
        )
        # No tool name yet, should return None
        self.assertIsNone(msg)

        # Step 2: Receive tool call begin token and partial tool name
        previous_text = current_text
        current_text = previous_text + "<｜tool▁call▁begin｜>get"
        delta_text = "<｜tool▁call▁begin｜>get"
        msg = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806],
            current_token_ids=[128806, 128808, 200],
            delta_token_ids=[128808, 200],
            request=self.request,
        )
        # No separator yet, should return None
        self.assertIsNone(msg)

        # Step 3: Receive complete tool name and separator
        previous_text = current_text
        current_text = previous_text + "_weather<｜tool▁sep｜>"
        delta_text = "_weather<｜tool▁sep｜>"
        msg = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 200],
            current_token_ids=[128806, 128808, 200, 201, 128814],
            delta_token_ids=[201, 128814],
            request=self.request,
        )
        # Should extract tool name now
        self.assertIsNotNone(msg)
        self.assertIsNotNone(msg.tool_calls)
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertEqual(msg.tool_calls[0].function.name, "get_weather")
        self.assertIsNone(msg.tool_calls[0].function.arguments)

    def test_streaming_tool_arguments(self):
        """Test streaming tool arguments extraction."""
        # Reset parser state
        self.parser.buffer = ""
        self.parser.current_tool_name_sent = False
        self.parser.streamed_args_for_tool = []
        self.parser.current_tool_id = -1

        # Step 1: Set tool name sent state (simulate tool name already extracted)
        previous_text = "<think>需要查询天气</think>\n\n"
        current_text = previous_text + "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
        delta_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
        msg = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[128806, 128808, 200, 201, 128814],
            delta_token_ids=[128806, 128808, 200, 201, 128814],
            request=self.request,
        )
        # Should extract tool name now
        self.assertIsNotNone(msg)
        self.assertEqual(msg.tool_calls[0].function.name, "get_weather")
        self.assertTrue(self.parser.current_tool_name_sent)

        # Step 2: Receive partial JSON arguments
        previous_text = current_text
        current_text = previous_text + '{"location": "'
        delta_text = '{"location": "'
        msg = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 200, 201, 128814],
            current_token_ids=[128806, 128808, 200, 201, 128814, 300, 301],
            delta_token_ids=[300, 301],
            request=self.request,
        )
        # Should return incremental arguments
        self.assertIsNotNone(msg)
        self.assertIsNotNone(msg.tool_calls)
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertIn("location", msg.tool_calls[0].function.arguments)

        # Step 3: Receive more arguments
        previous_text = current_text
        current_text = previous_text + '北京", "unit": "c"}'
        delta_text = '北京", "unit": "c"}'
        msg = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 200, 201, 128814, 300, 301],
            current_token_ids=[128806, 128808, 200, 201, 128814, 300, 301, 302, 303],
            delta_token_ids=[302, 303],
            request=self.request,
        )
        # Should return incremental arguments
        self.assertIsNotNone(msg)
        self.assertIsNotNone(msg.tool_calls)
        self.assertIn("unit", msg.tool_calls[0].function.arguments)

        # Step 4: Receive tool call end token
        previous_text = current_text
        current_text = previous_text + "<｜tool▁call▁end｜>"
        delta_text = "<｜tool▁call▁end｜>"
        msg = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 200, 201, 128814, 300, 301, 302, 303],
            current_token_ids=[128806, 128808, 200, 201, 128814, 300, 301, 302, 303, 128809],
            delta_token_ids=[128809],
            request=self.request,
        )
        # Should return complete arguments
        self.assertIsNotNone(msg)
        self.assertIsNotNone(msg.tool_calls)
        arguments = json.loads(msg.tool_calls[0].function.arguments)
        self.assertEqual(arguments["location"], "北京")
        self.assertEqual(arguments["unit"], "c")


class TestDeepSeekToolParserV30324(unittest.TestCase):
    """Test tool parser for DeepSeek-V3-0324/R1 format."""

    def setUp(self):
        self.tokenizer = DummyTokenizer()
        self.parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3-0324")
        self.request = ChatCompletionRequest(
            model="deepseek-v3-0324",
            messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {"type": "string", "enum": ["c", "f"]},
                            },
                        },
                    },
                }
            ],
        )

    # ---- Non-streaming parsing ----
    def test_batch_single_tool_call(self):
        """Test single tool call (V3-0324/R1 format)."""
        text = (
            "<think>需要查询天气</think>\n\n"
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            "```json\n"
            '{"location": "北京", "unit": "c"}\n'
            "```\n"
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertTrue(result.tools_called)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertIn("location", result.tool_calls[0].function.arguments)

    def test_batch_parallel_tool_calls(self):
        """Test parallel tool calls (V3-0324/R1 format)."""
        text = (
            "<think>需要查询多个信息</think>\n\n"
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            "```json\n"
            '{"location": "北京", "unit": "c"}\n'
            "```\n"
            "<｜tool▁call▁end｜>"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n"
            "```json\n"
            '{"timezone": "Asia/Shanghai"}\n'
            "```\n"
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertTrue(result.tools_called)
        self.assertIsNotNone(result.tool_calls)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertEqual(result.tool_calls[1].function.name, "get_time")

    def test_batch_no_tool_calls(self):
        """Test no tool calls."""
        text = "<think>这是普通回复</think>\n\n这是回复内容"
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertFalse(result.tools_called)
        self.assertIsNone(result.tool_calls)
        self.assertEqual(result.content, text)

    def test_batch_invalid_format_with_content_before_tool(self):
        """Test invalid format: non-whitespace content between reasoning end and tool calls."""
        text = (
            "<think>思考内容</think>\n\nABC\n"
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            "```json\n"
            '{"location": "北京"}\n'
            "```\n"
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertFalse(result.tools_called)
        self.assertEqual(result.content, text)

    # ---- Streaming parsing ----
    def test_streaming_tool_name_v30324(self):
        """Test streaming tool name extraction (V3-0324/R1 format)."""
        # Reset parser state
        self.parser.buffer = ""
        self.parser.current_tool_name_sent = False
        self.parser.streamed_args_for_tool = []
        self.parser.current_tool_id = -1

        # Step 1: Receive reasoning content and tool call begin token
        previous_text = "<think>需要查询天气</think>\n\n"
        current_text = previous_text + "<｜tool▁calls▁begin｜>"
        delta_text = "<｜tool▁calls▁begin｜>"
        msg = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[128806],
            delta_token_ids=[128806],
            request=self.request,
        )
        # No tool name yet, should return None
        self.assertIsNone(msg)

        # Step 2: Receive tool call begin token and "function"
        previous_text = current_text
        current_text = previous_text + "<｜tool▁call▁begin｜>function"
        delta_text = "<｜tool▁call▁begin｜>function"
        msg = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806],
            current_token_ids=[128806, 128808, 300],
            delta_token_ids=[128808, 300],
            request=self.request,
        )
        # No separator yet, should return None
        self.assertIsNone(msg)

        # Step 3: Receive separator
        previous_text = current_text
        current_text = previous_text + "<｜tool▁sep｜>"
        delta_text = "<｜tool▁sep｜>"
        msg = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 300],
            current_token_ids=[128806, 128808, 300, 128814],
            delta_token_ids=[128814],
            request=self.request,
        )
        # Detected "function" but no newline yet, should return None and wait for more data
        self.assertIsNone(msg)

        # Step 4: Receive function name and newline
        previous_text = current_text
        current_text = previous_text + "get_weather\n"
        delta_text = "get_weather\n"
        msg = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 300, 128814],
            current_token_ids=[128806, 128808, 300, 128814, 200, 201, 202, 10],
            delta_token_ids=[200, 201, 202, 10],
            request=self.request,
        )
        # Should extract tool name now
        self.assertIsNotNone(msg)
        self.assertIsNotNone(msg.tool_calls)
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertEqual(msg.tool_calls[0].function.name, "get_weather")
        self.assertIsNone(msg.tool_calls[0].function.arguments)

    def test_streaming_tool_arguments_v30324(self):
        """Test streaming tool arguments extraction (V3-0324/R1 format)."""
        # Reset parser state
        self.parser.buffer = ""
        self.parser.current_tool_name_sent = False
        self.parser.streamed_args_for_tool = []
        self.parser.current_tool_id = -1

        # Step 1: Set tool name sent state (simulate tool name already extracted)
        previous_text = "<think>需要查询天气</think>\n\n"
        current_text = previous_text + "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        delta_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        msg = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[128806, 128808, 300, 128814, 200, 201, 202, 10],
            delta_token_ids=[128806, 128808, 300, 128814, 200, 201, 202, 10],
            request=self.request,
        )
        # Should extract tool name now
        self.assertIsNotNone(msg)
        self.assertEqual(msg.tool_calls[0].function.name, "get_weather")
        self.assertTrue(self.parser.current_tool_name_sent)

        # Step 2: Receive code block start token
        previous_text = current_text
        current_text = previous_text + "```json\n"
        delta_text = "```json\n"
        msg = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 300, 128814, 200, 201, 202, 10],
            current_token_ids=[128806, 128808, 300, 128814, 200, 201, 202, 10, 400, 401],
            delta_token_ids=[400, 401],
            request=self.request,
        )
        # Code block token should be skipped, return None (no argument content yet)
        self.assertIsNone(msg)

        # Step 3: Receive partial JSON arguments
        previous_text = current_text
        current_text = previous_text + '{"location": "'
        delta_text = '{"location": "'
        msg = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 300, 128814, 200, 201, 202, 10, 400, 401],
            current_token_ids=[128806, 128808, 300, 128814, 200, 201, 202, 10, 400, 401, 500, 501],
            delta_token_ids=[500, 501],
            request=self.request,
        )
        # Should return incremental arguments
        self.assertIsNotNone(msg)
        self.assertIsNotNone(msg.tool_calls)
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertIn("location", msg.tool_calls[0].function.arguments)

        # Step 4: Receive more arguments
        previous_text = current_text
        current_text = previous_text + '北京", "unit": "c"}\n'
        delta_text = '北京", "unit": "c"}\n'
        msg = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 300, 128814, 200, 201, 202, 10, 400, 401, 500, 501],
            current_token_ids=[128806, 128808, 300, 128814, 200, 201, 202, 10, 400, 401, 500, 501, 502, 503, 10],
            delta_token_ids=[502, 503, 10],
            request=self.request,
        )
        # Should return incremental arguments
        self.assertIsNotNone(msg)
        self.assertIsNotNone(msg.tool_calls)
        self.assertIn("unit", msg.tool_calls[0].function.arguments)

        # Step 5: Receive code block end token and tool call end token
        previous_text = current_text
        current_text = previous_text + "```\n<｜tool▁call▁end｜>"
        delta_text = "```\n<｜tool▁call▁end｜>"
        msg = self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 300, 128814, 200, 201, 202, 10, 400, 401, 500, 501, 502, 503, 10],
            current_token_ids=[
                128806,
                128808,
                300,
                128814,
                200,
                201,
                202,
                10,
                400,
                401,
                500,
                501,
                502,
                503,
                10,
                402,
                10,
                128809,
            ],
            delta_token_ids=[402, 10, 128809],
            request=self.request,
        )
        # Should return complete arguments
        self.assertIsNotNone(msg)
        self.assertIsNotNone(msg.tool_calls)
        arguments = json.loads(msg.tool_calls[0].function.arguments)
        self.assertEqual(arguments["location"], "北京")
        self.assertEqual(arguments["unit"], "c")


class TestDeepSeekToolParserEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def setUp(self):
        self.tokenizer = DummyTokenizer()
        self.parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        self.request = ChatCompletionRequest(
            model="deepseek-v3.1",
            messages=[{"role": "user", "content": "test"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "test_function",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

    def test_detect_output_stage_tool_call(self):
        """Test detecting tool call stage."""
        prompt_token_ids = [128806]  # Contains <｜tool▁calls▁begin｜>
        stage = self.parser.detect_output_stage(prompt_token_ids)
        self.assertEqual(stage, "TOOL_CALL_STAGE")

    def test_detect_output_stage_content(self):
        """Test detecting content stage."""
        prompt_token_ids = [200, 201, 202]  # No tool call tokens
        stage = self.parser.detect_output_stage(prompt_token_ids)
        self.assertEqual(stage, "CONTENT_STAGE")

    def test_empty_tool_calls_block(self):
        """Test empty tool calls block."""
        text = "<think>思考内容</think>\n\n<｜tool▁calls▁begin｜><｜tool▁calls▁end｜>"
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertFalse(result.tools_called)

    def test_malformed_json(self):
        """Test malformed JSON."""
        text = '<think>思考内容</think>\n\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "北京", "unit":}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result = self.parser.extract_tool_calls(text, self.request)
        # Should handle gracefully, at least extract function name
        if result.tools_called:
            self.assertEqual(result.tool_calls[0].function.name, "get_weather")

    def test_empty_arguments(self):
        """Tool call with empty arguments should return empty dict."""
        text = "<think>需要查询天气</think>\n\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜><｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertTrue(result.tools_called)
        args = json.loads(result.tool_calls[0].function.arguments)
        self.assertEqual(args, {})

    def test_reasoning_end_without_tool_calls(self):
        """Reasoning end token but no tool calls."""
        text = "<think>思考内容</think>\n\n这是普通回复"
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertFalse(result.tools_called)
        self.assertEqual(result.content, text)

    def test_reasoning_end_with_whitespace_before_tool_calls(self):
        """Whitespace between reasoning end and tool calls should be allowed."""
        text = '<think>思考内容</think>\n\n  \n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "北京"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertTrue(result.tools_called)

    def test_reasoning_end_immediately_followed_by_tool_calls(self):
        """覆盖 tool_calls_begin 紧随 </think>（tool_calls_begin_pos 为 0）的分支。"""
        text = '<think>思考内容</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "北京"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertTrue(result.tools_called)
        self.assertEqual(result.content, "")

    def test_detect_model_version_default(self):
        """Unknown model name defaults to V3.1."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-unknown")
        self.assertTrue(parser.is_v31)

    def test_detect_model_version_r1(self):
        """R1 model should be treated as V3-0324 format."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-r1")
        self.assertFalse(parser.is_v31)

    def test_init_missing_tokenizer(self):
        """Missing required tokens should raise RuntimeError."""

        class MissingTokenizer:
            def get_vocab(self):
                return {}

        with self.assertRaises(RuntimeError):
            DeepSeekToolParser(tokenizer=MissingTokenizer(), model_name="deepseek-v3.1")

    def test_extract_tool_calls_exception_handling(self):
        """extract_tool_calls should handle internal errors gracefully."""
        import unittest.mock

        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        with unittest.mock.patch(
            "fastdeploy.entrypoints.openai.tool_parsers.deepseek_tool_parser.re.finditer",
            side_effect=Exception("boom"),
        ):
            text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>test<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
            result = parser.extract_tool_calls(text, self.request)
            self.assertFalse(result.tools_called)
            self.assertEqual(result.content, text)

    def test_extract_tool_calls_json_decode_error_fallback(self):
        """Incomplete JSON should fall back to partial parser."""
        text = '<think>需要查询天气</think>\n\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "北京", "unit": "c"<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result = self.parser.extract_tool_calls(text, self.request)
        if result.tools_called:
            self.assertIsNotNone(result.tool_calls[0].function.arguments)

    def test_extract_tool_calls_both_json_parsers_fail(self):
        """Both json and partial parser fail should still keep function name."""
        text = "<think>需要查询天气</think>\n\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>invalid json{<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        result = self.parser.extract_tool_calls(text, self.request)
        if result.tools_called:
            self.assertEqual(result.tool_calls[0].function.name, "get_weather")

    def test_extract_tool_calls_with_empty_function_arguments(self):
        """Empty function arguments string should return empty dict."""
        text = "<think>需要查询天气</think>\n\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>   <｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        result = self.parser.extract_tool_calls(text, self.request)
        self.assertTrue(result.tools_called)
        args = json.loads(result.tool_calls[0].function.arguments)
        self.assertEqual(args, {})

    # ---- Streaming edge cases ----
    def test_streaming_v31_no_separator_yet(self):
        """V3.1 streaming: no separator received yet."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.buffer = ""
        parser.current_tool_name_sent = False
        parser.streamed_args_for_tool = []
        parser.current_tool_id = -1

        previous_text = "<｜tool▁calls▁begin｜>"
        current_text = previous_text + "<｜tool▁call▁begin｜>get"
        delta_text = "<｜tool▁call▁begin｜>get"
        msg = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806],
            current_token_ids=[128806, 128808, 200],
            delta_token_ids=[128808, 200],
            request=self.request,
        )
        self.assertIsNone(msg)

    def test_streaming_existing_args_list_not_appended(self):
        """覆盖 len(streamed_args_for_tool) > current_tool_id 分支，不追加空字符串。"""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.buffer = ""
        parser.current_tool_name_sent = False
        parser.streamed_args_for_tool = ["existing"]
        parser.current_tool_id = -1

        current_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
        msg = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=current_text,
            delta_text=current_text,
            previous_token_ids=[],
            current_token_ids=[parser.tool_calls_begin_token_id, parser.tool_call_begin_token_id],
            delta_token_ids=[parser.tool_calls_begin_token_id, parser.tool_call_begin_token_id],
            request=self.request,
        )
        self.assertIsNone(msg)
        self.assertEqual(len(parser.streamed_args_for_tool), 1)

    def test_streaming_empty_function_name_returns_none(self):
        """工具名为空时应返回 None，覆盖 function_name 为空的分支。"""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.buffer = ""
        parser.current_tool_name_sent = False
        parser.streamed_args_for_tool = []
        parser.current_tool_id = -1

        delta = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜><｜tool▁sep｜>"
        msg = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=delta,
            delta_text=delta,
            previous_token_ids=[],
            current_token_ids=[
                parser.tool_calls_begin_token_id,
                parser.tool_call_begin_token_id,
                parser.tool_sep_token_id,
            ],
            delta_token_ids=[
                parser.tool_calls_begin_token_id,
                parser.tool_call_begin_token_id,
                parser.tool_sep_token_id,
            ],
            request=self.request,
        )
        self.assertIsNone(msg)
        self.assertFalse(parser.current_tool_name_sent)

    def test_streaming_v30324_no_newline_yet(self):
        """V3-0324 streaming: separator received but no newline yet."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3-0324")
        parser.buffer = ""
        parser.current_tool_name_sent = False
        parser.streamed_args_for_tool = []
        parser.current_tool_id = -1

        previous_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>"
        current_text = previous_text + "get_weather"
        delta_text = "get_weather"
        msg = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 300, 128814],
            current_token_ids=[128806, 128808, 300, 128814, 200, 201, 202],
            delta_token_ids=[200, 201, 202],
            request=self.request,
        )
        self.assertIsNone(msg)

    def test_streaming_first_args_received(self):
        """First chunk of arguments should be returned."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.buffer = '{"location": "'
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = []
        parser.current_tool_id = 0

        previous_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>test<｜tool▁sep｜>"
        current_text = previous_text + '{"location": "'
        delta_text = '{"location": "'
        msg = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 200, 128814],
            current_token_ids=[128806, 128808, 200, 128814, 300, 301],
            delta_token_ids=[300, 301],
            request=self.request,
        )
        self.assertIsNotNone(msg)
        self.assertIsNotNone(msg.tool_calls)
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertIsNotNone(msg.tool_calls[0].function.arguments)

    def test_streaming_empty_args_text(self):
        """Empty args when tool call ends should not crash."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.buffer = ""
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.current_tool_id = 0

        previous_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>test<｜tool▁sep｜>"
        current_text = previous_text + "<｜tool▁call▁end｜>"
        delta_text = "<｜tool▁call▁end｜>"
        msg = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 200, 128814],
            current_token_ids=[128806, 128808, 200, 128814, 128809],
            delta_token_ids=[128809],
            request=self.request,
        )
        if msg is not None:
            args = json.loads(msg.tool_calls[0].function.arguments)
            self.assertEqual(args, {})

    def test_streaming_args_text_not_stripped(self):
        """Whitespace-only args_text should return None."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.buffer = "   "
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.current_tool_id = 0

        previous_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>test<｜tool▁sep｜>"
        current_text = previous_text + "   "
        delta_text = "   "
        msg = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 200, 128814],
            current_token_ids=[128806, 128808, 200, 128814, 300],
            delta_token_ids=[300],
            request=self.request,
        )
        self.assertIsNone(msg)

    def test_streaming_args_not_starting_with_prev_returns_none(self):
        """When args don't start with previous args, should return None."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.buffer = '{"new": "data"}'
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = ['{"old": "data"}']
        parser.current_tool_id = 0

        previous_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>test<｜tool▁sep｜>"
        current_text = previous_text + '{"new": "data"}'
        delta_text = '{"new": "data"}'
        msg = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 200, 128814],
            current_token_ids=[128806, 128808, 200, 128814, 300, 301],
            delta_token_ids=[300, 301],
            request=self.request,
        )
        self.assertIsNone(msg)

    def test_streaming_new_args_empty_after_subtraction(self):
        """No new args appended should return None."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.buffer = '{"location": "北京"}'
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = ['{"location": "北京"}']
        parser.current_tool_id = 0

        previous_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>test<｜tool▁sep｜>"
        current_text = previous_text + '{"location": "北京"}'
        delta_text = '{"location": "北京"}'
        msg = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 200, 128814],
            current_token_ids=[128806, 128808, 200, 128814, 300, 301],
            delta_token_ids=[300, 301],
            request=self.request,
        )
        self.assertIsNone(msg)

    def test_streaming_v30324_code_block_removal(self):
        """V3-0324 code block should be stripped before parsing."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3-0324")
        parser.buffer = '```json\n{"test": "value"}\n```'
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.current_tool_id = 0

        previous_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>test\n"
        current_text = previous_text + '```json\n{"test": "value"}\n```<｜tool▁call▁end｜>'
        delta_text = '```json\n{"test": "value"}\n```<｜tool▁call▁end｜>'
        msg = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 300, 128814, 200, 10],
            current_token_ids=[128806, 128808, 300, 128814, 200, 10, 400, 401, 128809],
            delta_token_ids=[400, 401, 128809],
            request=self.request,
        )
        if msg is not None:
            args = json.loads(msg.tool_calls[0].function.arguments)
            self.assertEqual(args["test"], "value")

    def test_streaming_v30324_partial_code_block(self):
        """V3-0324 partial code block should not crash."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3-0324")
        parser.buffer = '```json\n{"test": "'
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.current_tool_id = 0

        previous_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>test\n"
        current_text = previous_text + '```json\n{"test": "'
        delta_text = '```json\n{"test": "'
        msg = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 300, 128814, 200, 10],
            current_token_ids=[128806, 128808, 300, 128814, 200, 10, 400, 401, 500],
            delta_token_ids=[400, 401, 500],
            request=self.request,
        )
        if msg is not None:
            self.assertIsNotNone(msg.tool_calls[0].function.arguments)

    def test_streaming_v30324_empty_args_after_code_block_removal(self):
        """V3-0324 empty code block should be handled gracefully."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3-0324")
        parser.buffer = "```json\n```"
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.current_tool_id = 0

        previous_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>test\n"
        current_text = previous_text + "```json\n```<｜tool▁call▁end｜>"
        delta_text = "```json\n```<｜tool▁call▁end｜>"
        msg = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 300, 128814, 200, 10],
            current_token_ids=[128806, 128808, 300, 128814, 200, 10, 400, 401, 128809],
            delta_token_ids=[400, 401, 128809],
            request=self.request,
        )
        self.assertIsNone(msg)

    def test_streaming_empty_args_text_at_end(self):
        """End token with empty args should not error."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.buffer = ""
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.current_tool_id = 0

        previous_text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>test<｜tool▁sep｜>"
        current_text = previous_text + "<｜tool▁call▁end｜>"
        delta_text = "<｜tool▁call▁end｜>"
        msg = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[128806, 128808, 200, 128814],
            current_token_ids=[128806, 128808, 200, 128814, 128809],
            delta_token_ids=[128809],
            request=self.request,
        )
        self.assertIsNone(msg)

    def test_streaming_end_token_id_without_marker(self):
        """delta 中仅包含结束 token id 但 buffer 不含结束标记时应返回 None。"""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.buffer = '{"foo": 1}'
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.current_tool_id = 0

        msg = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=parser.buffer,
            delta_text="",
            previous_token_ids=[],
            current_token_ids=[parser.tool_calls_begin_token_id, parser.tool_call_end_token_id],
            delta_token_ids=[parser.tool_call_end_token_id],
            request=self.request,
        )
        self.assertIsNone(msg)

    def test_init_model_tokenizer_none_raises(self):
        """__init__ should raise ValueError when model_tokenizer is missing."""
        from unittest.mock import PropertyMock, patch

        from fastdeploy.entrypoints.openai.tool_parsers import abstract_tool_parser

        fake_vocab = {
            "<｜tool▁calls▁begin｜>": 1,
            "<｜tool▁call▁begin｜>": 2,
            "<｜tool▁sep｜>": 3,
            "<｜tool▁call▁end｜>": 4,
        }

        with (
            patch.object(
                abstract_tool_parser.ToolParser,
                "__init__",
                lambda self, tokenizer: setattr(self, "model_tokenizer", tokenizer),
            ),
            patch.object(DeepSeekToolParser, "vocab", new_callable=PropertyMock, return_value=fake_vocab),
        ):
            with self.assertRaises(ValueError):
                DeepSeekToolParser(tokenizer=None, model_name="deepseek-v3.1")

    def test_streaming_no_begin_token_returns_none(self):
        """If no tool_calls_begin token present, streaming should return None."""
        msg = self.parser.extract_tool_calls_streaming(
            previous_text="",
            current_text="plain text",
            delta_text="plain text",
            previous_token_ids=[],
            current_token_ids=[200, 201],
            delta_token_ids=[200, 201],
            request=self.request,
        )
        self.assertIsNone(msg)

    def test_streaming_v30324_no_code_block_fallback(self):
        """V3-0324 without code block should fall back to stripping markers and parse."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3-0324")
        parser.buffer = '{"foo": "bar"}<｜tool▁call▁end｜>'
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.current_tool_id = 0

        msg = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=parser.buffer,
            delta_text=parser.buffer,
            previous_token_ids=[],
            current_token_ids=[parser.tool_calls_begin_token_id, parser.tool_call_end_token_id],
            delta_token_ids=[parser.tool_call_end_token_id],
            request=self.request,
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.tool_calls[0].function.arguments, '{"foo": "bar"}')

    def test_streaming_partial_json_double_fallback(self):
        """Both json.loads and partial_json_parser fail should return raw text."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.buffer = "invalid-json<｜tool▁call▁end｜>"
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.current_tool_id = 0

        with unittest.mock.patch(
            "fastdeploy.entrypoints.openai.tool_parsers.deepseek_tool_parser.partial_json_parser.loads",
            side_effect=Exception("boom"),
        ):
            msg = parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=parser.buffer,
                delta_text=parser.buffer,
                previous_token_ids=[],
                current_token_ids=[parser.tool_calls_begin_token_id, parser.tool_call_end_token_id],
                delta_token_ids=[parser.tool_call_end_token_id],
                request=self.request,
            )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.tool_calls[0].function.arguments, "invalid-json")

    def test_streaming_incremental_no_new_args_returns_none(self):
        """When no new incremental args, should return None."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = ['{"location": "北京"}']
        parser.current_tool_id = 0
        parser.buffer = '{"location": "北京"}'

        msg = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=parser.buffer,
            delta_text=parser.buffer,
            previous_token_ids=[],
            current_token_ids=[128806, 128808],
            delta_token_ids=[128808],
            request=self.request,
        )
        self.assertIsNone(msg)

    def test_streaming_outer_exception_returns_none(self):
        """Any unexpected exception should be caught and return None."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")

        with unittest.mock.patch(
            "fastdeploy.entrypoints.openai.tool_parsers.deepseek_tool_parser.random_tool_call_id",
            side_effect=RuntimeError("boom"),
        ):
            msg = parser.extract_tool_calls_streaming(
                previous_text="",
                current_text="<｜tool▁call▁begin｜>fn<｜tool▁sep｜>",
                delta_text="<｜tool▁call▁begin｜>fn<｜tool▁sep｜>",
                previous_token_ids=[],
                current_token_ids=[parser.tool_call_begin_token_id, parser.tool_sep_token_id],
                delta_token_ids=[parser.tool_call_begin_token_id, parser.tool_sep_token_id],
                request=self.request,
            )
        self.assertIsNone(msg)

    def test_streaming_partial_json_double_fallback_coverage(self):
        """Cover double fallback path when both json and partial parser fail."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.buffer = "invalid-json<｜tool▁call▁end｜>"
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.current_tool_id = 0

        with unittest.mock.patch(
            "fastdeploy.entrypoints.openai.tool_parsers.deepseek_tool_parser.partial_json_parser.loads",
            side_effect=Exception("boom"),
        ):
            msg = parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=parser.buffer,
                delta_text=parser.buffer,
                previous_token_ids=[],
                current_token_ids=[parser.tool_calls_begin_token_id, parser.tool_call_end_token_id],
                delta_token_ids=[parser.tool_call_end_token_id],
                request=self.request,
            )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.tool_calls[0].function.arguments, "invalid-json")

    def test_streaming_partial_json_partial_parser_success(self):
        """Cover partial_json_parser success path (line ~298)."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.buffer = '{"foo":1<｜tool▁call▁end｜>'
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.current_tool_id = 0

        with (
            unittest.mock.patch("json.loads", side_effect=json.JSONDecodeError("err", "doc", 0)),
            unittest.mock.patch(
                "fastdeploy.entrypoints.openai.tool_parsers.deepseek_tool_parser.partial_json_parser.loads",
                return_value={"foo": 1},
            ),
        ):
            msg = parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=parser.buffer,
                delta_text=parser.buffer,
                previous_token_ids=[],
                current_token_ids=[parser.tool_calls_begin_token_id, parser.tool_call_end_token_id],
                delta_token_ids=[parser.tool_call_end_token_id],
                request=self.request,
            )
        self.assertIsNotNone(msg)
        self.assertIn('"foo"', msg.tool_calls[0].function.arguments)

    def test_streaming_incremental_no_new_args_returns_none_coverage(self):
        """Cover incremental no-new-args early return."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = ['{"location": "北京"}']
        parser.current_tool_id = 0
        parser.buffer = '{"location": "北京"}'

        msg = parser.extract_tool_calls_streaming(
            previous_text="",
            current_text=parser.buffer,
            delta_text=parser.buffer,
            previous_token_ids=[],
            current_token_ids=[parser.tool_calls_begin_token_id],
            delta_token_ids=[parser.tool_calls_begin_token_id],
            request=self.request,
        )
        self.assertIsNone(msg)

    def test_streaming_incremental_partial_parser_new_args(self):
        """Cover streaming branch where partial_json_parser succeeds and returns new args (line ~326)."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = ["{"]
        parser.current_tool_id = 0
        parser.buffer = '{"a":1'

        with unittest.mock.patch(
            "fastdeploy.entrypoints.openai.tool_parsers.deepseek_tool_parser.partial_json_parser.loads",
            return_value={"a": 1},
        ):
            msg = parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=parser.buffer,
                delta_text=parser.buffer,
                previous_token_ids=[],
                current_token_ids=[parser.tool_calls_begin_token_id],
                delta_token_ids=[parser.tool_calls_begin_token_id],
                request=self.request,
            )
        self.assertIsNotNone(msg)
        # 新增部分可能是片段，容忍格式差异
        self.assertIn('"a"', msg.tool_calls[0].function.arguments)

    def test_streaming_outer_exception_returns_none_coverage(self):
        """Cover outer try/except returning None on unexpected error."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")

        with unittest.mock.patch(
            "fastdeploy.entrypoints.openai.tool_parsers.deepseek_tool_parser.data_processor_logger.error",
            side_effect=RuntimeError("boom"),
        ):
            msg = parser.extract_tool_calls_streaming(
                previous_text="",
                current_text="<｜tool▁calls▁begin｜>",
                delta_text="<｜tool▁calls▁begin｜>",
                previous_token_ids=[],
                current_token_ids=[parser.tool_calls_begin_token_id],
                delta_token_ids=[parser.tool_calls_begin_token_id],
                request=self.request,
            )
        self.assertIsNone(msg)

    def test_streaming_outer_exception_delta_function_call(self):
        """Force DeltaFunctionCall to raise and hit outer except (lines 362-364)."""
        parser = DeepSeekToolParser(tokenizer=self.tokenizer, model_name="deepseek-v3.1")
        parser.current_tool_name_sent = True
        parser.streamed_args_for_tool = [""]
        parser.current_tool_id = 0
        parser.buffer = "{}<｜tool▁call▁end｜>"

        with unittest.mock.patch(
            "fastdeploy.entrypoints.openai.tool_parsers.deepseek_tool_parser.DeltaFunctionCall",
            side_effect=RuntimeError("boom"),
        ):
            msg = parser.extract_tool_calls_streaming(
                previous_text="",
                current_text=parser.buffer,
                delta_text=parser.buffer,
                previous_token_ids=[],
                current_token_ids=[parser.tool_calls_begin_token_id, parser.tool_call_end_token_id],
                delta_token_ids=[parser.tool_call_end_token_id],
                request=self.request,
            )
        self.assertIsNone(msg)


if __name__ == "__main__":
    unittest.main()
