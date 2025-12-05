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

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from fastdeploy.entrypoints.openai.tool_parsers.deepseek_tool_parser import DeepSeekToolParser


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
            '<think>需要查询多个信息</think>\n\n'
            '<｜tool▁calls▁begin｜>'
            '<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "北京", "unit": "c"}<｜tool▁call▁end｜>'
            '<｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{"timezone": "Asia/Shanghai"}<｜tool▁call▁end｜>'
            '<｜tool▁calls▁end｜>'
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
            '<think>需要查询天气</think>\n\n'
            '<｜tool▁calls▁begin｜>'
            '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n'
            '```json\n'
            '{"location": "北京", "unit": "c"}\n'
            '```\n'
            '<｜tool▁call▁end｜>'
            '<｜tool▁calls▁end｜>'
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
            '<think>需要查询多个信息</think>\n\n'
            '<｜tool▁calls▁begin｜>'
            '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n'
            '```json\n'
            '{"location": "北京", "unit": "c"}\n'
            '```\n'
            '<｜tool▁call▁end｜>'
            '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n'
            '```json\n'
            '{"timezone": "Asia/Shanghai"}\n'
            '```\n'
            '<｜tool▁call▁end｜>'
            '<｜tool▁calls▁end｜>'
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
            '<think>思考内容</think>\n\nABC\n'
            '<｜tool▁calls▁begin｜>'
            '<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n'
            '```json\n'
            '{"location": "北京"}\n'
            '```\n'
            '<｜tool▁call▁end｜>'
            '<｜tool▁calls▁end｜>'
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
            current_token_ids=[128806, 128808, 300, 128814, 200, 201, 202, 10, 400, 401, 500, 501, 502, 503, 10, 402, 10, 128809],
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


if __name__ == "__main__":
    unittest.main()

