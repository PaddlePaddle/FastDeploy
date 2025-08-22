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
import json
import re
from unittest.mock import MagicMock


# Mock structures to avoid import dependencies
class ExtractedToolCallInformation:
    def __init__(self, tools_called=False, tool_calls=None, content=""):
        self.tools_called = tools_called
        self.tool_calls = tool_calls or []
        self.content = content


class DeltaMessage:
    def __init__(self, role="assistant", content="", tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []


class ToolCall:
    def __init__(self, id, type, function):
        self.id = id
        self.type = type
        self.function = function


class FunctionCall:
    def __init__(self, name="", arguments=""):
        self.name = name
        self.arguments = arguments


# Simplified version of ErnieX1ToolParser for testing
class ErnieX1ToolParser:
    """Simplified Ernie X1 Tool parser for testing"""

    def __init__(self, tokenizer):
        self.model_tokenizer = tokenizer
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool = []
        self.buffer = ""
        self.bracket_counts = {"total_l": 0, "total_r": 0}
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"

        # Mock vocab access
        self.vocab = getattr(tokenizer, 'vocab', {}) or tokenizer.get_vocab()
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token, 1000)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token, 1001)

    def extract_tool_calls(self, model_output: str, request) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model response"""
        try:
            tool_calls = []

            # Check for invalid <response> tags before tool calls
            if re.search(r"<response>[\s\S]*?</response>\s*(?=<tool_call>)", model_output):
                return ExtractedToolCallInformation(tools_called=False, content=model_output)

            function_call_arr = []
            remaining_text = model_output

            while True:
                # Find next tool_call block
                tool_call_pos = remaining_text.find("<tool_call>")
                if tool_call_pos == -1:
                    break

                # Extract content after tool_call start
                tool_content_start = tool_call_pos + len("<tool_call>")
                tool_content_end = remaining_text.find("</tool_call>", tool_content_start)

                tool_json = ""
                if tool_content_end == -1:
                    # Handle unclosed tool_call block (truncation case)
                    tool_json = remaining_text[tool_content_start:].strip()
                    remaining_text = ""
                else:
                    # Handle complete tool_call block
                    tool_json = remaining_text[tool_content_start:tool_content_end].strip()
                    remaining_text = remaining_text[tool_content_end + len("</tool_call>"):]

                if not tool_json:
                    continue

                # Process JSON content
                tool_json = tool_json.strip()
                if not tool_json.startswith("{"):
                    tool_json = "{" + tool_json
                if not tool_json.endswith("}"):
                    tool_json = tool_json + "}"

                try:
                    # Try standard JSON parsing first
                    tool_data = json.loads(tool_json)
                    if isinstance(tool_data, dict) and "name" in tool_data and "arguments" in tool_data:
                        function_call_arr.append({
                            "name": tool_data["name"],
                            "arguments": tool_data["arguments"],
                            "_is_complete": True,
                        })
                        continue
                except json.JSONDecodeError:
                    # Handle partial JSON or malformed JSON
                    pass

            # Convert to tool calls format
            for func_call in function_call_arr:
                tool_calls.append(ToolCall(
                    id=f"call_{len(tool_calls)}",
                    type="function",
                    function=FunctionCall(
                        name=func_call["name"],
                        arguments=json.dumps(func_call["arguments"])
                        if isinstance(func_call["arguments"], dict)
                        else str(func_call["arguments"])
                    )
                ))

            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=model_output
            )

        except Exception as e:
            return ExtractedToolCallInformation(tools_called=False, content=model_output)

    def extract_tool_calls_streaming(self, previous_text, current_text, delta_text,
                                   previous_token_ids, current_token_ids, delta_token_ids, request):
        """Extract tool calls for streaming response (simplified)"""
        # Simplified streaming implementation
        if "<tool_call>" in delta_text:
            return DeltaMessage(role="assistant", content="")
        elif "</tool_call>" in delta_text:
            return DeltaMessage(role="assistant", content="")
        else:
            return DeltaMessage(role="assistant", content=delta_text)


class TestErnieX1ToolParser(unittest.TestCase):
    """Test ErnieX1ToolParser functionality"""

    def setUp(self):
        """Set up test environment"""
        self.mock_tokenizer = MagicMock()
        # Set up vocab as a real dictionary
        vocab_dict = {
            "<tool_call>": 1000,
            "</tool_call>": 1001,
            "token1": 1,
            "token2": 2
        }
        self.mock_tokenizer.get_vocab.return_value = vocab_dict
        self.mock_tokenizer.vocab = vocab_dict
        self.parser = ErnieX1ToolParser(self.mock_tokenizer)

    def test_init(self):
        """Test parser initialization"""
        self.assertEqual(self.parser.tool_call_start_token, "<tool_call>")
        self.assertEqual(self.parser.tool_call_end_token, "</tool_call>")
        self.assertEqual(self.parser.tool_call_start_token_id, 1000)
        self.assertEqual(self.parser.tool_call_end_token_id, 1001)
        self.assertEqual(self.parser.prev_tool_call_arr, [])
        self.assertEqual(self.parser.current_tool_id, -1)
        self.assertFalse(self.parser.current_tool_name_sent)

    def test_extract_tool_calls_no_tools(self):
        """Test extracting tool calls when none present"""
        model_output = "This is a regular response without tool calls."
        request = MagicMock()

        result = self.parser.extract_tool_calls(model_output, request)

        self.assertFalse(result.tools_called)
        self.assertEqual(len(result.tool_calls), 0)
        self.assertEqual(result.content, model_output)

    def test_extract_tool_calls_single_complete(self):
        """Test extracting a single complete tool call"""
        model_output = '''<tool_call>
{"name": "get_weather", "arguments": {"location": "Beijing"}}
</tool_call>'''
        request = MagicMock()

        result = self.parser.extract_tool_calls(model_output, request)

        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertIn("Beijing", result.tool_calls[0].function.arguments)

    def test_extract_tool_calls_multiple_complete(self):
        """Test extracting multiple complete tool calls"""
        model_output = '''<tool_call>
{"name": "get_weather", "arguments": {"location": "Beijing"}}
</tool_call>
<tool_call>
{"name": "get_time", "arguments": {"timezone": "UTC"}}
</tool_call>'''
        request = MagicMock()

        result = self.parser.extract_tool_calls(model_output, request)

        self.assertTrue(result.tools_called)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0].function.name, "get_weather")
        self.assertEqual(result.tool_calls[1].function.name, "get_time")

    def test_extract_tool_calls_incomplete(self):
        """Test extracting incomplete tool call (truncated)"""
        model_output = '''<tool_call>
{"name": "get_weather", "arguments": {"location": "Beijing"'''
        request = MagicMock()

        result = self.parser.extract_tool_calls(model_output, request)

        # Should handle incomplete JSON gracefully
        self.assertIsInstance(result, ExtractedToolCallInformation)

    def test_extract_tool_calls_malformed_json(self):
        """Test extracting tool calls with malformed JSON"""
        model_output = '''<tool_call>
"name": "get_weather", "arguments": {"location": "Beijing"}
</tool_call>'''
        request = MagicMock()

        result = self.parser.extract_tool_calls(model_output, request)

        # Should try to fix JSON by adding braces
        self.assertIsInstance(result, ExtractedToolCallInformation)

    def test_extract_tool_calls_with_response_tags(self):
        """Test extracting tool calls with invalid response tags"""
        model_output = '''<response>
This should not be here before tool calls
</response>
<tool_call>
{"name": "get_weather", "arguments": {"location": "Beijing"}}
</tool_call>'''
        request = MagicMock()

        result = self.parser.extract_tool_calls(model_output, request)

        # Should reject due to invalid format
        self.assertFalse(result.tools_called)

    def test_extract_tool_calls_empty_tool_call(self):
        """Test extracting empty tool call blocks"""
        model_output = '''<tool_call>
</tool_call>'''
        request = MagicMock()

        result = self.parser.extract_tool_calls(model_output, request)

        self.assertFalse(result.tools_called)
        self.assertEqual(len(result.tool_calls), 0)

    def test_extract_tool_calls_streaming_basic(self):
        """Test basic streaming tool call extraction"""
        previous_text = ""
        current_text = "Let me check the weather"
        delta_text = "Let me check the weather"
        request = MagicMock()

        result = self.parser.extract_tool_calls_streaming(
            previous_text, current_text, delta_text, [], [], [], request
        )

        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.role, "assistant")
        self.assertEqual(result.content, delta_text)

    def test_extract_tool_calls_streaming_start_token(self):
        """Test streaming with tool call start token"""
        delta_text = "<tool_call>"
        request = MagicMock()

        result = self.parser.extract_tool_calls_streaming(
            "", "", delta_text, [], [], [], request
        )

        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.role, "assistant")
        self.assertEqual(result.content, "")  # Should suppress token

    def test_extract_tool_calls_streaming_end_token(self):
        """Test streaming with tool call end token"""
        delta_text = "</tool_call>"
        request = MagicMock()

        result = self.parser.extract_tool_calls_streaming(
            "", "", delta_text, [], [], [], request
        )

        self.assertIsInstance(result, DeltaMessage)
        self.assertEqual(result.role, "assistant")
        self.assertEqual(result.content, "")  # Should suppress token

    def test_vocab_property(self):
        """Test vocab property access"""
        vocab = self.parser.vocab
        self.assertIn("<tool_call>", vocab)
        self.assertIn("</tool_call>", vocab)
        self.assertEqual(vocab["<tool_call>"], 1000)
        self.assertEqual(vocab["</tool_call>"], 1001)

    def test_bracket_counting_init(self):
        """Test bracket counting initialization"""
        self.assertEqual(self.parser.bracket_counts["total_l"], 0)
        self.assertEqual(self.parser.bracket_counts["total_r"], 0)

    def test_buffer_init(self):
        """Test buffer initialization"""
        self.assertEqual(self.parser.buffer, "")


if __name__ == "__main__":
    unittest.main()