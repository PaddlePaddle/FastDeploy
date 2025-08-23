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
from unittest.mock import MagicMock

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest, ExtractedToolCallInformation
from fastdeploy.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager


class MockToolParser(ToolParser):
    """Mock implementation of ToolParser for testing"""

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        return ExtractedToolCallInformation(tools_called=False)

    def extract_tool_calls_streaming(
        self, previous_text, current_text, delta_text, previous_token_ids, current_token_ids, delta_token_ids, request
    ):
        return None


class InvalidToolParser:
    """Invalid parser that doesn't inherit from ToolParser"""

    pass


class TestToolParser(unittest.TestCase):
    """Unit tests for ToolParser abstract class"""

    def setUp(self):
        """Set up test environment"""
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.get_vocab.return_value = {"token1": 1, "token2": 2}

    def test_tool_parser_init(self):
        """Test ToolParser initialization"""
        parser = MockToolParser(self.mock_tokenizer)

        self.assertEqual(parser.prev_tool_call_arr, [])
        self.assertEqual(parser.current_tool_id, -1)
        self.assertFalse(parser.current_tool_name_sent)
        self.assertEqual(parser.streamed_args_for_tool, [])
        self.assertEqual(parser.model_tokenizer, self.mock_tokenizer)

    def test_vocab_property(self):
        """Test vocab cached property"""
        parser = MockToolParser(self.mock_tokenizer)
        vocab = parser.vocab

        self.assertEqual(vocab, {"token1": 1, "token2": 2})
        self.mock_tokenizer.get_vocab.assert_called_once()

        # Test that it's cached (get_vocab should not be called again)
        vocab_again = parser.vocab
        self.assertEqual(vocab_again, {"token1": 1, "token2": 2})
        self.mock_tokenizer.get_vocab.assert_called_once()

    def test_adjust_request(self):
        """Test adjust_request method returns request unchanged"""
        parser = MockToolParser(self.mock_tokenizer)
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "test"}])

        adjusted_request = parser.adjust_request(request)
        self.assertEqual(adjusted_request, request)

    def test_extract_tool_calls_not_implemented(self):
        """Test that base ToolParser raises NotImplementedError for extract_tool_calls"""
        parser = ToolParser(self.mock_tokenizer)
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "test"}])

        with self.assertRaises(NotImplementedError) as context:
            parser.extract_tool_calls("output", request)

        self.assertIn("AbstractToolParser.extract_tool_calls has not been implemented", str(context.exception))

    def test_extract_tool_calls_streaming_not_implemented(self):
        """Test that base ToolParser raises NotImplementedError for extract_tool_calls_streaming"""
        parser = ToolParser(self.mock_tokenizer)
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "test"}])

        with self.assertRaises(NotImplementedError) as context:
            parser.extract_tool_calls_streaming("prev", "curr", "delta", [], [], [], request)

        self.assertIn("AbstractToolParser.extract_tool_calls_streaming has not been", str(context.exception))


class TestToolParserManager(unittest.TestCase):
    """Unit tests for ToolParserManager"""

    def setUp(self):
        """Set up test environment"""
        # Clear the registry before each test
        ToolParserManager.tool_parsers.clear()

    def test_register_module_decorator(self):
        """Test registering a module using decorator syntax"""

        @ToolParserManager.register_module("test_parser")
        class TestParser(ToolParser):
            pass

        self.assertIn("test_parser", ToolParserManager.tool_parsers)
        self.assertEqual(ToolParserManager.tool_parsers["test_parser"], TestParser)

    def test_register_module_function(self):
        """Test registering a module using function call"""

        class TestParser(ToolParser):
            pass

        result = ToolParserManager.register_module("test_parser", module=TestParser)

        self.assertEqual(result, TestParser)
        self.assertIn("test_parser", ToolParserManager.tool_parsers)
        self.assertEqual(ToolParserManager.tool_parsers["test_parser"], TestParser)

    def test_register_module_multiple_names(self):
        """Test registering a module with multiple names"""

        class TestParser(ToolParser):
            pass

        ToolParserManager.register_module(["name1", "name2"], module=TestParser)

        self.assertIn("name1", ToolParserManager.tool_parsers)
        self.assertIn("name2", ToolParserManager.tool_parsers)
        self.assertEqual(ToolParserManager.tool_parsers["name1"], TestParser)
        self.assertEqual(ToolParserManager.tool_parsers["name2"], TestParser)

    def test_register_invalid_module(self):
        """Test registering invalid module raises TypeError"""
        with self.assertRaises(TypeError) as context:
            ToolParserManager.register_module("invalid", module=InvalidToolParser)

        self.assertIn("module must be subclass of ToolParser", str(context.exception))

    def test_register_duplicate_name_no_force(self):
        """Test registering duplicate name without force raises KeyError"""

        class Parser1(ToolParser):
            pass

        class Parser2(ToolParser):
            pass

        ToolParserManager.register_module("duplicate", module=Parser1)

        with self.assertRaises(KeyError) as context:
            ToolParserManager.register_module("duplicate", force=False, module=Parser2)

        self.assertIn("duplicate is already registered", str(context.exception))

    def test_register_duplicate_name_with_force(self):
        """Test registering duplicate name with force overwrites"""

        class Parser1(ToolParser):
            pass

        class Parser2(ToolParser):
            pass

        ToolParserManager.register_module("duplicate", module=Parser1)
        ToolParserManager.register_module("duplicate", force=True, module=Parser2)

        self.assertEqual(ToolParserManager.tool_parsers["duplicate"], Parser2)

    def test_get_tool_parser_success(self):
        """Test getting registered tool parser"""

        class TestParser(ToolParser):
            pass

        ToolParserManager.register_module("test", module=TestParser)

        result = ToolParserManager.get_tool_parser("test")
        self.assertEqual(result, TestParser)

    def test_get_tool_parser_not_found(self):
        """Test getting unregistered tool parser raises KeyError"""
        with self.assertRaises(KeyError) as context:
            ToolParserManager.get_tool_parser("nonexistent")

        self.assertIn("tool helper: 'nonexistent' not found", str(context.exception))

    def test_register_invalid_force_type(self):
        """Test registering with invalid force type raises TypeError"""
        with self.assertRaises(TypeError) as context:
            ToolParserManager.register_module("test", force="invalid")

        self.assertIn("force must be a boolean", str(context.exception))

    def test_register_invalid_name_type(self):
        """Test registering with invalid name type raises TypeError"""

        class TestParser(ToolParser):
            pass

        with self.assertRaises(TypeError) as context:
            ToolParserManager.register_module(123, module=TestParser)

        self.assertIn("name must be None, an instance of str, or a sequence of str", str(context.exception))


if __name__ == "__main__":
    unittest.main()
