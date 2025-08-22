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
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)


class MockToolParser(ToolParser):
    """Mock tool parser for testing"""
    
    def extract_tool_calls(self, model_output, request):
        return {"tool_calls": [], "content": model_output}
    
    def extract_tool_calls_streaming(self, previous_text, current_text, delta_text, 
                                   previous_token_ids, current_token_ids, delta_token_ids, request):
        return {"role": "assistant", "content": delta_text}


class TestToolParser(unittest.TestCase):
    """Test ToolParser base class"""

    def setUp(self):
        """Set up test environment"""
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.get_vocab.return_value = {"token1": 1, "token2": 2}
        
    def test_tool_parser_init(self):
        """Test ToolParser initialization"""
        parser = MockToolParser(self.mock_tokenizer)
        
        self.assertEqual(parser.prev_tool_call_arr, [])
        self.assertEqual(parser.current_tool_id, -1)
        self.assertEqual(parser.current_tool_name_sent, False)
        self.assertEqual(parser.streamed_args_for_tool, [])
        self.assertEqual(parser.model_tokenizer, self.mock_tokenizer)

    def test_tool_parser_vocab_property(self):
        """Test vocab property caching"""
        parser = MockToolParser(self.mock_tokenizer)
        
        # First access
        vocab1 = parser.vocab
        self.assertEqual(vocab1, {"token1": 1, "token2": 2})
        self.mock_tokenizer.get_vocab.assert_called_once()
        
        # Second access should use cached value
        vocab2 = parser.vocab
        self.assertEqual(vocab2, {"token1": 1, "token2": 2})
        self.mock_tokenizer.get_vocab.assert_called_once()  # Still only called once

    def test_adjust_request_default(self):
        """Test default adjust_request method"""
        parser = MockToolParser(self.mock_tokenizer)
        mock_request = MagicMock()
        
        result = parser.adjust_request(mock_request)
        self.assertEqual(result, mock_request)

    def test_extract_tool_calls_implemented(self):
        """Test that extract_tool_calls is implemented in mock"""
        parser = MockToolParser(self.mock_tokenizer)
        mock_request = MagicMock()
        
        result = parser.extract_tool_calls("test output", mock_request)
        self.assertEqual(result, {"tool_calls": [], "content": "test output"})

    def test_extract_tool_calls_streaming_implemented(self):
        """Test that extract_tool_calls_streaming is implemented in mock"""
        parser = MockToolParser(self.mock_tokenizer)
        mock_request = MagicMock()
        
        result = parser.extract_tool_calls_streaming(
            "prev", "curr", "delta", [1, 2], [1, 2, 3], [3], mock_request
        )
        self.assertEqual(result, {"role": "assistant", "content": "delta"})

    def test_base_tool_parser_abstract_methods(self):
        """Test that base ToolParser raises NotImplementedError for abstract methods"""
        parser = ToolParser(self.mock_tokenizer)
        mock_request = MagicMock()
        
        with self.assertRaises(NotImplementedError):
            parser.extract_tool_calls("test", mock_request)
            
        with self.assertRaises(NotImplementedError):
            parser.extract_tool_calls_streaming(
                "prev", "curr", "delta", [1], [1, 2], [2], mock_request
            )


class TestToolParserManager(unittest.TestCase):
    """Test ToolParserManager class"""

    def setUp(self):
        """Set up test environment"""
        # Clear any existing parsers
        ToolParserManager.tool_parsers = {}

    def tearDown(self):
        """Clean up after tests"""
        # Clear parsers to avoid interference
        ToolParserManager.tool_parsers = {}

    def test_register_module_as_method(self):
        """Test registering module as method call"""
        ToolParserManager.register_module("test_parser", module=MockToolParser)
        
        self.assertIn("test_parser", ToolParserManager.tool_parsers)
        self.assertEqual(ToolParserManager.tool_parsers["test_parser"], MockToolParser)

    def test_register_module_as_decorator(self):
        """Test registering module as decorator"""
        @ToolParserManager.register_module("decorated_parser")
        class DecoratedParser(ToolParser):
            pass
        
        self.assertIn("decorated_parser", ToolParserManager.tool_parsers)
        self.assertEqual(ToolParserManager.tool_parsers["decorated_parser"], DecoratedParser)

    def test_register_module_multiple_names(self):
        """Test registering module with multiple names"""
        ToolParserManager.register_module(["name1", "name2"], module=MockToolParser)
        
        self.assertIn("name1", ToolParserManager.tool_parsers)
        self.assertIn("name2", ToolParserManager.tool_parsers)
        self.assertEqual(ToolParserManager.tool_parsers["name1"], MockToolParser)
        self.assertEqual(ToolParserManager.tool_parsers["name2"], MockToolParser)

    def test_register_module_default_name(self):
        """Test registering module with default name"""
        ToolParserManager.register_module(module=MockToolParser)
        
        self.assertIn("MockToolParser", ToolParserManager.tool_parsers)
        self.assertEqual(ToolParserManager.tool_parsers["MockToolParser"], MockToolParser)

    def test_register_module_force_false_existing(self):
        """Test registering module with force=False when name exists"""
        ToolParserManager.tool_parsers["existing"] = MockToolParser
        
        class AnotherParser(ToolParser):
            pass
        
        with self.assertRaises(KeyError):
            ToolParserManager.register_module("existing", force=False, module=AnotherParser)

    def test_register_module_invalid_type(self):
        """Test registering invalid module type"""
        class NotAToolParser:
            pass
        
        with self.assertRaises(TypeError):
            ToolParserManager.register_module("invalid", module=NotAToolParser)

    def test_register_module_invalid_force_type(self):
        """Test registering with invalid force parameter"""
        with self.assertRaises(TypeError):
            ToolParserManager.register_module("test", force="not_bool", module=MockToolParser)

    def test_register_module_invalid_name_type(self):
        """Test registering with invalid name parameter"""
        with self.assertRaises(TypeError):
            ToolParserManager.register_module(123, module=MockToolParser)

    def test_get_tool_parser_existing(self):
        """Test getting existing tool parser"""
        ToolParserManager.tool_parsers["test_parser"] = MockToolParser
        
        result = ToolParserManager.get_tool_parser("test_parser")
        self.assertEqual(result, MockToolParser)

    def test_get_tool_parser_nonexistent(self):
        """Test getting non-existent tool parser"""
        with self.assertRaises(KeyError) as cm:
            ToolParserManager.get_tool_parser("nonexistent")
        
        self.assertIn("'nonexistent' not found in tool_parsers", str(cm.exception))

    @patch('fastdeploy.entrypoints.openai.tool_parsers.abstract_tool_parser.import_from_path')
    @patch('fastdeploy.entrypoints.openai.tool_parsers.abstract_tool_parser.data_processor_logger')
    def test_import_tool_parser_success(self, mock_logger, mock_import):
        """Test successful tool parser import"""
        plugin_path = "/path/to/plugin.py"
        
        ToolParserManager.import_tool_parser(plugin_path)
        
        mock_import.assert_called_once_with("plugin", plugin_path)

    @patch('fastdeploy.entrypoints.openai.tool_parsers.abstract_tool_parser.import_from_path')
    @patch('fastdeploy.entrypoints.openai.tool_parsers.abstract_tool_parser.data_processor_logger')
    def test_import_tool_parser_failure(self, mock_logger, mock_import):
        """Test failed tool parser import"""
        plugin_path = "/path/to/plugin.py"
        mock_import.side_effect = ImportError("Failed to import")
        
        ToolParserManager.import_tool_parser(plugin_path)
        
        mock_import.assert_called_once_with("plugin", plugin_path)
        mock_logger.exception.assert_called_once()

    def test_import_tool_parser_module_name_extraction(self):
        """Test module name extraction from path"""
        with patch('fastdeploy.entrypoints.openai.tool_parsers.abstract_tool_parser.import_from_path') as mock_import:
            ToolParserManager.import_tool_parser("/complex/path/to/my_parser.py")
            mock_import.assert_called_once_with("my_parser", "/complex/path/to/my_parser.py")


if __name__ == "__main__":
    unittest.main()