"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)


class MockToolParser(ToolParser):
    """Mock implementation of ToolParser for testing"""
    
    def extract_tool_calls(self, model_output, request):
        return {"tool_calls": [], "content": model_output}
    
    def extract_tool_calls_streaming(self, previous_text, current_text, delta_text, 
                                   previous_token_ids, current_token_ids, delta_token_ids, request):
        return {"role": "assistant", "content": delta_text}


class TestToolParser(unittest.TestCase):
    """Test case for ToolParser abstract class"""

    def setUp(self):
        """Set up test environment"""
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.get_vocab.return_value = {"token1": 1, "token2": 2}
        self.parser = MockToolParser(self.mock_tokenizer)

    def test_tool_parser_init(self):
        """Test ToolParser initialization"""
        self.assertEqual(self.parser.prev_tool_call_arr, [])
        self.assertEqual(self.parser.current_tool_id, -1)
        self.assertFalse(self.parser.current_tool_name_sent)
        self.assertEqual(self.parser.streamed_args_for_tool, [])
        self.assertEqual(self.parser.model_tokenizer, self.mock_tokenizer)

    def test_vocab_property(self):
        """Test vocab cached property"""
        vocab = self.parser.vocab
        self.assertEqual(vocab, {"token1": 1, "token2": 2})
        self.mock_tokenizer.get_vocab.assert_called_once()
        
        # Test caching - should not call get_vocab again
        vocab2 = self.parser.vocab
        self.assertEqual(vocab, vocab2)
        self.mock_tokenizer.get_vocab.assert_called_once()  # Still called only once

    def test_adjust_request(self):
        """Test adjust_request method"""
        mock_request = MagicMock()
        result = self.parser.adjust_request(mock_request)
        self.assertEqual(result, mock_request)  # Default implementation returns request unchanged

    def test_extract_tool_calls_abstract(self):
        """Test extract_tool_calls abstract method"""
        # The base ToolParser should raise NotImplementedError
        base_parser = ToolParser(self.mock_tokenizer)
        
        with self.assertRaises(NotImplementedError) as context:
            base_parser.extract_tool_calls("test", MagicMock())
        
        self.assertIn("extract_tool_calls has not been implemented", str(context.exception))

    def test_extract_tool_calls_streaming_abstract(self):
        """Test extract_tool_calls_streaming abstract method"""
        # The base ToolParser should raise NotImplementedError
        base_parser = ToolParser(self.mock_tokenizer)
        
        with self.assertRaises(NotImplementedError) as context:
            base_parser.extract_tool_calls_streaming("", "", "", [], [], [], MagicMock())
        
        self.assertIn("extract_tool_calls_streaming has not been", str(context.exception))

    def test_mock_implementation_extract_tool_calls(self):
        """Test that mock implementation works"""
        result = self.parser.extract_tool_calls("test output", MagicMock())
        self.assertEqual(result["content"], "test output")
        self.assertEqual(result["tool_calls"], [])

    def test_mock_implementation_extract_tool_calls_streaming(self):
        """Test that mock implementation streaming works"""
        result = self.parser.extract_tool_calls_streaming(
            "prev", "current", "delta", [1], [1, 2], [2], MagicMock()
        )
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["content"], "delta")


class TestToolParserManager(unittest.TestCase):
    """Test case for ToolParserManager"""

    def setUp(self):
        """Set up test environment"""
        # Clear the tool_parsers dict for clean testing
        ToolParserManager.tool_parsers = {}

    def test_register_module_with_decorator(self):
        """Test registering module with decorator"""
        @ToolParserManager.register_module("test_parser")
        class TestParser(ToolParser):
            def extract_tool_calls(self, model_output, request):
                return {}
            
            def extract_tool_calls_streaming(self, *args):
                return {}
        
        # Verify registration
        self.assertIn("test_parser", ToolParserManager.tool_parsers)
        self.assertEqual(ToolParserManager.tool_parsers["test_parser"], TestParser)

    def test_register_module_directly(self):
        """Test registering module directly"""
        class DirectParser(ToolParser):
            def extract_tool_calls(self, model_output, request):
                return {}
            
            def extract_tool_calls_streaming(self, *args):
                return {}
        
        ToolParserManager.register_module("direct_parser", module=DirectParser)
        
        # Verify registration
        self.assertIn("direct_parser", ToolParserManager.tool_parsers)
        self.assertEqual(ToolParserManager.tool_parsers["direct_parser"], DirectParser)

    def test_register_module_multiple_names(self):
        """Test registering module with multiple names"""
        class MultiParser(ToolParser):
            def extract_tool_calls(self, model_output, request):
                return {}
            
            def extract_tool_calls_streaming(self, *args):
                return {}
        
        ToolParserManager.register_module(["multi1", "multi2"], module=MultiParser)
        
        # Verify both names are registered
        self.assertIn("multi1", ToolParserManager.tool_parsers)
        self.assertIn("multi2", ToolParserManager.tool_parsers)
        self.assertEqual(ToolParserManager.tool_parsers["multi1"], MultiParser)
        self.assertEqual(ToolParserManager.tool_parsers["multi2"], MultiParser)

    def test_register_module_auto_name(self):
        """Test registering module with automatic naming"""
        class AutoNameParser(ToolParser):
            def extract_tool_calls(self, model_output, request):
                return {}
            
            def extract_tool_calls_streaming(self, *args):
                return {}
        
        ToolParserManager.register_module(module=AutoNameParser)
        
        # Verify registration with class name
        self.assertIn("AutoNameParser", ToolParserManager.tool_parsers)

    def test_register_module_force_false(self):
        """Test registering module with force=False"""
        class FirstParser(ToolParser):
            def extract_tool_calls(self, model_output, request):
                return {}
            
            def extract_tool_calls_streaming(self, *args):
                return {}
        
        class SecondParser(ToolParser):
            def extract_tool_calls(self, model_output, request):
                return {}
            
            def extract_tool_calls_streaming(self, *args):
                return {}
        
        # Register first parser
        ToolParserManager.register_module("conflict_parser", module=FirstParser)
        
        # Try to register second parser with same name and force=False
        with self.assertRaises(KeyError) as context:
            ToolParserManager.register_module("conflict_parser", force=False, module=SecondParser)
        
        self.assertIn("conflict_parser is already registered", str(context.exception))

    def test_register_module_invalid_type(self):
        """Test registering invalid module type"""
        class NotToolParser:
            pass
        
        with self.assertRaises(TypeError) as context:
            ToolParserManager.register_module("invalid", module=NotToolParser)
        
        self.assertIn("module must be subclass of ToolParser", str(context.exception))

    def test_register_module_invalid_force_type(self):
        """Test registering with invalid force type"""
        class ValidParser(ToolParser):
            def extract_tool_calls(self, model_output, request):
                return {}
            
            def extract_tool_calls_streaming(self, *args):
                return {}
        
        with self.assertRaises(TypeError) as context:
            ToolParserManager.register_module("valid", force="not_bool", module=ValidParser)
        
        self.assertIn("force must be a boolean", str(context.exception))

    def test_register_module_invalid_name_type(self):
        """Test registering with invalid name type"""
        class ValidParser(ToolParser):
            def extract_tool_calls(self, model_output, request):
                return {}
            
            def extract_tool_calls_streaming(self, *args):
                return {}
        
        with self.assertRaises(TypeError) as context:
            ToolParserManager.register_module(123, module=ValidParser)
        
        self.assertIn("name must be None, an instance of str", str(context.exception))

    def test_get_tool_parser_success(self):
        """Test getting registered tool parser"""
        class GetParser(ToolParser):
            def extract_tool_calls(self, model_output, request):
                return {}
            
            def extract_tool_calls_streaming(self, *args):
                return {}
        
        ToolParserManager.register_module("get_test", module=GetParser)
        
        retrieved = ToolParserManager.get_tool_parser("get_test")
        self.assertEqual(retrieved, GetParser)

    def test_get_tool_parser_not_found(self):
        """Test getting non-existent tool parser"""
        with self.assertRaises(KeyError) as context:
            ToolParserManager.get_tool_parser("nonexistent")
        
        self.assertIn("tool helper: 'nonexistent' not found", str(context.exception))

    @patch('fastdeploy.entrypoints.openai.tool_parsers.abstract_tool_parser.import_from_path')
    def test_import_tool_parser_success(self, mock_import_from_path):
        """Test importing tool parser from path"""
        plugin_path = "/path/to/plugin.py"
        
        ToolParserManager.import_tool_parser(plugin_path)
        
        mock_import_from_path.assert_called_once_with("plugin", plugin_path)

    @patch('fastdeploy.entrypoints.openai.tool_parsers.abstract_tool_parser.import_from_path')
    @patch('fastdeploy.entrypoints.openai.tool_parsers.abstract_tool_parser.data_processor_logger')
    def test_import_tool_parser_failure(self, mock_logger, mock_import_from_path):
        """Test importing tool parser failure"""
        plugin_path = "/path/to/plugin.py"
        mock_import_from_path.side_effect = Exception("Import failed")
        
        ToolParserManager.import_tool_parser(plugin_path)
        
        mock_logger.exception.assert_called_once()
        call_args = mock_logger.exception.call_args[0]
        self.assertIn("Failed to load module", call_args[0])
        self.assertEqual(call_args[1], "plugin")
        self.assertEqual(call_args[2], plugin_path)

    def test_import_tool_parser_module_name_extraction(self):
        """Test module name extraction from path"""
        test_cases = [
            ("/path/to/file.py", "file"),
            ("simple_file.py", "simple_file"),
            ("/complex/path/with.dots.py", "with.dots"),
            ("no_extension", "no_extension")
        ]
        
        for path, expected_name in test_cases:
            actual_name = os.path.splitext(os.path.basename(path))[0]
            self.assertEqual(actual_name, expected_name)


if __name__ == "__main__":
    unittest.main()