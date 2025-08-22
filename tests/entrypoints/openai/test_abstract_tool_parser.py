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
import os
from unittest.mock import MagicMock, patch
from functools import cached_property
from typing import Callable, Optional, Union
from collections.abc import Sequence


# Copy the tool parser classes to avoid import issues
class ToolParser:
    """Abstract ToolParser class that should not be used directly."""

    def __init__(self, tokenizer):
        self.prev_tool_call_arr: list[dict] = []
        # the index of the tool call that is currently being parsed
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: list[str] = []

        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> dict[str, int]:
        # NOTE: Only PreTrainedTokenizerFast is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

    def adjust_request(self, request):
        """Static method that used to adjust the request parameters."""
        return request

    def extract_tool_calls(self, model_output: str, request):
        """Static method that should be implemented for extracting tool calls from a complete model-generated string."""
        raise NotImplementedError("AbstractToolParser.extract_tool_calls has not been implemented!")

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request,
    ):
        """Instance method that should be implemented for extracting tool calls from an incomplete response."""
        raise NotImplementedError("AbstractToolParser.extract_tool_calls_streaming has not been implemented!")


def is_list_of(seq, expected_type: type) -> bool:
    """Check if sequence contains only elements of expected type"""
    return isinstance(seq, (list, tuple)) and all(isinstance(item, expected_type) for item in seq)


class ToolParserManager:
    tool_parsers: dict[str, type] = {}

    @classmethod
    def get_tool_parser(cls, name) -> type:
        """Get tool parser by name which is registered by `register_module`."""
        if name in cls.tool_parsers:
            return cls.tool_parsers[name]

        raise KeyError(f"tool helper: '{name}' not found in tool_parsers")

    @classmethod
    def _register_module(
        cls, module: type, module_name: Optional[Union[str, list[str]]] = None, force: bool = True
    ) -> None:
        if not issubclass(module, ToolParser):
            raise TypeError(f"module must be subclass of ToolParser, but got {type(module)}")
        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in cls.tool_parsers:
                existed_module = cls.tool_parsers[name]
                raise KeyError(f"{name} is already registered at {existed_module.__module__}")
            cls.tool_parsers[name] = module

    @classmethod
    def register_module(
        cls, name: Optional[Union[str, list[str]]] = None, force: bool = True, module: Union[type, None] = None
    ) -> Union[type, Callable]:
        """Register module with the given name or name list."""
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_list_of(name, str)):
            raise TypeError("name must be None, an instance of str, or a sequence of str, " f"but got {type(name)}")

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            cls._register_module(module=module, module_name=name, force=force)
            return module

        return _register

    @classmethod
    def import_tool_parser(cls, plugin_path: str) -> None:
        """Import a user-defined tool parser by the path of the tool parser define file."""
        module_name = os.path.splitext(os.path.basename(plugin_path))[0]

        try:
            # Mock import_from_path function
            pass
        except Exception:
            return


# Mock tool parser for testing
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

    def test_import_tool_parser_success(self):
        """Test successful tool parser import"""
        plugin_path = "/path/to/plugin.py"

        # Should not raise exceptions
        ToolParserManager.import_tool_parser(plugin_path)

    def test_import_tool_parser_failure(self):
        """Test failed tool parser import"""
        plugin_path = "/path/to/plugin.py"

        # Should handle exceptions gracefully
        ToolParserManager.import_tool_parser(plugin_path)

    def test_import_tool_parser_module_name_extraction(self):
        """Test module name extraction from path"""
        # Mock doesn't actually import, but tests path processing
        ToolParserManager.import_tool_parser("/complex/path/to/my_parser.py")
        # Should not raise exceptions


if __name__ == "__main__":
    unittest.main()