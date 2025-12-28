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

import json
import os
import sys
import unittest
from unittest.mock import Mock, patch


# Mock torch module for standalone testing
class MockTorch:
    class Tensor:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    @staticmethod
    def zeros(shape, dtype):
        return MockTorch.Tensor(shape, dtype)

    @staticmethod
    def from_numpy(array):
        return Mock()

    # Add torch data types
    int32 = "int32"


torch = MockTorch()

# Mock all the dependencies before importing target module
mock_xgrammar = Mock()
mock_config = Mock()
mock_request = Mock()
mock_base_classes = Mock()
mock_utils = Mock()

# Mock torch in sys.modules before importing the target module
sys.modules["torch"] = torch

# Import the target module in standalone mode
import importlib.util

# Setup mock module structure
sys.modules["xgrammar"] = mock_xgrammar
sys.modules["fastdeploy"] = Mock()
sys.modules["fastdeploy.config"] = mock_config
sys.modules["fastdeploy.engine"] = Mock()
sys.modules["fastdeploy.engine.request"] = mock_request
sys.modules["fastdeploy.model_executor"] = Mock()
sys.modules["fastdeploy.model_executor.guided_decoding"] = mock_base_classes
sys.modules["fastdeploy.utils"] = mock_utils
sys.modules["fastdeploy.platforms"] = Mock()
sys.modules["fastdeploy.platforms"].current_platform = Mock()


# Create mock classes
class MockCompiledGrammar:
    pass


class MockGrammar:
    @staticmethod
    def from_json_schema(schema_str, any_whitespace=True):
        return Mock()

    @staticmethod
    def from_ebnf(grammar_str):
        return Mock()

    @staticmethod
    def from_structural_tag(tags, triggers):
        return Mock()


class MockGrammarMatcher:
    def __init__(
        self,
        compiled_grammar,
        max_rollback_tokens=200,
        terminate_without_stop_token=False,
        override_stop_tokens=None,
    ):
        self.max_rollback_tokens = max_rollback_tokens
        self.terminate_without_stop_token = terminate_without_stop_token
        self.override_stop_tokens = override_stop_tokens
        # Make methods mockable by replacing them with Mock objects
        self.fill_next_token_bitmask = Mock()
        self.reset = Mock()
        self.accept_token = Mock(return_value=True)
        self.is_terminated = Mock(return_value=False)


class MockGrammarCompiler:
    def __init__(self, tokenizer_info, max_threads=8, cache_enabled=True, cache_limit_bytes=4 * 1024 * 1024):
        self.tokenizer_info = tokenizer_info
        self.max_threads = max_threads
        self.cache_enabled = cache_enabled
        self.cache_limit_bytes = cache_limit_bytes

    def compile_json_schema(self, schema_str, any_whitespace=True):
        return MockCompiledGrammar()

    def compile_regex(self, pattern_str):
        return MockCompiledGrammar()

    def compile_grammar(self, grammar_str):
        return MockCompiledGrammar()

    def compile_structural_tag(self, tags, triggers):
        return MockCompiledGrammar()


class MockTokenizerInfo:
    @staticmethod
    def from_huggingface(tokenizer, vocab_size=None):
        return Mock()


class MockStructuralTagItem:
    def __init__(self, begin, schema, end):
        self.begin = begin
        self.schema = schema
        self.end = end


def mock_allocate_token_bitmask(batch_size, vocab_size):
    return torch.zeros((batch_size, vocab_size), dtype=torch.int32)


def mock_apply_token_bitmask_inplace(logits, bitmask, indices=None):
    pass


# Setup mock module
mock_xgrammar.CompiledGrammar = MockCompiledGrammar
mock_xgrammar.Grammar = MockGrammar
mock_xgrammar.GrammarMatcher = MockGrammarMatcher
mock_xgrammar.GrammarCompiler = MockGrammarCompiler
mock_xgrammar.TokenizerInfo = MockTokenizerInfo
mock_xgrammar.StructuralTagItem = MockStructuralTagItem
mock_xgrammar.allocate_token_bitmask = mock_allocate_token_bitmask
mock_xgrammar.apply_token_bitmask_inplace = mock_apply_token_bitmask_inplace


# Create mock base classes
class MockLogitsProcessorBase:
    def __init__(self, enable_reasoning=False):
        self.enable_reasoning = enable_reasoning


class MockBackendBase:
    def __init__(self, fd_config=None):
        self.fd_config = fd_config
        self.hf_tokenizer = Mock()


class MockBaseChecker:
    def __init__(self):
        pass


class MockRequest:
    def __init__(self):
        self.guided_json = None
        self.guided_grammar = None
        self.guided_json_object = None
        self.guided_choice = None
        self.structural_tag = None


mock_base_classes.LogitsProcessorBase = MockLogitsProcessorBase
mock_base_classes.BackendBase = MockBackendBase
mock_base_classes.BaseChecker = MockBaseChecker

# Mock config and request
mock_request.Request = MockRequest
mock_config.FDConfig = Mock
mock_utils.llm_logger = Mock()

# Import the target module
spec = importlib.util.spec_from_file_location(
    "xgrammar_backend",
    os.path.join(os.path.dirname(__file__), "../../../fastdeploy/model_executor/guided_decoding/xgrammar_backend.py"),
)
xgrammar_backend = importlib.util.module_from_spec(spec)
sys.modules["fastdeploy.model_executor.guided_decoding.xgrammar_backend"] = xgrammar_backend
spec.loader.exec_module(xgrammar_backend)

XGrammarProcessor = xgrammar_backend.XGrammarProcessor
XGrammarBackend = xgrammar_backend.XGrammarBackend
XGrammarChecker = xgrammar_backend.XGrammarChecker


class TestXGrammarProcessor(unittest.TestCase):
    """Test cases for XGrammarProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_compiled_grammar = Mock()
        self.processor = XGrammarProcessor(
            compiled_grammar=self.mock_compiled_grammar,
            terminate_without_stop_token=False,
            override_stop_tokens=None,
            vocab_size=1000,
            batch_size=4,
            enable_thinking=False,
        )

    def test_processor_initialization(self):
        """Test processor initialization with various parameters."""
        # Test default parameters
        processor = XGrammarProcessor(compiled_grammar=self.mock_compiled_grammar)
        self.assertIsNone(processor.vocab_size)
        self.assertIsNone(processor.batch_size)
        self.assertFalse(processor.terminate_without_stop_token)
        self.assertIsNone(processor.override_stop_tokens)
        self.assertFalse(processor.enable_reasoning)

        # Test custom parameters
        processor = XGrammarProcessor(
            compiled_grammar=self.mock_compiled_grammar,
            terminate_without_stop_token=True,
            override_stop_tokens=[1, 2, 3],
            vocab_size=2000,
            batch_size=8,
            enable_thinking=True,
        )
        self.assertTrue(processor.terminate_without_stop_token)
        self.assertEqual(processor.override_stop_tokens, [1, 2, 3])
        self.assertEqual(processor.vocab_size, 2000)
        self.assertEqual(processor.batch_size, 8)
        self.assertTrue(processor.enable_reasoning)

    def test_allocate_token_bitmask(self):
        """Test token bitmask allocation."""
        if self.processor.batch_size is None or self.processor.vocab_size is None:
            # Skip test if batch_size or vocab_size is None
            return

        bitmask = self.processor.allocate_token_bitmask()

        # Should return a torch tensor
        self.assertIsInstance(bitmask, torch.Tensor)
        # Should have correct shape
        self.assertEqual(bitmask.shape, (self.processor.batch_size, self.processor.vocab_size))

    def test_fill_token_bitmask(self):
        """Test filling token bitmask."""
        mock_bitmask = Mock()
        idx = 2

        # Should call matcher's fill_next_token_bitmask method
        self.processor.fill_token_bitmask(mock_bitmask, idx)
        self.processor.matcher.fill_next_token_bitmask.assert_called_once_with(mock_bitmask, idx)

    def test_reset(self):
        """Test resetting processor state."""
        self.processor.reset()
        self.processor.matcher.reset.assert_called_once()

    def test_accept_token(self):
        """Test accepting a token."""
        token = 42

        # Mock matcher to return True
        self.processor.matcher.accept_token.return_value = True

        # Should not raise assertion
        self.processor.accept_token(token)
        self.processor.matcher.accept_token.assert_called_once_with(token)

    def test_accept_token_failure(self):
        """Test accepting a token that fails validation."""
        token = 999

        # Mock matcher to return False
        self.processor.matcher.accept_token.return_value = False

        # The actual implementation returns False instead of raising AssertionError
        result = self.processor.accept_token(token)
        self.assertFalse(result)
        self.processor.matcher.accept_token.assert_called_once_with(token)
        # Should reset matcher on failure
        self.processor.matcher.reset.assert_called_once()

    def test_is_terminated(self):
        """Test checking if processor is terminated."""
        # The processor's is_terminated attribute should reflect the matcher's state

        # Initially should be False (set in __init__)
        self.assertFalse(self.processor.is_terminated)

        # Set matcher to return True
        self.processor.matcher.is_terminated.return_value = True

        # Call accept_token to trigger the check
        self.processor.accept_token(42)

        # Now should be True
        self.assertTrue(self.processor.is_terminated)

    def test_copy(self):
        """Test creating a copy of the processor."""
        copied_processor = self.processor.copy()

        # Should be a new instance
        self.assertIsNot(copied_processor, self.processor)
        # Should have same attributes
        self.assertEqual(copied_processor.compiled_grammar, self.processor.compiled_grammar)
        self.assertEqual(copied_processor.terminate_without_stop_token, self.processor.terminate_without_stop_token)
        self.assertEqual(copied_processor.override_stop_tokens, self.processor.override_stop_tokens)
        self.assertEqual(copied_processor.vocab_size, self.processor.vocab_size)
        self.assertEqual(copied_processor.batch_size, self.processor.batch_size)


class TestXGrammarBackend(unittest.TestCase):
    """Test cases for XGrammarBackend class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_fd_config = Mock()
        self.mock_fd_config.model_config.vocab_size = 1000
        self.mock_fd_config.scheduler_config.max_num_seqs = 4
        self.mock_fd_config.structured_outputs_config.disable_any_whitespace = False

        with patch("xgrammar.TokenizerInfo.from_huggingface"), patch("xgrammar.GrammarCompiler"):
            self.backend = XGrammarBackend(self.mock_fd_config)

    def test_backend_initialization(self):
        """Test backend initialization."""
        self.assertEqual(self.backend.vocab_size, 1000)
        self.assertEqual(self.backend.batch_size, 4)
        self.assertTrue(self.backend.any_whitespace)  # disable_any_whitespace is False

        # Test with disable_any_whitespace=True
        mock_fd_config = Mock()
        mock_fd_config.model_config.vocab_size = 2000
        mock_fd_config.scheduler_config.max_num_seqs = 8
        mock_fd_config.structured_outputs_config.disable_any_whitespace = True

        with patch("xgrammar.TokenizerInfo.from_huggingface"), patch("xgrammar.GrammarCompiler"):
            backend = XGrammarBackend(mock_fd_config)
            self.assertFalse(backend.any_whitespace)

    def test_backend_initialization_tokenizer_failure(self):
        """Test backend initialization with tokenizer failure."""
        mock_fd_config = Mock()
        mock_fd_config.model_config.vocab_size = 1000
        mock_fd_config.scheduler_config.max_num_seqs = 4
        mock_fd_config.structured_outputs_config.disable_any_whitespace = False

        with patch("xgrammar.TokenizerInfo.from_huggingface", side_effect=Exception("Tokenizer error")):
            with self.assertRaises(Exception) as context:
                XGrammarBackend(mock_fd_config)

            self.assertIn("Failed to load XGrammar tokenizer", str(context.exception))

    def test_create_processor(self):
        """Test creating a processor instance."""
        mock_compiled_grammar = Mock()

        processor = self.backend._create_processor(
            compiled_grammar=mock_compiled_grammar,
            terminate_without_stop_token=True,
            override_stop_tokens=[1, 2, 3],
            enable_thinking=True,
        )

        self.assertIsInstance(processor, XGrammarProcessor)
        self.assertEqual(processor.compiled_grammar, mock_compiled_grammar)
        self.assertTrue(processor.terminate_without_stop_token)
        self.assertEqual(processor.override_stop_tokens, [1, 2, 3])
        self.assertEqual(processor.vocab_size, self.backend.vocab_size)
        self.assertEqual(processor.batch_size, self.backend.batch_size)
        self.assertTrue(processor.enable_reasoning)

    def test_json_processor_success(self):
        """Test successful JSON processor creation."""
        schema = '{"type": "object", "properties": {"name": {"type": "string"}}}'

        with patch.object(self.backend.grammar_compiler, "compile_json_schema") as mock_compile:
            mock_compiled_grammar = Mock()
            mock_compile.return_value = mock_compiled_grammar

            processor = self.backend._json_processor(schema)

            self.assertIsNotNone(processor)
            mock_compile.assert_called_once_with(schema, any_whitespace=self.backend.any_whitespace)

    def test_json_processor_failure(self):
        """Test JSON processor creation with compilation failure."""
        schema = "invalid schema"

        with patch.object(
            self.backend.grammar_compiler, "compile_json_schema", side_effect=Exception("Compilation error")
        ) as mock_compile:
            processor = self.backend._json_processor(schema)

            self.assertIsNone(processor)
            mock_compile.assert_called_once_with(schema, any_whitespace=self.backend.any_whitespace)

    def test_regex_processor_success(self):
        """Test successful regex processor creation."""
        pattern = r"[a-z]+"

        with patch.object(self.backend.grammar_compiler, "compile_regex") as mock_compile:
            mock_compiled_grammar = Mock()
            mock_compile.return_value = mock_compiled_grammar

            processor = self.backend._regex_processor(pattern)

            self.assertIsNotNone(processor)
            mock_compile.assert_called_once_with(pattern)

    def test_regex_processor_failure(self):
        """Test regex processor creation with compilation failure."""
        pattern = "[invalid regex"

        with patch.object(
            self.backend.grammar_compiler, "compile_regex", side_effect=Exception("Compilation error")
        ) as mock_compile:
            processor = self.backend._regex_processor(pattern)

            self.assertIsNone(processor)
            mock_compile.assert_called_once_with(pattern)

    def test_grammar_processor_success(self):
        """Test successful grammar processor creation."""
        grammar = 'root ::= "hello" "world"'

        with patch.object(self.backend.grammar_compiler, "compile_grammar") as mock_compile:
            mock_compiled_grammar = Mock()
            mock_compile.return_value = mock_compiled_grammar

            processor = self.backend._grammar_processor(grammar)

            self.assertIsNotNone(processor)
            mock_compile.assert_called_once_with(grammar)

    def test_grammar_processor_failure(self):
        """Test grammar processor creation with compilation failure."""
        grammar = "invalid grammar"

        with patch.object(
            self.backend.grammar_compiler, "compile_grammar", side_effect=Exception("Compilation error")
        ) as mock_compile:
            processor = self.backend._grammar_processor(grammar)

            self.assertIsNone(processor)
            mock_compile.assert_called_once_with(grammar)

    def test_structural_tag_processor_success(self):
        """Test successful structural tag processor creation."""
        structural_tag = json.dumps(
            {
                "structures": [{"begin": "<tool>", "schema": {"type": "string"}, "end": "</tool>"}],
                "triggers": ["<tool>"],
            }
        )

        with patch.object(self.backend.grammar_compiler, "compile_structural_tag") as mock_compile:
            mock_compiled_grammar = Mock()
            mock_compile.return_value = mock_compiled_grammar

            processor = self.backend._structural_tag_processor(structural_tag)

            self.assertIsNotNone(processor)
            mock_compile.assert_called_once()

    def test_structural_tag_processor_invalid_json(self):
        """Test structural tag processor with invalid JSON."""
        invalid_structural_tag = "invalid json"

        # The actual implementation catches all exceptions and returns None
        processor = self.backend._structural_tag_processor(invalid_structural_tag)
        self.assertIsNone(processor)

    def test_structural_tag_processor_compilation_failure(self):
        """Test structural tag processor with compilation failure."""
        structural_tag = json.dumps(
            {
                "structures": [{"begin": "<tool>", "schema": {"type": "string"}, "end": "</tool>"}],
                "triggers": ["<tool>"],
            }
        )

        with patch.object(
            self.backend.grammar_compiler, "compile_structural_tag", side_effect=Exception("Compilation error")
        ) as mock_compile:
            processor = self.backend._structural_tag_processor(structural_tag)

            self.assertIsNone(processor)
            mock_compile.assert_called_once()


class TestXGrammarChecker(unittest.TestCase):
    """Test cases for XGrammarChecker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.checker = XGrammarChecker(disable_any_whitespace=False)

    def test_checker_initialization(self):
        """Test checker initialization."""
        # Test default (disable_any_whitespace=True)
        checker1 = XGrammarChecker()
        self.assertFalse(checker1.any_whitespace)

        # Test with disable_any_whitespace=False
        checker2 = XGrammarChecker(disable_any_whitespace=False)
        self.assertTrue(checker2.any_whitespace)

        # Test with disable_any_whitespace=True
        checker3 = XGrammarChecker(disable_any_whitespace=True)
        self.assertFalse(checker3.any_whitespace)

    def test_unsupported_json_schema_with_multipleOf(self):
        """Test detection of unsupported multipleOf in JSON schema."""
        schema = {"type": "object", "properties": {"age": {"type": "number", "multipleOf": 2}}}

        self.assertTrue(self.checker._unsupported_json_schema(schema))

    def test_unsupported_json_schema_with_array_features(self):
        """Test detection of unsupported array features in JSON schema."""
        # Test uniqueItems
        schema1 = {"type": "array", "uniqueItems": True}
        self.assertTrue(self.checker._unsupported_json_schema(schema1))

        # Test contains
        schema2 = {"type": "array", "contains": {"type": "number"}}
        self.assertTrue(self.checker._unsupported_json_schema(schema2))

        # Test minContains
        schema3 = {"type": "array", "minContains": 1}
        self.assertTrue(self.checker._unsupported_json_schema(schema3))

        # Test maxContains
        schema4 = {"type": "array", "maxContains": 5}
        self.assertTrue(self.checker._unsupported_json_schema(schema4))

    def test_unsupported_json_schema_with_string_format(self):
        """Test detection of unsupported format in JSON schema."""
        schema = {"type": "string", "format": "date-time"}

        self.assertTrue(self.checker._unsupported_json_schema(schema))

    def test_unsupported_json_schema_with_object_features(self):
        """Test detection of unsupported object features in JSON schema."""
        # Test minProperties
        schema1 = {"type": "object", "minProperties": 1}
        self.assertTrue(self.checker._unsupported_json_schema(schema1))

        # Test maxProperties
        schema2 = {"type": "object", "maxProperties": 10}
        self.assertTrue(self.checker._unsupported_json_schema(schema2))

        # Test propertyNames
        schema3 = {"type": "object", "propertyNames": {"type": "string"}}
        self.assertTrue(self.checker._unsupported_json_schema(schema3))

        # Test patternProperties
        schema4 = {"type": "object", "patternProperties": {"^S_": {"type": "string"}}}
        self.assertTrue(self.checker._unsupported_json_schema(schema4))

    def test_unsupported_json_schema_nested(self):
        """Test detection of unsupported features in nested JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"value": {"type": "number", "multipleOf": 0.1}}},
                }
            },
        }

        self.assertTrue(self.checker._unsupported_json_schema(schema))

    def test_supported_json_schema(self):
        """Test that supported JSON schema is not flagged as unsupported."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {"street": {"type": "string"}, "city": {"type": "string"}},
                },
            },
        }

        self.assertFalse(self.checker._unsupported_json_schema(schema))

    def test_schema_format_guided_json_success(self):
        """Test successful formatting of guided_json."""
        request = Mock()
        original_json = {"type": "object", "properties": {"name": {"type": "string"}}}
        request.guided_json = original_json
        request.guided_grammar = None
        request.guided_json_object = None
        request.guided_choice = None
        request.structural_tag = None

        with patch("xgrammar.Grammar.from_json_schema"):
            result_request, error = self.checker.schema_format(request)

            self.assertIsNone(error)
            # The implementation converts JSON object to string
            expected_json = json.dumps(original_json)
            self.assertEqual(result_request.guided_json, expected_json)

    def test_schema_format_guided_json_invalid_format(self):
        """Test formatting of invalid guided_json."""
        request = Mock()
        request.guided_json = "invalid json"
        request.guided_grammar = None
        request.guided_json_object = None
        request.guided_choice = None
        request.structural_tag = None

        with patch("xgrammar.Grammar.from_json_schema", side_effect=RuntimeError("Invalid JSON")):
            result_request, error = self.checker.schema_format(request)

            self.assertIsNotNone(error)
            self.assertIn("Invalid JSON format", error)

    def test_schema_format_guided_json_unsupported_schema(self):
        """Test formatting of guided_json with unsupported schema."""
        request = Mock()
        unsupported_json = {"type": "number", "multipleOf": 2}
        request.guided_json = unsupported_json
        request.guided_grammar = None
        request.guided_json_object = None
        request.guided_choice = None
        request.structural_tag = None

        # Mock the _unsupported_json_schema method to return True for our test
        with patch("xgrammar.Grammar.from_json_schema"):
            with patch.object(self.checker, "_unsupported_json_schema", return_value=True):
                result_request, error = self.checker.schema_format(request)

                self.assertIsNotNone(error)
                self.assertIn("unsupported JSON schema", error)

    def test_schema_format_guided_grammar_success(self):
        """Test successful formatting of guided_grammar."""
        request = Mock()
        request.guided_json = None
        request.guided_grammar = 'root ::= "hello" "world"'
        request.guided_json_object = None
        request.guided_choice = None
        request.structural_tag = None

        with patch("xgrammar.Grammar.from_ebnf"):
            result_request, error = self.checker.schema_format(request)

            self.assertIsNone(error)
            self.assertEqual(result_request.guided_grammar, request.guided_grammar)

    def test_schema_format_guided_grammar_invalid_format(self):
        """Test formatting of invalid guided_grammar."""
        request = Mock()
        request.guided_json = None
        request.guided_grammar = "invalid grammar"
        request.guided_json_object = None
        request.guided_choice = None
        request.structural_tag = None

        with patch("xgrammar.Grammar.from_ebnf", side_effect=RuntimeError("Invalid grammar")):
            result_request, error = self.checker.schema_format(request)

            self.assertIsNotNone(error)
            self.assertIn("Invalid grammar format", error)

    def test_schema_format_guided_json_object(self):
        """Test formatting of guided_json_object."""
        request = Mock()
        request.guided_json = None
        request.guided_grammar = None
        request.guided_json_object = True
        request.guided_choice = None
        request.structural_tag = None

        result_request, error = self.checker.schema_format(request)

        self.assertIsNone(error)
        self.assertEqual(result_request.guided_json, '{"type": "object"}')

    def test_schema_format_guided_choice_success(self):
        """Test successful formatting of guided_choice."""
        request = Mock()
        request.guided_json = None
        request.guided_grammar = None
        request.guided_json_object = None
        request.guided_choice = ["hello", "world", "test"]
        request.structural_tag = None

        with patch("xgrammar.Grammar.from_ebnf"):
            result_request, error = self.checker.schema_format(request)

            self.assertIsNone(error)
            self.assertIsNotNone(result_request.guided_grammar)
            self.assertIn("hello", result_request.guided_grammar)
            self.assertIn("world", result_request.guided_grammar)
            self.assertIn("test", result_request.guided_grammar)

    def test_schema_format_guided_choice_invalid_format(self):
        """Test formatting that results in invalid guided_choice grammar."""
        request = Mock()
        request.guided_json = None
        request.guided_grammar = None
        request.guided_json_object = None
        request.guided_choice = ["hello", "world"]
        request.structural_tag = None

        with patch("xgrammar.Grammar.from_ebnf", side_effect=RuntimeError("Invalid choice")):
            result_request, error = self.checker.schema_format(request)

            self.assertIsNotNone(error)
            self.assertIn("Invalid choice format", error)

    def test_schema_format_structural_tag_success(self):
        """Test successful formatting of structural_tag."""
        structural_tag_data = {
            "structures": [{"begin": "<tool>", "schema": {"type": "string"}, "end": "</tool>"}],
            "triggers": ["<tool>"],
        }

        request = Mock()
        request.guided_json = None
        request.guided_grammar = None
        request.guided_json_object = None
        request.guided_choice = None
        request.structural_tag = json.dumps(structural_tag_data)

        with patch("xgrammar.Grammar.from_structural_tag"):
            result_request, error = self.checker.schema_format(request)

            self.assertIsNone(error)

    def test_schema_format_structural_tag_invalid_json(self):
        """Test formatting of invalid structural_tag JSON."""
        request = Mock()
        request.guided_json = None
        request.guided_grammar = None
        request.guided_json_object = None
        request.guided_choice = None
        request.structural_tag = "invalid json"

        # The actual implementation doesn't catch JSONDecodeError, so we expect it to raise
        with self.assertRaises(json.JSONDecodeError):
            self.checker.schema_format(request)

    def test_schema_format_structural_tag_invalid_grammar(self):
        """Test formatting of structural_tag with invalid grammar."""
        structural_tag_data = {
            "structures": [{"begin": "<tool>", "schema": {"type": "string"}, "end": "</tool>"}],
            "triggers": ["<tool>"],
        }

        request = Mock()
        request.guided_json = None
        request.guided_grammar = None
        request.guided_json_object = None
        request.guided_choice = None
        request.structural_tag = json.dumps(structural_tag_data)

        with patch("xgrammar.Grammar.from_structural_tag", side_effect=RuntimeError("Invalid structural tag")):
            result_request, error = self.checker.schema_format(request)

            self.assertIsNotNone(error)
            self.assertIn("Invalid structural_tag format", error)

    def test_schema_format_regex_passthrough(self):
        """Test that regex requests are passed through unchanged."""
        request = Mock()
        request.guided_json = None
        request.guided_grammar = None
        request.guided_json_object = None
        request.guided_choice = None
        request.structural_tag = None
        # Note: regex is handled separately, not through schema_format

        result_request, error = self.checker.schema_format(request)

        self.assertIsNone(error)
        # Request should be unchanged
        self.assertEqual(result_request, request)


if __name__ == "__main__":
    unittest.main(verbosity=2)
