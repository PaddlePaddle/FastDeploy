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

import importlib.util
import json
import os
import sys
import unittest
from unittest.mock import Mock, patch

sys.modules["torch"] = Mock()
sys.modules["xgrammar"] = Mock()
sys.modules["fastdeploy"] = Mock()
sys.modules["fastdeploy.config"] = Mock()
sys.modules["fastdeploy.engine"] = Mock()
sys.modules["fastdeploy.engine.request"] = Mock()
sys.modules["fastdeploy.model_executor"] = Mock()
sys.modules["fastdeploy.model_executor.guided_decoding"] = Mock()
sys.modules["fastdeploy.utils"] = Mock()
sys.modules["fastdeploy.platforms"] = Mock()


class MockGrammarMatcher:
    def __init__(self, compiled_grammar, **kwargs):
        self.fill_next_token_bitmask = Mock()
        self.reset = Mock()
        self.accept_token = Mock(return_value=True)
        self.is_terminated = Mock(return_value=False)


def mock_allocate_token_bitmask(batch_size, vocab_size):
    mock_tensor = Mock()
    mock_tensor.shape = (batch_size, vocab_size)
    mock_tensor.dtype = "int32"
    return mock_tensor


sys.modules["xgrammar"].CompiledGrammar = type("CompiledGrammar", (), {})
sys.modules["xgrammar"].Grammar = type(
    "Grammar",
    (),
    {
        "from_json_schema": staticmethod(Mock),
        "from_ebnf": staticmethod(Mock),
        "from_structural_tag": staticmethod(Mock),
    },
)
sys.modules["xgrammar"].GrammarMatcher = MockGrammarMatcher
sys.modules["xgrammar"].GrammarCompiler = Mock
sys.modules["xgrammar"].TokenizerInfo = Mock
sys.modules["xgrammar"].StructuralTagItem = Mock
sys.modules["xgrammar"].allocate_token_bitmask = mock_allocate_token_bitmask
sys.modules["xgrammar"].apply_token_bitmask_inplace = lambda logits, bitmask, indices=None: None


class MockLogitsProcessorBase:
    def __init__(self, enable_reasoning=False):
        self.enable_reasoning = enable_reasoning


class MockBackendBase:
    def __init__(self, fd_config=None):
        self.hf_tokenizer = Mock()


class MockBaseChecker:
    pass


sys.modules["fastdeploy.model_executor.guided_decoding"].LogitsProcessorBase = MockLogitsProcessorBase
sys.modules["fastdeploy.model_executor.guided_decoding"].BackendBase = MockBackendBase
sys.modules["fastdeploy.model_executor.guided_decoding"].BaseChecker = MockBaseChecker


class MockGrammarCompiler:
    def __init__(self, tokenizer_info, **kwargs):
        self.tokenizer_info = tokenizer_info

    def compile_json_schema(self, schema_str, **kwargs):
        return Mock()

    def compile_regex(self, pattern_str):
        return Mock()

    def compile_grammar(self, grammar_str):
        return Mock()

    def compile_structural_tag(self, tags, triggers):
        return Mock()


sys.modules["xgrammar"].GrammarCompiler = MockGrammarCompiler
sys.modules["xgrammar"].TokenizerInfo = type("TokenizerInfo", (), {"from_huggingface": staticmethod(Mock)})
sys.modules["xgrammar"].StructuralTagItem = lambda begin, schema, end: Mock(begin=begin, schema=schema, end=end)
sys.modules["fastdeploy.utils"].llm_logger = type("llm_logger", (), {"error": Mock, "info": Mock, "warning": Mock})()

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
        """Test processor initialization (covers lines 75-88)."""
        # Default parameters
        processor = XGrammarProcessor(compiled_grammar=self.mock_compiled_grammar)
        self.assertIsNone(processor.vocab_size)
        self.assertIsNone(processor.batch_size)
        self.assertFalse(processor.terminate_without_stop_token)
        self.assertIsNone(processor.override_stop_tokens)
        self.assertFalse(processor.enable_reasoning)

        # Custom parameters
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
        """Test token bitmask allocation (covers line 97)."""
        if self.processor.batch_size is None or self.processor.vocab_size is None:
            return
        bitmask = self.processor.allocate_token_bitmask()
        self.assertIsInstance(bitmask, Mock)
        self.assertEqual(bitmask.shape, (self.processor.batch_size, self.processor.vocab_size))

    def test_fill_token_bitmask(self):
        """Test filling token bitmask (covers line 110)."""
        mock_bitmask = Mock()
        self.processor.fill_token_bitmask(mock_bitmask, 2)
        self.processor.matcher.fill_next_token_bitmask.assert_called_once_with(mock_bitmask, 2)

    def test_reset(self):
        """Test resetting processor state (covers line 119)."""
        self.processor.reset()
        self.processor.matcher.reset.assert_called_once()

    def test_accept_token_when_already_terminated(self):
        """Test accept_token when already terminated (covers lines 130-132)."""
        self.processor.is_terminated = True
        result = self.processor.accept_token(42)
        self.assertFalse(result)
        self.processor.matcher.accept_token.assert_not_called()

    def test_accept_token_sets_terminated_flag(self):
        """Test accept_token sets terminated flag (covers lines 136-138)."""
        self.processor.matcher.accept_token.return_value = True
        self.processor.matcher.is_terminated.side_effect = [False, True]
        result = self.processor.accept_token(42)
        self.assertTrue(result)
        self.assertTrue(self.processor.is_terminated)

    def test_copy(self):
        """Test creating a copy of the processor (covers line 147)."""
        copied = self.processor.copy()
        self.assertIsNot(copied, self.processor)
        self.assertEqual(copied.compiled_grammar, self.processor.compiled_grammar)
        self.assertEqual(copied.terminate_without_stop_token, self.processor.terminate_without_stop_token)
        self.assertEqual(copied.override_stop_tokens, self.processor.override_stop_tokens)
        self.assertEqual(copied.vocab_size, self.processor.vocab_size)
        self.assertEqual(copied.batch_size, self.processor.batch_size)


class TestXGrammarBackend(unittest.TestCase):
    """Test cases for XGrammarBackend class."""

    def setUp(self):
        self.mock_fd_config = Mock()
        self.mock_fd_config.model_config.vocab_size = 1000
        self.mock_fd_config.scheduler_config.max_num_seqs = 4
        self.mock_fd_config.structured_outputs_config.disable_any_whitespace = False

        with patch("xgrammar.TokenizerInfo.from_huggingface"), patch("xgrammar.GrammarCompiler"):
            self.backend = XGrammarBackend(self.mock_fd_config)

    def _get_sample_structural_tag(self):
        return json.dumps(
            {
                "structures": [{"begin": "<tool>", "schema": {"type": "string"}, "end": "</tool>"}],
                "triggers": ["<tool>"],
            }
        )

    def test_backend_initialization(self):
        """Test backend initialization (covers lines 175-196)."""
        test_configs = [
            (1000, 4, False, True),
            (2000, 8, True, False),
        ]
        for vocab_size, max_num_seqs, disable_any_white, expected_any_white in test_configs:
            mock_fd_config = Mock()
            mock_fd_config.model_config.vocab_size = vocab_size
            mock_fd_config.scheduler_config.max_num_seqs = max_num_seqs
            mock_fd_config.structured_outputs_config.disable_any_whitespace = disable_any_white
            with patch("xgrammar.TokenizerInfo.from_huggingface"), patch("xgrammar.GrammarCompiler"):
                backend = XGrammarBackend(mock_fd_config)
                self.assertEqual(backend.vocab_size, vocab_size)
                self.assertEqual(backend.batch_size, max_num_seqs)
                self.assertEqual(backend.any_whitespace, expected_any_white)

    def test_backend_initialization_tokenizer_failure(self):
        """Test backend initialization with tokenizer failure (covers lines 175-196)."""
        mock_fd_config = Mock()
        mock_fd_config.model_config.vocab_size = 1000
        mock_fd_config.scheduler_config.max_num_seqs = 4
        mock_fd_config.structured_outputs_config.disable_any_whitespace = False
        with patch("xgrammar.TokenizerInfo.from_huggingface", side_effect=Exception("Tokenizer error")):
            with self.assertRaises(Exception) as context:
                XGrammarBackend(mock_fd_config)
            self.assertIn("Failed to load XGrammar tokenizer", str(context.exception))

    def test_create_processor(self):
        """Test creating a processor instance (covers line 217)."""
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

    def test_structural_tag_processor_parsing_and_tag_construction(self):
        """Test structural tag parsing (covers lines 290-305)."""
        complex_tag = json.dumps(
            {
                "structures": [
                    {"begin": "<tool>", "schema": {"type": "string"}, "end": "</tool>"},
                    {"begin": "<function>", "schema": {"type": "object"}, "end": "</function>"},
                ],
                "triggers": ["<tool>", "<function>"],
            }
        )
        with patch.object(self.backend.grammar_compiler, "compile_structural_tag") as mock_compile:
            mock_compile.return_value = Mock()
            processor = self.backend._structural_tag_processor(complex_tag)
            self.assertIsNotNone(processor)
            mock_compile.assert_called_once()

    def test_processor_error_logging(self):
        """Test processors log errors (covers lines 237-241, 255-259, 273-277)."""
        from fastdeploy.utils import llm_logger

        test_cases = [
            ("json", "compile_json_schema", "invalid schema", "_json_processor"),
            ("regex", "compile_regex", "[invalid", "_regex_processor"),
            ("grammar", "compile_grammar", "invalid", "_grammar_processor"),
        ]
        for name, compile_method, test_input, processor_method in test_cases:
            with self.subTest(name=name):
                with patch.object(self.backend.grammar_compiler, compile_method, side_effect=Exception("Failed")):
                    with patch.object(llm_logger, "error") as mock_logger:
                        processor = getattr(self.backend, processor_method)(test_input)
                        self.assertIsNone(processor)
                        mock_logger.assert_called_once()


class TestXGrammarChecker(unittest.TestCase):
    """Test cases for XGrammarChecker class."""

    def setUp(self):
        self.checker = XGrammarChecker(disable_any_whitespace=False)

    def _create_mock_request(self):
        request = Mock()
        request.guided_json = None
        request.guided_grammar = None
        request.guided_json_object = None
        request.guided_choice = None
        request.structural_tag = None
        return request

    def _get_sample_structural_tag(self):
        return json.dumps(
            {
                "structures": [{"begin": "<tool>", "schema": {"type": "string"}, "end": "</tool>"}],
                "triggers": ["<tool>"],
            }
        )

    def _test_schema_format_error(self, field_name, field_value, patch_target, error_substring):
        request = self._create_mock_request()
        setattr(request, field_name, field_value)
        with patch(patch_target, side_effect=RuntimeError("test error")):
            _, error = self.checker.schema_format(request)
            self.assertIsNotNone(error)
            self.assertIn(error_substring, error)

    def test_unsupported_json_schema(self):
        """Test detection of unsupported features in JSON schema (covers lines 339-375)."""
        test_schemas = [
            {"type": "object", "properties": {"age": {"type": "number", "multipleOf": 2}}},
            {"type": "array", "uniqueItems": True},
            {"type": "array", "contains": {"type": "number"}},
            {"type": "array", "minContains": 1},
            {"type": "array", "maxContains": 5},
            {"type": "string", "format": "date-time"},
            {"type": "object", "minProperties": 1},
            {"type": "object", "maxProperties": 10},
            {"type": "object", "propertyNames": {"type": "string"}},
            {"type": "object", "patternProperties": {"^S_": {"type": "string"}}},
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"value": {"type": "number", "multipleOf": 0.1}}},
                    }
                },
            },
        ]
        for schema in test_schemas:
            self.assertTrue(self.checker._unsupported_json_schema(schema), f"Failed for: {schema}")

    def test_schema_format_guided_json_success(self):
        """Test successful formatting of guided_json (covers lines 388, 391-393)."""
        request = self._create_mock_request()
        original_json = {"type": "object", "properties": {"name": {"type": "string"}}}
        request.guided_json = original_json
        with patch("xgrammar.Grammar.from_json_schema"):
            result_request, error = self.checker.schema_format(request)
            self.assertIsNone(error)
            expected_json = json.dumps(original_json)
            self.assertEqual(result_request.guided_json, expected_json)

    def test_schema_format_guided_json_invalid_format(self):
        """Test formatting of invalid guided_json (covers lines 391-393)."""
        self._test_schema_format_error(
            "guided_json", "invalid json", "xgrammar.Grammar.from_json_schema", "Invalid JSON format"
        )

    def test_schema_format_guided_json_unsupported_schema(self):
        """Test formatting of guided_json with unsupported schema (covers lines 396-397)."""
        request = self._create_mock_request()
        request.guided_json = {"type": "number", "multipleOf": 2}
        with patch("xgrammar.Grammar.from_json_schema"):
            with patch.object(self.checker, "_unsupported_json_schema", return_value=True):
                _, error = self.checker.schema_format(request)
                self.assertIsNotNone(error)
                self.assertIn("unsupported JSON schema", error)

    def test_schema_format_guided_grammar_invalid_format(self):
        """Test formatting of invalid guided_grammar (covers lines 406-408)."""
        self._test_schema_format_error(
            "guided_grammar", "invalid grammar", "xgrammar.Grammar.from_ebnf", "Invalid grammar format"
        )

    def test_schema_format_guided_json_object(self):
        """Test formatting of guided_json_object (covers lines 412-413)."""
        request = self._create_mock_request()
        request.guided_json_object = True
        result_request, error = self.checker.schema_format(request)
        self.assertIsNone(error)
        self.assertEqual(result_request.guided_json, '{"type": "object"}')

    def test_schema_format_structural_tag_success(self):
        """Test successful formatting of structural_tag (covers lines 439-440)."""
        request = self._create_mock_request()
        request.structural_tag = self._get_sample_structural_tag()
        with patch("xgrammar.Grammar.from_structural_tag"):
            _, error = self.checker.schema_format(request)
            self.assertIsNone(error)

    def test_schema_format_structural_tag_invalid_grammar(self):
        """Test formatting of structural_tag with invalid grammar (covers lines 438-440)."""
        self._test_schema_format_error(
            "structural_tag",
            self._get_sample_structural_tag(),
            "xgrammar.Grammar.from_structural_tag",
            "Invalid structural_tag format",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
