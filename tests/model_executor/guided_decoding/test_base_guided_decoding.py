"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from fastdeploy.model_executor.guided_decoding.base_guided_decoding import (
    BackendBase,
    LogitsProcessorBase,
)


class TestLogitsProcessorBase(unittest.TestCase):
    """Test LogitsProcessorBase class."""

    def test_init_with_reasoning_disabled(self):
        """__init__ with enable_reasoning=False."""
        proc = LogitsProcessorBase(enable_reasoning=False)
        self.assertFalse(proc.reasoning_ended)
        self.assertFalse(proc.enable_reasoning)

    def test_init_with_reasoning_enabled(self):
        """__init__ with enable_reasoning=True."""
        proc = LogitsProcessorBase(enable_reasoning=True)
        self.assertFalse(proc.reasoning_ended)
        self.assertTrue(proc.enable_reasoning)

    def test_fill_token_bitmask_not_implemented(self):
        """fill_token_bitmask raises NotImplementedError."""
        proc = LogitsProcessorBase(enable_reasoning=False)
        with self.assertRaises(NotImplementedError):
            proc.fill_token_bitmask(None, None)

    def test_apply_token_mask_not_implemented(self):
        """apply_token_mask raises NotImplementedError."""
        proc = LogitsProcessorBase(enable_reasoning=False)
        with self.assertRaises(NotImplementedError):
            proc.apply_token_mask(None, None)

    def test_allocate_token_bitmask_not_implemented(self):
        """allocate_token_bitmask raises NotImplementedError."""
        proc = LogitsProcessorBase(enable_reasoning=False)
        with self.assertRaises(NotImplementedError):
            proc.allocate_token_bitmask(1, 32000)

    def test_accept_token_not_implemented(self):
        """accept_token raises NotImplementedError."""
        proc = LogitsProcessorBase(enable_reasoning=False)
        with self.assertRaises(NotImplementedError):
            proc.accept_token(42)

    def test_is_terminated_not_implemented(self):
        """is_terminated raises NotImplementedError."""
        proc = LogitsProcessorBase(enable_reasoning=False)
        with self.assertRaises(NotImplementedError):
            proc.is_terminated()

    def test_reset_not_implemented(self):
        """reset raises NotImplementedError."""
        proc = LogitsProcessorBase(enable_reasoning=False)
        with self.assertRaises(NotImplementedError):
            proc.reset()

    def test_copy_not_implemented(self):
        """copy raises NotImplementedError."""
        proc = LogitsProcessorBase(enable_reasoning=False)
        with self.assertRaises(NotImplementedError):
            proc.copy()


class TestBackendBaseInit(unittest.TestCase):
    """Test BackendBase.__init__."""

    @patch("fastdeploy.model_executor.guided_decoding.base_guided_decoding.ReasoningParserManager")
    @patch.object(BackendBase, "_get_tokenizer_hf")
    def test_init_without_reasoning_parser(self, mock_get_tokenizer, mock_parser_mgr):
        """__init__ without reasoning_parser configured."""
        mock_get_tokenizer.return_value = MagicMock()

        fd_config = MagicMock()
        fd_config.structured_outputs_config.reasoning_parser = None

        backend = BackendBase(fd_config)

        self.assertIs(backend.fd_config, fd_config)
        self.assertIsNotNone(backend.executor)
        self.assertEqual(backend.max_cache_size, 2048)
        self.assertIsNone(backend.reasoning_parser)
        mock_parser_mgr.get_reasoning_parser.assert_not_called()

    @patch("fastdeploy.model_executor.guided_decoding.base_guided_decoding.ReasoningParserManager")
    @patch.object(BackendBase, "_get_tokenizer_hf")
    def test_init_with_reasoning_parser(self, mock_get_tokenizer, mock_parser_mgr):
        """__init__ with reasoning_parser configured creates parser instance."""
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer

        mock_parser_cls = MagicMock()
        mock_parser_instance = MagicMock()
        mock_parser_cls.return_value = mock_parser_instance
        mock_parser_mgr.get_reasoning_parser.return_value = mock_parser_cls

        fd_config = MagicMock()
        fd_config.structured_outputs_config.reasoning_parser = "deepseek_r1"

        backend = BackendBase(fd_config)

        mock_parser_mgr.get_reasoning_parser.assert_called_once_with("deepseek_r1")
        mock_parser_cls.assert_called_once_with(mock_tokenizer)
        self.assertIs(backend.reasoning_parser, mock_parser_instance)


class TestBackendBaseUnsupportedProcessorType(unittest.TestCase):
    """Test BackendBase._unsupported_processor_type."""

    @patch.object(BackendBase, "_get_tokenizer_hf", return_value=MagicMock())
    def test_raises_exception(self, mock_tokenizer):
        """_unsupported_processor_type raises Exception."""
        fd_config = MagicMock()
        fd_config.structured_outputs_config.reasoning_parser = None

        backend = BackendBase(fd_config)

        with self.assertRaises(Exception) as ctx:
            backend._unsupported_processor_type("unknown_type", "{}", False)
        self.assertIn("Unsupported processor type unknown_type", str(ctx.exception))


class TestBackendBaseGetReasoningParser(unittest.TestCase):
    """Test BackendBase.get_reasoning_parser."""

    @patch.object(BackendBase, "_get_tokenizer_hf", return_value=MagicMock())
    def test_returns_none_when_not_configured(self, mock_tokenizer):
        """get_reasoning_parser returns None when not configured."""
        fd_config = MagicMock()
        fd_config.structured_outputs_config.reasoning_parser = None

        backend = BackendBase(fd_config)

        self.assertIsNone(backend.get_reasoning_parser())

    @patch("fastdeploy.model_executor.guided_decoding.base_guided_decoding.ReasoningParserManager")
    @patch.object(BackendBase, "_get_tokenizer_hf", return_value=MagicMock())
    def test_returns_parser_when_configured(self, mock_tokenizer, mock_parser_mgr):
        """get_reasoning_parser returns parser when configured."""
        mock_parser_cls = MagicMock()
        mock_parser_instance = MagicMock()
        mock_parser_cls.return_value = mock_parser_instance
        mock_parser_mgr.get_reasoning_parser.return_value = mock_parser_cls

        fd_config = MagicMock()
        fd_config.structured_outputs_config.reasoning_parser = "deepseek_r1"

        backend = BackendBase(fd_config)

        self.assertIs(backend.get_reasoning_parser(), mock_parser_instance)


class TestBackendBaseInitLogitsProcessor(unittest.TestCase):
    """Test BackendBase._init_logits_processor."""

    def _make_backend(self):
        """Create a BackendBase with mocked init."""
        with patch.object(BackendBase, "_get_tokenizer_hf", return_value=MagicMock()):
            fd_config = MagicMock()
            fd_config.structured_outputs_config.reasoning_parser = None
            return BackendBase(fd_config)

    @patch.object(BackendBase, "_json_processor", return_value="json_proc")
    def test_json_type(self, mock_json):
        """_init_logits_processor routes 'json' type correctly."""
        backend = self._make_backend()
        result = backend._init_logits_processor(("json", '{"type": "object"}'), enable_thinking=True)

        mock_json.assert_called_once_with('{"type": "object"}', True)
        self.assertEqual(result, "json_proc")

    @patch.object(BackendBase, "_regex_processor", return_value="regex_proc")
    def test_regex_type(self, mock_regex):
        """_init_logits_processor routes 'regex' type correctly."""
        backend = self._make_backend()
        result = backend._init_logits_processor(("regex", "[0-9]+"))

        mock_regex.assert_called_once_with("[0-9]+", False)
        self.assertEqual(result, "regex_proc")

    @patch.object(BackendBase, "_grammar_processor", return_value="grammar_proc")
    def test_grammar_type(self, mock_grammar):
        """_init_logits_processor routes 'grammar' type correctly."""
        backend = self._make_backend()
        result = backend._init_logits_processor(("grammar", "root ::= 'a'"))

        mock_grammar.assert_called_once_with("root ::= 'a'", False)
        self.assertEqual(result, "grammar_proc")

    @patch.object(BackendBase, "_structural_tag_processor", return_value="tag_proc")
    def test_structural_tag_type(self, mock_tag):
        """_init_logits_processor routes 'structural_tag' type correctly."""
        backend = self._make_backend()
        result = backend._init_logits_processor(("structural_tag", "<tag>"))

        mock_tag.assert_called_once_with("<tag>", False)
        self.assertEqual(result, "tag_proc")

    @patch("fastdeploy.model_executor.guided_decoding.base_guided_decoding.llm_logger")
    def test_unsupported_type_returns_none(self, mock_logger):
        """_init_logits_processor returns None for unsupported type."""
        backend = self._make_backend()
        result = backend._init_logits_processor(("xml", "<root/>"))

        self.assertIsNone(result)
        mock_logger.error.assert_called_once()
        self.assertIn("Unsupported processor type xml", mock_logger.error.call_args[0][0])


class TestBackendBaseGetLogitsProcessor(unittest.TestCase):
    """Test BackendBase.get_logits_processor."""

    @patch.object(BackendBase, "_init_logits_processor", return_value="mock_proc")
    @patch.object(BackendBase, "_get_tokenizer_hf", return_value=MagicMock())
    def test_returns_future(self, mock_tokenizer, mock_init):
        """get_logits_processor returns a Future."""
        fd_config = MagicMock()
        fd_config.structured_outputs_config.reasoning_parser = None

        backend = BackendBase(fd_config)
        result = backend.get_logits_processor(("json", "{}"), enable_thinking=True)

        self.assertIsInstance(result, Future)
        self.assertEqual(result.result(timeout=5), "mock_proc")


class TestBackendBaseGetTokenizerHf(unittest.TestCase):
    """Test BackendBase._get_tokenizer_hf."""

    @patch("fastdeploy.model_executor.guided_decoding.base_guided_decoding.ErnieArchitectures")
    @patch("fastdeploy.model_executor.guided_decoding.base_guided_decoding.os.path.exists")
    def test_non_ernie_model_uses_auto_tokenizer(self, mock_exists, mock_ernie_arch):
        """Non-Ernie model uses AutoTokenizer from transformers."""
        mock_ernie_arch.contains_ernie_arch.return_value = False

        fd_config = MagicMock()
        fd_config.model_config.architectures = ["LlamaForCausalLM"]
        fd_config.model_config.model = "/path/to/model"
        fd_config.structured_outputs_config.reasoning_parser = None
        fd_config.structured_outputs_config.guided_decoding_backend = None

        from transformers import PreTrainedTokenizerFast

        mock_fast_tokenizer = MagicMock(spec=PreTrainedTokenizerFast)
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_fast_tokenizer) as mock_from:
            backend = BackendBase(fd_config)
            mock_from.assert_called_once_with("/path/to/model", use_fast=True)
            self.assertIs(backend.hf_tokenizer, mock_fast_tokenizer)

    @patch("fastdeploy.model_executor.guided_decoding.base_guided_decoding.ErnieArchitectures")
    @patch("fastdeploy.model_executor.guided_decoding.base_guided_decoding.os.path.exists")
    def test_non_ernie_slow_tokenizer_wraps_in_fast(self, mock_exists, mock_ernie_arch):
        """Non-Ernie slow tokenizer is wrapped in PreTrainedTokenizerFast."""
        mock_ernie_arch.contains_ernie_arch.return_value = False

        fd_config = MagicMock()
        fd_config.model_config.architectures = ["LlamaForCausalLM"]
        fd_config.model_config.model = "/path/to/model"
        fd_config.structured_outputs_config.reasoning_parser = None
        fd_config.structured_outputs_config.guided_decoding_backend = None

        # Return a plain object that is NOT a PreTrainedTokenizerFast
        mock_slow_tokenizer = object()
        mock_wrapped = MagicMock()

        # Create a fake class to use as PreTrainedTokenizerFast replacement
        class FakeFastTokenizer:
            def __new__(cls, **kwargs):
                return mock_wrapped

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_slow_tokenizer),
            patch("transformers.PreTrainedTokenizerFast", FakeFastTokenizer),
        ):
            backend = BackendBase(fd_config)
            self.assertIs(backend.hf_tokenizer, mock_wrapped)

    @patch("fastdeploy.model_executor.guided_decoding.base_guided_decoding.ErnieArchitectures")
    @patch("fastdeploy.model_executor.guided_decoding.base_guided_decoding.os.path.exists")
    def test_ernie_model_uses_ernie_tokenizer(self, mock_exists, mock_ernie_arch):
        """Ernie model uses Ernie4_5Tokenizer."""
        mock_ernie_arch.contains_ernie_arch.return_value = True

        fd_config = MagicMock()
        fd_config.model_config.architectures = ["Ernie4_5ForCausalLM"]
        fd_config.model_config.model = "/path/to/ernie_model"
        fd_config.structured_outputs_config.reasoning_parser = None
        fd_config.structured_outputs_config.guided_decoding_backend = None

        mock_exists.side_effect = lambda path: "tokenizer.model" in path

        with patch(
            "fastdeploy.model_executor.guided_decoding.base_guided_decoding.os.path.join",
            side_effect=lambda *args: "/".join(args),
        ):
            with patch(
                "fastdeploy.model_executor.guided_decoding.ernie_tokenizer.Ernie4_5Tokenizer"
            ) as mock_ernie_tok:
                mock_ernie_tok.vocab_files_names = {"vocab_file": ""}
                mock_ernie_tok.from_pretrained.return_value = MagicMock()

                BackendBase(fd_config)

                mock_ernie_tok.from_pretrained.assert_called_once_with("/path/to/ernie_model")

    @patch("fastdeploy.model_executor.guided_decoding.base_guided_decoding.ErnieArchitectures")
    def test_tokenizer_init_failure_raises(self, mock_ernie_arch):
        """_get_tokenizer_hf raises Exception on failure."""
        mock_ernie_arch.contains_ernie_arch.side_effect = Exception("config error")

        fd_config = MagicMock()
        fd_config.model_config.architectures = ["SomeModel"]
        fd_config.structured_outputs_config.reasoning_parser = None
        fd_config.structured_outputs_config.guided_decoding_backend = None

        with self.assertRaises(Exception) as ctx:
            BackendBase(fd_config)
        self.assertIn("Fail to initialize hf tokenizer", str(ctx.exception))

    @patch("fastdeploy.model_executor.guided_decoding.base_guided_decoding.ErnieArchitectures")
    def test_guidance_backend_forces_auto_tokenizer(self, mock_ernie_arch):
        """guidance backend uses AutoTokenizer even for Ernie models."""
        mock_ernie_arch.contains_ernie_arch.return_value = True

        fd_config = MagicMock()
        fd_config.model_config.architectures = ["Ernie4_5ForCausalLM"]
        fd_config.model_config.model = "/path/to/model"
        fd_config.structured_outputs_config.reasoning_parser = None
        fd_config.structured_outputs_config.guided_decoding_backend = "guidance"

        from transformers import PreTrainedTokenizerFast

        mock_fast_tokenizer = MagicMock(spec=PreTrainedTokenizerFast)
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_fast_tokenizer) as mock_from:
            BackendBase(fd_config)
            mock_from.assert_called_once()


if __name__ == "__main__":
    unittest.main()
