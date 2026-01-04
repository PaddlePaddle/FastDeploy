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

import sys
import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.model_executor.guided_decoding import BackendBase

mock_llguidance = MagicMock()
mock_llguidancehf = MagicMock()
mock_llguidancetorch = MagicMock()
mock_torch = MagicMock()

setattr(mock_llguidance, "hf", mock_llguidancehf)

sys.modules["llguidance"] = mock_llguidance
sys.modules["llguidance.hf"] = mock_llguidancehf
sys.modules["llguidance.torch"] = mock_llguidancetorch
sys.modules["torch"] = mock_torch

# Import the module to be tested
from fastdeploy.model_executor.guided_decoding.guidance_backend import (
    LLGuidanceBackend,
    LLGuidanceProcessor,
    process_for_additional_properties,
)


class TestProcessForAdditionalProperties(unittest.TestCase):
    def test_process_json_string(self):
        # Test string input
        json_str = '{"type": "object", "properties": {"name": {"type": "string"}}}'
        result = process_for_additional_properties(json_str)
        self.assertFalse(result["additionalProperties"])

    def test_process_json_dict(self):
        # Test dictionary input
        json_dict = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = process_for_additional_properties(json_dict)
        self.assertFalse(result["additionalProperties"])
        # Ensure the original dictionary is not modified
        self.assertNotIn("additionalProperties", json_dict)

    def test_nested_objects(self):
        # Test nested objects
        json_dict = {
            "type": "object",
            "properties": {"person": {"type": "object", "properties": {"name": {"type": "string"}}}},
        }
        result = process_for_additional_properties(json_dict)
        self.assertFalse(result["additionalProperties"])
        self.assertFalse(result["properties"]["person"]["additionalProperties"])


@patch("llguidance.LLMatcher")
@patch("llguidance.LLTokenizer")
class TestLLGuidanceProcessor(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.batch_size = 2

    def test_initialization(self, mock_tokenizer, mock_matcher):
        # Test initialization
        processor = LLGuidanceProcessor(
            ll_matcher=mock_matcher,
            ll_tokenizer=mock_tokenizer,
            serialized_grammar="test_grammar",
            vocab_size=self.vocab_size,
            batch_size=self.batch_size,
        )

        self.assertEqual(processor.vocab_size, self.vocab_size)
        self.assertEqual(processor.batch_size, self.batch_size)
        self.assertFalse(processor.is_terminated)

    def test_reset(self, mock_tokenizer, mock_matcher):
        # Test reset functionality
        processor = LLGuidanceProcessor(
            ll_matcher=mock_matcher,
            ll_tokenizer=mock_tokenizer,
            serialized_grammar="test_grammar",
            vocab_size=self.vocab_size,
            batch_size=self.batch_size,
        )

        processor.is_terminated = True
        processor.reset()

        mock_matcher.reset.assert_called_once()
        self.assertFalse(processor.is_terminated)

    def test_accept_token(self, mock_tokenizer, mock_matcher):
        # Test accept_token functionality
        mock_matcher.is_stopped.return_value = False
        mock_matcher.consume_tokens.return_value = True
        mock_tokenizer.eos_token = 1

        processor = LLGuidanceProcessor(
            ll_matcher=mock_matcher,
            ll_tokenizer=mock_tokenizer,
            serialized_grammar="test_grammar",
            vocab_size=self.vocab_size,
            batch_size=self.batch_size,
        )

        # Normal token
        result = processor.accept_token(0)
        self.assertTrue(result)
        mock_matcher.consume_tokens.assert_called_with([0])

        # EOS token
        result = processor.accept_token(1)
        self.assertTrue(result)
        self.assertTrue(processor.is_terminated)


@patch("llguidance.LLMatcher")
@patch("llguidance.hf.from_tokenizer")
class TestLLGuidanceBackend(unittest.TestCase):
    def setUp(self):
        # Create a mock FDConfig
        self.fd_config = MagicMock()
        self.fd_config.model_config.vocab_size = 100
        self.fd_config.scheduler_config.max_num_seqs = 2
        self.fd_config.structured_outputs_config.disable_any_whitespace = False
        self.fd_config.structured_outputs_config.disable_additional_properties = False
        self.fd_config.structured_outputs_config.reasoning_parser = None

    def test_initialization(self, mock_from_tokenizer, mock_matcher):
        # Test backend initialization
        mock_tokenizer = MagicMock()
        with patch.object(BackendBase, "_get_tokenizer_hf", return_value=mock_tokenizer):
            backend = LLGuidanceBackend(fd_config=self.fd_config)

            self.assertEqual(backend.vocab_size, 100)
            self.assertEqual(backend.batch_size, 2)
            self.assertTrue(backend.any_whitespace)

    @patch("llguidance.LLMatcher")
    def test_create_processor(self, mock_matcher_class, mock_from_tokenizer, mock_matcher):
        # Test creating a processor
        with patch.object(LLGuidanceBackend, "__init__", return_value=None):
            backend = LLGuidanceBackend(fd_config=None)  # Arguments are not important because __init__ is mocked

            # Manually set all required attributes
            backend.hf_tokenizer = MagicMock()
            backend.ll_tokenizer = MagicMock()
            backend.vocab_size = 100
            backend.batch_size = 2
            backend.any_whitespace = True
            backend.disable_additional_properties = False

            mock_matcher = MagicMock()
            mock_matcher_class.return_value = mock_matcher

            processor = backend._create_processor("test_grammar")

            self.assertIsInstance(processor, LLGuidanceProcessor)
            self.assertEqual(processor.vocab_size, 100)
            self.assertEqual(processor.batch_size, 2)


if __name__ == "__main__":
    unittest.main()
