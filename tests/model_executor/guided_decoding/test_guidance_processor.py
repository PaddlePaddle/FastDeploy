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

# --- Mocking Setup ---
# Prioritize mocking these lazy-loaded modules to facilitate testing in environments where these libraries are not installed.
mock_torch = MagicMock()
mock_llguidance = MagicMock()
mock_llguidance_hf = MagicMock()
mock_llguidance_torch = MagicMock()

mock_torch.__spec__ = MagicMock()
mock_torch.distributed = MagicMock()

sys.modules["torch"] = mock_torch
sys.modules["llguidance"] = mock_llguidance
sys.modules["llguidance.hf"] = mock_llguidance_hf
sys.modules["llguidance.torch"] = mock_llguidance_torch

# Import the module to be tested after the mock setup is complete
from fastdeploy.model_executor.guided_decoding.guidance_backend import (
    LLGuidanceProcessor,
)


def MockFDConfig():
    """Create a mock FDConfig object for testing"""
    config = MagicMock()
    # --- Fix point 1: Explicitly set model as a string to pass HF validation ---
    config.model_config.model = "test-model-path"
    config.model_config.architectures = []  # Set to empty list to prevent errors when iterating over the Mock

    config.model_config.vocab_size = 1000
    config.scheduler_config.max_num_seqs = 4
    config.structured_outputs_config.disable_any_whitespace = False
    # Ensure the backend check logic passes
    config.structured_outputs_config.guided_decoding_backend = "guidance"
    return config


def MockHFTokenizer():
    """Create a mock Hugging Face Tokenizer object for testing"""
    return MagicMock()


class TestLLGuidanceProcessorMocked(unittest.TestCase):
    """
    Unit tests for LLGuidanceProcessor using Mock.
    This test class is suitable for environments where the llguidance library is not installed.
    """

    def setUp(self):
        """Set up a new LLGuidanceProcessor instance for each test case"""
        self.mock_matcher = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.eos_token = 2  # Example EOS token ID
        self.processor = LLGuidanceProcessor(
            ll_matcher=self.mock_matcher,
            ll_tokenizer=self.mock_tokenizer,
            serialized_grammar="test_grammar",
            vocab_size=1000,
            batch_size=4,
            enable_thinking=False,
        )

    def test_init(self):
        """Test the constructor of LLGuidanceProcessor"""
        self.assertIs(self.processor.matcher, self.mock_matcher)
        self.assertEqual(self.processor.vocab_size, 1000)
        self.assertEqual(self.processor.batch_size, 4)
        self.assertFalse(self.processor.is_terminated)

    @patch("fastdeploy.utils.llm_logger.warning")
    def test_check_error_logs_warning_once(self, mock_log_warning):
        """Test that the _check_error method logs a warning when the matcher errors, and only logs it once"""
        self.mock_matcher.get_error.return_value = "A test error."

        # First call, should log the message
        self.processor._check_error()
        mock_log_warning.assert_called_once_with("LLGuidance Matcher error: A test error.")

        # Second call, should not log repeatedly
        self.processor._check_error()
        mock_log_warning.assert_called_once()

    @patch("fastdeploy.model_executor.guided_decoding.guidance_backend.llguidance.torch")
    def test_allocate_token_bitmask(self, mock_backend_torch):
        """
        Test the allocation of token bitmask.
        Note: We patch the llguidance_torch variable imported in the guidance_backend module here,
        instead of the global mock in sys.modules, to resolve inconsistent references caused by LazyLoader.
        """
        mock_backend_torch.allocate_token_bitmask.return_value = "fake_bitmask_tensor"

        result = self.processor.allocate_token_bitmask()

        mock_backend_torch.allocate_token_bitmask.assert_called_once_with(4, 1000)
        self.assertEqual(result, "fake_bitmask_tensor")

    @patch("fastdeploy.model_executor.guided_decoding.guidance_backend.llguidance.torch")
    def test_fill_token_bitmask(self, mock_backend_torch):
        """Test the filling of token bitmask"""
        mock_bitmask = MagicMock()

        self.processor.fill_token_bitmask(mock_bitmask, idx=2)

        mock_backend_torch.fill_next_token_bitmask.assert_called_once_with(self.mock_matcher, mock_bitmask, 2)
        self.mock_matcher.get_error.assert_called_once()

    def test_reset(self):
        """Test the state reset of the processor"""
        self.processor.is_terminated = True
        self.processor._printed_error = True
        self.mock_matcher.get_error.return_value = ""

        self.processor.reset()

        self.mock_matcher.reset.assert_called_once()
        self.assertFalse(self.processor.is_terminated)
        self.assertFalse(self.processor._printed_error)

    def test_accept_token_when_terminated(self):
        """Test that accept_token returns False immediately when status is is_terminated"""
        self.processor.is_terminated = True
        self.assertFalse(self.processor.accept_token(123))

    def test_accept_token_when_matcher_stopped(self):
        """Test that accept_token returns False and updates status when the matcher is stopped"""
        self.mock_matcher.is_stopped.return_value = True
        self.assertTrue(self.processor.accept_token(123))
        self.assertFalse(self.processor.is_terminated)

    def test_accept_token_is_eos(self):
        """Test the behavior when an EOS token is received"""
        self.mock_matcher.is_stopped.return_value = False
        self.assertTrue(self.processor.accept_token(self.mock_tokenizer.eos_token))
        self.assertTrue(self.processor.is_terminated)

    def test_accept_token_consumes_and_succeeds(self):
        """Test successfully consuming a token"""
        self.mock_matcher.is_stopped.return_value = False
        self.mock_matcher.consume_tokens.return_value = True
        self.assertTrue(self.processor.accept_token(123))
        self.mock_matcher.consume_tokens.assert_called_once_with([123])
        self.mock_matcher.get_error.assert_called_once()

    def test_accept_token_consumes_and_fails(self):
        """Test failing to consume a token"""
        self.mock_matcher.is_stopped.return_value = False
        self.mock_matcher.consume_tokens.return_value = False
        self.assertFalse(self.processor.accept_token(123))
        self.mock_matcher.consume_tokens.assert_called_once_with([123])


if __name__ == "__main__":
    unittest.main()
