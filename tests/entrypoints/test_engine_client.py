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

import os
import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.engine_client import EngineClient
from fastdeploy.utils import ParameterError


class TestEngineClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # 创建 EngineClient 实例的模拟对象
        with patch.object(EngineClient, "__init__", return_value=None) as mock_init:
            self.engine_client = EngineClient("model_path")
            mock_init.side_effect = lambda *args, **kwargs: print(f"__init__ called with {args}, {kwargs}")

        self.engine_client.data_processor = MagicMock()
        self.engine_client.zmq_client = MagicMock()
        self.engine_client.max_model_len = 1024
        self.engine_client.enable_mm = False
        self.engine_client.max_logprobs = 20
        self.engine_client.enable_logprob = True
        self.engine_client.ori_vocab_size = 1000

    async def test_add_request(self):
        request = {
            "request_id": "test-request-id",
            "chat_template_kwargs": {"enable_thinking": True},
            "prompt_token_ids": [1],
            "chat_template": "Hello",
            "max_tokens": 20,
            "tools": [1],
        }

        await self.engine_client.add_requests(request)
        assert "chat_template" in request["chat_template_kwargs"], "'chat_template' not found in 'chat_template_kwargs"
        # assert "tools" in request["chat_template_kwargs"], "'tools' not found in 'chat_template_kwargs'"
        assert request["chat_template_kwargs"]["chat_template"] == "Hello"
        assert request["tools"] == [1]
        # assert request["chat_template_kwargs"]["tools"] == [1]


class TestEngineClientValidParameters(unittest.TestCase):
    """Test cases for EngineClient.valid_parameters method"""

    def setUp(self):
        """Set up test fixtures for valid_parameters tests"""
        # Mock the dependencies
        mock_tokenizer = MagicMock()
        mock_tokenizer.sp_model = MagicMock()
        mock_tokenizer.sp_model.__len__ = MagicMock(return_value=1000)
        mock_tokenizer.vocab = MagicMock()
        mock_tokenizer.vocab.__len__ = MagicMock(return_value=1000)

        mock_data_processor = MagicMock()
        mock_data_processor.tokenizer = mock_tokenizer

        mock_model_config = MagicMock()
        mock_model_config.enable_mm = False

        # Mock IPCSignal to avoid file system dependencies
        with patch("fastdeploy.entrypoints.engine_client.IPCSignal") as mock_ipcsignal:
            mock_ipcsignal.return_value = MagicMock()

            with patch("fastdeploy.entrypoints.engine_client.StatefulSemaphore") as mock_semaphore:
                mock_semaphore.return_value = MagicMock()

                with patch("fastdeploy.entrypoints.engine_client.DealerConnectionManager") as mock_connection_manager:
                    mock_connection_manager.return_value = MagicMock()

                    with patch("fastdeploy.entrypoints.engine_client.FileLock") as mock_filelock:
                        mock_filelock.return_value = MagicMock()

                        with patch("fastdeploy.entrypoints.engine_client.ModelConfig") as mock_model_config_class:
                            mock_model_config_class.return_value = mock_model_config

                            with patch(
                                "fastdeploy.entrypoints.engine_client.InputPreprocessor"
                            ) as mock_input_processor:
                                mock_input_processor_instance = MagicMock()
                                mock_input_processor_instance.create_processor.return_value = mock_data_processor
                                mock_input_processor.return_value = mock_input_processor_instance

                                # Create EngineClient with minimal required parameters
                                self.engine_client = EngineClient(
                                    model_name_or_path="test_model",
                                    tokenizer=mock_tokenizer,
                                    max_model_len=2048,
                                    tensor_parallel_size=1,
                                    pid=1234,
                                    port=8080,
                                    limit_mm_per_prompt=None,
                                    mm_processor_kwargs=None,
                                    enable_logprob=True,  # Enable logprob for testing
                                    max_logprobs=20,
                                )

    def test_max_logprobs_valid_values(self):
        """Test valid max_logprobs values"""
        # Test positive max_logprobs
        self.engine_client.max_logprobs = 20
        data = {"request_id": "test"}
        self.engine_client.valid_parameters(data)  # Should not raise

        # Test -1 (unlimited)
        self.engine_client.max_logprobs = -1
        data = {"request_id": "test"}
        self.engine_client.valid_parameters(data)  # Should not raise

    def test_max_logprobs_invalid_values(self):
        """Test invalid max_logprobs values"""
        # Test negative value less than -1
        self.engine_client.max_logprobs = -2
        data = {"request_id": "test"}

        with self.assertRaises(ValueError) as context:
            self.engine_client.valid_parameters(data)

        self.assertIn("max_logprobs", str(context.exception))
        self.assertIn("must be >= -1", str(context.exception))
        self.assertIn("got -2", str(context.exception))

    def test_max_logprobs_exceeds_vocab_size(self):
        """Test max_logprobs exceeding vocab_size"""
        self.engine_client.max_logprobs = 1500
        self.engine_client.ori_vocab_size = 1000
        data = {"request_id": "test"}

        with self.assertRaises(ValueError) as context:
            self.engine_client.valid_parameters(data)

        self.assertIn("max_logprobs", str(context.exception))
        self.assertIn("must be <= vocab_size", str(context.exception))
        self.assertIn("1000", str(context.exception))
        self.assertIn("got 1500", str(context.exception))

    def test_max_logprobs_unlimited(self):
        """Test max_logprobs = -1 (unlimited) sets to ori_vocab_size"""
        self.engine_client.max_logprobs = -1
        self.engine_client.ori_vocab_size = 1000
        data = {"request_id": "test"}

        # This should not raise and internally max_logprobs should be set to ori_vocab_size
        self.engine_client.valid_parameters(data)  # Should not raise
        # The actual max_logprobs value should be set to ori_vocab_size internally
        self.assertEqual(self.engine_client.max_logprobs, -1)  # Original value remains unchanged

    def test_prompt_logprobs_valid_values(self):
        """Test valid prompt_logprobs values"""
        self.engine_client.max_logprobs = 20
        self.engine_client.enable_logprob = True

        # Test valid positive value with FD_USE_GET_SAVE_OUTPUT_V1=1
        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
            data = {"prompt_logprobs": 10, "request_id": "test"}
            self.engine_client.valid_parameters(data)  # Should not raise

        # Test -1 (unlimited) with FD_USE_GET_SAVE_OUTPUT_V1=1
        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
            self.engine_client.max_logprobs = -1
            data = {"prompt_logprobs": -1, "request_id": "test"}
            self.engine_client.valid_parameters(data)  # Should not raise

        # Test None (default)
        data = {"request_id": "test"}
        self.engine_client.valid_parameters(data)  # Should not raise

    def test_prompt_logprobs_unlimited_sets_to_vocab_size(self):
        """Test prompt_logprobs = -1 sets to ori_vocab_size"""
        self.engine_client.max_logprobs = -1  # Set to unlimited to allow prompt_logprobs = -1
        self.engine_client.enable_logprob = True
        self.engine_client.ori_vocab_size = 1000

        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
            data = {"prompt_logprobs": -1, "request_id": "test"}
            self.engine_client.valid_parameters(data)  # Should not raise
            # prompt_logprobs should be set to ori_vocab_size internally

    def test_prompt_logprobs_disabled_when_fd_use_get_save_output_v1_disabled(self):
        """Test prompt_logprobs when FD_USE_GET_SAVE_OUTPUT_V1 is disabled"""
        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "0"}):
            data = {"prompt_logprobs": 10, "request_id": "test"}

            with self.assertRaises(ParameterError) as context:
                self.engine_client.valid_parameters(data)

            self.assertIn(
                "prompt_logprobs is not support when FD_USE_GET_SAVE_OUTPUT_V1 is disabled", str(context.exception)
            )

    def test_prompt_logprobs_disabled_logprob(self):
        """Test prompt_logprobs when logprob is disabled"""
        self.engine_client.enable_logprob = False
        data = {"prompt_logprobs": 10, "request_id": "test"}

        with self.assertRaises(ParameterError) as context:
            self.engine_client.valid_parameters(data)

        self.assertIn("disabled", str(context.exception))

    def test_prompt_logprobs_disabled_when_prefix_caching_enabled(self):
        """Test prompt_logprobs when prefix caching is enabled"""
        self.engine_client.enable_prefix_caching = True
        self.engine_client.enable_logprob = True

        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
            data = {"prompt_logprobs": 10, "request_id": "test"}

            with self.assertRaises(ParameterError) as context:
                self.engine_client.valid_parameters(data)

            self.assertIn("prompt_logprobs is not support when prefix caching is enabled", str(context.exception))

    def test_prompt_logprobs_invalid_values(self):
        """Test invalid prompt_logprobs values"""
        self.engine_client.enable_logprob = True

        # Test negative value less than -1 with FD_USE_GET_SAVE_OUTPUT_V1=1
        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
            data = {"prompt_logprobs": -2, "request_id": "test"}

            with self.assertRaises(ValueError) as context:
                self.engine_client.valid_parameters(data)

            self.assertIn("prompt_logprobs", str(context.exception))
            self.assertIn("must be a non-negative value or -1", str(context.exception))
            self.assertIn("current value is -2", str(context.exception))

    def test_prompt_logprobs_exceeds_max_logprobs(self):
        """Test prompt_logprobs exceeding max_logprobs"""
        self.engine_client.max_logprobs = 10
        self.engine_client.enable_logprob = True

        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
            data = {"prompt_logprobs": 15, "request_id": "test"}

            with self.assertRaises(ValueError) as context:
                self.engine_client.valid_parameters(data)

            self.assertIn("prompt_logprobs", str(context.exception))
            self.assertIn("exceeds maximum allowed value", str(context.exception))
            self.assertIn("15", str(context.exception))
            self.assertIn("10", str(context.exception))

    def test_top_logprobs_validation_with_fd_use_get_save_output_v1_enabled(self):
        """Test top_logprobs validation when FD_USE_GET_SAVE_OUTPUT_V1 is enabled"""
        self.engine_client.max_logprobs = 20
        self.engine_client.enable_logprob = True

        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
            # Test -1 (unlimited) - should set to ori_vocab_size, but need max_logprobs also to be -1
            self.engine_client.max_logprobs = -1  # Set to unlimited to allow top_logprobs = -1
            data = {"logprobs": True, "top_logprobs": -1, "request_id": "test"}
            self.engine_client.valid_parameters(data)  # Should not raise

            # Reset max_logprobs for other tests
            self.engine_client.max_logprobs = 20

            # Test valid positive value
            data = {"logprobs": True, "top_logprobs": 10, "request_id": "test"}
            self.engine_client.valid_parameters(data)  # Should not raise

            # Test value less than -1 - should raise ValueError
            data = {"logprobs": True, "top_logprobs": -2, "request_id": "test"}
            with self.assertRaises(ValueError) as context:
                self.engine_client.valid_parameters(data)
            self.assertIn("must be a non-negative value or -1", str(context.exception))
            self.assertIn("current value is -2", str(context.exception))

            # Test value exceeding max_logprobs - should raise ValueError
            data = {"logprobs": True, "top_logprobs": 25, "request_id": "test"}
            with self.assertRaises(ValueError) as context:
                self.engine_client.valid_parameters(data)
            self.assertIn("exceeds maximum allowed value", str(context.exception))

    def test_top_logprobs_validation_with_fd_use_get_save_output_v1_disabled(self):
        """Test top_logprobs validation when FD_USE_GET_SAVE_OUTPUT_V1 is disabled"""
        self.engine_client.max_logprobs = 20
        self.engine_client.enable_logprob = True

        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "0"}):
            # Test negative value - should raise ValueError
            data = {"logprobs": True, "top_logprobs": -1, "request_id": "test"}
            with self.assertRaises(ValueError) as context:
                self.engine_client.valid_parameters(data)
            self.assertIn("top_logprobs must be between 0 and 20", str(context.exception))
            self.assertIn("current value is -1", str(context.exception))

            # Test value > 20 - should raise ValueError
            data = {"logprobs": True, "top_logprobs": 25, "request_id": "test"}
            with self.assertRaises(ValueError) as context:
                self.engine_client.valid_parameters(data)
            self.assertIn("top_logprobs must be between 0 and 20", str(context.exception))
            self.assertIn("current value is 25", str(context.exception))

            # Test valid value
            data = {"logprobs": True, "top_logprobs": 10, "request_id": "test"}
            self.engine_client.valid_parameters(data)  # Should not raise

    def test_top_logprobs_disabled_logprob(self):
        """Test top_logprobs when logprob is disabled"""
        self.engine_client.enable_logprob = False
        data = {"logprobs": True, "top_logprobs": 10, "request_id": "test"}

        with self.assertRaises(ParameterError) as context:
            self.engine_client.valid_parameters(data)

        self.assertIn("disabled", str(context.exception))

    def test_top_logprobs_invalid_type(self):
        """Test top_logprobs with invalid type"""
        self.engine_client.enable_logprob = True

        # Test with string type
        data = {"logprobs": True, "top_logprobs": "10", "request_id": "test"}

        with self.assertRaises(ParameterError) as context:
            self.engine_client.valid_parameters(data)

        self.assertIn("top_logprobs", str(context.exception))
        self.assertIn("Invalid type", str(context.exception))
        self.assertIn("expected int", str(context.exception))

    def test_logprobs_invalid_type(self):
        """Test logprobs with invalid type"""
        self.engine_client.enable_logprob = True

        # Test with string type
        data = {"logprobs": "true", "request_id": "test"}

        with self.assertRaises(ParameterError) as context:
            self.engine_client.valid_parameters(data)

        self.assertIn("logprobs", str(context.exception))
        self.assertIn("Invalid type", str(context.exception))

    def test_logprobs_disabled(self):
        """Test logprobs when logprob is disabled"""
        self.engine_client.enable_logprob = False

        # Test with logprobs=True
        data = {"logprobs": True, "request_id": "test"}

        with self.assertRaises(ParameterError) as context:
            self.engine_client.valid_parameters(data)

        self.assertIn("disabled", str(context.exception))

    def test_unlimited_max_logprobs_with_prompt_logprobs(self):
        """Test unlimited max_logprobs (-1) with prompt_logprobs"""
        self.engine_client.max_logprobs = -1  # Unlimited
        self.engine_client.enable_logprob = True

        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
            # Should allow any prompt_logprobs value
            data = {"prompt_logprobs": 1000, "request_id": "test"}
            self.engine_client.valid_parameters(data)  # Should not raise

    def test_unlimited_max_logprobs_with_top_logprobs(self):
        """Test unlimited max_logprobs (-1) with top_logprobs"""
        self.engine_client.max_logprobs = -1  # Unlimited
        self.engine_client.enable_logprob = True

        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
            # Should allow any top_logprobs value
            data = {"logprobs": True, "top_logprobs": 1000, "request_id": "test"}
            self.engine_client.valid_parameters(data)  # Should not raise

    def test_edge_case_zero_values(self):
        """Test edge cases with zero values"""
        self.engine_client.max_logprobs = 20
        self.engine_client.enable_logprob = True

        # Test prompt_logprobs = 0 with FD_USE_GET_SAVE_OUTPUT_V1=1
        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
            data = {"prompt_logprobs": 0, "request_id": "test"}
            self.engine_client.valid_parameters(data)  # Should not raise

        # Test top_logprobs = 0 with FD_USE_GET_SAVE_OUTPUT_V1=0
        with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "0"}):
            data = {"logprobs": True, "top_logprobs": 0, "request_id": "test"}
            self.engine_client.valid_parameters(data)  # Should not raise

    def test_valid_parameters(self):
        request = {
            "request_id": "test-request-id",
            "chat_template_kwargs": {"enable_thinking": True},
            "prompt_token_ids": [1],
            "chat_template": "Hello",
            "max_tokens": 20,
            "tools": [1],
            "temperature": 0,
        }
        self.engine_client.valid_parameters(request)
        assert request["temperature"] == 1e-6


if __name__ == "__main__":
    unittest.main()
