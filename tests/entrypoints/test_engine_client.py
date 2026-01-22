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
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from fastdeploy.entrypoints.engine_client import EngineClient
from fastdeploy.inter_communicator import (
    KVCacheStatus,
    ModelWeightsStatus,
    PrefixTreeStatus,
    RearrangeExpertStatus,
)
from fastdeploy.utils import EngineError, ParameterError


class DummyConfig(SimpleNamespace):
    def __getattr__(self, name):
        return None


# ============ Pytest Fixtures and Helpers ============


def create_mock_tokenizer(vocab_size=1000):
    """Create a mock tokenizer with specified vocab size."""
    mock_tokenizer = Mock()
    mock_tokenizer.sp_model = Mock()
    mock_tokenizer.sp_model.__len__ = Mock(return_value=vocab_size)
    mock_tokenizer.vocab = Mock()
    mock_tokenizer.vocab.__len__ = Mock(return_value=vocab_size)
    mock_tokenizer.__len__ = Mock(return_value=vocab_size)
    return mock_tokenizer


def create_mock_fd_config(
    enable_mm=True,
    enable_logprob=True,
    max_model_len=1024,
    enable_prefix_caching=True,
    max_processor_cache=10,
    enable_eplb=False,
    tensor_parallel_size=1,
    tensor_parallel_rank=0,
    local_data_parallel_id=0,
    splitwise_role="mixed",
    limit_mm_per_prompt=5,
    **kwargs,
):
    """Create a mock FDConfig with common settings."""
    mock_config = Mock()
    mock_config.model_config = Mock()
    mock_config.model_config.enable_mm = enable_mm
    mock_config.model_config.enable_logprob = enable_logprob
    mock_config.model_config.max_model_len = max_model_len
    mock_config.model_config.enable_mm = enable_mm

    mock_config.cache_config = Mock()
    mock_config.cache_config.max_processor_cache = max_processor_cache
    mock_config.cache_config.enable_prefix_caching = enable_prefix_caching

    mock_config.eplb_config = Mock()
    mock_config.eplb_config.enable_eplb = enable_eplb
    mock_config.eplb_config.redundant_expert_api_user = kwargs.get("eplb_user", "test_user")
    mock_config.eplb_config.redundant_expert_api_password = kwargs.get("eplb_password", "test_pass")
    mock_config.eplb_config.redundant_expert_ip_shm_size = kwargs.get("eplb_shm_size", 1024)
    mock_config.eplb_config.redundant_expert_meta_dir = kwargs.get("eplb_meta_dir", "/tmp/meta")

    mock_config.parallel_config = Mock()
    mock_config.parallel_config.tensor_parallel_size = tensor_parallel_size
    mock_config.parallel_config.tensor_parallel_rank = tensor_parallel_rank
    mock_config.parallel_config.local_data_parallel_id = local_data_parallel_id

    mock_config.scheduler_config = Mock()
    mock_config.scheduler_config.splitwise_role = splitwise_role

    mock_config.limit_mm_per_prompt = limit_mm_per_prompt
    mock_config.mm_processor_kwargs = kwargs.get("mm_processor_kwargs", {})

    mock_config.structured_outputs_config = Mock()
    mock_config.structured_outputs_config.reasoning_parser = None
    mock_config.tool_parser = None

    return mock_config


def create_mock_eplb_config(
    enable_eplb=True, user="test_user", password="test_pass", shm_size=1024, meta_dir="/tmp/meta"
):
    """Create a mock EPLB config with common settings."""
    mock_config = Mock()
    mock_config.enable_eplb = enable_eplb
    mock_config.redundant_expert_api_user = user
    mock_config.redundant_expert_api_password = password
    mock_config.redundant_expert_ip_shm_size = shm_size
    mock_config.redundant_expert_meta_dir = meta_dir
    return mock_config


def create_mock_signals():
    """Create common mock signal objects."""
    return {
        "rearrange_experts_signal": Mock(value=np.array([0])),
        "rearrange_experts_ips_size_signal": Mock(value=np.array([0])),
        "signal_update_weight_from_tensor_array": Mock(value=np.array([0])),
        "model_weights_status_signal": Mock(value=np.array([0])),
        "kv_cache_status_signal": Mock(value=np.array([0])),
        "prefix_tree_status_signal": Mock(value=np.array([0])),
    }


@pytest.fixture
def mock_fd_config():
    """Provide a mock FDConfig with default settings."""
    return create_mock_fd_config()


@pytest.fixture
def mock_fd_config_with_eplb():
    """Provide a mock FDConfig with EPLB enabled."""
    return create_mock_fd_config(enable_eplb=True)


class TestEngineClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        """Set up test fixtures before each test method."""
        # Create a properly configured tokenizer mock first
        mock_tokenizer = Mock()
        mock_tokenizer.sp_model = Mock()
        mock_tokenizer.sp_model.__len__ = Mock(return_value=1000)
        mock_tokenizer.vocab = Mock()
        mock_tokenizer.vocab.__len__ = Mock(return_value=1000)
        # Add len() method directly to the tokenizer mock
        mock_tokenizer.__len__ = Mock(return_value=1000)

        # Create a proper ModelConfig mock with enable_mm attribute
        mock_model_config = Mock()
        mock_model_config.enable_mm = True  # Match engine_config.model_config.enable_mm
        mock_model_config.enable_logprob = True  # Match engine_config.model_config.enable_logprob
        mock_model_config.max_model_len = 1024

        # Create a mock FDConfig that contains the model_config
        mock_config = Mock()
        mock_config.model_config = mock_model_config
        mock_config.cache_config = Mock()
        mock_config.cache_config.max_processor_cache = 10
        mock_config.cache_config.enable_prefix_caching = True
        mock_config.eplb_config = Mock()
        mock_config.eplb_config.enable_eplb = False
        mock_config.parallel_config = Mock()
        mock_config.parallel_config.tensor_parallel_rank = 0
        mock_config.parallel_config.local_data_parallel_id = 0
        mock_config.parallel_config.tensor_parallel_size = 1
        mock_config.scheduler_config = Mock()
        mock_config.scheduler_config.splitwise_role = None
        mock_config.limit_mm_per_prompt = 5
        mock_config.mm_processor_kwargs = {}
        mock_config.tool_parser = None
        mock_config.structured_outputs_config = Mock()
        mock_config.structured_outputs_config.reasoning_parser = None

        # Create mocks for all the external dependencies
        mock_input_processor = Mock()
        mock_processor = Mock()
        mock_processor.tokenizer = mock_tokenizer  # Set the tokenizer on the processor
        mock_input_processor.create_processor.return_value = mock_processor

        # Mock current platform
        mock_platform = Mock()
        mock_platform.is_iluvatar.return_value = False
        mock_platform.max_chips_per_node = 8

        # Create mock IPCSignal that behaves properly
        mock_ipcsignal = Mock()
        mock_signal_instance = Mock()
        mock_signal_instance.value = np.array([0])
        mock_ipcsignal.return_value = mock_signal_instance

        # Mock envs for FD_SUPPORT_MAX_CONNECTIONS
        mock_envs = Mock()
        mock_envs.FD_SUPPORT_MAX_CONNECTIONS = 100

        # Mock all the dependencies and external components
        with (
            patch("fastdeploy.entrypoints.engine_client.IPCSignal"),
            patch("fastdeploy.entrypoints.engine_client.DealerConnectionManager"),
            patch.multiple(
                "fastdeploy.entrypoints.engine_client",
                InputPreprocessor=Mock(return_value=mock_input_processor),
                ZmqIpcClient=Mock,
                IPCSignal=mock_ipcsignal,
                StatefulSemaphore=Mock,
                DealerConnectionManager=Mock,
                FileLock=Mock,
                current_platform=mock_platform,
                envs=mock_envs,
            ),
            patch("os.getenv", return_value="50"),
        ):
            self.engine_config = DummyConfig(
                model_config=DummyConfig(enable_mm=True, enable_logprob=True, max_model_len=1024),
                cache_config=DummyConfig(enable_prefix_caching=True, max_processor_cache=10),
                scheduler_config=DummyConfig(splitwise_role="mixed", max_num_seqs=128),
                parallel_config=DummyConfig(tensor_parallel_size=1),
                structured_outputs_config=DummyConfig(reasoning_parser="reasoning_parser"),
                eplb_config=DummyConfig(enable_eplb=True, eplb_max_tokens=1024),
            )
            # Create EngineClient instance with mocked dependencies
            self.engine_client = EngineClient(pid=1234, port=8080, fd_config=mock_config, workers=1)
            self.engine_client.zmq_client = MagicMock()
            self.engine_client.zmq_client = MagicMock()

    def test_engine_client_initialized_by_fd_config(self):
        for config_group_name, config_group in self.engine_config.__dict__.items():
            for config_name, config_value in config_group.__dict__.items():
                if hasattr(self.engine_client, config_name):
                    # Skip enable_mm, enable_logprob, and enable_prefix_caching checks as they're handled differently in EngineClient
                    if config_name in ["enable_mm", "enable_logprob", "enable_prefix_caching"]:
                        continue
                    assert getattr(self.engine_client, config_name) == config_value

        # Check enable_mm separately since it's copied from model_config
        assert getattr(self.engine_client, "enable_mm") == self.engine_config.model_config.enable_mm
        # Check enable_logprob separately since it's copied from model_config
        assert getattr(self.engine_client, "enable_logprob") == self.engine_config.model_config.enable_logprob
        # Check enable_prefix_caching separately since it's copied from cache_config
        assert (
            getattr(self.engine_client, "enable_prefix_caching")
            == self.engine_config.cache_config.enable_prefix_caching
        )

        # Set up mock attributes
        self.engine_client.data_processor = Mock()
        self.engine_client.data_processor.process_request_dict = Mock()
        self.engine_client.zmq_client = Mock()
        self.engine_client.zmq_client.send_json = Mock()
        self.engine_client.zmq_client.send_pyobj = Mock()
        self.engine_client.max_model_len = 1024
        self.engine_client.enable_mm = False
        self.engine_client.max_logprobs = 20
        self.engine_client.enable_logprob = True
        self.engine_client.ori_vocab_size = 1000
        self.engine_client.enable_prefix_caching = False
        self.engine_client.enable_splitwise = False
        self.engine_client.disable_prefix_mm = False

        # Set up mock attributes for TestEngineClientValidParameters class too
        if hasattr(self, "engine_client_valid"):
            self.engine_client_valid.zmq_client = Mock()
            self.engine_client_valid.zmq_client.send_json = Mock()
            self.engine_client_valid.zmq_client.send_pyobj = Mock()

        # Mock IPC signals
        self.engine_client.worker_healthy_live_signal = Mock()
        self.engine_client.worker_healthy_live_signal.value = np.array([time.time()])
        self.engine_client.model_weights_status_signal = Mock()
        self.engine_client.model_weights_status_signal.value = np.array([0])  # NORMAL
        self.engine_client.prefix_tree_status_signal = Mock()
        self.engine_client.prefix_tree_status_signal.value = np.array([0])  # NORMAL
        self.engine_client.kv_cache_status_signal = Mock()
        self.engine_client.kv_cache_status_signal.value = np.array([0])  # NORMAL

        # Mock file lock
        self.engine_client.clear_update_lock = Mock()
        self.engine_client.clear_update_lock.__enter__ = Mock(return_value=None)
        self.engine_client.clear_update_lock.__exit__ = Mock(return_value=None)

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

        # Mock config object
        mock_config = MagicMock()
        mock_config.model_config = mock_model_config
        mock_config.eplb_config = MagicMock()
        mock_config.eplb_config.enable_eplb = False
        mock_config.parallel_config = MagicMock()
        mock_config.parallel_config.tensor_parallel_rank = 0
        mock_config.parallel_config.local_data_parallel_id = 0
        mock_config.parallel_config.tensor_parallel_size = 1  # Add this missing attribute
        mock_config.scheduler_config = MagicMock()
        mock_config.scheduler_config.splitwise_role = None
        mock_config.cache_config = MagicMock()  # Add cache_config
        mock_config.cache_config.enable_prefix_caching = False
        mock_config.cache_config.max_processor_cache = 0
        mock_config.cache_config.swap_space = False  # Critical: must be False for update/clear tests
        mock_config.limit_mm_per_prompt = 5  # Add this attribute
        mock_config.mm_processor_kwargs = {}  # Add this attribute
        mock_config.structured_outputs_config = MagicMock()  # Add this
        mock_config.structured_outputs_config.reasoning_parser = None
        mock_config.tool_parser = None  # Add this attribute

        # Mock IPCSignal to avoid file system dependencies
        with patch("fastdeploy.entrypoints.engine_client.IPCSignal") as mock_ipcsignal:
            mock_ipcsignal.return_value = MagicMock()

            with patch("fastdeploy.entrypoints.engine_client.StatefulSemaphore") as mock_semaphore:
                mock_semaphore.return_value = MagicMock()

                with patch("fastdeploy.entrypoints.engine_client.DealerConnectionManager") as mock_connection_manager:
                    mock_connection_manager.return_value = MagicMock()

                    with patch("fastdeploy.entrypoints.engine_client.FileLock") as mock_filelock:
                        mock_filelock.return_value = MagicMock()

                        with patch("fastdeploy.entrypoints.engine_client.InputPreprocessor") as mock_input_processor:
                            mock_input_processor_instance = MagicMock()
                            mock_input_processor_instance.create_processor.return_value = mock_data_processor
                            mock_input_processor.return_value = mock_input_processor_instance

                            # Create EngineClient with minimal required parameters
                            self.engine_client = EngineClient(
                                pid=1234,
                                port=8080,
                                fd_config=mock_config,
                                workers=1,
                            )

                            # Set up mock attributes for TestEngineClientValidParameters class
                            self.engine_client.zmq_client = Mock()
                            self.engine_client.zmq_client.send_json = Mock()
                            self.engine_client.zmq_client.send_pyobj = Mock()
                            self.engine_client.max_logprobs = 20
                            self.engine_client.enable_logprob = True
                            self.engine_client.ori_vocab_size = 1000
                            self.engine_client.enable_prefix_caching = False
                            self.engine_client.enable_splitwise = False
                            self.engine_client.disable_prefix_mm = False
                            self.engine_client.max_model_len = 1024
                            self.engine_client.enable_mm = False
                            self.engine_client.config = mock_config
                            self.engine_client.max_chips_per_node = 8
                            self.engine_client.tensor_parallel_size = 1
                            self.engine_client.is_master = True
                            self.engine_client.worker_healthy_live_signal = Mock()
                            self.engine_client.worker_healthy_live_signal.value = np.array([0])
                            self.engine_client.model_weights_status_signal = Mock()
                            self.engine_client.model_weights_status_signal.value = np.array([0])
                            self.engine_client.clear_update_lock = Mock()
                            self.engine_client.clear_update_lock.__enter__ = Mock(return_value=None)
                            self.engine_client.clear_update_lock.__exit__ = Mock(return_value=None)
                            self.engine_client.kv_cache_status_signal = Mock()
                            self.engine_client.kv_cache_status_signal.value = np.array([0])
                            self.engine_client.prefix_tree_status_signal = Mock()
                            self.engine_client.prefix_tree_status_signal.value = np.array([0])

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

    def test_max_logprobs_exceeds_vocab_size(self):
        """Test max_logprobs exceeding vocab_size"""
        self.engine_client.max_logprobs = 1500
        self.engine_client.ori_vocab_size = 1000
        data = {"request_id": "test"}

        with self.assertRaises(ValueError) as context:
            self.engine_client.valid_parameters(data)

        self.assertIn("max_logprobs", str(context.exception))
        self.assertIn("must be <= vocab_size", str(context.exception))

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

            # Test value > 20 - should raise ValueError
            data = {"logprobs": True, "top_logprobs": 25, "request_id": "test"}
            with self.assertRaises(ValueError) as context:
                self.engine_client.valid_parameters(data)
            self.assertIn(
                "Number of top_logprobs requested (25) exceeds maximum allowed value (20)", str(context.exception)
            )

            # Test valid value
            data = {"logprobs": True, "top_logprobs": 10, "request_id": "test"}
            self.engine_client.valid_parameters(data)  # Should not raise

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

    async def test_init_basic_parameters(self):
        """Test EngineClient initialization with basic parameters."""
        # Create a proper ModelConfig mock with enable_mm attribute
        mock_model_config = Mock()
        mock_model_config.enable_mm = False

        # Create mocks for all the external dependencies
        mock_input_processor = Mock()
        mock_processor = Mock()
        mock_input_processor.create_processor.return_value = mock_processor

        # Mock current platform
        mock_platform = Mock()
        mock_platform.is_iluvatar.return_value = False

        # Create mock IPCSignal that behaves properly
        mock_ipcsignal = Mock()
        mock_signal_instance = Mock()
        mock_signal_instance.value = np.array([0])
        mock_ipcsignal.return_value = mock_signal_instance

        # Mock envs for FD_SUPPORT_MAX_CONNECTIONS
        mock_envs = Mock()
        mock_envs.FD_SUPPORT_MAX_CONNECTIONS = 100

        with (
            patch.multiple(
                "fastdeploy.entrypoints.engine_client",
                InputPreprocessor=Mock(return_value=mock_input_processor),
                current_platform=mock_platform,
                IPCSignal=mock_ipcsignal,
                StatefulSemaphore=Mock,
                DealerConnectionManager=Mock,
                FileLock=Mock,
                envs=mock_envs,
            ),
            patch("os.getenv", return_value="50"),
        ):
            # Create a mock config for this test
            mock_config = Mock()
            mock_config.model_config = Mock()
            mock_config.model_config.enable_mm = False
            mock_config.model_config.enable_logprob = False
            mock_config.model_config.max_model_len = 2048
            mock_config.cache_config = Mock()
            mock_config.cache_config.enable_prefix_caching = True
            mock_config.cache_config.max_processor_cache = 100
            mock_config.eplb_config = Mock()
            mock_config.eplb_config.enable_eplb = False
            mock_config.parallel_config = Mock()
            mock_config.parallel_config.tensor_parallel_size = 2
            mock_config.parallel_config.tensor_parallel_rank = 0
            mock_config.parallel_config.local_data_parallel_id = 0
            mock_config.scheduler_config = Mock()
            mock_config.scheduler_config.splitwise_role = "master"
            mock_config.limit_mm_per_prompt = 3
            mock_config.mm_processor_kwargs = {"test": "value"}
            mock_config.structured_outputs_config = Mock()
            mock_config.structured_outputs_config.reasoning_parser = None
            mock_config.tool_parser = None

            client = EngineClient(
                pid=5678,
                port=9090,
                fd_config=mock_config,
                workers=2,
            )

        self.assertEqual(client.max_model_len, 2048)
        self.assertEqual(client.enable_logprob, False)
        self.assertEqual(client.enable_prefix_caching, True)
        self.assertEqual(client.enable_splitwise, True)

    async def test_format_and_add_data_without_request_id(self):
        """Test format_and_add_data adds request_id when missing."""
        prompts = {"prompt_token_ids": [1, 2, 3], "max_tokens": 50}

        with patch.object(self.engine_client, "add_requests") as mock_add:
            mock_add.return_value = None

            result = await self.engine_client.format_and_add_data(prompts)

            self.assertIn("request_id", prompts)
            self.assertEqual(result, prompts["prompt_token_ids"])
            mock_add.assert_called_once_with(prompts)

    async def test_format_and_add_data_with_max_tokens_default(self):
        """Test format_and_add_data sets default max_tokens when missing."""
        prompts = {"request_id": "test-id", "prompt_token_ids": [1, 2, 3]}

        with patch.object(self.engine_client, "add_requests") as mock_add:
            mock_add.return_value = None

            await self.engine_client.format_and_add_data(prompts)

            self.assertEqual(prompts["max_tokens"], self.engine_client.max_model_len - 1)

    async def test_check_mm_disable_prefix_cache_with_disabled_cache(self):
        """Test _check_mm_disable_prefix_cache when prefix cache is disabled."""
        self.engine_client.disable_prefix_mm = False
        task = {"multimodal_inputs": {"token_type_ids": [1, 2, 3]}}

        result = self.engine_client._check_mm_disable_prefix_cache(task)

        self.assertFalse(result)

    async def test_check_mm_disable_prefix_cache_with_no_multimodal_data(self):
        """Test _check_mm_disable_prefix_cache with no multimodal inputs."""
        self.engine_client.disable_prefix_mm = True
        task = {"multimodal_inputs": []}

        result = self.engine_client._check_mm_disable_prefix_cache(task)

        self.assertFalse(result)

    async def test_check_mm_disable_prefix_cache_with_multimodal_data(self):
        """Test _check_mm_disable_prefix_cache detects multimodal data."""
        self.engine_client.disable_prefix_mm = True
        task = {"multimodal_inputs": {"token_type_ids": [1, 0, 2]}}

        result = self.engine_client._check_mm_disable_prefix_cache(task)

        self.assertTrue(result)

    async def test_add_requests_successful_processing(self):
        """Test successful request processing in add_requests."""
        task = {
            "request_id": "test-id",
            "chat_template_kwargs": {"existing": "value"},
            "chat_template": "test_template",
            "prompt_token_ids": [1, 2, 3, 4, 5],
            "max_tokens": 100,
            "min_tokens": 1,
            "messages": "test message",
        }

        self.engine_client.data_processor.process_request_dict = Mock()

        with patch.object(self.engine_client, "_send_task") as mock_send:
            await self.engine_client.add_requests(task)

            self.assertEqual(task["chat_template_kwargs"]["chat_template"], "test_template")
            self.assertEqual(task["prompt_token_ids_len"], 5)
            self.assertNotIn("messages", task)
            mock_send.assert_called_once()

    async def test_add_requests_with_coroutine_processor(self):
        """Test add_requests with async processor."""
        task = {"request_id": "test-id", "prompt_token_ids": [1, 2, 3], "max_tokens": 100}

        async_mock = AsyncMock()
        self.engine_client.data_processor.process_request_dict = async_mock

        with patch.object(self.engine_client, "_send_task"):
            await self.engine_client.add_requests(task)

            async_mock.assert_called_once()

    async def test_add_requests_with_multimodal_prefix_cache_error(self):
        """Test add_requests raises error for multimodal data with prefix cache."""
        self.engine_client.enable_mm = True
        self.engine_client.enable_prefix_caching = True
        self.engine_client.disable_prefix_mm = True

        task = {
            "request_id": "test-id",
            "prompt_token_ids": [1, 2, 3],
            "multimodal_inputs": {"token_type_ids": [1, 0, 1]},
        }

        with self.assertRaises(Exception):  # EngineError
            await self.engine_client.add_requests(task)

    async def test_add_requests_input_length_validation_error(self):
        """Test add_requests validation for input length."""
        task = {"request_id": "test-id", "prompt_token_ids": list(range(1024)), "min_tokens": 1}  # At max length

        with self.assertRaises(Exception):  # EngineError
            await self.engine_client.add_requests(task)

    async def test_add_requests_stop_sequences_validation(self):
        """Test add_requests validation for stop sequences."""
        task = {
            "request_id": "test-id",
            "prompt_token_ids": [1, 2, 3],
            "stop_seqs_len": list(range(25)),  # Exceeds default limit
        }

        with self.assertRaises(Exception):  # EngineError
            await self.engine_client.add_requests(task)

    async def test_add_requests_with_n_parameter_multiple_requests(self):
        """Test add_requests with n parameter for multiple requests."""
        task = {"request_id": "test-id_1", "prompt_token_ids": [1, 2, 3], "n": 3, "max_tokens": 100}

        with patch.object(self.engine_client, "_send_task") as mock_send:
            await self.engine_client.add_requests(task)

            # Should send 3 tasks with indices 3, 4, 5 (1*3 to (1+1)*3)
            self.assertEqual(mock_send.call_count, 3)

    def test_send_task_without_multimodal(self):
        """Test _send_task for non-multimodal content."""
        self.engine_client.enable_mm = False
        task = {"test": "data"}

        self.engine_client._send_task(task)

        self.engine_client.zmq_client.send_json.assert_called_once_with(task)

    def test_send_task_with_multimodal(self):
        """Test _send_task for multimodal content."""
        self.engine_client.enable_mm = True
        task = {"test": "multimodal_data"}

        self.engine_client._send_task(task)

        self.engine_client.zmq_client.send_pyobj.assert_called_once_with(task)

    def test_valid_parameters_max_tokens_valid(self):
        """Test valid_parameters accepts valid max_tokens."""
        data = {"max_tokens": 100}

        # Should not raise exception
        self.engine_client.valid_parameters(data)

    def test_valid_parameters_max_tokens_too_small(self):
        """Test valid_parameters rejects max_tokens < 1."""
        data = {"max_tokens": 0}

        with self.assertRaises(Exception):  # ParameterError
            self.engine_client.valid_parameters(data)

    def test_valid_parameters_max_tokens_too_large(self):
        """Test valid_parameters rejects max_tokens >= max_model_len."""
        data = {"max_tokens": 2048}  # Equal to max_model_len, should raise exception

        with self.assertRaises(Exception):  # ParameterError
            self.engine_client.valid_parameters(data)

    def test_valid_parameters_reasoning_max_tokens_adjustment(self):
        """Test valid_parameters adjusts reasoning_max_tokens when needed."""
        data = {"max_tokens": 50, "reasoning_max_tokens": 100, "request_id": "test-id"}  # Larger than max_tokens

        with patch("fastdeploy.entrypoints.engine_client.api_server_logger") as mock_logger:
            self.engine_client.valid_parameters(data)

            self.assertEqual(data["reasoning_max_tokens"], 50)
            mock_logger.warning.assert_called_once()

    def test_valid_parameters_temperature_zero_adjustment(self):
        """Test valid_parameters adjusts zero temperature."""
        data = {"temperature": 0}

        self.engine_client.valid_parameters(data)

        self.assertEqual(data["temperature"], 1e-6)

    def test_valid_parameters_logprobs_disabled_when_enabled(self):
        """Test valid_parameters rejects logprobs when disabled."""
        self.engine_client.enable_logprob = False
        data = {"logprobs": True}

        with self.assertRaises(Exception):  # ParameterError
            self.engine_client.valid_parameters(data)

    def test_valid_parameters_logprobs_with_invalid_type(self):
        """Test valid_parameters rejects invalid logprobs type."""
        data = {"logprobs": "invalid"}

        with self.assertRaises(Exception):  # ParameterError
            self.engine_client.valid_parameters(data)

    def test_valid_parameters_top_logprobs_disabled(self):
        """Test valid_parameters rejects top_logprobs when disabled."""
        self.engine_client.enable_logprob = False
        data = {"logprobs": True, "top_logprobs": 5}

        with self.assertRaises(Exception):  # ParameterError
            self.engine_client.valid_parameters(data)

    def test_valid_parameters_top_logprobs_invalid_type(self):
        """Test valid_parameters rejects invalid top_logprobs type."""
        self.engine_client.enable_logprob = True
        data = {"logprobs": True, "top_logprobs": "invalid"}

        with self.assertRaises(Exception):  # ParameterError
            self.engine_client.valid_parameters(data)

    def test_valid_parameters_top_logprobs_negative(self):
        """Test valid_parameters rejects negative top_logprobs."""
        self.engine_client.enable_logprob = True
        data = {"logprobs": True, "top_logprobs": -1}

        with self.assertRaises(Exception):  # ParameterError
            self.engine_client.valid_parameters(data)

    def test_valid_parameters_top_logprobs_too_large(self):
        """Test valid_parameters rejects top_logprobs > 20."""
        self.engine_client.enable_logprob = True
        data = {"logprobs": True, "top_logprobs": 25}

        with self.assertRaises(Exception):  # ParameterError
            self.engine_client.valid_parameters(data)

    def test_valid_parameters_top_logprobs_valid(self):
        """Test valid_parameters accepts valid top_logprobs."""
        self.engine_client.enable_logprob = True
        data = {"logprobs": True, "top_logprobs": 10}

        # Should not raise exception
        self.engine_client.valid_parameters(data)

    def test_check_health_healthy(self):
        """Test check_health returns healthy status."""
        self.engine_client.worker_healthy_live_signal.value = np.array([time.time()])

        result, message = self.engine_client.check_health()

        self.assertTrue(result)
        self.assertEqual(message, "")

    def test_check_health_unhealthy_timeout(self):
        """Test check_health returns unhealthy due to timeout."""
        # Set signal to old time (more than 30 seconds ago)
        old_time = time.time() - 60
        self.engine_client.worker_healthy_live_signal.value = np.array([old_time])

        result, message = self.engine_client.check_health(time_interval_threashold=30)

        self.assertFalse(result)
        self.assertEqual(message, "Worker Service Not Healthy")

    def test_is_workers_alive_normal(self):
        """Test is_workers_alive returns True when weights are normal."""
        self.engine_client.model_weights_status_signal.value = np.array([ModelWeightsStatus.NORMAL])

        result, message = self.engine_client.is_workers_alive()

        self.assertTrue(result)
        self.assertEqual(message, "")

    def test_is_workers_alive_no_weights(self):
        """Test is_workers_alive returns False when no weights."""
        self.engine_client.model_weights_status_signal.value = np.array([1])

        result, message = self.engine_client.is_workers_alive()

        self.assertFalse(result)
        self.assertEqual(message, "No model weight enabled")

    def test_update_model_weight_already_normal(self):
        """Test update_model_weight when weights are already normal."""
        # Use real enum value instead of mock
        self.engine_client.model_weights_status_signal.value = np.array([ModelWeightsStatus.NORMAL])

        result, message = self.engine_client.update_model_weight()

        self.assertTrue(result)
        self.assertEqual(message, "")

    def test_update_model_weight_already_updating(self):
        """Test update_model_weight when already updating."""
        # Use real enum value
        self.engine_client.model_weights_status_signal.value = np.array([ModelWeightsStatus.UPDATING])

        result, message = self.engine_client.update_model_weight()

        self.assertFalse(result)
        self.assertEqual(message, "worker is updating model weight already")

    def test_update_model_weight_clearing(self):
        """Test update_model_weight when clearing weights."""
        # Use real enum value
        self.engine_client.model_weights_status_signal.value = np.array([ModelWeightsStatus.CLEARING])

        result, message = self.engine_client.update_model_weight()

        self.assertFalse(result)
        self.assertEqual(message, "worker is clearing model weight, cannot update now")

    def test_update_model_weight_timeout(self):
        """Test update_model_weight timeout scenario."""
        # No need to mock enum classes, use real values directly
        self.engine_client.enable_prefix_caching = True

        # Start with CLEARED status to enter the updating loop
        # Create mutable numpy arrays that can be modified during test
        self.engine_client.model_weights_status_signal.value = np.array([ModelWeightsStatus.CLEARED])
        self.engine_client.kv_cache_status_signal.value = np.array([KVCacheStatus.CLEARED])

        # For prefix_tree, set to NORMAL to avoid getting stuck in prefix tree loop
        self.engine_client.prefix_tree_status_signal.value = np.array([PrefixTreeStatus.NORMAL])

        result, message = self.engine_client.update_model_weight(timeout=1)

        self.assertFalse(result)
        self.assertEqual(message, "Update model weight timeout")

    def test_clear_load_weight_already_cleared(self):
        """Test clear_load_weight when weights are already cleared."""
        # Use real enum value instead of mock
        self.engine_client.model_weights_status_signal.value = np.array([ModelWeightsStatus.CLEARED])

        result, message = self.engine_client.clear_load_weight()

        self.assertTrue(result)
        self.assertEqual(message, "")

    def test_clear_load_weight_already_clearing(self):
        """Test clear_load_weight when already clearing."""
        # Use real enum value
        self.engine_client.model_weights_status_signal.value = np.array([ModelWeightsStatus.CLEARING])

        result, message = self.engine_client.clear_load_weight()

        self.assertFalse(result)
        self.assertEqual(message, "worker is clearing model weight already")

    def test_clear_load_weight_updating(self):
        """Test clear_load_weight when updating weights."""
        # Use real enum value
        self.engine_client.model_weights_status_signal.value = np.array([ModelWeightsStatus.UPDATING])

        result, message = self.engine_client.clear_load_weight()

        self.assertFalse(result)
        self.assertEqual(message, "worker is updating model weight, cannot clear now")

    def test_clear_load_weight_timeout(self):
        """Test clear_load_weight timeout scenario."""
        # No need to mock enum classes, use real values directly
        self.engine_client.enable_prefix_caching = True

        # Start with NORMAL status to enter the clearing loop
        self.engine_client.model_weights_status_signal.value = np.array([ModelWeightsStatus.NORMAL])

        # For prefix_tree, set to CLEARED to avoid getting stuck in prefix tree loop
        self.engine_client.prefix_tree_status_signal.value = np.array([PrefixTreeStatus.CLEARED])

        result, message = self.engine_client.clear_load_weight(timeout=1)

        self.assertFalse(result)
        self.assertEqual(message, "Clear model weight timeout")

    def test_check_model_weight_status(self):
        """Test check_model_weight_status returns correct status."""
        # Status < 0 indicates abnormal
        self.engine_client.model_weights_status_signal.value = np.array([-1])
        result = self.engine_client.check_model_weight_status()
        self.assertTrue(result)

        # Status >= 0 indicates normal
        self.engine_client.model_weights_status_signal.value = np.array([0])
        result = self.engine_client.check_model_weight_status()
        self.assertFalse(result)

    def test_create_zmq_client(self):
        """Test create_zmq_client method."""
        mock_zmq_client = Mock()
        with patch("fastdeploy.entrypoints.engine_client.ZmqIpcClient", return_value=mock_zmq_client) as mock_zmq:
            self.engine_client.create_zmq_client("test_model", "test_mode")

            mock_zmq.assert_called_once_with("test_model", "test_mode")
            mock_zmq_client.connect.assert_called_once()
            self.assertEqual(self.engine_client.zmq_client, mock_zmq_client)

    async def test_init_with_multimodal_prefix_cache(self):
        """Test EngineClient initialization with multimodal prefix cache enabled."""
        mock_model_config = Mock()
        mock_model_config.enable_mm = True

        mock_config = Mock()
        mock_config.model_config = mock_model_config
        mock_config.eplb_config = Mock()
        mock_config.eplb_config.enable_eplb = False

        with (
            patch("fastdeploy.entrypoints.engine_client.InputPreprocessor") as mock_processor_class,
            patch("fastdeploy.entrypoints.engine_client.current_platform") as mock_platform,
            patch("fastdeploy.entrypoints.engine_client.IPCSignal") as mock_ipcsignal,
            patch("fastdeploy.entrypoints.engine_client.envs") as mock_envs,
            patch("os.getenv", return_value="50"),
            patch("fastdeploy.cache_manager.cache_data.is_mm_model_disable_prefix_cache", return_value=True),
        ):
            mock_platform.is_iluvatar.return_value = False
            mock_input_processor = Mock()
            mock_processor_class.return_value = mock_input_processor
            mock_processor = Mock()
            mock_input_processor.create_processor.return_value = mock_processor

            mock_signal_instance = Mock()
            mock_signal_instance.value = np.array([0])
            mock_ipcsignal.return_value = mock_signal_instance
            mock_envs.FD_SUPPORT_MAX_CONNECTIONS = 100

            mock_config.model_config.max_model_len = 2048
            mock_config.model_config.enable_logprob = True
            mock_config.cache_config.enable_prefix_caching = True
            mock_config.cache_config.max_processor_cache = 0
            mock_config.parallel_config.tensor_parallel_size = 1
            mock_config.parallel_config.tensor_parallel_rank = 0
            mock_config.parallel_config.local_data_parallel_id = 0
            mock_config.scheduler_config.splitwise_role = None
            mock_config.limit_mm_per_prompt = 5
            mock_config.mm_processor_kwargs = {}
            mock_config.structured_outputs_config.reasoning_parser = None
            mock_config.tool_parser = None

            client = EngineClient(
                pid=5678,
                port=8080,
                fd_config=mock_config,
                workers=1,
            )

        self.assertTrue(client.enable_mm)
        self.assertTrue(client.enable_prefix_caching)
        self.assertTrue(client.disable_prefix_mm)

    async def test_init_as_worker_node(self):
        """Test EngineClient initialization as worker node (not master)."""
        mock_model_config = Mock()
        mock_model_config.enable_mm = False

        mock_config = Mock()
        mock_config.model_config = mock_model_config
        mock_config.eplb_config = Mock()
        mock_config.eplb_config.enable_eplb = False

        with (
            patch("fastdeploy.entrypoints.engine_client.InputPreprocessor") as mock_processor_class,
            patch("fastdeploy.entrypoints.engine_client.current_platform") as mock_platform,
            patch("fastdeploy.entrypoints.engine_client.IPCSignal") as mock_ipcsignal,
            patch("fastdeploy.entrypoints.engine_client.envs") as mock_envs,
            patch("os.getenv", return_value="50"),
        ):
            mock_platform.is_iluvatar.return_value = False
            mock_platform.max_chips_per_node = 8
            mock_input_processor = Mock()
            mock_processor_class.return_value = mock_input_processor
            mock_processor = Mock()
            mock_input_processor.create_processor.return_value = mock_processor

            mock_signal_instance = Mock()
            mock_signal_instance.value = np.array([0])
            mock_ipcsignal.return_value = mock_signal_instance
            mock_envs.FD_SUPPORT_MAX_CONNECTIONS = 100

            # Use tensor_parallel_size > max_chips_per_node to make it a worker
            mock_config.model_config.max_model_len = 2048
            mock_config.model_config.enable_logprob = True
            mock_config.cache_config.enable_prefix_caching = False
            mock_config.cache_config.max_processor_cache = 0
            mock_config.parallel_config.tensor_parallel_size = 16
            mock_config.parallel_config.tensor_parallel_rank = 0
            mock_config.parallel_config.local_data_parallel_id = 0
            mock_config.scheduler_config.splitwise_role = None
            mock_config.limit_mm_per_prompt = 5
            mock_config.mm_processor_kwargs = {}
            mock_config.structured_outputs_config.reasoning_parser = None
            mock_config.tool_parser = None

            client = EngineClient(
                pid=5678,
                port=8080,
                fd_config=mock_config,
                workers=1,
            )

        self.assertFalse(client.is_master)

    async def test_format_and_add_data(self):
        """Test format_and_add_data method."""
        prompts = {"prompt_token_ids": [1, 2, 3], "max_tokens": 50}

        with patch.object(self.engine_client, "add_requests") as mock_add:
            mock_add.return_value = None

            await self.engine_client.format_and_add_data(prompts)

            mock_add.assert_called_once()
            call_args = mock_add.call_args[0][0]
            self.assertIn("request_id", call_args)
            self.assertEqual(call_args["prompt_token_ids"], [1, 2, 3])
            self.assertEqual(call_args["max_tokens"], 50)

    async def test_get_per_expert_tokens_stats_success(self):
        """Test get_per_expert_tokens_stats successful response."""
        mock_eplb_config = Mock()
        mock_eplb_config.enable_eplb = True
        mock_eplb_config.redundant_expert_api_user = "test_user"
        mock_eplb_config.redundant_expert_api_password = "test_pass"

        mock_parallel_config = Mock()
        mock_parallel_config.tensor_parallel_rank = 0

        mock_config = Mock()
        mock_config.eplb_config = mock_eplb_config
        mock_config.parallel_config = mock_parallel_config

        self.engine_client.config = mock_config

        # Set up mock arrays
        mock_local_stats = Mock()
        mock_local_stats.value = np.array([1, 2, 3])
        self.engine_client.local_experts_token_stats_array_list = [mock_local_stats]
        self.engine_client.signal_clear_experts_token_stats_list = []

        request_dict = {"user": "test_user", "passwd": "test_pass"}

        content, status_code = await self.engine_client.get_per_expert_tokens_stats(request_dict)

        self.assertEqual(content["code"], 0)
        self.assertEqual(content["msg"], "ok")
        self.assertIn("data", content)
        self.assertEqual(content["data"], [[1, 2, 3]])
        self.assertEqual(status_code, 200)

    async def test_check_redundant_disabled(self):
        """Test check_redundant when EPLB is disabled."""
        mock_config = Mock()
        mock_config.eplb_config = Mock()
        mock_config.eplb_config.enable_eplb = False

        self.engine_client.config = mock_config

        request_dict = {"user": "test", "passwd": "test"}
        content, status_code = await self.engine_client.check_redundant(request_dict)

        self.assertEqual(content["code"], 1)
        self.assertEqual(content["msg"], "redundant expert is disabled")
        self.assertEqual(status_code, 400)

    async def test_check_redundant_status_known(self):
        """Test check_redundant with known status."""
        mock_eplb_config = Mock()
        mock_eplb_config.enable_eplb = True
        mock_eplb_config.redundant_expert_api_user = "test_user"
        mock_eplb_config.redundant_expert_api_password = "test_pass"

        mock_parallel_config = Mock()
        mock_parallel_config.tensor_parallel_rank = 0

        mock_config = Mock()
        mock_config.eplb_config = mock_eplb_config
        mock_config.parallel_config = mock_parallel_config

        self.engine_client.config = mock_config
        self.engine_client.rearrange_experts_signal = Mock()
        self.engine_client.rearrange_experts_signal.value = np.array([RearrangeExpertStatus.FREE.value])

        request_dict = {"user": "test_user", "passwd": "test_pass", "action": ""}

        content, status_code = await self.engine_client.check_redundant(request_dict)

        self.assertEqual(content["code"], 0)
        self.assertEqual(content["msg"], "ok")
        self.assertEqual(content["status"], "FREE")
        self.assertEqual(status_code, 200)

    def test_init_eplb_signals_non_zero_rank(self):
        """Test init_eplb_signals returns early for non-zero tensor parallel rank."""
        mock_parallel_config = Mock()
        mock_parallel_config.tensor_parallel_rank = 1  # Non-zero rank
        mock_parallel_config.local_data_parallel_id = 0

        mock_config = Mock()
        mock_config.parallel_config = mock_parallel_config

        # Set fd_config to ensure the method checks the correct config
        self.engine_client.fd_config = mock_config
        self.engine_client.config = mock_config

        # Mock IPCSignal to prevent actual file system calls
        with patch("fastdeploy.entrypoints.engine_client.IPCSignal") as mock_ipcsignal:
            # Should return early without initializing signals
            self.engine_client.init_eplb_signals("test_suffix")

            # Should not create any IPCSignal instances
            mock_ipcsignal.assert_not_called()

            # Should return None (implicitly) and not create any signals
            self.assertFalse(hasattr(self.engine_client, "rearrange_experts_signal"))
            self.assertFalse(hasattr(self.engine_client, "signal_clear_experts_token_stats_list"))

    def test_init_eplb_signals_rank_zero_success(self):
        """Test init_eplb_signals successful initialization for rank 0."""
        mock_model_config = Mock()
        mock_model_config.num_hidden_layers = 12
        mock_model_config.moe_num_experts = 8

        mock_eplb_config = Mock()
        mock_eplb_config.redundant_expert_ip_shm_size = 1024

        mock_parallel_config = Mock()
        mock_parallel_config.tensor_parallel_rank = 0
        mock_parallel_config.local_data_parallel_id = 2
        mock_parallel_config.tensor_parallel_size = 4

        mock_config = Mock()
        mock_config.model_config = mock_model_config
        mock_config.eplb_config = mock_eplb_config
        mock_config.parallel_config = mock_parallel_config

        self.engine_client.config = mock_config
        self.engine_client.fd_config = mock_config  # Also set fd_config for proper access
        self.engine_client.tensor_parallel_size = 4  # Set this to match the config

        with patch("fastdeploy.entrypoints.engine_client.IPCSignal") as mock_ipcsignal:
            mock_signal = Mock()
            mock_ipcsignal.return_value = mock_signal

            self.engine_client.init_eplb_signals("8080")

            # Check that IPCSignal was called with correct parameters
            # Based on the actual implementation: 4 base signals + 4 TP ranks * 5 signals each = 24 total
            self.assertEqual(mock_ipcsignal.call_count, 24)  # 4 TP ranks * 5 signals each + 4 base signals = 24 total

            # Check that the suffix includes data parallel ID
            call_args_list = mock_ipcsignal.call_args_list
            dp_suffix_found = any("8080_dp2" in str(call) for call in call_args_list)
            self.assertTrue(dp_suffix_found)

            # Check that all required signal lists were created
            self.assertEqual(len(self.engine_client.signal_clear_experts_token_stats_list), 4)
            self.assertEqual(len(self.engine_client.local_experts_token_stats_array_list), 4)
            self.assertEqual(len(self.engine_client.expert_tokens_stats_array_list), 4)
            self.assertEqual(len(self.engine_client.signal_update_weight_from_disk_array_list), 4)
            self.assertEqual(len(self.engine_client.update_weight_from_disk_result_list), 4)

            # Check that base signals were created
            self.assertTrue(hasattr(self.engine_client, "rearrange_experts_signal"))
            self.assertTrue(hasattr(self.engine_client, "rearrange_experts_ips_size_signal"))
            self.assertTrue(hasattr(self.engine_client, "shm_rearrange_experts_ips_list"))
            self.assertTrue(hasattr(self.engine_client, "signal_update_weight_from_tensor_array"))

    def test_init_eplb_signals_array_dimensions(self):
        """Test init_eplb_signals creates arrays with correct dimensions."""
        mock_model_config = Mock()
        mock_model_config.num_hidden_layers = 6
        mock_model_config.moe_num_experts = 4

        mock_eplb_config = Mock()
        mock_eplb_config.redundant_expert_ip_shm_size = 512

        mock_parallel_config = Mock()
        mock_parallel_config.tensor_parallel_rank = 0
        mock_parallel_config.local_data_parallel_id = 1
        mock_parallel_config.tensor_parallel_size = 2

        mock_config = Mock()
        mock_config.model_config = mock_model_config
        mock_config.eplb_config = mock_eplb_config
        mock_config.parallel_config = mock_parallel_config

        self.engine_client.config = mock_config
        self.engine_client.tensor_parallel_size = 2  # Set this to match mock_parallel_config.tensor_parallel_size
        self.engine_client.fd_config = mock_config  # Also set fd_config to ensure proper access

        with patch("fastdeploy.entrypoints.engine_client.IPCSignal") as mock_ipcsignal:
            mock_signal = Mock()
            mock_ipcsignal.return_value = mock_signal

            self.engine_client.init_eplb_signals("9090")

            # Check that IPCSignal was called with arrays of correct shape
            call_args_list = mock_ipcsignal.call_args_list

            # Find calls for expert token stats arrays (should be 6x4 shape for 2D arrays)
            all_experts_token_stats_calls = [call for call in call_args_list if "all_experts_token_stats" in str(call)]
            local_experts_token_stats_calls = [
                call for call in call_args_list if "local_experts_token_stats" in str(call)
            ]

            # These should be 2D arrays with shape (6, 4)
            for call in all_experts_token_stats_calls:
                array_arg = call[1]["array"]
                self.assertEqual(array_arg.shape, (6, 4))  # (num_hidden_layers, moe_num_experts)

            for call in local_experts_token_stats_calls:
                array_arg = call[1]["array"]
                self.assertEqual(array_arg.shape, (6, 4))  # (num_hidden_layers, moe_num_experts)

            # Check that single-element signals have shape (1,)
            single_element_calls = [
                call
                for call in call_args_list
                if "rearrange_experts_status" in str(call)
                or "rearrange_experts_ips_size" in str(call)
                or "signal_update_weight_from_tensor" in str(call)
            ]

            for call in single_element_calls:
                array_arg = call[1]["array"]
                self.assertEqual(array_arg.shape, (1,))  # Single element array

    async def test_init_iluvatar_platform(self):
        """Test EngineClient initialization on Iluvatar platform."""
        mock_model_config = Mock()
        mock_model_config.enable_mm = False

        mock_config = Mock()
        mock_config.model_config = mock_model_config
        mock_config.eplb_config = Mock()
        mock_config.eplb_config.enable_eplb = False

        with (
            patch("fastdeploy.entrypoints.engine_client.InputPreprocessor") as mock_processor_class,
            patch("fastdeploy.entrypoints.engine_client.current_platform") as mock_platform,
            patch("fastdeploy.entrypoints.engine_client.IPCSignal") as mock_ipcsignal,
            patch("fastdeploy.entrypoints.engine_client.envs") as mock_envs,
            patch("os.getenv", return_value="50"),
        ):
            mock_platform.is_iluvatar.return_value = True  # Iluvatar platform
            mock_input_processor = Mock()
            mock_processor_class.return_value = mock_input_processor
            mock_processor = Mock()
            mock_input_processor.create_processor.return_value = mock_processor

            mock_signal_instance = Mock()
            mock_signal_instance.value = np.array([0])
            mock_ipcsignal.return_value = mock_signal_instance
            mock_envs.FD_SUPPORT_MAX_CONNECTIONS = 100

            mock_config.model_config.max_model_len = 2048
            mock_config.model_config.enable_logprob = True
            mock_config.cache_config.enable_prefix_caching = False
            mock_config.cache_config.max_processor_cache = 0
            mock_config.parallel_config.tensor_parallel_size = 1
            mock_config.parallel_config.tensor_parallel_rank = 0
            mock_config.parallel_config.local_data_parallel_id = 0
            mock_config.scheduler_config.splitwise_role = None
            mock_config.limit_mm_per_prompt = 5
            mock_config.mm_processor_kwargs = {}
            mock_config.structured_outputs_config.reasoning_parser = None
            mock_config.tool_parser = None

            client = EngineClient(
                pid=5678,
                port=8080,
                fd_config=mock_config,
                workers=1,
            )

        self.assertTrue(client.is_master)  # With 1 tensor_parallel_size, should be master even on Iluvatar

    async def test_add_requests_multimodal_prefix_cache_error(self):
        """Test add_requests with multimodal data when prefix cache is enabled."""
        self.engine_client.enable_mm = True
        self.engine_client.enable_prefix_caching = True
        self.engine_client.disable_prefix_mm = True
        self.engine_client.data_processor = Mock()
        self.engine_client.data_processor.process_request_dict = Mock()

        task = {
            "request_id": "test_request",
            "user": "test_user",
            "multimodal_inputs": {"token_type_ids": [1, 1, 0, 1]},  # Multimodal data present
            "prompt_token_ids": [1, 2, 3],
            "max_tokens": 100,
        }

        with self.assertRaises(EngineError) as context:
            await self.engine_client.add_requests(task)

        self.assertIn("does not support processing requests containing multimodal data", str(context.exception))
        self.assertEqual(context.exception.error_code, 400)

    async def test_add_requests_input_too_long_error(self):
        """Test add_requests with input length too long."""
        self.engine_client.max_model_len = 10
        self.engine_client.data_processor = Mock()
        self.engine_client.data_processor.process_request_dict = Mock()

        task = {
            "request_id": "test_request",
            "user": "test_user",
            "prompt_token_ids": [1, 2, 3, 4, 5, 6, 7, 8],  # length = 8
            "max_tokens": 5,  # 8 + 5 = 13 >= 10
            "min_tokens": 2,
        }

        with self.assertRaises(EngineError) as context:
            await self.engine_client.add_requests(task)

        self.assertIn("Input text is too long", str(context.exception))
        self.assertIn("input_ids_len (8) + min_tokens(2) >= max_model_len(10)", str(context.exception))
        self.assertEqual(context.exception.error_code, 400)

    @patch("fastdeploy.entrypoints.engine_client.envs.FD_MAX_STOP_SEQS_NUM", 3)
    async def test_add_requests_stop_seqs_num_exceeds_limit(self):
        """Test add_requests with stop sequences number exceeding limit."""
        self.engine_client.data_processor = Mock()
        self.engine_client.data_processor.process_request_dict = Mock()

        task = {
            "request_id": "test_request",
            "user": "test_user",
            "prompt_token_ids": [1, 2, 3],
            "max_tokens": 10,
            "stop_seqs_len": [10, 20, 30, 40],  # 4 sequences > limit of 3
        }

        with self.assertRaises(EngineError) as context:
            await self.engine_client.add_requests(task)

        self.assertIn(
            "Length of stop ([10, 20, 30, 40]) exceeds the limit max_stop_seqs_num(3)", str(context.exception)
        )
        self.assertIn("Please reduce the number of stop or set a lager max_stop_seqs_num", str(context.exception))
        self.assertEqual(context.exception.error_code, 400)

    @patch("fastdeploy.entrypoints.engine_client.envs.FD_STOP_SEQS_MAX_LEN", 5)
    async def test_add_requests_single_stop_seq_len_exceeds_limit(self):
        """Test add_requests with single stop sequence length exceeding limit."""
        self.engine_client.data_processor = Mock()
        self.engine_client.data_processor.process_request_dict = Mock()

        task = {
            "request_id": "test_request",
            "user": "test_user",
            "prompt_token_ids": [1, 2, 3],
            "max_tokens": 10,
            "stop_seqs_len": [3, 10, 2],  # 10 > limit of 5
        }

        with self.assertRaises(EngineError) as context:
            await self.engine_client.add_requests(task)

        self.assertIn("Length of stop_seqs(10) exceeds the limit stop_seqs_max_len(5)", str(context.exception))
        self.assertIn(
            "Please reduce the length of stop sequences or set a larger stop_seqs_max_len", str(context.exception)
        )
        self.assertEqual(context.exception.error_code, 400)

    # ========== Phase 1: Critical EPLB Core Functionality Tests ==========

    async def test_rearrange_experts_action_start_with_ips(self):
        """Test rearrange_experts start action with valid IP list."""
        # Use helper to create config
        mock_config = create_mock_fd_config(enable_eplb=True)

        self.engine_client.config = mock_config
        self.engine_client.fd_config = mock_config

        # Setup signals
        self.engine_client.rearrange_experts_signal = Mock(value=np.array([RearrangeExpertStatus.FREE.value]))
        self.engine_client.rearrange_experts_ips_size_signal = Mock(value=np.array([0]))
        self.engine_client.signal_update_weight_from_tensor_array = Mock(value=np.array([0]))
        self.engine_client.shm_rearrange_experts_ips_list = Mock()
        self.engine_client.shm_rearrange_experts_ips_list.shm.buf = bytearray(1024)

        content, status_code = await self.engine_client.rearrange_experts(
            {"user": "test_user", "passwd": "test_pass", "action": "", "ips": ["10.0.0.1:8000", "10.0.0.2:8000"]}
        )

        self.assertEqual(content["code"], 0)
        self.assertEqual(status_code, 200)

    async def test_rearrange_experts_recv_expert_weight(self):
        """Test rearrange_experts recv_expert_weight action."""
        mock_config = create_mock_fd_config(enable_eplb=True, splitwise_role="prefill")

        self.engine_client.config = mock_config
        self.engine_client.fd_config = mock_config
        self.engine_client.rearrange_experts_signal = Mock(value=np.array([2]))
        self.engine_client.expert_tokens_stats_array_list = [Mock(value=np.array([0]))]
        self.engine_client.signal_update_weight_from_disk_array_list = [Mock(value=np.array([0]))]

        content, status_code = await self.engine_client.rearrange_experts(
            {
                "user": "test_user",
                "passwd": "test_pass",
                "action": "recv_expert_weight",
                "data": [[1, 2, 3], [4, 5, 6]],
            }
        )

        self.assertEqual(content["code"], 0)
        self.assertEqual(status_code, 200)

    async def test_rearrange_experts_update_weight_from_tensor(self):
        """Test rearrange_experts update_weight_from_tensor action."""
        mock_config = create_mock_fd_config(enable_eplb=True, splitwise_role="prefill")

        self.engine_client.config = mock_config
        self.engine_client.fd_config = mock_config
        self.engine_client.rearrange_experts_signal = Mock(value=np.array([2]))
        self.engine_client.signal_update_weight_from_tensor_array = Mock(value=np.array([0]))

        content, status_code = await self.engine_client.rearrange_experts(
            {"user": "test_user", "passwd": "test_pass", "action": "update_weight_from_tensor"}
        )

        self.assertEqual(content["code"], 0)
        self.assertEqual(status_code, 200)

    async def test_rearrange_experts_invalid_action(self):
        """Test rearrange_experts with invalid action string."""
        mock_config = create_mock_fd_config(enable_eplb=True)
        self.engine_client.config = mock_config
        self.engine_client.fd_config = mock_config

        content, status_code = await self.engine_client.rearrange_experts(
            {"user": "test_user", "passwd": "test_pass", "action": "invalid_action"}
        )

        self.assertEqual(content["code"], 1)
        self.assertEqual(content["msg"], "invalid action invalid_action")
        self.assertEqual(status_code, 400)

    async def test_rearrange_experts_action_start_ips_too_large(self):
        """Test rearrange_experts when IP list exceeds SHM size."""
        mock_config = create_mock_fd_config(enable_eplb=True, eplb_shm_size=10)
        self.engine_client.config = mock_config
        self.engine_client.fd_config = mock_config
        self.engine_client.rearrange_experts_signal = Mock(value=np.array([RearrangeExpertStatus.FREE.value]))
        self.engine_client.shm_rearrange_experts_ips_list = Mock()
        self.engine_client.shm_rearrange_experts_ips_list.shm.buf = bytearray(10)

        content, status_code = await self.engine_client.rearrange_experts(
            {
                "user": "test_user",
                "passwd": "test_pass",
                "action": "",
                "ips": ["10.0.0.1:8000", "10.0.0.2:8000"],  # > 10 bytes
            }
        )

        self.assertEqual(content["code"], 1)
        self.assertIn("max limit", content["msg"])
        self.assertEqual(status_code, 500)

    async def test_add_requests_preprocessing_exception(self):
        """Test add_requests with preprocessing error raises EngineError."""
        self.engine_client.data_processor = Mock(process_request_dict=Mock(side_effect=Exception("Processing failed")))

        with self.assertRaises(EngineError) as context:
            await self.engine_client.add_requests(
                {"request_id": "test-id", "prompt_token_ids": [1, 2, 3], "max_tokens": 100}
            )

        self.assertIn("Processing failed", str(context.exception))
        self.assertEqual(context.exception.error_code, 400)

    # ========== Phase 2: Error Handling Tests ==========

    async def test_rearrange_experts_invalid_credentials(self):
        """Test rearrange_experts with invalid credentials."""
        mock_config = create_mock_fd_config(enable_eplb=True)
        self.engine_client.config = mock_config
        self.engine_client.fd_config = mock_config

        content, status_code = await self.engine_client.rearrange_experts(
            {"user": "invalid_user", "passwd": "invalid_pass"}
        )

        self.assertEqual(content["code"], 1)
        self.assertEqual(status_code, 401)

    async def test_get_per_expert_tokens_stats_invalid_auth(self):
        """Test get_per_expert_tokens_stats with invalid credentials."""
        mock_config = create_mock_fd_config(enable_eplb=True)
        self.engine_client.config = mock_config

        content, status_code = await self.engine_client.get_per_expert_tokens_stats(
            {"user": "wrong_user", "passwd": "wrong_pass"}
        )

        self.assertEqual(content["code"], 1)
        self.assertEqual(status_code, 401)

    async def test_check_redundant_invalid_credentials(self):
        """Test check_redundant with invalid credentials."""
        mock_config = create_mock_fd_config(enable_eplb=True)
        self.engine_client.config = mock_config

        content, status_code = await self.engine_client.check_redundant({"user": "wrong_user", "passwd": "wrong_pass"})

        self.assertEqual(content["code"], 1)
        self.assertEqual(status_code, 401)

    async def test_add_requests_send_failure(self):
        """Test add_requests when ZMQ send fails."""
        self.engine_client.enable_mm = False
        self.engine_client.data_processor = Mock(process_request_dict=Mock())
        self.engine_client.zmq_client.send_json = Mock(side_effect=Exception("ZMQ send failed"))

        with self.assertRaises(Exception) as context:
            await self.engine_client.add_requests(
                {"request_id": "test-id", "prompt_token_ids": [1, 2, 3], "max_tokens": 100}
            )

        self.assertIn("ZMQ send failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
