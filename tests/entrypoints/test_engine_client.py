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

import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from fastdeploy.entrypoints.engine_client import EngineClient
from fastdeploy.utils import EngineError


class TestEngineClient(unittest.IsolatedAsyncioTestCase):
    """Test cases for EngineClient class."""

    async def asyncSetUp(self):
        """Set up test fixtures before each test method."""
        # Create a proper ModelConfig mock with enable_mm attribute
        mock_model_config = Mock()
        mock_model_config.enable_mm = False

        # Create a mock FDConfig that contains the model_config
        mock_config = Mock()
        mock_config.model_config = mock_model_config

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

        # Mock all the dependencies and external components
        with (
            patch.multiple(
                "fastdeploy.entrypoints.engine_client",
                InputPreprocessor=Mock(return_value=mock_input_processor),
                ZmqIpcClient=Mock,
                IPCSignal=mock_ipcsignal,
                StatefulSemaphore=Mock,
                DealerConnectionManager=Mock,
                FileLock=Mock,
                work_process_metrics=Mock(),
                current_platform=mock_platform,
                envs=mock_envs,
            ),
            patch("os.getenv", return_value="50"),
        ):
            # Create EngineClient instance with mocked dependencies
            self.engine_client = EngineClient(
                model_name_or_path="test_model",
                tokenizer=Mock(),
                max_model_len=1024,
                tensor_parallel_size=1,
                pid=1234,
                port=8080,
                limit_mm_per_prompt=5,
                mm_processor_kwargs={},
                config=mock_config,
                reasoning_parser=None,
                data_parallel_size=1,
                enable_logprob=True,
                workers=1,
                tool_parser=None,
                enable_prefix_caching=False,
                splitwise_role=None,
                max_processor_cache=0,
            )

        # Set up mock attributes
        self.engine_client.data_processor = Mock()
        self.engine_client.data_processor.process_request_dict = Mock()
        self.engine_client.zmq_client = Mock()
        self.engine_client.zmq_client.send_json = Mock()
        self.engine_client.zmq_client.send_pyobj = Mock()
        self.engine_client.max_model_len = 1024
        self.engine_client.enable_mm = False
        self.engine_client.enable_logprob = True
        self.engine_client.enable_prefix_caching = False
        self.engine_client.enable_splitwise = False
        self.engine_client.disable_prefix_mm = False

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
        assert request["chat_template_kwargs"]["chat_template"] == "Hello"
        assert request["tools"] == [1]

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
                work_process_metrics=Mock(),
                envs=mock_envs,
            ),
            patch("os.getenv", return_value="50"),
        ):
            # Create a mock config for this test
            mock_config = Mock()
            mock_config.model_config = Mock()
            mock_config.model_config.enable_mm = False

            client = EngineClient(
                model_name_or_path="test_model",
                tokenizer=Mock(),
                max_model_len=2048,
                tensor_parallel_size=2,
                pid=5678,
                port=9090,
                limit_mm_per_prompt=3,
                mm_processor_kwargs={"test": "value"},
                config=mock_config,
                reasoning_parser=None,
                data_parallel_size=1,
                enable_logprob=False,
                workers=2,
                tool_parser=None,
                enable_prefix_caching=True,
                splitwise_role="master",
                max_processor_cache=100,
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

        with patch("fastdeploy.entrypoints.engine_client.envs") as mock_envs:
            mock_envs.FD_MAX_STOP_SEQS_NUM = 20
            mock_envs.FD_STOP_SEQS_MAX_LEN = 100

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
        data = {"max_tokens": 1024}

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
        with patch("fastdeploy.entrypoints.engine_client.ModelWeightsStatus") as mock_status:
            mock_status.NORMAL = 0
            self.engine_client.model_weights_status_signal.value = np.array([0])

            result, message = self.engine_client.is_workers_alive()

            self.assertTrue(result)
            self.assertEqual(message, "")

    def test_is_workers_alive_no_weights(self):
        """Test is_workers_alive returns False when no weights."""
        with patch("fastdeploy.entrypoints.engine_client.ModelWeightsStatus") as mock_status:
            mock_status.NORMAL = 0
            self.engine_client.model_weights_status_signal.value = np.array([1])

            result, message = self.engine_client.is_workers_alive()

            self.assertFalse(result)
            self.assertEqual(message, "No model weight enabled")

    def test_update_model_weight_already_normal(self):
        """Test update_model_weight when weights are already normal."""
        with patch("fastdeploy.entrypoints.engine_client.ModelWeightsStatus") as mock_status:
            mock_status.NORMAL = 0
            self.engine_client.model_weights_status_signal.value = np.array([0])

            result, message = self.engine_client.update_model_weight()

            self.assertTrue(result)
            self.assertEqual(message, "")

    def test_update_model_weight_already_updating(self):
        """Test update_model_weight when already updating."""
        with patch("fastdeploy.entrypoints.engine_client.ModelWeightsStatus") as mock_status:
            mock_status.NORMAL = 0
            mock_status.UPDATING = 1
            self.engine_client.model_weights_status_signal.value = np.array([1])

            result, message = self.engine_client.update_model_weight()

            self.assertFalse(result)
            self.assertEqual(message, "worker is updating model weight already")

    def test_update_model_weight_clearing(self):
        """Test update_model_weight when clearing weights."""
        with patch("fastdeploy.entrypoints.engine_client.ModelWeightsStatus") as mock_status:
            mock_status.NORMAL = 0
            mock_status.CLEARING = -1
            self.engine_client.model_weights_status_signal.value = np.array([-1])

            result, message = self.engine_client.update_model_weight()

            self.assertFalse(result)
            self.assertEqual(message, "worker is clearing model weight, cannot update now")

    def test_update_model_weight_timeout(self):
        """Test update_model_weight timeout scenario."""
        with patch("fastdeploy.entrypoints.engine_client.ModelWeightsStatus") as mock_status:
            with patch("fastdeploy.entrypoints.engine_client.KVCacheStatus") as mock_kv_status:
                with patch("fastdeploy.entrypoints.engine_client.PrefixTreeStatus") as mock_prefix_status:
                    mock_status.NORMAL = 0
                    mock_status.UPDATING = 1
                    mock_kv_status.NORMAL = 0
                    mock_kv_status.UPDATING = 1
                    mock_prefix_status.NORMAL = 0
                    mock_prefix_status.UPDATING = 1

                    self.engine_client.enable_prefix_caching = True
                    # Start with CLEARED status to enter the updating loop
                    self.engine_client.model_weights_status_signal.value = np.array([-2])
                    self.engine_client.kv_cache_status_signal.value = np.array([1])
                    self.engine_client.prefix_tree_status_signal.value = np.array([1])

                    result, message = self.engine_client.update_model_weight(timeout=1)

                    self.assertFalse(result)
                    self.assertEqual(message, "Update model weight timeout")

    def test_clear_load_weight_already_cleared(self):
        """Test clear_load_weight when weights are already cleared."""
        with patch("fastdeploy.entrypoints.engine_client.ModelWeightsStatus") as mock_status:
            mock_status.CLEARED = -2
            self.engine_client.model_weights_status_signal.value = np.array([-2])

            result, message = self.engine_client.clear_load_weight()

            self.assertTrue(result)
            self.assertEqual(message, "")

    def test_clear_load_weight_already_clearing(self):
        """Test clear_load_weight when already clearing."""
        with patch("fastdeploy.entrypoints.engine_client.ModelWeightsStatus") as mock_status:
            mock_status.CLEARED = -2
            mock_status.CLEARING = -1
            self.engine_client.model_weights_status_signal.value = np.array([-1])

            result, message = self.engine_client.clear_load_weight()

            self.assertFalse(result)
            self.assertEqual(message, "worker is clearing model weight already")

    def test_clear_load_weight_updating(self):
        """Test clear_load_weight when updating weights."""
        with patch("fastdeploy.entrypoints.engine_client.ModelWeightsStatus") as mock_status:
            mock_status.CLEARED = -2
            mock_status.CLEARING = -1
            mock_status.UPDATING = 1
            self.engine_client.model_weights_status_signal.value = np.array([1])

            result, message = self.engine_client.clear_load_weight()

            self.assertFalse(result)
            self.assertEqual(message, "worker is updating model weight, cannot clear now")

    def test_clear_load_weight_timeout(self):
        """Test clear_load_weight timeout scenario."""
        with patch("fastdeploy.entrypoints.engine_client.ModelWeightsStatus") as mock_status:
            with patch("fastdeploy.entrypoints.engine_client.KVCacheStatus") as mock_kv_status:
                with patch("fastdeploy.entrypoints.engine_client.PrefixTreeStatus") as mock_prefix_status:
                    mock_status.NORMAL = 0
                    mock_status.CLEARED = -2
                    mock_status.CLEARING = -1
                    mock_kv_status.CLEARED = -2
                    mock_kv_status.CLEARING = -1
                    mock_prefix_status.CLEARED = -2
                    mock_prefix_status.CLEARING = -1

                    self.engine_client.enable_prefix_caching = True
                    # Start with NORMAL status to enter the clearing loop
                    self.engine_client.model_weights_status_signal.value = np.array([0])
                    self.engine_client.kv_cache_status_signal.value = np.array([-1])
                    self.engine_client.prefix_tree_status_signal.value = np.array([-1])

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

            client = EngineClient(
                model_name_or_path="test_model",
                tokenizer=Mock(),
                max_model_len=2048,
                tensor_parallel_size=1,
                pid=5678,
                port=8080,
                limit_mm_per_prompt=5,
                mm_processor_kwargs={},
                config=mock_config,
                reasoning_parser=None,
                data_parallel_size=1,
                enable_logprob=True,
                workers=1,
                tool_parser=None,
                enable_prefix_caching=True,  # Enable prefix caching
                splitwise_role=None,
                max_processor_cache=0,
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
            client = EngineClient(
                model_name_or_path="test_model",
                tokenizer=Mock(),
                max_model_len=2048,
                tensor_parallel_size=16,  # Large number to make it a worker
                pid=5678,
                port=8080,
                limit_mm_per_prompt=5,
                mm_processor_kwargs={},
                config=mock_config,
                reasoning_parser=None,
                data_parallel_size=1,
                enable_logprob=True,
                workers=1,
                tool_parser=None,
                enable_prefix_caching=False,
                splitwise_role=None,
                max_processor_cache=0,
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

    async def test_rearrange_experts_disabled(self):
        """Test rearrange_experts when EPLB is disabled."""
        mock_config = Mock()
        mock_config.eplb_config = Mock()
        mock_config.eplb_config.enable_eplb = False

        self.engine_client.config = mock_config

        request_dict = {"user": "test", "passwd": "test"}
        content, status_code = await self.engine_client.rearrange_experts(request_dict)

        self.assertEqual(content["code"], 1)
        self.assertEqual(content["msg"], "redundant expert is disabled")
        self.assertEqual(status_code, 400)

    async def test_get_per_expert_tokens_stats_disabled(self):
        """Test get_per_expert_tokens_stats when EPLB is disabled."""
        mock_config = Mock()
        mock_config.eplb_config = Mock()
        mock_config.eplb_config.enable_eplb = False

        self.engine_client.config = mock_config

        request_dict = {"user": "test", "passwd": "test"}
        content, status_code = await self.engine_client.get_per_expert_tokens_stats(request_dict)

        self.assertEqual(content["code"], 1)
        self.assertEqual(content["msg"], "redundant expert is disabled")
        self.assertEqual(status_code, 400)

    async def test_get_per_expert_tokens_stats_invalid_auth(self):
        """Test get_per_expert_tokens_stats with invalid authentication."""
        mock_eplb_config = Mock()
        mock_eplb_config.enable_eplb = True
        mock_eplb_config.redundant_expert_api_user = "correct_user"
        mock_eplb_config.redundant_expert_api_password = "correct_pass"

        mock_parallel_config = Mock()
        mock_parallel_config.tensor_parallel_rank = 0

        mock_config = Mock()
        mock_config.eplb_config = mock_eplb_config
        mock_config.parallel_config = mock_parallel_config

        self.engine_client.config = mock_config

        request_dict = {"user": "wrong_user", "passwd": "wrong_pass"}
        content, status_code = await self.engine_client.get_per_expert_tokens_stats(request_dict)

        self.assertEqual(content["code"], 1)
        self.assertEqual(content["msg"], "user or passwd is invalid")
        self.assertEqual(status_code, 401)

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

    async def test_get_per_expert_tokens_stats_clear_stat(self):
        """Test get_per_expert_tokens_stats with clear_stat flag."""
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

        # Set up mock arrays and signals
        mock_clear_signal = Mock()
        mock_clear_signal.value = np.array([0])
        self.engine_client.signal_clear_experts_token_stats_list = [mock_clear_signal]

        mock_local_stats = Mock()
        mock_local_stats.value = np.array([1, 2, 3])
        self.engine_client.local_experts_token_stats_array_list = [mock_local_stats]

        request_dict = {"user": "test_user", "passwd": "test_pass", "clear_stat": True}

        content, status_code = await self.engine_client.get_per_expert_tokens_stats(request_dict)

        self.assertEqual(content["code"], 0)
        self.assertEqual(content["msg"], "ok")
        self.assertEqual(mock_clear_signal.value[0], 1)  # Clear signal should be set
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

    async def test_check_redundant_invalid_auth(self):
        """Test check_redundant with invalid authentication."""
        mock_eplb_config = Mock()
        mock_eplb_config.enable_eplb = True
        mock_eplb_config.redundant_expert_api_user = "correct_user"
        mock_eplb_config.redundant_expert_api_password = "correct_pass"

        mock_parallel_config = Mock()
        mock_parallel_config.tensor_parallel_rank = 0

        mock_config = Mock()
        mock_config.eplb_config = mock_eplb_config
        mock_config.parallel_config = mock_parallel_config

        self.engine_client.config = mock_config

        request_dict = {"user": "wrong_user", "passwd": "wrong_pass"}
        content, status_code = await self.engine_client.check_redundant(request_dict)

        self.assertEqual(content["code"], 1)
        self.assertEqual(content["msg"], "user or passwd is invalid")
        self.assertEqual(status_code, 401)

    async def test_check_redundant_wrong_rank(self):
        """Test check_redundant with wrong tensor parallel rank."""
        mock_eplb_config = Mock()
        mock_eplb_config.enable_eplb = True
        mock_eplb_config.redundant_expert_api_user = "test_user"
        mock_eplb_config.redundant_expert_api_password = "test_pass"

        mock_parallel_config = Mock()
        mock_parallel_config.tensor_parallel_rank = 1  # Not rank 0

        mock_config = Mock()
        mock_config.eplb_config = mock_eplb_config
        mock_config.parallel_config = mock_parallel_config

        self.engine_client.config = mock_config

        request_dict = {"user": "test_user", "passwd": "test_pass"}
        content, status_code = await self.engine_client.check_redundant(request_dict)

        self.assertEqual(content["code"], 1)
        self.assertIn("actual rank 1, expect rank 0", content["msg"])
        self.assertEqual(status_code, 400)

    async def test_check_redundant_status_unknown(self):
        """Test check_redundant with unknown status (invalid signal value)."""
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
        self.engine_client.rearrange_experts_signal.value = np.array([999])  # Invalid status

        with patch("fastdeploy.entrypoints.engine_client.RearrangeExpertStatus") as mock_status:
            mock_status.side_effect = Exception("Invalid status")

            request_dict = {"user": "test_user", "passwd": "test_pass", "action": ""}

            content, status_code = await self.engine_client.check_redundant(request_dict)

            self.assertEqual(content["code"], 0)
            self.assertEqual(content["msg"], "ok")
            self.assertEqual(content["status"], "unknown")  # Should fallback to unknown
            self.assertEqual(status_code, 200)

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
        self.engine_client.rearrange_experts_signal.value = np.array([0])  # FREE status

        with patch("fastdeploy.entrypoints.engine_client.RearrangeExpertStatus") as mock_status:
            mock_status_instance = Mock()
            mock_status_instance.name = "FREE"
            mock_status.return_value = mock_status_instance

            request_dict = {"user": "test_user", "passwd": "test_pass", "action": ""}

            content, status_code = await self.engine_client.check_redundant(request_dict)

            self.assertEqual(content["code"], 0)
            self.assertEqual(content["msg"], "ok")
            self.assertEqual(content["status"], "FREE")
            self.assertEqual(status_code, 200)

    async def test_check_redundant_check_load_weight_result(self):
        """Test check_redundant with check_load_weight_result action."""
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

        # Set up mock update_weight_from_disk_result_list
        mock_result1 = Mock()
        mock_result1.value = np.array([1, 2, 3])
        mock_result2 = Mock()
        mock_result2.value = np.array([4, 5, 6])
        self.engine_client.update_weight_from_disk_result_list = [mock_result1, mock_result2]

        request_dict = {"user": "test_user", "passwd": "test_pass", "action": "check_load_weight_result"}

        content, status_code = await self.engine_client.check_redundant(request_dict)

        self.assertEqual(content["code"], 0)
        self.assertEqual(content["msg"], "ok")
        self.assertIn("data", content)
        # Code does: update_weight_result.value[0].tolist(), so only first elements
        self.assertEqual(content["data"], [1, 4])
        self.assertEqual(status_code, 200)

    async def test_check_redundant_invalid_action(self):
        """Test check_redundant with invalid action."""
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

        request_dict = {"user": "test_user", "passwd": "test_pass", "action": "invalid_action"}

        content, status_code = await self.engine_client.check_redundant(request_dict)

        # For invalid action, content remains None and status_code is HTTPStatus.OK
        self.assertIsNone(content)
        self.assertEqual(status_code, 200)

    def test_init_eplb_signals_non_zero_rank(self):
        """Test init_eplb_signals returns early for non-zero tensor parallel rank."""
        mock_parallel_config = Mock()
        mock_parallel_config.tensor_parallel_rank = 1  # Non-zero rank
        mock_parallel_config.local_data_parallel_id = 0

        mock_config = Mock()
        mock_config.parallel_config = mock_parallel_config

        self.engine_client.config = mock_config

        # Should return early without initializing signals
        self.engine_client.init_eplb_signals("test_suffix")

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

        with patch("fastdeploy.entrypoints.engine_client.IPCSignal") as mock_ipcsignal:
            mock_signal = Mock()
            mock_ipcsignal.return_value = mock_signal

            self.engine_client.init_eplb_signals("8080")

            # Check that IPCSignal was called with correct parameters
            self.assertEqual(
                mock_ipcsignal.call_count, 1 + 1 + 1 + 1 + (4 * 5)
            )  # 4 TP ranks * 5 signals each + 4 base signals

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

    def test_init_eplb_signals_suffix_format(self):
        """Test init_eplb_signals uses correct suffix format."""
        mock_model_config = Mock()
        mock_model_config.num_hidden_layers = 4
        mock_model_config.moe_num_experts = 2

        mock_eplb_config = Mock()
        mock_eplb_config.redundant_expert_ip_shm_size = 256

        mock_parallel_config = Mock()
        mock_parallel_config.tensor_parallel_rank = 0
        mock_parallel_config.local_data_parallel_id = 3
        mock_parallel_config.tensor_parallel_size = 1

        mock_config = Mock()
        mock_config.model_config = mock_model_config
        mock_config.eplb_config = mock_eplb_config
        mock_config.parallel_config = mock_parallel_config

        self.engine_client.config = mock_config

        with patch("fastdeploy.entrypoints.engine_client.IPCSignal") as mock_ipcsignal:
            mock_signal = Mock()
            mock_ipcsignal.return_value = mock_signal

            self.engine_client.init_eplb_signals("7777")

            # Check suffix format
            call_args_list = mock_ipcsignal.call_args_list

            # Check DP suffix
            dp_calls = [call for call in call_args_list if "rearrange_experts_status" in str(call)]
            self.assertEqual(len(dp_calls), 1)
            self.assertEqual(dp_calls[0][1]["suffix"], "7777_dp3")

            # Check TP suffix for TP rank 0
            tp_calls = [call for call in call_args_list if "signal_clear_experts_token_stats" in str(call)]
            self.assertEqual(len(tp_calls), 1)
            self.assertEqual(tp_calls[0][1]["suffix"], "7777_dp3_tp0")

    def test_init_eplb_signals_list_initialization(self):
        """Test init_eplb_signals properly initializes all signal lists."""
        mock_model_config = Mock()
        mock_model_config.num_hidden_layers = 2
        mock_model_config.moe_num_experts = 2

        mock_eplb_config = Mock()
        mock_eplb_config.redundant_expert_ip_shm_size = 128

        mock_parallel_config = Mock()
        mock_parallel_config.tensor_parallel_rank = 0
        mock_parallel_config.local_data_parallel_id = 0
        mock_parallel_config.tensor_parallel_size = 3

        mock_config = Mock()
        mock_config.model_config = mock_model_config
        mock_config.eplb_config = mock_eplb_config
        mock_config.parallel_config = mock_parallel_config

        self.engine_client.config = mock_config

        # Ensure lists start empty
        self.engine_client.signal_clear_experts_token_stats_list = []
        self.engine_client.local_experts_token_stats_array_list = []
        self.engine_client.expert_tokens_stats_array_list = []
        self.engine_client.signal_update_weight_from_disk_array_list = []
        self.engine_client.update_weight_from_disk_result_list = []

        with patch("fastdeploy.entrypoints.engine_client.IPCSignal") as mock_ipcsignal:
            mock_signal = Mock()
            mock_ipcsignal.return_value = mock_signal

            self.engine_client.init_eplb_signals("6666")

            # Check that all lists have correct length (3 TP ranks)
            self.assertEqual(len(self.engine_client.signal_clear_experts_token_stats_list), 3)
            self.assertEqual(len(self.engine_client.local_experts_token_stats_array_list), 3)
            self.assertEqual(len(self.engine_client.expert_tokens_stats_array_list), 3)
            self.assertEqual(len(self.engine_client.signal_update_weight_from_disk_array_list), 3)
            self.assertEqual(len(self.engine_client.update_weight_from_disk_result_list), 3)

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

            client = EngineClient(
                model_name_or_path="test_model",
                tokenizer=Mock(),
                max_model_len=2048,
                tensor_parallel_size=1,
                pid=5678,
                port=8080,
                limit_mm_per_prompt=5,
                mm_processor_kwargs={},
                config=mock_config,
                reasoning_parser=None,
                data_parallel_size=1,
                enable_logprob=True,
                workers=1,
                tool_parser=None,
                enable_prefix_caching=False,
                splitwise_role=None,
                max_processor_cache=0,
            )

        self.assertTrue(client.is_master)  # With 1 tensor_parallel_size, should be master even on Iluvatar

    def test_check_mm_disable_prefix_cache_without_multimodal_data(self):
        """Test _check_mm_disable_prefix_cache without multimodal data."""
        self.engine_client.disable_prefix_mm = True

        task = {"multimodal_inputs": {"token_type_ids": [0, 0, 0]}}  # Sum = 0

        result = self.engine_client._check_mm_disable_prefix_cache(task)
        self.assertFalse(result)

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

    async def test_rearrange_experts_eplb_disabled(self):
        """Test rearrange_experts when EPLB is disabled."""
        # Mock eplb_config with enable_eplb = False
        mock_eplb_config = Mock()
        mock_eplb_config.enable_eplb = False

        mock_config = Mock()
        mock_config.eplb_config = mock_eplb_config

        self.engine_client.config = mock_config

        request_dict = {"user": "test_user", "passwd": "test_pass"}

        content, status_code = await self.engine_client.rearrange_experts(request_dict)

        expected_content = {"code": 1, "msg": "redundant expert is disabled"}
        self.assertEqual(content, expected_content)
        self.assertEqual(status_code.value, 400)  # BAD_REQUEST

    async def test_rearrange_experts_invalid_credentials(self):
        """Test rearrange_experts with invalid user/password."""
        # Mock eplb_config with enable_eplb = True
        mock_eplb_config = Mock()
        mock_eplb_config.enable_eplb = True
        mock_eplb_config.redundant_expert_api_user = "valid_user"
        mock_eplb_config.redundant_expert_api_password = "valid_pass"

        mock_config = Mock()
        mock_config.eplb_config = mock_eplb_config
        mock_config.parallel_config.tensor_parallel_rank = 0

        self.engine_client.config = mock_config

        request_dict = {"user": "invalid_user", "passwd": "invalid_pass"}

        content, status_code = await self.engine_client.rearrange_experts(request_dict)

        expected_content = {"code": 1, "msg": "user or passwd is invalid"}
        self.assertEqual(content, expected_content)
        self.assertEqual(status_code.value, 401)  # UNAUTHORIZED

    async def test_rearrange_experts_non_rank_zero(self):
        """Test rearrange_experts from non-zero rank."""
        # Mock eplb_config with enable_eplb = True
        mock_eplb_config = Mock()
        mock_eplb_config.enable_eplb = True
        mock_eplb_config.redundant_expert_api_user = "test_user"
        mock_eplb_config.redundant_expert_api_password = "test_pass"

        mock_config = Mock()
        mock_config.eplb_config = mock_eplb_config
        mock_config.parallel_config.tensor_parallel_rank = 2  # Non-zero rank

        self.engine_client.config = mock_config

        request_dict = {"user": "test_user", "passwd": "test_pass"}

        content, status_code = await self.engine_client.rearrange_experts(request_dict)

        expected_content = {"code": 1, "msg": "actual rank 2, expect rank 0"}
        self.assertEqual(content, expected_content)
        self.assertEqual(status_code.value, 400)  # BAD_REQUEST

    async def test_rearrange_experts_recv_expert_weight_invalid_data(self):
        """Test rearrange_experts recv_expert_weight action with invalid data."""
        # Mock eplb_config
        mock_eplb_config = Mock()
        mock_eplb_config.enable_eplb = True
        mock_eplb_config.redundant_expert_api_user = "test_user"
        mock_eplb_config.redundant_expert_api_password = "test_pass"

        mock_config = Mock()
        mock_config.eplb_config = mock_eplb_config
        mock_config.parallel_config.tensor_parallel_rank = 0

        self.engine_client.config = mock_config

        request_dict = {
            "user": "test_user",
            "passwd": "test_pass",
            "action": "recv_expert_weight",
            # Missing "data" field
        }

        content, status_code = await self.engine_client.rearrange_experts(request_dict)

        expected_content = {"code": 1, "msg": "data not in request or data is not a list"}
        self.assertEqual(content, expected_content)
        self.assertEqual(status_code.value, 400)  # BAD_REQUEST

    async def test_rearrange_experts_invalid_action(self):
        """Test rearrange_experts with invalid action."""
        # Mock eplb_config
        mock_eplb_config = Mock()
        mock_eplb_config.enable_eplb = True
        mock_eplb_config.redundant_expert_api_user = "test_user"
        mock_eplb_config.redundant_expert_api_password = "test_pass"

        mock_config = Mock()
        mock_config.eplb_config = mock_eplb_config
        mock_config.parallel_config.tensor_parallel_rank = 0

        self.engine_client.config = mock_config

        request_dict = {"user": "test_user", "passwd": "test_pass", "action": "invalid_action"}

        content, status_code = await self.engine_client.rearrange_experts(request_dict)

        expected_content = {"code": 1, "msg": "invalid action invalid_action"}
        self.assertEqual(content, expected_content)
        self.assertEqual(status_code.value, 400)  # BAD_REQUEST


if __name__ == "__main__":
    unittest.main()
