"""
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
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.engine.common_engine import EngineService

# Note: This test requires dependencies like paddlenlp, paddle, etc.
# In CI environment, these should be available.
# For local testing without dependencies, you may need to install them or use CI.


class TestEngineService(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Mock configuration
        self.mock_cfg = MagicMock()
        self.mock_cfg.scheduler_config = MagicMock()
        self.mock_cfg.scheduler_config.splitwise_role = "mixed"
        self.mock_cfg.scheduler_config.scheduler = MagicMock(return_value=MagicMock())
        self.mock_cfg.scheduler_config.max_num_seqs = 8
        self.mock_cfg.scheduler_config.max_num_batched_tokens = 4096
        self.mock_cfg.cache_config = MagicMock()
        self.mock_cfg.cache_config.enable_prefix_caching = False
        self.mock_cfg.cache_config.cache_queue_port = 12345
        self.mock_cfg.cache_config.block_size = 16
        self.mock_cfg.cache_config.enc_dec_block_num = 0
        self.mock_cfg.parallel_config = MagicMock()
        self.mock_cfg.parallel_config.enable_expert_parallel = False
        self.mock_cfg.parallel_config.local_data_parallel_id = 0
        self.mock_cfg.parallel_config.tensor_parallel_size = 1
        self.mock_cfg.parallel_config.data_parallel_size = 1
        self.mock_cfg.parallel_config.engine_worker_queue_port = ["12345"]
        self.mock_cfg.max_num_partial_prefills = 4
        self.mock_cfg.max_prefill_batch = 4
        self.mock_cfg.model_config = MagicMock()
        self.mock_cfg.model_config.enable_mm = False
        self.mock_cfg.structured_outputs_config = MagicMock()
        self.mock_cfg.structured_outputs_config.guided_decoding_backend = "off"
        self.mock_cfg.structured_outputs_config.disable_any_whitespace = False
        self.mock_cfg.structured_outputs_config.reasoning_parser = None
        self.mock_cfg.limit_mm_per_prompt = {}
        self.mock_cfg.mm_processor_kwargs = {}
        self.mock_cfg.tool_parser = None
        self.mock_cfg.master_ip = "127.0.0.1"
        self.mock_cfg.host_ip = "127.0.0.1"
        self.mock_cfg.worker_num_per_node = 1
        self.mock_cfg.node_rank = 0
        self.mock_cfg.router_config = MagicMock()
        self.mock_cfg.router_config.router = None
        self.mock_cfg.router_config.api_server_host = "127.0.0.1"
        self.mock_cfg.router_config.api_server_port = 8000
        self.mock_cfg.register_info = {}
        self.mock_cfg.splitwise_version = "v1"

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_init(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test EngineService initialization with mocked dependencies.

        Purpose:
            Verify that EngineService can be initialized correctly when provided with a mocked configuration and all dependent components are mocked.
        Scenario:
            - All external dependencies (such as ResourceManager, EngineWorkerQueue, etc.) are mocked.
            - The configuration object is a MagicMock with required attributes set.
        Expected Behavior:
            - EngineService instance is created successfully.
            - The configuration of the created EngineService matches the provided mock configuration.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager_instance = MagicMock()
        mock_resource_manager.return_value = mock_resource_manager_instance

        mock_queue_instance = MagicMock()
        mock_engine_queue.return_value = mock_queue_instance

        mock_connector_instance = MagicMock()
        mock_splitwise_connector.return_value = mock_connector_instance

        mock_token_processor_instance = MagicMock()
        mock_token_processor.return_value = mock_token_processor_instance

        mock_ipc_signal_instance = MagicMock()
        mock_ipc_signal.return_value = mock_ipc_signal_instance

        engine = EngineService(self.mock_cfg, start_queue=False)
        self.assertIsNotNone(engine)
        self.assertEqual(engine.cfg, self.mock_cfg)

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_create_data_processor(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test create_data_processor method creates InputPreprocessor and data_processor correctly.
        Verifies that the processor is initialized with correct configuration parameters and
        the data_processor attribute is set.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager.return_value = MagicMock()
        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()
        mock_ipc_signal.return_value = MagicMock()

        with patch("fastdeploy.engine.common_engine.InputPreprocessor") as mock_input_processor:
            mock_processor_instance = MagicMock()
            mock_processor_instance.create_processor.return_value = MagicMock()
            mock_input_processor.return_value = mock_processor_instance

            engine = EngineService(self.mock_cfg, start_queue=False)
            engine.create_data_processor()

            self.assertIsNotNone(engine.data_processor)
            mock_input_processor.assert_called_once()

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_task_is_finished(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test that task_is_finished correctly identifies finished and unfinished tasks
        by checking the stop_flags array at the given index.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager_instance = MagicMock()
        mock_resource_manager_instance.stop_flags = np.array([True, False, True])
        mock_resource_manager.return_value = mock_resource_manager_instance

        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()
        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)
        self.assertTrue(engine.task_is_finished(0))
        self.assertFalse(engine.task_is_finished(1))

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_all_tasks_finished(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test the all_tasks_finished method of EngineService.

        Verifies that all_tasks_finished returns True when all tasks are finished
        (i.e., all stop_flags are True), and returns False when at least one task
        is still running (i.e., at least one stop_flag is False). This is tested
        by setting different stop_flags configurations in the resource manager.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager_instance = MagicMock()
        mock_resource_manager_instance.stop_flags = np.array([True, True, True])
        mock_resource_manager.return_value = mock_resource_manager_instance

        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()
        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)
        self.assertTrue(engine.all_tasks_finished())

        mock_resource_manager_instance.stop_flags = np.array([True, False, True])
        self.assertFalse(engine.all_tasks_finished())

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_update_requests_chunk_size(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test update_requests_chunk_size method with both chunked prefill enabled and disabled.
        Verifies that when chunked prefill is enabled, chunk size information is properly calculated
        and set on requests; when disabled, no chunk info is set.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager.return_value = MagicMock()
        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()
        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)

        # Test with chunked prefill disabled
        self.mock_cfg.cache_config.enable_chunked_prefill = False
        requests = []
        engine.update_requests_chunk_size(requests)

        # Test with chunked prefill enabled
        self.mock_cfg.cache_config.enable_chunked_prefill = True
        mock_request = MagicMock()
        mock_request.prompt_token_ids_len = 100
        requests = [mock_request]
        engine.update_requests_chunk_size(requests)
        self.assertTrue(hasattr(mock_request, "prefill_chunk_info"))
        self.assertIsNotNone(mock_request.prefill_chunk_info)

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_check_and_free_block_tables(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test that EngineService.check_and_free_block_tables delegates to ResourceManager.
        Verifies that the method properly calls the resource manager's check_and_free_block_tables method exactly once.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager_instance = MagicMock()
        mock_resource_manager.return_value = mock_resource_manager_instance

        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()
        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)
        engine.check_and_free_block_tables()
        mock_resource_manager_instance.check_and_free_block_tables.assert_called_once()

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_clear_data(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test clear_data method clears all internal data structures.

        Verifies that:
        - The token_processor and engine_worker_queue are properly cleared by calling their clear_data methods.
        - The send_response_server and recv_request_server request dictionaries are reset.
        - The clear_data method returns True on success.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager.return_value = MagicMock()

        mock_queue_instance = MagicMock()
        mock_engine_queue.return_value = mock_queue_instance

        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor_instance = MagicMock()
        mock_token_processor.return_value = mock_token_processor_instance

        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)
        engine.send_response_server = MagicMock()
        engine.send_response_server.req_dict = {}
        engine.recv_request_server = MagicMock()
        engine.recv_request_server.req_dict = {}

        result = engine.clear_data()
        self.assertTrue(result)
        mock_token_processor_instance.clear_data.assert_called_once()
        mock_queue_instance.clear_data.assert_called_once()

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_has_features_info(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test _has_features_info method in EngineService.

        Scenarios:
        - Verifies that the method returns False when task.multimodal_inputs is None or empty.
        - Verifies that the method returns True when image_feature_urls, video_feature_urls, or audio_feature_urls are present in task.multimodal_inputs.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager.return_value = MagicMock()
        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()
        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)

        # Test with no multimodal inputs
        task = MagicMock()
        task.multimodal_inputs = None
        self.assertFalse(engine._has_features_info(task))

        # Test with empty multimodal inputs
        task.multimodal_inputs = {}
        self.assertFalse(engine._has_features_info(task))

        # Test with image feature URLs
        task.multimodal_inputs = {"image_feature_urls": ["url1", "url2"]}
        self.assertTrue(engine._has_features_info(task))

        # Test with video feature URLs
        task.multimodal_inputs = {"video_feature_urls": ["url1"]}
        self.assertTrue(engine._has_features_info(task))

        # Test with audio feature URLs
        task.multimodal_inputs = {"audio_feature_urls": ["url1"]}
        self.assertTrue(engine._has_features_info(task))

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_init_with_v1_scheduler(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test EngineService initialization with V1 KVCache Scheduler enabled.
        Verifies that ResourceManagerV1 is used instead of ResourceManager.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = True
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        with patch("fastdeploy.engine.common_engine.ResourceManagerV1") as mock_resource_manager_v1:
            mock_resource_manager_v1.return_value = MagicMock()
            mock_engine_queue.return_value = MagicMock()
            mock_splitwise_connector.return_value = MagicMock()
            mock_token_processor.return_value = MagicMock()
            mock_ipc_signal.return_value = MagicMock()

            engine = EngineService(self.mock_cfg, start_queue=False)
            self.assertIsNotNone(engine)
            mock_resource_manager_v1.assert_called_once()

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_init_with_expert_parallel(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test EngineService initialization with expert parallel enabled.
        Verifies that a specific logger is created for the rank.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False

        self.mock_cfg.parallel_config.enable_expert_parallel = True
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager.return_value = MagicMock()
        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()
        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)
        self.assertIsNotNone(engine)
        mock_get_logger.assert_called()

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_init_with_splitwise_not_mixed(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test EngineService initialization with splitwise role not 'mixed'.
        Verifies cache_queue_port is properly handled as string or list.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        # Test with splitwise role as 'prefill'
        self.mock_cfg.scheduler_config.splitwise_role = "prefill"
        self.mock_cfg.cache_config.cache_queue_port = "12345,12346"

        mock_resource_manager.return_value = MagicMock()
        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()
        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)
        self.assertIsNotNone(engine)

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_init_with_guided_decoding(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test EngineService initialization with guided decoding enabled.
        Verifies that schema_checker is properly initialized.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        self.mock_cfg.structured_outputs_config.guided_decoding_backend = "xgrammar"
        mock_schema_checker.return_value = MagicMock()

        mock_resource_manager.return_value = MagicMock()
        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()
        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)
        self.assertIsNotNone(engine)
        mock_schema_checker.assert_called_once()

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_insert_tasks_basic(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test insert_tasks method with basic task insertion.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager_instance = MagicMock()
        mock_resource_manager_instance.stop_flags = np.array([True, True, True])
        mock_resource_manager_instance.allocate_resources_for_new_tasks.return_value = []
        mock_resource_manager.return_value = mock_resource_manager_instance

        mock_queue_instance = MagicMock()
        mock_engine_queue.return_value = mock_queue_instance

        mock_connector_instance = MagicMock()
        mock_splitwise_connector.return_value = mock_connector_instance

        mock_token_processor_instance = MagicMock()
        mock_token_processor_instance.number_of_tasks = 0
        mock_token_processor_instance.number_of_input_tokens = 0
        mock_token_processor.return_value = mock_token_processor_instance

        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)

        # Test with empty tasks after allocation
        mock_task = MagicMock()
        mock_task.request_id = "test-req-1"
        mock_task.prompt_token_ids_len = 100

        with self.assertRaises(Exception):
            # Should raise EngineError when tasks list is empty after allocation
            engine.insert_tasks([mock_task])

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_insert_tasks_success(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test insert_tasks method with successful task insertion.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_task = MagicMock()
        mock_task.request_id = "test-req-1"
        mock_task.prompt_token_ids_len = 100
        mock_task.disaggregate_info = None

        mock_resource_manager_instance = MagicMock()
        mock_resource_manager_instance.stop_flags = np.array([True, True, True])
        mock_resource_manager_instance.allocate_resources_for_new_tasks.return_value = [mock_task]
        mock_resource_manager_instance.real_bsz = 1
        mock_resource_manager.return_value = mock_resource_manager_instance

        mock_queue_instance = MagicMock()
        mock_engine_queue.return_value = mock_queue_instance

        mock_connector_instance = MagicMock()
        mock_splitwise_connector.return_value = mock_connector_instance

        mock_token_processor_instance = MagicMock()
        mock_token_processor_instance.number_of_tasks = 0
        mock_token_processor_instance.number_of_input_tokens = 0
        mock_token_processor.return_value = mock_token_processor_instance

        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)
        self.mock_cfg.cache_config.enable_chunked_prefill = False

        result = engine.insert_tasks([mock_task])
        self.assertTrue(result)
        mock_queue_instance.put_tasks.assert_called_once()

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_insert_tasks_with_disaggregate_decode(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test insert_tasks method with disaggregate decode task.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_task = MagicMock()
        mock_task.request_id = "test-req-1"
        mock_task.prompt_token_ids_len = 100
        mock_task.disaggregate_info = {"role": "decode"}

        mock_resource_manager_instance = MagicMock()
        mock_resource_manager_instance.stop_flags = np.array([True, True, True])
        mock_resource_manager_instance.allocate_resources_for_new_tasks.return_value = [mock_task]
        mock_resource_manager_instance.real_bsz = 1
        mock_resource_manager.return_value = mock_resource_manager_instance

        mock_queue_instance = MagicMock()
        mock_engine_queue.return_value = mock_queue_instance

        mock_connector_instance = MagicMock()
        mock_splitwise_connector.return_value = mock_connector_instance

        mock_token_processor_instance = MagicMock()
        mock_token_processor_instance.number_of_tasks = 0
        mock_token_processor_instance.number_of_input_tokens = 0
        mock_token_processor.return_value = mock_token_processor_instance

        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)
        self.mock_cfg.cache_config.enable_chunked_prefill = False

        result = engine.insert_tasks([mock_task])
        self.assertTrue(result)

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_decode_token(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test _decode_token method.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_envs.FD_ENABLE_RETURN_TEXT = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager.return_value = MagicMock()
        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()
        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)

        # Test with FD_ENABLE_RETURN_TEXT disabled
        delta_text, token_ids = engine._decode_token([1, 2, 3], "req-1", False)
        self.assertEqual(delta_text, "")

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_start_cache_service(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test start_cache_service method.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_cache_manager = MagicMock()
        mock_resource_manager_instance = MagicMock()
        mock_resource_manager_instance.cache_manager = mock_cache_manager
        mock_resource_manager.return_value = mock_resource_manager_instance

        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()
        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)
        engine.start_cache_service([0], 12345)
        mock_cache_manager.launch_cache_manager.assert_called_once()

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_exit_sub_services(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test _exit_sub_services method.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager.return_value = MagicMock()
        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()

        mock_ipc_signal_instance = MagicMock()
        mock_ipc_signal.return_value = mock_ipc_signal_instance

        engine = EngineService(self.mock_cfg, start_queue=False)
        engine.running = True
        engine.engine_worker_queue_server = MagicMock()
        engine.send_response_server = MagicMock()
        engine.recv_request_server = MagicMock()
        engine.recv_control_cmd_server = MagicMock()

        engine._exit_sub_services()
        self.assertFalse(engine.running)

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_register_to_router_disabled(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test _register_to_router when router is disabled (None).
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager.return_value = MagicMock()
        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()
        mock_ipc_signal.return_value = MagicMock()

        self.mock_cfg.router_config.router = None

        engine = EngineService(self.mock_cfg, start_queue=False)
        # Should not raise any exception
        engine._register_to_router()

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_start_zmq_service_no_pid(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test start_zmq_service when api_server_pid is None.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager.return_value = MagicMock()
        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()
        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)
        # Should return early without any setup
        engine.start_zmq_service(api_server_pid=None)
        self.assertFalse(hasattr(engine, "api_server_pid"))

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    @patch("fastdeploy.engine.common_engine.ZmqIpcServer")
    @patch("time.sleep", return_value=None)
    def test_start_zmq_service_with_pid(
        self,
        mock_sleep,
        mock_zmq_server,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test start_zmq_service with valid api_server_pid.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_envs.FD_ENABLE_INTERNAL_ADAPTER = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager.return_value = MagicMock()
        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()
        mock_ipc_signal.return_value = MagicMock()

        mock_zmq_instance = MagicMock()
        mock_zmq_server.return_value = mock_zmq_instance

        engine = EngineService(self.mock_cfg, start_queue=False)
        engine.running = True
        engine.scheduler = MagicMock()
        engine.scheduler.get_results.return_value = {}

        engine.start_zmq_service(api_server_pid="test_pid")
        self.assertEqual(engine.api_server_pid, "test_pid")

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_update_requests_chunk_size_empty(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test update_requests_chunk_size with empty requests list.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager.return_value = MagicMock()
        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()
        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)
        self.mock_cfg.cache_config.enable_chunked_prefill = True

        # Should return early without errors
        engine.update_requests_chunk_size([])

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_update_requests_chunk_size_multiple(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test update_requests_chunk_size with multiple requests.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager.return_value = MagicMock()
        mock_engine_queue.return_value = MagicMock()
        mock_splitwise_connector.return_value = MagicMock()
        mock_token_processor.return_value = MagicMock()
        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)
        self.mock_cfg.cache_config.enable_chunked_prefill = True

        mock_request1 = MagicMock()
        mock_request1.prompt_token_ids_len = 100
        mock_request2 = MagicMock()
        mock_request2.prompt_token_ids_len = 200

        engine.update_requests_chunk_size([mock_request1, mock_request2])
        mock_request1.set.assert_called()
        mock_request2.set.assert_called()

    @patch("fastdeploy.engine.common_engine.envs")
    @patch("fastdeploy.engine.common_engine.get_logger")
    @patch("fastdeploy.engine.common_engine.llm_logger")
    @patch("fastdeploy.engine.common_engine.ResourceManager")
    @patch("fastdeploy.engine.common_engine.EngineWorkerQueue")
    @patch("fastdeploy.engine.common_engine.SplitwiseConnector")
    @patch("fastdeploy.engine.common_engine.TokenProcessor")
    @patch("fastdeploy.engine.common_engine.IPCSignal")
    @patch("fastdeploy.engine.common_engine.schema_checker")
    def test_clear_data_exception(
        self,
        mock_schema_checker,
        mock_ipc_signal,
        mock_token_processor,
        mock_splitwise_connector,
        mock_engine_queue,
        mock_resource_manager,
        mock_llm_logger,
        mock_get_logger,
        mock_envs,
    ):
        """
        Test clear_data method when an exception occurs.
        """
        mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        mock_envs.FD_ENABLE_CACHE_TASK = "0"
        mock_envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        mock_get_logger.return_value = mock_llm_logger

        mock_resource_manager.return_value = MagicMock()

        mock_queue_instance = MagicMock()
        mock_engine_queue.return_value = mock_queue_instance

        mock_splitwise_connector.return_value = MagicMock()

        mock_token_processor_instance = MagicMock()
        mock_token_processor_instance.clear_data.side_effect = Exception("Test error")
        mock_token_processor.return_value = mock_token_processor_instance

        mock_ipc_signal.return_value = MagicMock()

        engine = EngineService(self.mock_cfg, start_queue=False)
        engine.send_response_server = MagicMock()
        engine.send_response_server.req_dict = {}
        engine.recv_request_server = MagicMock()
        engine.recv_request_server.req_dict = {}

        result = engine.clear_data()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
