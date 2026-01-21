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

import unittest
from unittest.mock import Mock, patch

import numpy as np

from fastdeploy.engine.engine import LLMEngine


class TestLLMEngine(unittest.TestCase):
    """Test case for LLMEngine functionality focusing on uncovered branches"""

    def setUp(self):
        """Set up test fixtures"""
        patch("fastdeploy.engine.engine.EngineService").start()
        patch("fastdeploy.engine.engine.main_process_metrics").start()
        patch("fastdeploy.engine.engine.tracing").start()

    def _create_mock_cfg(self, **kwargs):
        """Create a mock configuration for testing"""
        cfg = Mock()
        cfg.cache_config = Mock()
        cfg.cache_config.num_gpu_blocks_override = kwargs.get("num_gpu_blocks_override", 4)
        cfg.cache_config.enable_prefix_caching = kwargs.get("enable_prefix_caching", False)
        cfg.cache_config.block_size = 128
        cfg.cache_config.num_cpu_blocks = 0
        cfg.cache_config.total_block_num = 100

        cfg.parallel_config = Mock()
        cfg.parallel_config.device_ids = "0"
        cfg.parallel_config.engine_worker_queue_port = [6778]
        cfg.parallel_config.tensor_parallel_size = 1
        cfg.parallel_config.expert_parallel_size = 1
        cfg.parallel_config.chunked_moe_size = 1
        cfg.parallel_config.data_parallel_size = kwargs.get("data_parallel_size", 1)
        cfg.parallel_config.enable_expert_parallel = False
        cfg.parallel_config.enable_chunked_moe = False
        cfg.parallel_config.use_internode_ll_two_stage = False
        cfg.parallel_config.disable_custom_all_reduce = False
        cfg.parallel_config.disable_sequence_parallel_moe = False
        cfg.parallel_config.shutdown_comm_group_if_worker_idle = False

        cfg.scheduler_config = Mock()
        cfg.scheduler_config.splitwise_role = kwargs.get("splitwise_role", "mixed")
        cfg.scheduler_config.max_num_seqs = 32
        cfg.scheduler_config.max_num_batched_tokens = 8192
        cfg.scheduler_config.name = "local"

        cfg.model_config = Mock()
        cfg.model_config.max_model_len = 8192
        cfg.model_config.runner = "vllm"
        cfg.model_config.convert = "auto"
        cfg.model_config.override_pooler_config = None
        cfg.model_config.logprobs_mode = None
        cfg.model_config.max_logprobs = None
        cfg.model_config.enable_logprob = False
        cfg.model_config.enable_entropy = False
        cfg.model_config.lm_head_fp32 = False
        cfg.model_config.quantization = {}

        cfg.structured_outputs_config = Mock()
        cfg.structured_outputs_config.guided_decoding_backend = None
        cfg.structured_outputs_config.logits_processors = None
        cfg.structured_outputs_config.disable_any_whitespace = False
        cfg.structured_outputs_config.reasoning_parser = None

        cfg.load_config = Mock()
        cfg.load_config.dynamic_load_weight = False
        cfg.load_config.load_strategy = "auto"
        cfg.load_config.load_choices = None

        cfg.speculative_config = Mock()
        cfg.speculative_config.to_json_string = Mock(return_value='{"speculative": "config"}')

        cfg.graph_opt_config = Mock()
        cfg.graph_opt_config.to_json_string = Mock(return_value='{"graph_opt": "config"}')

        cfg.early_stop_config = Mock()
        cfg.early_stop_config.to_json_string = Mock(return_value='{"early_stop": "config"}')

        cfg.eplb_config = Mock()
        cfg.eplb_config.to_json_string = Mock(return_value='{"eplb": "config"}')

        cfg.routing_replay_config = Mock()
        cfg.routing_replay_config.to_json_string = Mock(return_value='{"routing_replay": "config"}')

        cfg.plas_attention_config = Mock()
        cfg.plas_attention_config.to_json_string = Mock(return_value='{"plas_attention": "config"}')

        cfg.worker_num_per_node = 1
        cfg.nnode = 1
        cfg.ips = None
        cfg.master_ip = "127.0.0.1"
        cfg.host_ip = "127.0.0.1"
        cfg.register_info = None

        cfg.print = Mock()
        return cfg

    def test_constructor_do_profile_false(self):
        """Test constructor when do_profile is False (line 98)"""
        cfg = self._create_mock_cfg(num_gpu_blocks_override=4)
        cfg.cache_config.num_gpu_blocks_override = 4

        with patch("fastdeploy.engine.engine.weakref.finalize") as mock_finalize:
            engine = LLMEngine(cfg)

        # Verify do_profile is set to 0 when num_gpu_blocks_override is not None
        self.assertEqual(engine.do_profile, 0)
        self.assertTrue(hasattr(engine, "_finalizer"))
        mock_finalize.assert_called_once()

    def test_start_cache_manager_prefill(self):
        """Test cache manager launch in prefill mode (lines 145-147)"""
        cfg = self._create_mock_cfg(splitwise_role="prefill")
        cfg.cache_config.enable_prefix_caching = True

        with (
            patch("fastdeploy.engine.engine.LLMEngine._init_worker_signals") as mock_init_signals,
            patch("fastdeploy.engine.engine.LLMEngine.launch_components") as mock_launch,
            patch("fastdeploy.engine.engine.EngineService.start") as mock_engine_start,
            patch("fastdeploy.engine.engine.EngineService.create_data_processor") as mock_create_dp,
            patch("fastdeploy.engine.engine.LLMEngine._start_worker_service") as mock_start_worker,
            patch("fastdeploy.engine.engine.LLMEngine.check_worker_initialize_status") as mock_check_status,
            patch("fastdeploy.platforms.current_platform.is_intel_hpu", return_value=False),
            patch("fastdeploy.engine.engine.time.sleep"),
            patch("fastdeploy.engine.engine.threading.Thread") as mock_thread,
        ):

            # Mock IPC signals
            mock_loaded_signal = Mock()
            mock_loaded_signal.value = np.array([1], dtype=np.int32)

            engine = LLMEngine(cfg)
            engine.loaded_model_signal = mock_loaded_signal
            engine.worker_init_status = {"finished": True}
            engine.ipc_signal_suffix = 6778

            # Mock cache manager start
            mock_cache_processes = [Mock()]
            engine.engine.start_cache_service = Mock(return_value=mock_cache_processes)

            # Mock check thread
            mock_check_thread = Mock()
            mock_check_thread.is_alive.return_value = False
            mock_check_thread.start = Mock()
            mock_check_thread.join = Mock()
            mock_thread.return_value = mock_check_thread

            # Call start with api_server_pid
            engine.start(api_server_pid=12345)

            # Verify cache manager was started for prefill mode
            engine.engine.start_cache_service.assert_called()

    def test_start_worker_launch_failure(self):
        """Test worker launch failure error handling (lines 199-200)"""
        cfg = self._create_mock_cfg()

        with (
            patch("fastdeploy.engine.engine.LLMEngine._init_worker_signals") as mock_init_signals,
            patch("fastdeploy.engine.engine.LLMEngine.launch_components") as mock_launch,
            patch("fastdeploy.engine.engine.EngineService.start") as mock_engine_start,
            patch("fastdeploy.engine.engine.EngineService.create_data_processor") as mock_create_dp,
            patch("fastdeploy.engine.engine.LLMEngine._start_worker_service") as mock_start_worker,
            patch("fastdeploy.engine.engine.LLMEngine.check_worker_initialize_status") as mock_check_status,
            patch("fastdeploy.engine.engine.time.sleep"),
            patch("fastdeploy.engine.engine.threading.Thread") as mock_thread,
            patch("fastdeploy.engine.engine.console_logger") as mock_logger,
        ):

            # Mock IPC signals
            mock_loaded_signal = Mock()
            mock_loaded_signal.value = np.array([1], dtype=np.int32)

            engine = LLMEngine(cfg)
            engine.loaded_model_signal = mock_loaded_signal
            engine.worker_init_status = {"finished": True}
            engine.ipc_signal_suffix = 6778

            # Mock check thread that indicates failure
            mock_check_thread = Mock()
            mock_check_thread.is_alive.return_value = False
            mock_check_thread.start = Mock()
            mock_check_thread.join = Mock()
            mock_thread.return_value = mock_check_thread

            # Mock result container with worker not alive
            result_container = {"worker_is_alive": False}

            def check_func(res):
                res["worker_is_alive"] = False

            engine.check_worker_initialize_status_func_thread = mock_check_thread

            # Manually set up the failure scenario
            engine.check_worker_initialize_status = lambda: True
            engine.worker_init_status = {"finished": True}

            # Mock the thread target to set failure
            def mock_target(res):
                res["worker_is_alive"] = False
                engine.running = False

            mock_check_thread.target = mock_target

            result = engine.start()

            # Should return False due to worker failure
            self.assertFalse(result)

    def test_get_generated_result(self):
        """Test _get_generated_result method (line 228)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service:
            engine = LLMEngine(cfg)

            # Mock scheduler get_results
            mock_results = [{"result": "test"}]
            engine.engine.scheduler.get_results = Mock(return_value=mock_results)

            result = engine._get_generated_result()
            self.assertEqual(result, mock_results)
            engine.engine.scheduler.get_results.assert_called_once()

    def test_add_requests_sampling_params(self):
        """Test add_requests sampling params handling (lines 261-263, 266-270)"""
        cfg = self._create_mock_cfg()

        with (
            patch("fastdeploy.engine.engine.EngineService") as mock_engine_service,
            patch("fastdeploy.engine.engine.Request.from_dict") as mock_request_from_dict,
        ):

            engine = LLMEngine(cfg)

            # Mock request creation
            mock_request = Mock()
            mock_request_from_dict.return_value = mock_request
            mock_request.get = Mock(
                side_effect=lambda key: {"max_tokens": 100, "min_tokens": 10, "stop_seqs_len": None}.get(key)
            )
            mock_request.set = Mock()

            # Mock data processor
            mock_data_processor = Mock()
            mock_data_processor.process_request = Mock(return_value=mock_request)
            engine.engine.data_processor = mock_data_processor

            # Mock scheduler
            mock_scheduler = Mock()
            engine.engine.scheduler = mock_scheduler

            # Test with sampling_params containing temperature
            sampling_params = Mock()
            sampling_params.temperature = 0.0  # Very small temperature
            sampling_params.asdict = Mock(return_value={"temperature": 0.0})

            task = {"prompt": "test", "max_tokens": 50}

            with (
                patch("fastdeploy.engine.engine.time.time", return_value=1234567890.0),
                patch("fastdeploy.engine.engine.llm_logger") as mock_llm_logger,
            ):

                engine.add_requests(task, sampling_params)

                # Verify temperature was adjusted
                self.assertEqual(sampling_params.temperature, 1e-06)
                mock_request.sampling_params = sampling_params

                # Verify request processing was called
                mock_data_processor.process_request.assert_called_once()
                mock_scheduler.put_requests.assert_called_once()

    def test_add_requests_length_validation(self):
        """Test input length validation in add_requests (lines 295-297)"""
        cfg = self._create_mock_cfg()
        cfg.model_config.max_model_len = 100

        with (
            patch("fastdeploy.engine.engine.EngineService") as mock_engine_service,
            patch("fastdeploy.engine.engine.Request.from_dict") as mock_request_from_dict,
            patch("fastdeploy.engine.engine.llm_logger") as mock_logger,
        ):

            engine = LLMEngine(cfg)

            # Mock request with very long input
            mock_request = Mock()
            mock_request_from_dict.return_value = mock_request
            mock_request.prompt_token_ids_len = 150  # Longer than max_model_len
            mock_request.get = Mock(
                side_effect=lambda key: {"max_tokens": 50, "min_tokens": 10, "stop_seqs_len": None}.get(key)
            )

            task = {"prompt": "very long prompt" * 50}

            with self.assertRaises(Exception) as context:
                engine.add_requests(task)

            # Verify error was raised
            mock_logger.error.assert_called()
            self.assertIn("exceeds the limit", str(context.exception))

    def test_add_requests_guided_decoding_none(self):
        """Test guided decoding when checker is None (lines 320-330)"""
        cfg = self._create_mock_cfg()

        with (
            patch("fastdeploy.engine.engine.EngineService") as mock_engine_service,
            patch("fastdeploy.engine.engine.Request.from_dict") as mock_request_from_dict,
            patch("fastdeploy.engine.engine.llm_logger") as mock_logger,
        ):

            engine = LLMEngine(cfg)
            engine.guided_decoding_checker = None  # No checker available

            # Mock request with guided input
            mock_request = Mock()
            mock_request_from_dict.return_value = mock_request
            mock_request.prompt_token_ids_len = 50
            mock_request.guided_json = {"test": "guided"}
            mock_request.get = Mock(
                side_effect=lambda key: {"max_tokens": 50, "min_tokens": 10, "stop_seqs_len": None}.get(key)
            )

            task = {"prompt": "test prompt", "guided_json": {"test": "guided"}}

            with self.assertRaises(Exception) as context:
                engine.add_requests(task)

            # Verify error was raised
            mock_logger.error.assert_called()
            self.assertIn("guided_backend is None", str(context.exception))

    def test_exit_sub_services_cleanup(self):
        """Test _exit_sub_services cleanup operations (lines 415-450)"""
        cfg = self._create_mock_cfg()

        with (
            patch("fastdeploy.engine.engine.EngineService") as mock_engine_service,
            patch("fastdeploy.engine.engine.os.getpgid") as mock_getpgid,
            patch("fastdeploy.engine.engine.os.killpg") as mock_killpg,
        ):

            engine = LLMEngine(cfg)
            engine.running = True

            # Mock cache manager processes
            mock_process1 = Mock()
            mock_process1.pid = 1234
            mock_process2 = Mock()
            mock_process2.pid = 5678

            engine.cache_manager_processes = [mock_process1, mock_process2]

            # Mock cache manager attributes
            mock_cache_mgr = Mock()
            mock_cache_mgr.shm_cache_task_flag_broadcast = Mock(clear=lambda: None)
            mock_cache_mgr.cache_ready_signal = Mock(clear=lambda: None)
            engine.engine.resource_manager.cache_manager = mock_cache_mgr

            # Mock worker process
            mock_worker_proc = Mock()
            mock_worker_proc.pid = 9999
            engine.worker_proc = mock_worker_proc

            # Mock zmq server
            mock_zmq_server = Mock()
            engine.zmq_server = mock_zmq_server

            # Mock dp processes
            mock_dp_proc1 = Mock()
            mock_dp_proc2 = Mock()
            engine.dp_processed = [mock_dp_proc1, mock_dp_proc2]

            # Mock dp queue servers
            mock_queue_server1 = Mock()
            mock_queue_server2 = Mock()
            engine.dp_engine_worker_queue_server = [mock_queue_server1, mock_queue_server2]

            mock_getpgid.side_effect = lambda pid: pid
            mock_killpg.return_value = None

            engine._exit_sub_services()

            # Verify all cleanup operations were called
            mock_cache_mgr.shm_cache_task_flag_broadcast.clear.assert_called()
            mock_cache_mgr.cache_ready_signal.clear.assert_called()
            self.assertEqual(mock_killpg.call_count, 2)  # cache manager processes
            mock_zmq_server.close.assert_called()
            mock_dp_proc1.join.assert_called()
            mock_dp_proc2.join.assert_called()
            mock_queue_server1.cleanup.assert_called()
            mock_queue_server2.cleanup.assert_called()

    def test_generate_method_basic(self):
        """Test generate method basic flow (lines 660-689)"""
        cfg = self._create_mock_cfg()

        with (
            patch("fastdeploy.engine.engine.EngineService") as mock_engine_service,
            patch("fastdeploy.engine.engine.llm_logger") as mock_llm_logger,
        ):

            engine = LLMEngine(cfg)

            prompts = {"prompt": "Test prompt"}
            stream = False

            # Mock _format_and_add_data
            with patch.object(engine, "_format_and_add_data", return_value="test-req-id") as mock_format:
                # Mock _get_generated_tokens
                with patch.object(
                    engine,
                    "_get_generated_tokens",
                    return_value=[Mock(finished=True, to_dict=lambda: {"result": "test"})],
                ) as mock_get_tokens:
                    # Mock data processor
                    mock_processor = Mock()
                    mock_processor.process_response = Mock(return_value=Mock(to_dict=lambda: {"output": "test"}))
                    engine.engine.data_processor = mock_processor

                    # Collect results
                    results = list(engine.generate(prompts, stream))

                    # Verify method calls
                    mock_format.assert_called_once_with(prompts)
                    mock_get_tokens.assert_called_once_with("test-req-id")
                    self.assertEqual(len(results), 1)

    def test_check_health_healthy(self):
        """Test check_health method healthy state (lines 711-716)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service:
            engine = LLMEngine(cfg)

            # Mock worker healthy signal
            mock_signal = Mock()
            import time

            mock_signal.value = np.array([time.time() - 10], dtype=np.float64)  # Recent timestamp
            engine.engine.worker_healthy_live_signal = mock_signal

            is_healthy, message = engine.check_health(time_interval_threashold=30)

            # Should be healthy
            self.assertTrue(is_healthy)
            self.assertEqual(message, "")

    def test_check_health_unhealthy(self):
        """Test check_health method unhealthy state (lines 711-716)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service:
            engine = LLMEngine(cfg)

            # Mock worker healthy signal with old timestamp
            mock_signal = Mock()
            import time

            mock_signal.value = np.array([time.time() - 100], dtype=np.float64)  # Old timestamp
            engine.engine.worker_healthy_live_signal = mock_signal

            is_healthy, message = engine.check_health(time_interval_threashold=50)

            # Should be unhealthy
            self.assertFalse(is_healthy)
            self.assertIn("Not Healthy", message)


if __name__ == "__main__":
    unittest.main()
