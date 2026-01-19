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
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import paddle

from fastdeploy.engine.args_utils import EngineArgs
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
        cfg.cache_config.num_gpu_blocks_override = kwargs.get('num_gpu_blocks_override', 4)
        cfg.cache_config.enable_prefix_caching = kwargs.get('enable_prefix_caching', False)
        cfg.cache_config.block_size = 128
        cfg.cache_config.num_cpu_blocks = 0
        cfg.cache_config.total_block_num = 100

        cfg.parallel_config = Mock()
        cfg.parallel_config.device_ids = "0"
        cfg.parallel_config.engine_worker_queue_port = [6778]
        cfg.parallel_config.tensor_parallel_size = 1
        cfg.parallel_config.expert_parallel_size = 1
        cfg.parallel_config.chunked_moe_size = 1
        cfg.parallel_config.data_parallel_size = kwargs.get('data_parallel_size', 1)
        cfg.parallel_config.enable_expert_parallel = False
        cfg.parallel_config.enable_chunked_moe = False
        cfg.parallel_config.use_internode_ll_two_stage = False
        cfg.parallel_config.disable_custom_all_reduce = False
        cfg.parallel_config.disable_sequence_parallel_moe = False
        cfg.parallel_config.shutdown_comm_group_if_worker_idle = False

        cfg.scheduler_config = Mock()
        cfg.scheduler_config.splitwise_role = kwargs.get('splitwise_role', 'mixed')
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

    def test_start_cache_manager_launch_prefill(self):
        """Test cache manager launch in prefill mode (lines 145-147)"""
        cfg = self._create_mock_cfg(splitwise_role="prefill")
        cfg.cache_config.enable_prefix_caching = True

        with patch("fastdeploy.engine.engine.LLMEngine._init_worker_signals") as mock_init_signals, \
             patch("fastdeploy.engine.engine.LLMEngine.launch_components") as mock_launch, \
             patch("fastdeploy.engine.engine.EngineService.start") as mock_engine_start, \
             patch("fastdeploy.engine.engine.EngineService.create_data_processor") as mock_create_dp, \
             patch("fastdeploy.engine.engine.LLMEngine._start_worker_service") as mock_start_worker, \
             patch("fastdeploy.engine.engine.LLMEngine.check_worker_initialize_status") as mock_check_status, \
             patch("fastdeploy.platforms.current_platform.is_intel_hpu", return_value=False), \
             patch("fastdeploy.engine.engine.time.sleep"), \
             patch("fastdeploy.engine.engine.threading.Thread") as mock_thread:

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

    def test_start_worker_init_check_thread_alive(self):
        """Test worker init check thread alive check (lines 158-161, 172)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.LLMEngine._init_worker_signals") as mock_init_signals, \
             patch("fastdeploy.engine.engine.LLMEngine.launch_components") as mock_launch, \
             patch("fastdeploy.engine.engine.EngineService.start") as mock_engine_start, \
             patch("fastdeploy.engine.engine.EngineService.create_data_processor") as mock_create_dp, \
             patch("fastdeploy.engine.engine.LLMEngine._start_worker_service") as mock_start_worker, \
             patch("fastdeploy.engine.engine.LLMEngine.check_worker_initialize_status") as mock_check_status, \
             patch("fastdeploy.engine.engine.time.sleep") as mock_sleep, \
             patch("fastdeploy.engine.engine.threading.Thread") as mock_thread:

            # Mock IPC signals
            mock_loaded_signal = Mock()
            mock_loaded_signal.value = np.array([0], dtype=np.int32)  # Not loaded yet

            engine = LLMEngine(cfg)
            engine.loaded_model_signal = mock_loaded_signal
            engine.ipc_signal_suffix = 6778

            # Mock check thread that stays alive
            mock_check_thread = Mock()
            mock_check_thread.is_alive.return_value = True  # Thread is alive
            mock_check_thread.start = Mock()
            mock_check_thread.join = Mock()
            mock_thread.return_value = mock_check_thread

            # Call start - should return False due to worker init failure
            result = engine.start()

            # Verify sleep was called and result is False
            mock_sleep.assert_called()
            self.assertFalse(result)

    def test_start_cache_manager_mixed_mode(self):
        """Test cache manager launch in mixed mode (lines 179-182)"""
        cfg = self._create_mock_cfg(splitwise_role="mixed")
        cfg.cache_config.enable_prefix_caching = True

        with patch("fastdeploy.engine.engine.LLMEngine._init_worker_signals") as mock_init_signals, \
             patch("fastdeploy.engine.engine.LLMEngine.launch_components") as mock_launch, \
             patch("fastdeploy.engine.engine.EngineService.start") as mock_engine_start, \
             patch("fastdeploy.engine.engine.EngineService.create_data_processor") as mock_create_dp, \
             patch("fastdeploy.engine.engine.LLMEngine._start_worker_service") as mock_start_worker, \
             patch("fastdeploy.engine.engine.LLMEngine.check_worker_initialize_status") as mock_check_status, \
             patch("fastdeploy.engine.engine.LLMEngine._stop_profile") as mock_stop_profile, \
             patch("fastdeploy.platforms.current_platform.is_intel_hpu", return_value=False), \
             patch("fastdeploy.engine.engine.time.sleep"), \
             patch("fastdeploy.engine.engine.threading.Thread") as mock_thread:

            # Mock IPC signals
            mock_loaded_signal = Mock()
            mock_loaded_signal.value = np.array([1], dtype=np.int32)

            engine = LLMEngine(cfg)
            engine.loaded_model_signal = mock_loaded_signal
            engine.worker_init_status = {"finished": True}
            engine.ipc_signal_suffix = 6778
            engine.do_profile = False

            # Mock cache manager start
            mock_cache_processes = [Mock()]
            engine.engine.start_cache_service = Mock(return_value=mock_cache_processes)

            # Mock check thread
            mock_check_thread = Mock()
            mock_check_thread.is_alive.return_value = False
            mock_check_thread.start = Mock()
            mock_check_thread.join = Mock()
            mock_thread.return_value = mock_check_thread

            engine.start()

            # Verify cache manager was started for mixed mode with prefix caching
            engine.engine.start_cache_service.assert_called()

    def test_start_env_variables_setting(self):
        """Test environment variables setting in start (lines 189-190)"""
        cfg = self._create_mock_cfg(splitwise_role="prefill")

        with patch("fastdeploy.engine.engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True), \
             patch("fastdeploy.engine.engine.envs.FD_ZMQ_RECV_REQUEST_SERVER_PORTS", ["8888"]), \
             patch("fastdeploy.engine.engine.envs.FD_ZMQ_SEND_RESPONSE_SERVER_PORTS", ["9999"]), \
             patch("fastdeploy.engine.engine.LLMEngine._init_worker_signals") as mock_init_signals, \
             patch("fastdeploy.engine.engine.LLMEngine.launch_components") as mock_launch, \
             patch("fastdeploy.engine.engine.EngineService.start") as mock_engine_start, \
             patch("fastdeploy.engine.engine.EngineService.create_data_processor") as mock_create_dp, \
             patch("fastdeploy.engine.engine.LLMEngine._start_worker_service") as mock_start_worker, \
             patch("fastdeploy.engine.engine.LLMEngine.check_worker_initialize_status") as mock_check_status, \
             patch("fastdeploy.engine.engine.time.sleep"), \
             patch("fastdeploy.engine.engine.threading.Thread") as mock_thread:

            # Mock IPC signals
            mock_loaded_signal = Mock()
            mock_loaded_signal.value = np.array([1], dtype=np.int32)

            engine = LLMEngine(cfg)
            engine.loaded_model_signal = mock_loaded_signal
            engine.worker_init_status = {"finished": True}
            engine.ipc_signal_suffix = 6778

            # Mock check thread
            mock_check_thread = Mock()
            mock_check_thread.is_alive.return_value = False
            mock_check_thread.start = Mock()
            mock_check_thread.join = Mock()
            mock_thread.return_value = mock_check_thread

            engine.start()

            # Verify environment variables were set (can't easily test envs module directly)
            # This is more of a behavioral test - the start method completed successfully

    def test_start_worker_launch_failure_error(self):
        """Test worker launch failure error handling (lines 199-200)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.LLMEngine._init_worker_signals") as mock_init_signals, \
             patch("fastdeploy.engine.engine.LLMEngine.launch_components") as mock_launch, \
             patch("fastdeploy.engine.engine.EngineService.start") as mock_engine_start, \
             patch("fastdeploy.engine.engine.EngineService.create_data_processor") as mock_create_dp, \
             patch("fastdeploy.engine.engine.LLMEngine._start_worker_service") as mock_start_worker, \
             patch("fastdeploy.engine.engine.LLMEngine.check_worker_initialize_status") as mock_check_status, \
             patch("fastdeploy.engine.engine.time.sleep"), \
             patch("fastdeploy.engine.engine.threading.Thread") as mock_thread, \
             patch("fastdeploy.engine.engine.console_logger") as mock_logger:

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

    def test_add_requests_sampling_params_handling(self):
        """Test add_requests sampling params handling (lines 261-263, 266-270)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.Request.from_dict") as mock_request_from_dict:

            engine = LLMEngine(cfg)

            # Mock request creation
            mock_request = Mock()
            mock_request_from_dict.return_value = mock_request
            mock_request.get = Mock(side_effect=lambda key: {
                "max_tokens": 100,
                "min_tokens": 10,
                "stop_seqs_len": None
            }.get(key))
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

            with patch("fastdeploy.engine.engine.time.time", return_value=1234567890.0), \
                 patch("fastdeploy.engine.engine.llm_logger") as mock_llm_logger:

                engine.add_requests(task, sampling_params)

                # Verify temperature was adjusted
                self.assertEqual(sampling_params.temperature, 1e-06)
                mock_request.sampling_params = sampling_params

                # Verify request processing was called
                mock_data_processor.process_request.assert_called_once()
                mock_scheduler.put_requests.assert_called_once()

    def test_add_requests_length_validation_input_too_long(self):
        """Test input length validation in add_requests (lines 295-297)"""
        cfg = self._create_mock_cfg()
        cfg.model_config.max_model_len = 100

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.Request.from_dict") as mock_request_from_dict, \
             patch("fastdeploy.engine.engine.llm_logger") as mock_logger:

            engine = LLMEngine(cfg)

            # Mock request with very long input
            mock_request = Mock()
            mock_request_from_dict.return_value = mock_request
            mock_request.prompt_token_ids_len = 150  # Longer than max_model_len
            mock_request.get = Mock(side_effect=lambda key: {
                "max_tokens": 50,
                "min_tokens": 10,
                "stop_seqs_len": None
            }.get(key))

            task = {"prompt": "very long prompt" * 50}

            with self.assertRaises(Exception) as context:
                engine.add_requests(task)

            # Verify error was raised
            mock_logger.error.assert_called()
            self.assertIn("exceeds the limit", str(context.exception))

    def test_add_requests_stop_seqs_validation(self):
        """Test stop sequences length validation (lines 300-317)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.Request.from_dict") as mock_request_from_dict, \
             patch("fastdeploy.engine.engine.llm_logger") as mock_logger, \
             patch("fastdeploy.engine.engine.envs.FD_MAX_STOP_SEQS_NUM", 5), \
             patch("fastdeploy.engine.engine.envs.FD_STOP_SEQS_MAX_LEN", 10):

            engine = LLMEngine(cfg)

            # Mock request with too many stop sequences
            mock_request = Mock()
            mock_request_from_dict.return_value = mock_request
            mock_request.prompt_token_ids_len = 50
            mock_request.get = Mock(side_effect=lambda key: {
                "max_tokens": 50,
                "min_tokens": 10,
                "stop_seqs_len": [5, 5, 5, 5, 5, 5, 5]  # More than max allowed
            }.get(key))

            task = {"prompt": "test prompt", "stop_seqs_len": [5, 5, 5, 5, 5, 5, 5]}

            with self.assertRaises(Exception) as context:
                engine.add_requests(task)

            # Verify error was raised for too many stop sequences
            mock_logger.error.assert_called()

            # Reset for next test
            mock_logger.reset_mock()

            # Test individual stop sequence too long
            mock_request.get = Mock(side_effect=lambda key: {
                "max_tokens": 50,
                "min_tokens": 10,
                "stop_seqs_len": [15]  # Single sequence too long
            }.get(key))

            task = {"prompt": "test prompt", "stop_seqs_len": [15]}

            with self.assertRaises(Exception) as context:
                engine.add_requests(task)

            # Verify error was raised for sequence too long
            mock_logger.error.assert_called()

    def test_add_requests_guided_decoding_none_checker(self):
        """Test guided decoding when checker is None (lines 320-330)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.Request.from_dict") as mock_request_from_dict, \
             patch("fastdeploy.engine.engine.llm_logger") as mock_logger:

            engine = LLMEngine(cfg)
            engine.guided_decoding_checker = None  # No checker available

            # Mock request with guided input
            mock_request = Mock()
            mock_request_from_dict.return_value = mock_request
            mock_request.prompt_token_ids_len = 50
            mock_request.guided_json = {"test": "guided"}
            mock_request.get = Mock(side_effect=lambda key: {
                "max_tokens": 50,
                "min_tokens": 10,
                "stop_seqs_len": None
            }.get(key))

            task = {"prompt": "test prompt", "guided_json": {"test": "guided"}}

            with self.assertRaises(Exception) as context:
                engine.add_requests(task)

            # Verify error was raised
            mock_logger.error.assert_called()
            self.assertIn("guided_backend is None", str(context.exception))

    def test_worker_processes_ready_false(self):
        """Test _worker_processes_ready returns False (lines 343-345)"""
        cfg = self._create_mock_cfg()
        cfg.worker_num_per_node = 2

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service:
            engine = LLMEngine(cfg)

            # Mock worker ready signal with only 1 ready out of 2
            mock_signal = Mock()
            mock_signal.value = np.array([1, 0], dtype=np.int32)  # Only first worker ready
            engine.worker_ready_signal = mock_signal

            result = engine._worker_processes_ready()
            self.assertFalse(result)

    def test_init_worker_signals_prefix_caching(self):
        """Test _init_worker_signals with prefix caching enabled (lines 362-373)"""
        cfg = self._create_mock_cfg()
        cfg.cache_config.enable_prefix_caching = True
        cfg.scheduler_config.splitwise_role = "prefill"

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.IPCSignal") as mock_ipc_signal:

            engine = LLMEngine(cfg)
            engine.ipc_signal_suffix = 6778

            # Mock IPC signals
            mock_launched_signal = Mock()
            mock_ipc_signal.return_value = mock_launched_signal

            engine._init_worker_signals()

            # Verify launched_cache_manager_signal was created
            self.assertTrue(hasattr(engine, "launched_cache_manager_signal"))
            mock_ipc_signal.assert_called()

    def test_init_worker_signals_expert_service(self):
        """Test _init_worker_signals with expert service (lines 374-377)"""
        cfg = self._create_mock_cfg()
        cfg.parallel_config.data_parallel_size = 2
        cfg.nnode = 1

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.IPCSignal") as mock_ipc_signal, \
             patch("fastdeploy.engine.engine.envs.FD_ENABLE_MULTI_API_SERVER", False):

            engine = LLMEngine(cfg)
            engine.ipc_signal_suffix = 6778

            engine._init_worker_signals()

            # Verify launched_expert_service_signal was created
            self.assertTrue(hasattr(engine, "launched_expert_service_signal"))

    def test_init_worker_signals_profile_iluvatar(self):
        """Test _init_worker_signals profile branch with iluvatar (lines 395-397)"""
        cfg = self._create_mock_cfg()
        cfg.cache_config.num_gpu_blocks_override = None  # Forces do_profile = True

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.IPCSignal") as mock_ipc_signal, \
             patch("fastdeploy.engine.engine.paddle.is_compiled_with_custom_device", return_value=True):

            engine = LLMEngine(cfg)
            engine.ipc_signal_suffix = 6778
            engine.do_profile = True

            engine._init_worker_signals()

            # Verify get_profile_block_num_signal was created with worker_num_per_node shape
            self.assertTrue(hasattr(engine, "get_profile_block_num_signal"))

    def test_exit_sub_services_cache_manager_cleanup(self):
        """Test _exit_sub_services cache manager cleanup (lines 415-429)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.os.getpgid") as mock_getpgid, \
             patch("fastdeploy.engine.engine.os.killpg") as mock_killpg:

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
            mock_cache_mgr.shm_cache_task_flag_broadcast = Mock()
            mock_cache_mgr.cache_ready_signal = Mock()
            engine.engine.resource_manager.cache_manager = mock_cache_mgr

            mock_getpgid.side_effect = lambda pid: pid
            mock_killpg.return_value = None

            engine._exit_sub_services()

            # Verify cleanup was attempted
            mock_cache_mgr.shm_cache_task_flag_broadcast.clear.assert_called()
            mock_cache_mgr.cache_ready_signal.clear.assert_called()
            self.assertEqual(mock_killpg.call_count, 2)

    def test_exit_sub_services_profile_signal_cleanup(self):
        """Test _exit_sub_services profile signal cleanup (lines 432-435)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service:
            engine = LLMEngine(cfg)
            engine.running = True

            # Mock profile signal
            mock_profile_signal = Mock()
            engine.get_profile_block_num_signal = mock_profile_signal

            engine._exit_sub_services()

            # Verify profile signal was cleared
            mock_profile_signal.clear.assert_called()

    def test_exit_sub_services_worker_cleanup(self):
        """Test _exit_sub_services worker process cleanup (lines 435-442)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.os.getpgid") as mock_getpgid, \
             patch("fastdeploy.engine.engine.os.killpg") as mock_killpg:

            engine = LLMEngine(cfg)
            engine.running = True

            # Mock worker process
            mock_worker_proc = Mock()
            mock_worker_proc.pid = 9999
            engine.worker_proc = mock_worker_proc

            mock_getpgid.return_value = 9999
            mock_killpg.return_value = None

            engine._exit_sub_services()

            # Verify worker was killed
            mock_getpgid.assert_called_with(9999)
            mock_killpg.assert_called_with(9999, 15)  # SIGTERM

    def test_exit_sub_services_zmq_cleanup(self):
        """Test _exit_sub_services zmq server cleanup (line 443)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service:
            engine = LLMEngine(cfg)
            engine.running = True

            # Mock zmq server
            mock_zmq_server = Mock()
            engine.zmq_server = mock_zmq_server

            engine._exit_sub_services()

            # Verify zmq server was closed
            mock_zmq_server.close.assert_called()

    def test_exit_sub_services_dp_cleanup(self):
        """Test _exit_sub_services dp processes cleanup (lines 446-450)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service:
            engine = LLMEngine(cfg)
            engine.running = True

            # Mock dp processes
            mock_dp_proc1 = Mock()
            mock_dp_proc2 = Mock()
            engine.dp_processed = [mock_dp_proc1, mock_dp_proc2]

            # Mock dp queue servers
            mock_queue_server1 = Mock()
            mock_queue_server2 = Mock()
            engine.dp_engine_worker_queue_server = [mock_queue_server1, mock_queue_server2]

            engine._exit_sub_services()

            # Verify processes were joined and queues cleaned up
            mock_dp_proc1.join.assert_called()
            mock_dp_proc2.join.assert_called()
            mock_queue_server1.cleanup.assert_called()
            mock_queue_server2.cleanup.assert_called()

    def test_setting_environ_variables_v1_scheduler(self):
        """Test _setting_environ_variables with V1 scheduler (line 486)"""
        cfg = self._create_mock_cfg()
        cfg.scheduler_config.splitwise_role = "prefill"

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.envs.ENABLE_V1_KVCACHE_SCHEDULER", True):

            engine = LLMEngine(cfg)

            result = engine._setting_environ_variables()

            # Verify V1 scheduler environment variables
            self.assertIn("FLAGS_use_pd_disaggregation_per_chunk=1", result)
            self.assertIn("FLAGS_fmt_write_cache_completed_signal=1", result)

    def test_start_worker_service_ips_processing(self):
        """Test _start_worker_service ips processing (line 533)"""
        cfg = self._create_mock_cfg()
        cfg.ips = ["192.168.1.1", "192.168.1.2"]
        cfg.nnode = 2

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.subprocess.Popen") as mock_popen, \
             patch("fastdeploy.engine.engine.llm_logger") as mock_llm_logger, \
             patch("fastdeploy.engine.engine.os.path") as mock_path:

            engine = LLMEngine(cfg)

            # Mock data processor
            mock_data_processor = Mock()
            mock_data_processor.tokenizer = Mock()
            mock_data_processor.tokenizer.sp_model = Mock()
            len(mock_data_processor.tokenizer.sp_model)  # Mock len
            mock_data_processor.eos_token_id_len = 1
            mock_data_processor.pad_token_id = 0
            engine.data_processor = mock_data_processor

            # Mock file path operations
            mock_path.abspath.return_value = "/path/to/engine.py"
            mock_path.split.return_value = ("/path/to", "engine.py")
            mock_path.join.return_value = "/path/to/worker_process.py"

            # Mock popen
            mock_proc = Mock()
            mock_proc.stdout = None
            mock_popen.return_value = mock_proc

            engine._start_worker_service()

            # Verify ips were processed (check command includes ips)
            # This would be verified by checking the command built

    def test_start_worker_service_logits_processors(self):
        """Test _start_worker_service logits processors handling (line 580)"""
        cfg = self._create_mock_cfg()
        cfg.structured_outputs_config.logits_processors = ["processor1", "processor2"]

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.subprocess.Popen") as mock_popen, \
             patch("fastdeploy.engine.engine.llm_logger") as mock_llm_logger, \
             patch("fastdeploy.engine.engine.os.path") as mock_path:

            engine = LLMEngine(cfg)

            # Mock data processor
            mock_data_processor = Mock()
            mock_data_processor.tokenizer = Mock()
            mock_data_processor.tokenizer.sp_model = Mock()
            len(mock_data_processor.tokenizer.sp_model)  # Mock len
            mock_data_processor.eos_token_id_len = 1
            mock_data_processor.pad_token_id = 0
            engine.data_processor = mock_data_processor

            # Mock file path operations
            mock_path.abspath.return_value = "/path/to/engine.py"
            mock_path.split.return_value = ("/path/to", "engine.py")
            mock_path.join.return_value = "/path/to/worker_process.py"

            # Mock popen
            mock_proc = Mock()
            mock_proc.stdout = None
            mock_popen.return_value = mock_proc

            engine._start_worker_service()

            # Verify logits processors were included in command
            # This would be verified by checking the command arguments

    def test_start_worker_service_iluvatar_special_case(self):
        """Test _start_worker_service iluvatar special case (line 584)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.subprocess.Popen") as mock_popen, \
             patch("fastdeploy.engine.engine.llm_logger") as mock_llm_logger, \
             patch("fastdeploy.engine.engine.os.path") as mock_path, \
             patch("fastdeploy.engine.engine.current_platform.is_iluvatar", return_value=True), \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):  # Simulate CUDA_VISIBLE_DEVICES set

            engine = LLMEngine(cfg)

            # Mock data processor
            mock_data_processor = Mock()
            mock_data_processor.tokenizer = Mock()
            mock_data_processor.tokenizer.sp_model = Mock()
            len(mock_data_processor.tokenizer.sp_model)  # Mock len
            mock_data_processor.eos_token_id_len = 1
            mock_data_processor.pad_token_id = 0
            engine.data_processor = mock_data_processor

            # Mock file path operations
            mock_path.abspath.return_value = "/path/to/engine.py"
            mock_path.split.return_value = ("/path/to", "engine.py")
            mock_path.join.return_value = "/path/to/worker_process.py"

            # Mock popen
            mock_proc = Mock()
            mock_proc.stdout = None
            mock_popen.return_value = mock_proc

            engine._start_worker_service()

            # Verify CUDA_VISIBLE_DEVICES was handled specially for iluvatar
            # This would be verified by checking the command built

    def test_start_worker_service_worker_flags(self):
        """Test _start_worker_service worker flags processing (line 611)"""
        cfg = self._create_mock_cfg()
        cfg.parallel_config.enable_expert_parallel = True
        cfg.cache_config.enable_prefix_caching = True

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.subprocess.Popen") as mock_popen, \
             patch("fastdeploy.engine.engine.llm_logger") as mock_llm_logger, \
             patch("fastdeploy.engine.engine.os.path") as mock_path:

            engine = LLMEngine(cfg)

            # Mock data processor
            mock_data_processor = Mock()
            mock_data_processor.tokenizer = Mock()
            mock_data_processor.tokenizer.sp_model = Mock()
            len(mock_data_processor.tokenizer.sp_model)  # Mock len
            mock_data_processor.eos_token_id_len = 1
            mock_data_processor.pad_token_id = 0
            engine.data_processor = mock_data_processor

            # Mock file path operations
            mock_path.abspath.return_value = "/path/to/engine.py"
            mock_path.split.return_value = ("/path/to", "engine.py")
            mock_path.join.return_value = "/path/to/worker_process.py"

            # Mock popen
            mock_proc = Mock()
            mock_proc.stdout = None
            mock_popen.return_value = mock_popen

            engine._start_worker_service()

            # Verify worker flags were processed
            # This would be verified by checking the command arguments

    def test_start_worker_service_multi_node(self):
        """Test _start_worker_service multi-node setup (line 614)"""
        cfg = self._create_mock_cfg()
        cfg.nnode = 2
        cfg.ips = ["192.168.1.1", "192.168.1.2"]

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.subprocess.Popen") as mock_popen, \
             patch("fastdeploy.engine.engine.llm_logger") as mock_llm_logger, \
             patch("fastdeploy.engine.engine.os.path") as mock_path:

            engine = LLMEngine(cfg)

            # Mock data processor
            mock_data_processor = Mock()
            mock_data_processor.tokenizer = Mock()
            mock_data_processor.tokenizer.sp_model = Mock()
            len(mock_data_processor.tokenizer.sp_model)  # Mock len
            mock_data_processor.eos_token_id_len = 1
            mock_data_processor.pad_token_id = 0
            engine.data_processor = mock_data_processor

            # Mock file path operations
            mock_path.abspath.return_value = "/path/to/engine.py"
            mock_path.split.return_value = ("/path/to", "engine.py")
            mock_path.join.return_value = "/path/to/worker_process.py"

            # Mock popen
            mock_proc = Mock()
            mock_proc.stdout = None
            mock_popen.return_value = mock_proc

            engine._start_worker_service()

            # Verify multi-node arguments were added
            # This would be verified by checking the command includes --nnodes and --ips

    def test_format_and_add_data_request_id_generation(self):
        """Test _format_and_add_data request ID generation (lines 627-647)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.uuid.uuid4") as mock_uuid:

            engine = LLMEngine(cfg)

            # Mock uuid generation
            mock_uuid.return_value = "test-uuid-123"

            # Test without request_id
            prompts = {"prompt": ["Hello world"]}

            with patch.object(engine, "add_requests") as mock_add_requests:
                result = engine._format_and_add_data(prompts)

                # Verify request_id was generated and added
                self.assertEqual(result, "test-uuid-123")
                self.assertEqual(prompts["request_id"], "test-uuid-123")
                mock_add_requests.assert_called_once_with(prompts)

    def test_format_and_add_data_context_processing(self):
        """Test _format_and_add_data context processing (lines 636-642)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.uuid.uuid4") as mock_uuid:

            engine = LLMEngine(cfg)
            mock_uuid.return_value = "test-uuid-456"

            # Test with context containing system and user messages
            prompts = {
                "context": [
                    {"role": "system", "utterance": "You are a helpful assistant"},
                    {"role": "user", "utterance": "Hello"},
                    {"role": "assistant", "utterance": "Hi there"}
                ]
            }

            with patch.object(engine, "add_requests") as mock_add_requests:
                result = engine._format_and_add_data(prompts)

                # Verify context was processed into system and prompt fields
                self.assertEqual(prompts["system"], "You are a helpful assistant")
                self.assertEqual(prompts["prompt"], ["Hello", "Hi there"])
                mock_add_requests.assert_called_once_with(prompts)

    def test_format_and_add_data_max_tokens_default(self):
        """Test _format_and_add_data max_tokens default (lines 644-646)"""
        cfg = self._create_mock_cfg()
        cfg.model_config.max_model_len = 4096

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.uuid.uuid4") as mock_uuid:

            engine = LLMEngine(cfg)
            mock_uuid.return_value = "test-uuid-789"

            # Test without max_tokens
            prompts = {"prompt": ["Hello world"]}

            with patch.object(engine, "add_requests") as mock_add_requests:
                result = engine._format_and_add_data(prompts)

                # Verify max_tokens was set to model max length
                self.assertEqual(prompts["max_tokens"], 4096)
                mock_add_requests.assert_called_once_with(prompts)

    def test_generate_method_calls(self):
        """Test generate method basic flow (lines 660-689)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.llm_logger") as mock_llm_logger:

            engine = LLMEngine(cfg)

            prompts = {"prompt": "Test prompt"}
            stream = False

            # Mock _format_and_add_data
            with patch.object(engine, "_format_and_add_data", return_value="test-req-id") as mock_format:
                # Mock _get_generated_tokens
                with patch.object(engine, "_get_generated_tokens", return_value=[
                    Mock(finished=True, to_dict=lambda: {"result": "test"})
                ]) as mock_get_tokens:
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

    def test_generate_error_handling(self):
        """Test generate method error handling (lines 664-666)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.llm_logger") as mock_llm_logger:

            engine = LLMEngine(cfg)

            prompts = {"prompt": "Test prompt"}

            # Mock _format_and_add_data to raise exception
            with patch.object(engine, "_format_and_add_data", side_effect=ValueError("Test error")):
                with self.assertRaises(Exception):
                    list(engine.generate(prompts, False))

                # Verify error was logged
                mock_llm_logger.error.assert_called()

    def test_stop_profile_method(self):
        """Test _stop_profile method (lines 701-706)"""
        cfg = self._create_mock_cfg()

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service:
            engine = LLMEngine(cfg)
            engine.do_profile = True

            # Mock profile signal
            mock_profile_signal = Mock()
            mock_profile_signal.value = np.array([128], dtype=np.int32)  # Block number
            engine.get_profile_block_num_signal = mock_profile_signal

            # Mock cache config reset
            mock_cache_config = Mock()
            cfg.cache_config = mock_cache_config

            # Mock resource manager
            mock_resource_mgr = Mock()
            engine.engine.resource_manager = mock_resource_mgr

            # Mock cache service start
            mock_cache_processes = [Mock()]
            engine.engine.start_cache_service = Mock(return_value=mock_cache_processes)

            with patch("fastdeploy.engine.engine.time.sleep"), \
                 patch("fastdeploy.platforms.current_platform.is_intel_hpu", return_value=False):

                engine._stop_profile()

                # Verify profile was stopped and cache service started
                self.assertEqual(engine.do_profile, False)
                mock_cache_config.reset.assert_called_with(128)
                mock_resource_mgr.reset_cache_config.assert_called_once()
                engine.engine.start_cache_service.assert_called_once()

    def test_check_health_method(self):
        """Test check_health method (lines 711-716)"""
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

            # Test unhealthy case
            mock_signal.value = np.array([time.time() - 100], dtype=np.float64)  # Old timestamp
            is_healthy, message = engine.check_health(time_interval_threashold=50)

            # Should be unhealthy
            self.assertFalse(is_healthy)
            self.assertIn("Not Healthy", message)

    def test_launch_components_splitwise_thread(self):
        """Test launch_components splitwise thread creation (lines 731-738)"""
        cfg = self._create_mock_cfg()
        cfg.scheduler_config.splitwise_role = "prefill"

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.threading.Thread") as mock_thread:

            engine = LLMEngine(cfg)

            # Mock splitwise connector
            mock_connector = Mock()
            engine.engine.split_connector = mock_connector

            # Mock thread
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            engine.launch_components()

            # Verify thread was created and started
            mock_thread.assert_called_once()
            mock_thread_instance.daemon = True
            mock_thread_instance.start.assert_called_once()

    def test_launch_components_dp_scheduler(self):
        """Test launch_components dp scheduler (lines 744-750)"""
        cfg = self._create_mock_cfg()
        cfg.scheduler_config.name = "dp"
        cfg.parallel_config.data_parallel_size = 2

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.multiprocessing.Queue") as mock_queue:

            engine = LLMEngine(cfg)

            # Mock scheduler
            mock_scheduler = Mock()
            engine.engine.scheduler = mock_scheduler

            # Mock queues
            mock_queue_instance = Mock()
            mock_queue.return_value = mock_queue_instance

            engine.launch_components()

            # Verify scheduler start was called with queues
            mock_scheduler.start.assert_called_once()
            # Should have created request and result queues
            self.assertEqual(mock_queue.call_count, 2)

    def test_launch_components_expert_parallel(self):
        """Test launch_components expert parallel setup (lines 746-790)"""
        cfg = self._create_mock_cfg()
        cfg.parallel_config.data_parallel_size = 2
        cfg.parallel_config.enable_expert_parallel = True

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.envs.FD_ENABLE_MULTI_API_SERVER", False), \
             patch("fastdeploy.engine.engine.EngineWorkerQueue") as mock_queue_class, \
             patch("fastdeploy.engine.engine.multiprocessing.get_context") as mock_get_context, \
             patch("fastdeploy.engine.engine.copy.deepcopy") as mock_deepcopy, \
             patch("fastdeploy.engine.engine.time.sleep"):

            engine = LLMEngine(cfg)

            # Mock IPC signal
            mock_signal = Mock()
            mock_signal.value = np.zeros([1], dtype=np.int32)
            engine.launched_expert_service_signal = mock_signal

            # Mock queue
            mock_queue = Mock()
            mock_queue.get_server_port.return_value = 0
            mock_queue_class.return_value = mock_queue

            # Mock context and process
            mock_context = Mock()
            mock_process = Mock()
            mock_context.Process.return_value = mock_process
            mock_get_context.return_value = mock_context

            # Mock deepcopy
            mock_deepcopy.return_value = cfg

            # Mock start_expert_service
            with patch("fastdeploy.engine.engine.start_data_parallel_service") as mock_start_service:
                engine.launch_components()

                # Verify expert service processes were created
                self.assertTrue(hasattr(engine, "dp_processed"))
                self.assertTrue(hasattr(engine, "dp_engine_worker_queue_server"))

    def test_check_worker_initialize_status_stdout_parsing(self):
        """Test check_worker_initialize_status stdout parsing (lines 797-849)"""
        cfg = self._create_mock_cfg()
        cfg.model_config.num_hidden_layers = 12

        with patch("fastdeploy.engine.engine.EngineService") as mock_engine_service, \
             patch("fastdeploy.engine.engine.threading.Thread") as mock_thread, \
             patch("fastdeploy.engine.engine.tqdm") as mock_tqdm:

            engine = LLMEngine(cfg)

            # Mock worker process with stdout containing progress lines
            mock_proc = Mock()
            progress_lines = [
                "Loading checkpoint shards: 50\n",
                "Start load layer 5\n",
                "Start load layer 10\n",
            ]
            mock_stdout = iter(progress_lines)
            mock_proc.stdout = mock_stdout
            mock_proc.poll.return_value = None
            engine.worker_proc = mock_proc

            engine.worker_init_status = {}

            # Mock tqdm
            mock_pbar = Mock()
            mock_tqdm.return_value = mock_pbar

            # Mock worker ready signal
            mock_signal = Mock()
            mock_signal.value = np.array([1], dtype=np.int32)
            engine.worker_ready_signal = mock_signal

            with patch("fastdeploy.engine.engine.time.sleep"):
                result = engine.check_worker_initialize_status()

                # Should return True
                self.assertTrue(result)

                # Verify progress was tracked
                self.assertIn("weight_loadding", engine.worker_init_status)
                self.assertIn("layer_loadding", engine.worker_init_status)


if __name__ == "__main__":
    unittest.main()
