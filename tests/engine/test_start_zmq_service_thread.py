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

import threading
import unittest
from unittest.mock import MagicMock, Mock, patch

from fastdeploy.engine.common_engine import EngineService


class TestStartZmqServiceThread(unittest.TestCase):
    """Test case for start_zmq_service method thread creation"""

    def setUp(self):
        """Set up for each test method"""
        # Create a mock config to avoid model loading issues
        self.cfg = MagicMock()
        self.cfg.parallel_config.local_engine_worker_queue_port = 6808
        self.cfg.parallel_config.local_data_parallel_id = 0
        self.cfg.scheduler_config.splitwise_role = "mixed"
        self.cfg.model_config.enable_mm = False

        # Mock the scheduler and other dependencies
        self.cfg.scheduler_config.scheduler.return_value = MagicMock()
        self.cfg.max_num_partial_prefills = 1
        self.cfg.scheduler_config.max_num_batched_tokens = 1024
        self.cfg.cache_config.block_size = 16
        self.cfg.cache_config.enable_prefix_caching = False
        self.cfg.structured_outputs_config.guided_decoding_backend = "off"
        self.cfg.eplb_config.enable_eplb = False
        self.cfg.router_config.router = None
        self.cfg.max_prefill_batch = 1
        self.cfg.limit_mm_per_prompt = 1
        self.cfg.mm_processor_kwargs = {}
        self.cfg.tool_parser = None
        self.cfg.cache_config.num_gpu_blocks_override = 4
        self.cfg.worker_num_per_node = 1
        self.cfg.parallel_config.tensor_parallel_size = 1
        self.cfg.parallel_config.data_parallel_size = 1
        self.cfg.parallel_config.device_ids = "0"
        self.cfg.parallel_config.engine_worker_queue_port = [6808]
        self.cfg.cache_config.local_cache_queue_port = 6809
        self.cfg.master_ip = "127.0.0.1"
        self.cfg.host_ip = "127.0.0.1"
        self.cfg.enable_decode_cache_task = False
        self.cfg.splitwise_version = "v1"
        self.cfg.register_info = {}
        self.cfg.parallel_config.enable_expert_parallel = False
        self.cfg.parallel_config.local_engine_worker_queue_port = 6808
        self.cfg.parallel_config.engine_worker_queue_port = 6808
        self.cfg.cache_config.enc_dec_block_num = 0
        self.cfg.cache_config.max_block_num_per_seq = 100
        self.cfg.model_config.max_model_len = 1024
        self.cfg.model_config.enable_mm = False
        self.cfg.structured_outputs_config.disable_any_whitespace = False
        self.cfg.structured_outputs_config.reasoning_parser = None
        self.cfg.scheduler_config.splitwise_role = "mixed"
        self.cfg.cache_config.enable_chunked_prefill = True
        self.cfg.scheduler_config.max_num_seqs = 1
        self.cfg.cache_config.num_cpu_blocks = 0
        self.cfg.cache_config.total_block_num = 100
        self.cfg.cache_config.prefill_kvcache_block_num = 100
        self.cfg.speculative_config = MagicMock()
        self.cfg.cache_config.max_block_num_per_seq = 100
        self.cfg.cache_config.enc_dec_block_num = 0
        self.cfg.cache_config.block_size = 16
        self.cfg.cache_config.enable_prefix_caching = False
        self.cfg.cache_config.num_gpu_blocks_override = 4
        self.cfg.cache_config.kv_cache_ratio = 0.75

    def test_start_zmq_service_thread_creation(self):
        """Test that insert_task_to_scheduler_thread is created correctly in start_zmq_service"""

        # Mock the dependencies to avoid actual ZMQ setup and queue connections
        with (
            patch("fastdeploy.engine.common_engine.ZmqTcpServer") as mock_zmq_tcp,
            patch("fastdeploy.engine.common_engine.ZmqIpcServer") as mock_zmq_ipc,
            patch("fastdeploy.engine.common_engine.InternalAdapter") as mock_internal_adapter,
            patch("fastdeploy.engine.common_engine.envs.FD_ENABLE_INTERNAL_ADAPTER", True),
            patch("fastdeploy.engine.common_engine.time.sleep"),
            patch("fastdeploy.engine.common_engine.EngineWorkerQueue") as mock_worker_queue,
            patch("fastdeploy.engine.common_engine.EngineCacheQueue") as mock_cache_queue,
        ):
            mock_zmq_ipc = mock_zmq_ipc if mock_zmq_ipc is not None else mock_zmq_ipc
            mock_internal_adapter = (
                mock_internal_adapter if mock_internal_adapter is not None else mock_internal_adapter
            )
            # Create mock queue instances
            mock_queue_instance = MagicMock()
            mock_queue_instance.get_server_port.return_value = 6808
            mock_worker_queue.return_value = mock_queue_instance
            mock_cache_queue.return_value = mock_queue_instance

            # Create engine service without starting full services
            engine = EngineService(self.cfg, start_queue=False, use_async_llm=False)
            engine.running = True  # Add running attribute to prevent thread errors

            # Mock the send_response_server.recv_result_handle method
            mock_recv_result_handle = Mock()

            # Create mock servers
            mock_recv_server = Mock()
            mock_send_server = Mock()
            mock_send_server.recv_result_handle = mock_recv_result_handle

            mock_zmq_tcp.side_effect = [mock_recv_server, mock_send_server]

            # Call start_zmq_service with a test PID
            api_server_pid = "test_pid_12345"
            engine.start_zmq_service(api_server_pid)

            # Verify that insert_task_to_scheduler_thread was created
            self.assertTrue(hasattr(engine, "insert_task_to_scheduler_thread"))
            self.assertIsInstance(engine.insert_task_to_scheduler_thread, threading.Thread)

            # Verify thread is configured correctly - use _target instead of target for Thread objects
            self.assertEqual(engine.insert_task_to_scheduler_thread._target, engine._run_insert_zmq_task_to_scheduler)
            self.assertTrue(engine.insert_task_to_scheduler_thread.daemon)

            # Verify thread was started
            self.assertTrue(
                engine.insert_task_to_scheduler_thread.is_alive()
                or engine.insert_task_to_scheduler_thread.ident is not None
            )

            # Clean up
            engine.running = False  # Stop the thread loop
            if engine.insert_task_to_scheduler_thread.is_alive():
                engine.insert_task_to_scheduler_thread.join(timeout=1)

    def test_run_insert_zmq_task_to_scheduler_success(self):
        """Test _run_insert_zmq_task_to_scheduler successful execution"""

        with (
            patch("fastdeploy.engine.common_engine.EngineWorkerQueue") as mock_worker_queue,
            patch("fastdeploy.engine.common_engine.EngineCacheQueue") as mock_cache_queue,
            patch("fastdeploy.engine.common_engine.asyncio.run") as mock_asyncio_run,
        ):

            # Create mock queue instances
            mock_queue_instance = MagicMock()
            mock_queue_instance.get_server_port.return_value = 6808
            mock_worker_queue.return_value = mock_queue_instance
            mock_cache_queue.return_value = mock_queue_instance

            engine = EngineService(self.cfg, start_queue=False, use_async_llm=False)
            engine.running = True

            # Mock the async method to avoid actual asyncio loop
            mock_asyncio_run.return_value = None

            # Call the method
            engine._run_insert_zmq_task_to_scheduler()

            # Verify asyncio.run was called once
            mock_asyncio_run.assert_called_once()
            # Verify it was called with a coroutine object (the async method)
            call_args = mock_asyncio_run.call_args[0]
            self.assertEqual(len(call_args), 1)
            # Check that the argument is a coroutine from the correct method
            import inspect

            self.assertTrue(inspect.iscoroutine(call_args[0]))
            # Check the coroutine's function name matches our target method
            self.assertEqual(call_args[0].cr_code.co_name, "_insert_zmq_task_to_scheduler")

    def test_run_insert_zmq_task_to_scheduler_exception_handling(self):
        """Test _run_insert_zmq_task_to_scheduler exception handling"""

        with (
            patch("fastdeploy.engine.common_engine.EngineWorkerQueue") as mock_worker_queue,
            patch("fastdeploy.engine.common_engine.EngineCacheQueue") as mock_cache_queue,
            patch("fastdeploy.engine.common_engine.asyncio.run") as mock_asyncio_run,
        ):

            # Create mock queue instances
            mock_queue_instance = MagicMock()
            mock_queue_instance.get_server_port.return_value = 6808
            mock_worker_queue.return_value = mock_queue_instance
            mock_cache_queue.return_value = mock_queue_instance

            engine = EngineService(self.cfg, start_queue=False, use_async_llm=False)
            engine.running = True

            # Mock asyncio.run to raise an exception
            test_exception = Exception("Test async loop error")
            mock_asyncio_run.side_effect = test_exception

            # Mock the logger
            engine.llm_logger = MagicMock()

            # Call the method
            engine._run_insert_zmq_task_to_scheduler()

            # Verify the exception was caught and logged
            mock_asyncio_run.assert_called_once()
            # Verify it was called with a coroutine object
            call_args = mock_asyncio_run.call_args[0]
            self.assertEqual(len(call_args), 1)
            import inspect

            self.assertTrue(inspect.iscoroutine(call_args[0]))
            self.assertEqual(call_args[0].cr_code.co_name, "_insert_zmq_task_to_scheduler")
            engine.llm_logger.error.assert_called_once_with("Async loop crashed: Test async loop error")

    def test_run_insert_zmq_task_to_scheduler_various_exceptions(self):
        """Test _run_insert_zmq_task_to_scheduler with different types of exceptions"""

        with (
            patch("fastdeploy.engine.common_engine.EngineWorkerQueue") as mock_worker_queue,
            patch("fastdeploy.engine.common_engine.EngineCacheQueue") as mock_cache_queue,
            patch("fastdeploy.engine.common_engine.asyncio.run") as mock_asyncio_run,
        ):

            # Create mock queue instances
            mock_queue_instance = MagicMock()
            mock_queue_instance.get_server_port.return_value = 6808
            mock_worker_queue.return_value = mock_queue_instance
            mock_cache_queue.return_value = mock_queue_instance

            engine = EngineService(self.cfg, start_queue=False, use_async_llm=False)
            engine.running = True

            # Test a couple of key exception types
            test_exceptions = [
                RuntimeError("Runtime error"),
                ValueError("Value error"),
            ]

            for test_exception in test_exceptions:
                # Reset the mock
                mock_asyncio_run.reset_mock()
                engine.llm_logger = MagicMock()

                # Mock asyncio.run to raise the test exception
                mock_asyncio_run.side_effect = test_exception

                # Call the method
                engine._run_insert_zmq_task_to_scheduler()

                # Verify the exception was caught and logged
                mock_asyncio_run.assert_called_once()
                # Verify it was called with a coroutine object
                call_args = mock_asyncio_run.call_args[0]
                self.assertEqual(len(call_args), 1)
                import inspect

                self.assertTrue(inspect.iscoroutine(call_args[0]))
                self.assertEqual(call_args[0].cr_code.co_name, "_insert_zmq_task_to_scheduler")
                engine.llm_logger.error.assert_called_once_with(f"Async loop crashed: {test_exception}")


if __name__ == "__main__":
    unittest.main()
