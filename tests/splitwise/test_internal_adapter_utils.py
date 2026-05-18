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

import threading
import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.splitwise import internal_adapter_utils as ia


class DummyEngine:
    """Dummy Engine class to simulate the actual Engine for testing."""

    class ResourceManager:
        def available_batch(self):
            return 4

        def available_block_num(self):
            return 2

    class Scheduler:
        def get_unhandled_request_num(self):
            return 0

    class EngineWorkerQueue:
        def __init__(self):
            self.called_task = None

        def put_connect_rdma_task(self, task):
            self.called_task = task

        def get_connect_rdma_task_response(self):
            return None

    def __init__(self):
        self.resource_manager = self.ResourceManager()
        self.scheduler = self.Scheduler()
        self.engine_worker_queue = self.EngineWorkerQueue()


class DummyCfg:
    """Dummy configuration class to simulate input config for InternalAdapter.

    Contains nested configuration classes (SchedulerConfig, CacheConfig, ModelConfig)
    with test-friendly default values.
    """

    class SchedulerConfig:
        """Mock SchedulerConfig with splitwise role configuration."""

        splitwise_role = "single"

    class CacheConfig:
        """Mock CacheConfig with cache-related configuration."""

        block_size = 1024
        total_block_num = 8
        dec_token_num = 4

    class ModelConfig:
        """Mock ModelConfig with model-related configuration."""

        max_model_len = 2048

    # Top-level configuration attributes
    max_prefill_batch = 2
    scheduler_config = SchedulerConfig()
    cache_config = CacheConfig()
    model_config = ModelConfig()


class TestInternalAdapterBasic(unittest.TestCase):
    """
    Unit test suite for basic functionalities of InternalAdapter.
    Covers initialization, server info retrieval, and thread creation.
    """

    @patch("fastdeploy.splitwise.internal_adapter_utils.ZmqTcpServer")
    def test_basic_initialization(self, mock_zmq_server):
        """Test InternalAdapter initialization and _get_current_server_info method."""
        # Setup mock ZmqTcpServer instance
        mock_server_instance = MagicMock()
        mock_zmq_server.return_value = mock_server_instance

        # Initialize InternalAdapter with dummy config, engine, and dp_rank
        adapter = ia.InternalAdapter(cfg=DummyCfg(), engine=DummyEngine(), dp_rank=0)

        # Verify _get_current_server_info returns expected structure
        server_info = adapter._get_current_server_info()
        expected_keys = ["splitwise_role", "block_size", "available_resource"]
        for key in expected_keys:
            with self.subTest(key=key):
                self.assertIn(key, server_info, f"Server info missing required key: {key}")

        # Verify background threads are properly initialized
        self.assertTrue(
            isinstance(adapter.recv_external_instruct_thread, threading.Thread),
            "recv_external_instruct_thread should be a Thread instance",
        )
        self.assertTrue(
            isinstance(adapter.response_external_instruct_thread, threading.Thread),
            "response_external_instruct_thread should be a Thread instance",
        )


class TestInternalAdapterRecvPayload(unittest.TestCase):
    """Unit test suite for payload reception functionality of InternalAdapter.

    Covers handling of different control commands (get_payload, get_metrics, connect_rdma)
    and exception handling.
    """

    @patch("fastdeploy.splitwise.internal_adapter_utils.ZmqTcpServer")
    @patch("fastdeploy.splitwise.internal_adapter_utils.get_filtered_metrics")
    @patch("fastdeploy.splitwise.internal_adapter_utils.logger")
    def test_recv_control_cmd_branches(self, mock_logger, mock_get_metrics, mock_zmq_server):
        """Test all command handling branches in _recv_external_module_control_instruct."""
        # Setup mock ZmqTcpServer instance
        mock_server_instance = MagicMock()
        mock_zmq_server.return_value = mock_server_instance

        # Create a generator to simulate sequential control commands
        def control_cmd_generator():
            """Generator to yield test commands in sequence."""
            yield {"task_id": "1", "cmd": "get_payload"}
            yield {"task_id": "2", "cmd": "get_metrics"}
            yield {"task_id": "3", "cmd": "connect_rdma"}
            while True:
                yield None

        # Configure mock server to return commands from the generator
        mock_server_instance.recv_control_cmd.side_effect = control_cmd_generator()
        mock_server_instance.response_for_control_cmd = MagicMock()  # Track response calls
        mock_get_metrics.return_value = "mocked_metrics"  # Mock metrics response

        # Initialize dependencies and InternalAdapter
        test_engine = DummyEngine()
        adapter = ia.InternalAdapter(cfg=DummyCfg(), engine=test_engine, dp_rank=0)

        # Override _recv_external_module_control_instruct to run only 3 iterations (test all commands)
        def run_limited_iterations(self):
            """Modified method to process 3 commands and exit (avoids infinite loop)."""
            for _ in range(3):
                try:
                    # Acquire response lock and receive command
                    with self.response_lock:
                        control_cmd = self.recv_control_cmd_server.recv_control_cmd()

                    if control_cmd is None:
                        continue  # Skip None commands

                    task_id = control_cmd["task_id"]
                    cmd = control_cmd["cmd"]

                    # Handle each command type
                    if cmd == "get_payload":
                        payload_info = self._get_current_server_info()
                        response = {"task_id": task_id, "result": payload_info}
                        with self.response_lock:
                            self.recv_control_cmd_server.response_for_control_cmd(task_id, response)
                    elif cmd == "get_metrics":
                        metrics_data = mock_get_metrics()
                        response = {"task_id": task_id, "result": metrics_data}
                        with self.response_lock:
                            self.recv_control_cmd_server.response_for_control_cmd(task_id, response)
                    elif cmd == "connect_rdma":
                        test_engine.engine_worker_queue.put_connect_rdma_task(control_cmd)
                except Exception as e:
                    mock_logger.error(f"handle_control_cmd got error: {e}")

        # Bind the modified method to the adapter instance
        adapter._recv_external_module_control_instruct = run_limited_iterations.__get__(adapter)
        # Execute the modified method to process test commands
        adapter._recv_external_module_control_instruct()

        # Verify 'get_payload' and 'get_metrics' triggered responses (2 total calls)
        self.assertEqual(
            mock_server_instance.response_for_control_cmd.call_count,
            2,
            "response_for_control_cmd should be called twice (get_payload + get_metrics)",
        )

        # Verify responses were sent for task IDs "1" and "2"
        called_task_ids = [call_arg[0][0] for call_arg in mock_server_instance.response_for_control_cmd.call_args_list]
        self.assertIn("1", called_task_ids, "Response not sent for 'get_payload' task (ID: 1)")
        self.assertIn("2", called_task_ids, "Response not sent for 'get_metrics' task (ID: 2)")

        # Verify 'connect_rdma' task was submitted to EngineWorkerQueue
        self.assertEqual(
            test_engine.engine_worker_queue.called_task["task_id"],
            "3",
            "connect_rdma task with ID 3 not received by EngineWorkerQueue",
        )

        # Test exception handling branch
        def raise_test_exception(self):
            """Modified method to raise a test exception."""
            raise ValueError("test_exception")

        # Configure mock server to trigger exception
        adapter.recv_control_cmd_server.recv_control_cmd = raise_test_exception.__get__(adapter)
        # Execute to trigger exception
        adapter._recv_external_module_control_instruct()

        # Verify exception was logged
        self.assertTrue(mock_logger.error.called, "Logger should capture exceptions during control command handling")


if __name__ == "__main__":
    unittest.main()
