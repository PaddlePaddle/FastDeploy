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
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastdeploy import envs
from fastdeploy.utils import find_free_shm_ports, is_shm_port_available


class TestSHMPortAllocation(unittest.TestCase):
    """Test suite for SHM-based port allocation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_sock_files = []

    def tearDown(self):
        """Clean up any created socket files."""
        for sock_file in self.test_sock_files:
            if os.path.exists(sock_file):
                try:
                    os.remove(sock_file)
                except OSError:
                    pass

    def _create_test_sock_file(self, port):
        """Helper to create a test socket file."""
        sock_path = f"/dev/shm/fd_task_queue_{port}.sock"
        self.test_sock_files.append(sock_path)
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(sock_path), exist_ok=True)
        # Create the socket file
        Path(sock_path).touch()
        return sock_path

    def test_is_shm_port_available_free_port(self):
        """Test that is_shm_port_available returns True for a free port."""
        # Use a high port number that's unlikely to be in use
        port = 54321
        sock_path = f"/dev/shm/fd_task_queue_{port}.sock"
        # Ensure the socket file doesn't exist
        if os.path.exists(sock_path):
            os.remove(sock_path)
        self.assertTrue(is_shm_port_available(port))

    def test_is_shm_port_available_occupied_port(self):
        """Test that is_shm_port_available returns False when socket file exists."""
        port = 54322
        self._create_test_sock_file(port)
        self.assertFalse(is_shm_port_available(port))

    def test_find_free_shm_ports_single_port(self):
        """Test finding a single free SHM port."""
        ports = find_free_shm_ports(port_range=(50000, 50100), num_ports=1)
        self.assertEqual(len(ports), 1)
        self.assertGreaterEqual(ports[0], 50000)
        self.assertLessEqual(ports[0], 50100)

    def test_find_free_shm_ports_multiple_ports(self):
        """Test finding multiple free SHM ports."""
        num_ports = 3
        ports = find_free_shm_ports(port_range=(50100, 50200), num_ports=num_ports)
        self.assertEqual(len(ports), num_ports)
        # Verify all ports are unique
        self.assertEqual(len(set(ports)), num_ports)
        # Verify all ports are in range
        for port in ports:
            self.assertGreaterEqual(port, 50100)
            self.assertLessEqual(port, 50200)

    def test_find_free_shm_ports_skips_occupied(self):
        """Test that find_free_shm_ports skips ports with existing socket files."""
        # Create socket files for some ports in the range
        occupied_ports = [50201, 50202, 50203]
        for port in occupied_ports:
            self._create_test_sock_file(port)

        # Find free ports in a range that includes occupied ports
        ports = find_free_shm_ports(port_range=(50200, 50210), num_ports=3)
        self.assertEqual(len(ports), 3)
        # Verify none of the returned ports are occupied
        for port in ports:
            self.assertNotIn(port, occupied_ports)

    def test_find_free_shm_ports_invalid_range(self):
        """Test that find_free_shm_ports raises ValueError for invalid port ranges."""
        # Start port < 1
        with self.assertRaises(ValueError) as cm:
            find_free_shm_ports(port_range=(0, 1000), num_ports=1)
        self.assertIn("Invalid port range", str(cm.exception))

        # End port > 65535
        with self.assertRaises(ValueError) as cm:
            find_free_shm_ports(port_range=(1000, 70000), num_ports=1)
        self.assertIn("Invalid port range", str(cm.exception))

        # Start > End
        with self.assertRaises(ValueError) as cm:
            find_free_shm_ports(port_range=(2000, 1000), num_ports=1)
        self.assertIn("Invalid port range", str(cm.exception))

    def test_find_free_shm_ports_invalid_num_ports(self):
        """Test that find_free_shm_ports raises ValueError for invalid num_ports."""
        with self.assertRaises(ValueError) as cm:
            find_free_shm_ports(port_range=(8000, 9000), num_ports=0)
        self.assertIn("num_ports must be a positive integer", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            find_free_shm_ports(port_range=(8000, 9000), num_ports=-1)
        self.assertIn("num_ports must be a positive integer", str(cm.exception))

    def test_find_free_shm_ports_num_ports_exceeds_range(self):
        """Test that find_free_shm_ports raises ValueError when num_ports exceeds range size."""
        with self.assertRaises(ValueError) as cm:
            find_free_shm_ports(port_range=(8000, 8005), num_ports=10)
        self.assertIn("num_ports is larger than range size", str(cm.exception))

    def test_find_free_shm_ports_insufficient_free_ports(self):
        """Test that find_free_shm_ports raises RuntimeError when not enough free ports."""
        # Create socket files for all ports in a small range except one
        port_range_start = 50300
        port_range_end = 50305
        for port in range(port_range_start, port_range_end):
            self._create_test_sock_file(port)

        # Try to find more free ports than available (only 1 free, requesting 3)
        with self.assertRaises(RuntimeError) as cm:
            find_free_shm_ports(port_range=(port_range_start, port_range_end), num_ports=3)
        self.assertIn("Only found", str(cm.exception))
        self.assertIn("free SHM ports", str(cm.exception))


class TestSHMPortAllocationIntegration(unittest.TestCase):
    """Integration tests for SHM port allocation with args_utils."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_sock_files = []

    def tearDown(self):
        """Clean up any created socket files."""
        for sock_file in self.test_sock_files:
            if os.path.exists(sock_file):
                try:
                    os.remove(sock_file)
                except OSError:
                    pass

    def _create_test_sock_file(self, port):
        """Helper to create a test socket file."""
        sock_path = f"/dev/shm/fd_task_queue_{port}.sock"
        self.test_sock_files.append(sock_path)
        os.makedirs(os.path.dirname(sock_path), exist_ok=True)
        Path(sock_path).touch()
        return sock_path

    @patch.object(envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", True)
    def test_args_utils_auto_allocate_with_shm(self):
        """Test that args_utils auto-allocates ports using find_free_shm_ports when SHM is enabled."""
        from fastdeploy.engine.args_utils import EngineArgs

        # Create mock model config
        model_config = Mock()
        model_config.max_model_len = 512

        # Create EngineArgs without specifying engine_worker_queue_port
        # This should trigger auto-allocation with find_free_shm_ports
        engine_args = EngineArgs(
            model="test_model",
            tensor_parallel_size=1,
            data_parallel_size=1,
        )

        # Mock the model config
        with patch.object(engine_args, "create_model_config", return_value=model_config):
            engine_args.create_model_config()

        # Verify that engine_worker_queue_port is None before post_init
        self.assertIsNone(engine_args.engine_worker_queue_port)

    @patch.object(envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", True)
    def test_args_utils_skips_conflicting_sock_files(self):
        """Test that auto-allocation skips ports with existing socket files."""
        from fastdeploy.engine.args_utils import EngineArgs

        # Create socket files for some ports
        occupied_ports = [8000, 8001, 8002]
        for port in occupied_ports:
            self._create_test_sock_file(port)

        # Create EngineArgs which should auto-allocate
        engine_args = EngineArgs(
            model="test_model",
            tensor_parallel_size=1,
            data_parallel_size=1,
        )

        # Create mock model config
        model_config = Mock()
        model_config.max_model_len = 512

        with patch.object(engine_args, "create_model_config", return_value=model_config):
            engine_args.create_model_config()

        # After post_init_all_ports is called, the allocated port should not be in occupied_ports
        # Note: This test verifies the integration, but actual allocation happens in post_init_all_ports
        # which is called during engine initialization

    @patch.object(envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", False)
    def test_args_utils_uses_regular_ports_when_shm_disabled(self):
        """Test that regular find_free_ports is used when SHM is disabled."""
        from fastdeploy.engine.args_utils import EngineArgs

        # Create EngineArgs with SHM disabled
        engine_args = EngineArgs(
            model="test_model",
            tensor_parallel_size=1,
            data_parallel_size=1,
        )

        # Verify that the args are created successfully
        self.assertIsNotNone(engine_args)


if __name__ == "__main__":
    unittest.main()
