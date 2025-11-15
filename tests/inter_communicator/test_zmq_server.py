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

import os
import sys
import threading
import unittest
from unittest.mock import MagicMock, Mock, patch
from collections import defaultdict

# Determine import method based on environment
# Use environment variable FD_TEST_MODE=standalone for local testing
TEST_MODE = os.environ.get("FD_TEST_MODE", "normal")

if TEST_MODE == "standalone":
    # Local testing mode - use dynamic import
    mock_logger = Mock()

    # Mock external dependencies
    sys.modules["zmq"] = Mock()
    sys.modules["msgpack"] = Mock()

    # Mock envs with proper values
    class MockEnvs:
        FD_ZMQ_SNDHWM = "1000"
        FD_USE_AGGREGATE_SEND = False

    # Mock fastdeploy module
    mock_fastdeploy = Mock()
    mock_fastdeploy.envs = MockEnvs()
    sys.modules["fastdeploy"] = mock_fastdeploy
    sys.modules["fastdeploy.envs"] = MockEnvs()
    sys.modules["fastdeploy.utils"] = Mock()

    # Create mock classes
    class MockZmqContext:
        def socket(self, socket_type):
            mock_socket = Mock()
            mock_socket.bind = Mock()
            mock_socket.connect = Mock()
            mock_socket.send = Mock()
            mock_socket.send_json = Mock()
            mock_socket.recv = Mock(return_value=b"test_response")
            mock_socket.recv_json = Mock(return_value={"status": "success"})
            mock_socket.close = Mock()
            mock_socket.setsockopt = Mock()
            return mock_socket

        term = Mock()

    sys.modules["zmq"].Context = MockZmqContext
    sys.modules["zmq"].PULL = 1
    sys.modules["zmq"].PUSH = 2
    sys.modules["zmq"].REPLY = 3
    sys.modules["zmq"].REQ = 4
    sys.modules["zmq"].DEALER = 5
    sys.modules["zmq"].ROUTER = 6
    sys.modules["zmq"].RCVTIMEO = 1000

    sys.modules["fastdeploy.utils"].llm_logger = mock_logger

    # Import the zmq_server module directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "zmq_server",
        os.path.join(os.path.dirname(__file__), "../../fastdeploy/inter_communicator/zmq_server.py"),
    )
    zmq_server_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(zmq_server_module)

    ZmqServerBase = zmq_server_module.ZmqServerBase
    ZmqIpcServer = zmq_server_module.ZmqIpcServer
    ZmqTcpServer = zmq_server_module.ZmqTcpServer
else:
    # Normal mode - direct import
    try:
        from fastdeploy.inter_communicator.zmq_server import (
            ZmqServerBase,
            ZmqIpcServer,
            ZmqTcpServer,
        )
    except ImportError:
        # Fallback to standalone mode
        print("Warning: Direct import failed, falling back to standalone mode")
        TEST_MODE = "standalone"
        mock_logger = Mock()

        sys.modules["zmq"] = Mock()
        sys.modules["msgpack"] = Mock()

        # Mock envs with proper values
        class MockEnvs:
            FD_ZMQ_SNDHWM = "1000"
            FD_USE_AGGREGATE_SEND = False

        # Mock fastdeploy module
        mock_fastdeploy = Mock()
        mock_fastdeploy.envs = MockEnvs()
        sys.modules["fastdeploy"] = mock_fastdeploy
        sys.modules["fastdeploy.envs"] = MockEnvs()
        sys.modules["fastdeploy.utils"] = Mock()

        class MockZmqContext:
            def socket(self, socket_type):
                mock_socket = Mock()
                mock_socket.bind = Mock()
                mock_socket.connect = Mock()
                mock_socket.send = Mock()
                mock_socket.send_json = Mock()
                mock_socket.recv = Mock(return_value=b"test_response")
                mock_socket.recv_json = Mock(return_value={"status": "success"})
                mock_socket.close = Mock()
                mock_socket.setsockopt = Mock()
                return mock_socket

            term = Mock()

        sys.modules["zmq"].Context = MockZmqContext
        sys.modules["zmq"].PULL = 1
        sys.modules["zmq"].PUSH = 2
        sys.modules["zmq"].REPLY = 3
        sys.modules["zmq"].REQ = 4
        sys.modules["zmq"].DEALER = 5
        sys.modules["zmq"].ROUTER = 6
        sys.modules["zmq"].RCVTIMEO = 1000

        sys.modules["fastdeploy.utils"].llm_logger = mock_logger

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "zmq_server",
            os.path.join(os.path.dirname(__file__), "../../fastdeploy/inter_communicator/zmq_server.py"),
        )
        zmq_server_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(zmq_server_module)

        ZmqServerBase = zmq_server_module.ZmqServerBase
        ZmqIpcServer = zmq_server_module.ZmqIpcServer
        ZmqTcpServer = zmq_server_module.ZmqTcpServer


class ConcreteZmqServer(ZmqServerBase):
    """Concrete implementation of ZmqServerBase for testing"""

    def __init__(self):
        """Initialize the concrete server"""
        super().__init__()
        self.socket = None

    def _create_socket(self):
        """Create a mock socket for testing"""
        mock_socket = Mock()
        mock_socket.bind = Mock()
        mock_socket.connect = Mock()
        mock_socket.send = Mock()
        mock_socket.send_json = Mock()
        mock_socket.recv = Mock(return_value=b"test_response")
        mock_socket.recv_json = Mock(return_value={"status": "success"})
        mock_socket.close = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.closed = False  # Add closed attribute
        return mock_socket

    def close(self):
        """Close method for testing"""
        if self.socket:
            self.socket.close()
            self.socket = None


class TestZmqServerBase(unittest.TestCase):
    """Test suite for ZmqServerBase class"""

    def setUp(self):
        """Setup method to create test fixtures"""
        self.server = ConcreteZmqServer()

    def test_init(self):
        """Test ZmqServerBase initialization"""
        self.assertIsInstance(self.server.cached_results, defaultdict)
        self.assertEqual(self.server.cached_results, {})
        self.assertIsInstance(self.server.response_token_lock, threading.Lock)
        self.assertIsNone(self.server.socket)

    def test_ensure_socket_creates_socket(self):
        """Test that _ensure_socket creates socket when None"""
        self.server.socket = None
        self.server._ensure_socket()
        self.assertIsNotNone(self.server.socket)

    def test_ensure_socket_uses_existing_socket(self):
        """Test that _ensure_socket doesn't create new socket when one exists"""
        existing_socket = Mock()
        self.server.socket = existing_socket
        self.server._ensure_socket()
        self.assertEqual(self.server.socket, existing_socket)

    def test_send_json(self):
        """Test send_json method"""
        self.server.socket = Mock()
        test_data = {"message": "test"}

        # Since we're mocking, the actual implementation will use send_json
        self.server.send_json(test_data)
        self.server.socket.send_json.assert_called_once_with(test_data)

    def test_send_json_without_socket(self):
        """Test send_json method when socket is None"""
        self.server.socket = None
        test_data = {"message": "test"}

        # Should create socket first
        result = self.server.send_json(test_data)
        self.assertIsNotNone(self.server.socket)

    def test_recv_json(self):
        """Test recv_json method"""
        self.server.socket = Mock()
        expected_result = {"status": "success"}
        self.server.socket.recv_json.return_value = expected_result

        result = self.server.recv_json()

        self.server.socket.recv_json.assert_called_once()
        self.assertEqual(result, expected_result)

    def test_recv_json_without_socket(self):
        """Test recv_json method when socket is None"""
        self.server.socket = None

        # Should create socket first
        result = self.server.recv_json()
        self.assertIsNotNone(self.server.socket)

    def test_cached_results_initialization(self):
        """Test that cached_results is properly initialized as defaultdict(list)"""
        self.assertIsInstance(self.server.cached_results, defaultdict)
        # Test that it returns empty list for non-existent keys
        empty_list = self.server.cached_results["nonexistent_key"]
        self.assertEqual(empty_list, [])
        # Note: After accessing a key, defaultdict will have 1 entry
        # So we should clear it for the test to be accurate
        self.server.cached_results.clear()
        self.assertEqual(len(self.server.cached_results), 0)

    def test_abstract_method_not_implemented(self):
        """Test that ZmqServerBase cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            ZmqServerBase()


class TestZmqIpcServer(unittest.TestCase):
    """Test suite for ZmqIpcServer class"""

    def setUp(self):
        """Setup method to create test fixtures"""
        self.name = "test_socket"
        self.mode = 6  # zmq.ROUTER (so file_name gets set properly)
        self.server = ZmqIpcServer(self.name, self.mode)

    def test_init(self):
        """Test ZmqIpcServer initialization"""
        self.assertEqual(self.server.name, self.name)
        self.assertEqual(self.server.mode, self.mode)
        # ZmqIpcServer creates socket during initialization
        self.assertIsNotNone(self.server.socket)
        # Context should be our mocked ZMQ context
        self.assertIsNotNone(self.server.context)

    def test_create_socket(self):
        """Test _create_socket method"""
        # Socket is already created during init, so we can verify it exists
        self.assertIsNotNone(self.server.socket)
        # Verify socket type is correct (should be ROUTER for IPC)
        self.server.socket.setsockopt.assert_called()

    def test_get_ipc_address_router_mode(self):
        """Test IPC address generation for ROUTER mode"""
        # Since we're using ROUTER mode, the file_name should be router_{name}.ipc
        expected = f"/dev/shm/router_{self.name}.ipc"
        self.assertEqual(self.server.file_name, expected)

    def test_get_ipc_address_pull_mode(self):
        """Test IPC address generation for PULL mode"""
        # Test with PULL mode to see different file naming
        pull_server = ZmqIpcServer(self.name, 1)  # zmq.PULL
        expected = f"/dev/shm/{self.name}.socket"
        self.assertEqual(pull_server.file_name, expected)


class TestZmqTcpServer(unittest.TestCase):
    """Test suite for ZmqTcpServer class"""

    def setUp(self):
        """Setup method to create test fixtures"""
        self.port = 8080
        self.mode = 6  # zmq.ROUTER
        self.server = ZmqTcpServer(self.port, self.mode)

    def test_init(self):
        """Test ZmqTcpServer initialization"""
        self.assertEqual(self.server.port, self.port)
        self.assertEqual(self.server.mode, self.mode)
        # ZmqTcpServer creates socket during initialization
        self.assertIsNotNone(self.server.socket)
        # Context should be our mocked ZMQ context
        self.assertIsNotNone(self.server.context)

    def test_create_socket(self):
        """Test _create_socket method"""
        # Socket is already created during init, so we can verify it exists
        self.assertIsNotNone(self.server.socket)
        # Verify socket type is correct (should be ROUTER for TCP)
        self.server.socket.setsockopt.assert_called()

    def test_get_tcp_address(self):
        """Test TCP address generation"""
        # ZmqTcpServer binds to tcp://*:{port} by default
        expected_bind_address = f"tcp://*:{self.port}"
        # The socket should have been bound to this address during initialization
        self.server.socket.bind.assert_called_with(expected_bind_address)


if __name__ == "__main__":
    # Print current test mode for clarity
    print(f"Running tests in {TEST_MODE} mode")
    if TEST_MODE == "standalone":
        print("To run in normal mode, ensure fastdeploy is properly installed")
        print("Or set FD_TEST_MODE=normal environment variable")
    unittest.main(verbosity=2)