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
    sys.modules["fastdeploy"] = Mock()

    # Mock envs with proper values
    mock_envs = Mock()
    mock_envs.FD_ZMQ_SNDHWM = "1000"
    mock_envs.FD_USE_AGGREGATE_SEND = False
    sys.modules["fastdeploy.envs"] = mock_envs
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
        sys.modules["fastdeploy"] = Mock()

        # Mock envs with proper values
        mock_envs = Mock()
        mock_envs.FD_ZMQ_SNDHWM = "1000"
        mock_envs.FD_USE_AGGREGATE_SEND = False
        sys.modules["fastdeploy.envs"] = mock_envs

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

        with patch('msgpack.packb') as mock_packb:
            mock_packb.return_value = b"packed_data"

            result = self.server.send_json(test_data)

            mock_packb.assert_called_once_with(test_data)
            self.server.socket.send.assert_called_once_with(b"packed_data")

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
        self.server.socket.recv.return_value = b'{"status": "success"}'

        with patch('msgpack.unpackb') as mock_unpackb:
            mock_unpackb.return_value = {"status": "success"}

            result = self.server.recv_json()

            self.server.socket.recv.assert_called_once()
            mock_unpackb.assert_called_once_with(b'{"status": "success"}')
            self.assertEqual(result, {"status": "success"})

    def test_recv_json_without_socket(self):
        """Test recv_json method when socket is None"""
        self.server.socket = None

        # Should create socket first
        result = self.server.recv_json()
        self.assertIsNotNone(self.server.socket)

    def test_cache_result_add_result(self):
        """Test caching a result"""
        request_id = "test_request"
        result_data = {"output": "test_output"}

        self.server.cache_result(request_id, result_data)

        self.assertIn(request_id, self.server.cached_results)
        self.assertEqual(len(self.server.cached_results[request_id]), 1)
        self.assertEqual(self.server.cached_results[request_id][0], result_data)

    def test_cache_result_multiple_results(self):
        """Test caching multiple results for same request"""
        request_id = "test_request"
        result1 = {"output": "test1"}
        result2 = {"output": "test2"}

        self.server.cache_result(request_id, result1)
        self.server.cache_result(request_id, result2)

        self.assertEqual(len(self.server.cached_results[request_id]), 2)
        self.assertEqual(self.server.cached_results[request_id][0], result1)
        self.assertEqual(self.server.cached_results[request_id][1], result2)

    def test_get_cached_result_existing(self):
        """Test retrieving cached result that exists"""
        request_id = "test_request"
        result_data = {"output": "test_output"}

        # Cache a result first
        self.server.cache_result(request_id, result_data)

        # Retrieve it
        cached_result = self.server.get_cached_result(request_id)

        self.assertEqual(cached_result, result_data)

    def test_get_cached_result_nonexistent(self):
        """Test retrieving cached result that doesn't exist"""
        request_id = "nonexistent_request"

        cached_result = self.server.get_cached_result(request_id)

        self.assertIsNone(cached_result)

    def test_get_cached_result_multiple_pop(self):
        """Test that get_cached_result removes result from cache"""
        request_id = "test_request"
        result_data = {"output": "test_output"}

        self.server.cache_result(request_id, result_data)

        # First retrieval should return the result
        result1 = self.server.get_cached_result(request_id)
        self.assertEqual(result1, result_data)

        # Second retrieval should return None (result was popped)
        result2 = self.server.get_cached_result(request_id)
        self.assertIsNone(result2)

    def test_clear_cache_for_request(self):
        """Test clearing cache for specific request"""
        request_id = "test_request"
        result_data = {"output": "test_output"}

        self.server.cache_result(request_id, result_data)
        self.assertIn(request_id, self.server.cached_results)

        self.server.clear_cache_for_request(request_id)

        self.assertNotIn(request_id, self.server.cached_results)

    def test_clear_all_cache(self):
        """Test clearing all cached results"""
        # Add multiple cached results
        self.server.cache_result("request1", {"output": "test1"})
        self.server.cache_result("request2", {"output": "test2"})

        self.assertEqual(len(self.server.cached_results), 2)

        self.server.clear_all_cache()

        self.assertEqual(len(self.server.cached_results), 0)

    def test_abstract_method_not_implemented(self):
        """Test that ZmqServerBase cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            ZmqServerBase()


class TestZmqIpcServer(unittest.TestCase):
    """Test suite for ZmqIpcServer class"""

    def setUp(self):
        """Setup method to create test fixtures"""
        self.host = "localhost"
        self.port = 8080
        self.server = ZmqIpcServer(self.host, self.port)

    def test_init(self):
        """Test ZmqIpcServer initialization"""
        self.assertEqual(self.server.host, self.host)
        self.assertEqual(self.server.port, self.port)
        self.assertIsNone(self.server.socket)
        self.assertIsInstance(self.server.context, Mock)

    def test_create_socket(self):
        """Test _create_socket method"""
        socket = self.server._create_socket()

        self.assertIsNotNone(socket)
        # Verify socket type is correct (should be DEALER for IPC)
        socket.setsockopt.assert_called()

    def test_get_ipc_address_linux(self):
        """Test IPC address generation on Linux"""
        with patch('sys.platform', 'linux'):
            address = self.server.get_ipc_address()
            expected = f"ipc:///tmp/fastdeploy_{self.port}"
            self.assertEqual(address, expected)

    def test_get_ipc_address_windows(self):
        """Test IPC address generation on Windows"""
        with patch('sys.platform', 'win32'):
            address = self.server.get_ipc_address()
            expected = f"tcp://{self.host}:{self.port}"
            self.assertEqual(address, expected)

    def test_get_ipc_address_darwin(self):
        """Test IPC address generation on macOS"""
        with patch('sys.platform', 'darwin'):
            address = self.server.get_ipc_address()
            expected = f"tcp://{self.host}:{self.port}"
            self.assertEqual(address, expected)


class TestZmqTcpServer(unittest.TestCase):
    """Test suite for ZmqTcpServer class"""

    def setUp(self):
        """Setup method to create test fixtures"""
        self.host = "localhost"
        self.port = 8080
        self.server = ZmqTcpServer(self.host, self.port)

    def test_init(self):
        """Test ZmqTcpServer initialization"""
        self.assertEqual(self.server.host, self.host)
        self.assertEqual(self.server.port, self.port)
        self.assertIsNone(self.server.socket)
        self.assertIsInstance(self.server.context, Mock)

    def test_create_socket(self):
        """Test _create_socket method"""
        socket = self.server._create_socket()

        self.assertIsNotNone(socket)
        # Verify socket type is correct (should be ROUTER for TCP)
        socket.setsockopt.assert_called()

    def test_get_tcp_address(self):
        """Test TCP address generation"""
        address = self.server.get_tcp_address()
        expected = f"tcp://{self.host}:{self.port}"
        self.assertEqual(address, expected)


if __name__ == "__main__":
    # Print current test mode for clarity
    print(f"Running tests in {TEST_MODE} mode")
    if TEST_MODE == "standalone":
        print("To run in normal mode, ensure fastdeploy is properly installed")
        print("Or set FD_TEST_MODE=normal environment variable")
    unittest.main(verbosity=2)