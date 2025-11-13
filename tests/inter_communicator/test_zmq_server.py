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
Unit tests for fastdeploy/inter_communicator/zmq_server.py
"""

import os
import sys
import threading
import unittest
from collections import defaultdict
from unittest.mock import Mock, patch

try:
    import zmq
except ImportError:
    # Mock zmq module for standalone testing
    zmq = Mock()
    zmq.PULL = 1
    zmq.ROUTER = 2
    zmq.SNDHWM = 1001
    zmq.SNDTIMEO = 2001
    zmq.NOBLOCK = 2002
    zmq.Again = Exception

# Determine import method based on environment
TEST_MODE = os.environ.get("FD_TEST_MODE", "normal")

if TEST_MODE == "standalone":
    # Local testing mode - use dynamic import
    mock_logger = Mock()

    # Create mock modules
    class MockUtils:
        llm_logger = mock_logger

    class MockEnvs:
        FD_ZMQ_SNDHWM = "1000"
        FD_USE_AGGREGATE_SEND = True

    sys.modules["fastdeploy"] = Mock()
    sys.modules["fastdeploy.utils"] = MockUtils()
    sys.modules["fastdeploy.envs"] = MockEnvs()

    # Import the zmq_server module directly
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "zmq_server", os.path.join(os.path.dirname(__file__), "../../fastdeploy/inter_communicator/zmq_server.py")
    )
    zmq_server_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(zmq_server_module)

    # Extract classes we want to test
    ZmqServerBase = zmq_server_module.ZmqServerBase
    ZmqIpcServer = zmq_server_module.ZmqIpcServer
    ZmqTcpServer = zmq_server_module.ZmqTcpServer
else:
    # Normal mode - direct import
    try:
        from fastdeploy.inter_communicator.zmq_server import (
            ZmqIpcServer,
            ZmqServerBase,
            ZmqTcpServer,
        )

        mock_logger = None
    except ImportError:
        print("Warning: Direct import failed, falling back to standalone mode")
        TEST_MODE = "standalone"
        mock_logger = Mock()

        class MockUtils:
            llm_logger = mock_logger

        class MockEnvs:
            FD_ZMQ_SNDHWM = "1000"
            FD_USE_AGGREGATE_SEND = True

        sys.modules["fastdeploy"] = Mock()
        sys.modules["fastdeploy.utils"] = MockUtils()
        sys.modules["fastdeploy.envs"] = MockEnvs()

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "zmq_server", os.path.join(os.path.dirname(__file__), "../../fastdeploy/inter_communicator/zmq_server.py")
        )
        zmq_server_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(zmq_server_module)

        ZmqServerBase = zmq_server_module.ZmqServerBase
        ZmqIpcServer = zmq_server_module.ZmqIpcServer
        ZmqTcpServer = zmq_server_module.ZmqTcpServer


class MockResponse:
    """Mock response object for testing pack_aggregated_data method."""

    def __init__(self, finished=False):
        self.finished = finished
        self.data = {"test": "data"}

    def add(self, other):
        """Mock add method."""
        return MockResponse(finished=self.finished or other.finished)

    def to_dict(self):
        """Mock to_dict method."""
        return self.data


class TestZmqServerBase(unittest.TestCase):
    """Test cases for ZmqServerBase abstract class."""

    def setUp(self):
        """Set up test fixtures."""

        # Create a concrete implementation of ZmqServerBase for testing
        class ConcreteZmqServer(ZmqServerBase):
            def __init__(self):
                super().__init__()
                self.socket = None

            def _create_socket(self):
                self.socket = Mock()
                return self.socket

            def close(self):
                if self.socket:
                    self.socket.close()

        self.server = ConcreteZmqServer()
        self.server.mutex = threading.Lock()
        self.server.req_dict = {}
        self.server.aggregate_send = False

    def test_init(self):
        """Test ZmqServerBase initialization."""
        self.assertIsInstance(self.server.cached_results, defaultdict)
        self.assertIsInstance(self.server.response_token_lock, type(threading.Lock()))

    def test_ensure_socket_creates_socket(self):
        """Test _ensure_socket creates socket when None."""
        self.server.socket = None
        self.server._ensure_socket()
        self.assertIsNotNone(self.server.socket)

    def test_ensure_socket_uses_existing_socket(self):
        """Test _ensure_socket uses existing socket."""
        existing_socket = Mock()
        self.server.socket = existing_socket
        self.server._ensure_socket()
        self.assertEqual(self.server.socket, existing_socket)

    @patch("msgpack.packb")
    def test_pack_aggregated_data_single_response(self, mock_packb):
        """Test pack_aggregated_data with single response."""
        response = MockResponse()
        data = [response]

        self.server.pack_aggregated_data(data)

        mock_packb.assert_called_once_with([response.to_dict()])

    @patch("msgpack.packb")
    def test_pack_aggregated_data_multiple_responses(self, mock_packb):
        """Test pack_aggregated_data with multiple responses."""
        response1 = MockResponse()
        response2 = MockResponse(finished=True)
        data = [response1, response2]

        # Mock the add method to return the second response
        response1.add.return_value = response2

        self.server.pack_aggregated_data(data)

        response1.add.assert_called_once_with(response2)
        mock_packb.assert_called_once_with([response2.to_dict()])

    def test_send_json(self):
        """Test send_json method."""
        mock_socket = Mock()
        self.server.socket = mock_socket

        test_data = {"test": "data"}
        self.server.send_json(test_data)

        mock_socket.send_json.assert_called_once_with(test_data)

    def test_recv_json(self):
        """Test recv_json method."""
        mock_socket = Mock()
        mock_socket.recv_json.return_value = {"response": "data"}
        self.server.socket = mock_socket

        result = self.server.recv_json()

        mock_socket.recv_json.assert_called_once()
        self.assertEqual(result, {"response": "data"})

    def test_send_pyobj(self):
        """Test send_pyobj method."""
        mock_socket = Mock()
        self.server.socket = mock_socket

        test_data = {"test": "data"}
        with patch("fastdeploy.inter_communicator.zmq_server.ForkingPickler") as mock_pickle:
            mock_pickle.dumps.return_value = b"serialized_data"

            self.server.send_pyobj(test_data)

            mock_pickle.dumps.assert_called_once_with(test_data)
            mock_socket.send.assert_called_once_with(b"serialized_data", copy=False)

    def test_recv_pyobj(self):
        """Test recv_pyobj method."""
        mock_socket = Mock()
        mock_socket.recv.return_value = b"serialized_data"
        self.server.socket = mock_socket

        with patch("fastdeploy.inter_communicator.zmq_server.ForkingPickler") as mock_pickle:
            mock_pickle.loads.return_value = {"deserialized": "data"}

            result = self.server.recv_pyobj()

            mock_pickle.loads.assert_called_once_with(b"serialized_data")
            self.assertEqual(result, {"deserialized": "data"})

    @patch("zmq.Again")
    def test_receive_json_once_success(self, mock_zmq_again):
        """Test receive_json_once successful reception."""
        mock_socket = Mock()
        mock_socket.recv_json.return_value = {"test": "data"}
        self.server.socket = mock_socket

        error, result = self.server.receive_json_once()

        self.assertIsNone(error)
        self.assertEqual(result, {"test": "data"})

    @patch("zmq.Again")
    def test_receive_json_once_no_block_timeout(self, mock_zmq_again):
        """Test receive_json_once with timeout when not blocking."""
        mock_socket = Mock()
        mock_socket.recv_json.side_effect = mock_zmq_again
        self.server.socket = mock_socket

        error, result = self.server.receive_json_once(block=False)

        self.assertIsNone(error)
        self.assertIsNone(result)

    def test_receive_json_once_closed_socket(self):
        """Test receive_json_once with closed socket."""
        mock_socket = Mock()
        mock_socket.closed = True
        self.server.socket = mock_socket

        error, result = self.server.receive_json_once()

        self.assertEqual(error, "zmp socket has closed")
        self.assertIsNone(result)

    @patch("zmq.Again")
    def test_receive_json_once_exception(self, mock_zmq_again):
        """Test receive_json_once with exception."""
        mock_socket = Mock()
        mock_socket.closed = False
        mock_socket.recv_json.side_effect = Exception("Test error")
        self.server.socket = mock_socket

        with patch.object(self.server, "close") as mock_close:
            error, result = self.server.receive_json_once()

            mock_close.assert_called_once()
            self.assertEqual(error, "Test error")
            self.assertIsNone(result)

    def test_send_response_no_request_handle(self):
        """Test send_response when no request handle exists."""
        req_id = "test_req_123"
        data = [MockResponse()]

        self.server.send_response(req_id, data)

        # Should cache the data
        self.assertIn(req_id, self.server.cached_results)
        self.assertEqual(self.server.cached_results[req_id], [data])

    def test_send_response_with_cached_data(self):
        """Test send_response with cached data and new request handle."""
        req_id = "test_req_123"
        cached_data = [MockResponse()]
        new_data = [MockResponse(finished=True)]

        # Add cached data and request handle
        self.server.cached_results[req_id].append(cached_data)
        self.server.req_dict[req_id] = b"client_identity"

        with patch.object(self.server, "pack_aggregated_data") as mock_pack:
            mock_pack.return_value = b"packed_data"
            mock_socket = Mock()
            self.server.socket = mock_socket

            self.server.send_response(req_id, new_data)

            # Should send cached data + new data
            self.assertNotIn(req_id, self.server.cached_results)
            # Request should be popped since finished=True
            self.assertNotIn(req_id, self.server.req_dict)

    def test_send_response_finished_request(self):
        """Test send_response with finished request."""
        req_id = "test_req_123"
        data = [MockResponse(finished=True)]

        self.server.req_dict[req_id] = b"client_identity"

        with patch.object(self.server, "pack_aggregated_data") as mock_pack:
            mock_pack.return_value = b"packed_data"
            mock_socket = Mock()
            self.server.socket = mock_socket

            self.server.send_response(req_id, data)

            # Request should be popped since finished=True
            self.assertNotIn(req_id, self.server.req_dict)

    def test_send_response_socket_not_created(self):
        """Test send_response when socket is not created."""
        req_id = "test_req_123"
        data = [MockResponse()]

        self.server.socket = None

        with self.assertRaises(RuntimeError) as context:
            self.server.send_response(req_id, data)

        self.assertIn("Router socket not created", str(context.exception))

    def test_context_manager(self):
        """Test __exit__ method calls close."""
        with patch.object(self.server, "close") as mock_close:
            self.server.__exit__(None, None, None)
            mock_close.assert_called_once()


class TestZmqIpcServer(unittest.TestCase):
    """Test cases for ZmqIpcServer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_name = "test_server"
        self.test_mode = 1  # PULL mode for testing

    @patch("zmq.Context")
    @patch("os.path.exists")
    def test_init_pull_mode(self, mock_exists, mock_context):
        """Test ZmqIpcServer initialization in PULL mode."""
        mock_context.return_value.socket.return_value = Mock()
        mock_exists.return_value = False

        server = ZmqIpcServer(self.test_name, zmq.PULL)

        self.assertEqual(server.name, self.test_name)
        self.assertEqual(server.mode, zmq.PULL)
        self.assertEqual(server.file_name, "/dev/shm/test_server.socket")
        self.assertTrue(server.running)
        mock_context.return_value.socket.assert_called_once_with(zmq.PULL)

    @patch("zmq.Context")
    def test_init_router_mode(self, mock_context):
        """Test ZmqIpcServer initialization in ROUTER mode."""
        mock_context.return_value.socket.return_value = Mock()

        server = ZmqIpcServer(self.test_name, zmq.ROUTER)

        self.assertEqual(server.mode, zmq.ROUTER)
        self.assertEqual(server.file_name, "/dev/shm/router_test_server.ipc")
        mock_context.return_value.socket.assert_called_once_with(zmq.ROUTER)

    @patch("zmq.Context")
    def test_create_socket(self, mock_context):
        """Test _create_socket method."""
        mock_socket = Mock()
        mock_context.return_value.socket.return_value = mock_socket

        ZmqIpcServer(self.test_name, zmq.PULL)

        # Verify socket configuration
        mock_socket.setsockopt.assert_any_call(1001, 1000)  # zmq.SNDHWM
        mock_socket.setsockopt.assert_any_call(2001, -1)  # zmq.SNDTIMEO
        mock_socket.bind.assert_called_once_with("ipc:///dev/shm/test_server.socket")

    @patch("os.path.exists")
    @patch("os.remove")
    def test_clear_ipc_file_exists(self, mock_remove, mock_exists):
        """Test _clear_ipc removes existing file."""
        mock_exists.return_value = True

        server = ZmqIpcServer(self.test_name, zmq.PULL)
        server._clear_ipc("/tmp/test.socket")

        mock_remove.assert_called_once_with("/tmp/test.socket")

    @patch("os.path.exists")
    def test_clear_ipc_file_not_exists(self, mock_exists):
        """Test _clear_ipc when file doesn't exist."""
        mock_exists.return_value = False

        server = ZmqIpcServer(self.test_name, zmq.PULL)
        server._clear_ipc("/tmp/nonexistent.socket")

        # Should not attempt to remove
        mock_exists.assert_called_once_with("/tmp/nonexistent.socket")

    @patch("os.path.exists")
    @patch("os.remove")
    def test_clear_ipc_os_error(self, mock_remove, mock_exists):
        """Test _clear_ipc handles OSError gracefully."""
        mock_exists.return_value = True
        mock_remove.side_effect = OSError("Permission denied")

        server = ZmqIpcServer(self.test_name, zmq.PULL)

        with patch("fastdeploy.inter_communicator.zmq_server.llm_logger") as mock_logger:
            server._clear_ipc("/tmp/test.socket")

            mock_logger.warning.assert_called_once()

    @patch("zmq.Context")
    def test_close_success(self, mock_context):
        """Test close method successful execution."""
        mock_socket = Mock()
        mock_socket.closed = False
        mock_context.return_value.socket.return_value = mock_socket
        mock_context.return_value.closed = False

        server = ZmqIpcServer(self.test_name, zmq.PULL)

        with patch.object(server, "_clear_ipc") as mock_clear:
            server.close()

            self.assertFalse(server.running)
            mock_socket.close.assert_called_once()
            mock_context.return_value.term.assert_called_once()
            mock_clear.assert_called_once()

    @patch("zmq.Context")
    def test_close_already_closed(self, mock_context):
        """Test close method when already closed."""
        mock_context.return_value.socket.return_value = Mock()

        server = ZmqIpcServer(self.test_name, zmq.PULL)
        server.running = False

        server.close()

        # Should not attempt to close again
        mock_socket = mock_context.return_value.socket.return_value
        mock_socket.close.assert_not_called()

    @patch("zmq.Context")
    def test_close_exception_handling(self, mock_context):
        """Test close method handles exceptions gracefully."""
        mock_socket = Mock()
        mock_socket.closed = False
        mock_socket.close.side_effect = Exception("Test error")
        mock_context.return_value.socket.return_value = mock_socket
        mock_context.return_value.closed = False

        server = ZmqIpcServer(self.test_name, zmq.PULL)

        with patch("fastdeploy.inter_communicator.zmq_server.llm_logger") as mock_logger:
            server.close()

            mock_logger.warning.assert_called_once()


class TestZmqTcpServer(unittest.TestCase):
    """Test cases for ZmqTcpServer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_port = 5555
        self.test_mode = 1  # ROUTER mode for testing

    @patch("zmq.Context")
    def test_init(self, mock_context):
        """Test ZmqTcpServer initialization."""
        mock_context.return_value.socket.return_value = Mock()

        server = ZmqTcpServer(self.test_port, zmq.ROUTER)

        self.assertEqual(server.port, self.test_port)
        self.assertEqual(server.mode, zmq.ROUTER)
        self.assertTrue(server.running)
        mock_context.return_value.socket.assert_called_once_with(zmq.ROUTER)

    @patch("zmq.Context")
    def test_create_socket(self, mock_context):
        """Test _create_socket method."""
        mock_socket = Mock()
        mock_context.return_value.socket.return_value = mock_socket

        ZmqTcpServer(self.test_port, zmq.ROUTER)

        # Verify socket configuration
        mock_socket.setsockopt.assert_any_call(1001, 1000)  # zmq.SNDHWM
        mock_socket.setsockopt.assert_any_call(2001, -1)  # zmq.SNDTIMEO
        mock_socket.bind.assert_called_once_with("tcp://*:5555")

    @patch("zmq.Context")
    @patch("msgpack.unpackb")
    def test_recv_control_cmd_success(self, mock_unpackb, mock_context):
        """Test recv_control_cmd successful reception."""
        mock_socket = Mock()
        mock_socket.recv_multipart.return_value = [b"client", b"", b"task_data"]
        mock_context.return_value.socket.return_value = mock_socket

        mock_unpackb.return_value = {"task_id": "task_123", "command": "test"}

        server = ZmqTcpServer(self.test_port, zmq.ROUTER)
        result = server.recv_control_cmd()

        self.assertEqual(result, {"task_id": "task_123", "command": "test"})
        self.assertIn("task_123", server.req_dict)

    @patch("zmq.Context")
    def test_recv_control_cmd_timeout(self, mock_context):
        """Test recv_control_cmd with timeout."""
        mock_socket = Mock()
        mock_socket.recv_multipart.side_effect = Exception("Resource temporarily unavailable")
        mock_context.return_value.socket.return_value = mock_socket

        server = ZmqTcpServer(self.test_port, zmq.ROUTER)

        with patch("zmq.Again"):
            result = server.recv_control_cmd()

        self.assertIsNone(result)

    @patch("zmq.Context")
    @patch("msgpack.packb")
    def test_response_for_control_cmd_success(self, mock_packb, mock_context):
        """Test response_for_control_cmd successful execution."""
        mock_socket = Mock()
        mock_context.return_value.socket.return_value = mock_socket

        server = ZmqTcpServer(self.test_port, zmq.ROUTER)
        server.req_dict["task_123"] = b"client_identity"

        test_result = {"status": "success", "data": "response_data"}
        mock_packb.return_value = b"packed_result"

        server.response_for_control_cmd("task_123", test_result)

        mock_packb.assert_called_once_with(test_result)
        mock_socket.send_multipart.assert_called_once_with([b"client_identity", b"", b"packed_result"])
        self.assertNotIn("task_123", server.req_dict)

    @patch("zmq.Context")
    def test_response_for_control_cmd_no_socket(self, mock_context):
        """Test response_for_control_cmd when socket is None."""
        mock_context.return_value.socket.return_value = Mock()

        server = ZmqTcpServer(self.test_port, zmq.ROUTER)
        server.socket = None

        test_result = {"status": "success"}

        with self.assertRaises(RuntimeError) as context:
            server.response_for_control_cmd("task_123", test_result)

        self.assertIn("Router socket not created", str(context.exception))

    @patch("zmq.Context")
    @patch("msgpack.packb")
    def test_response_for_control_cmd_exception(self, mock_packb, mock_context):
        """Test response_for_control_cmd handles exceptions."""
        mock_socket = Mock()
        mock_socket.send_multipart.side_effect = Exception("Connection error")
        mock_context.return_value.socket.return_value = mock_socket

        server = ZmqTcpServer(self.test_port, zmq.ROUTER)
        server.req_dict["task_123"] = b"client_identity"

        test_result = {"status": "success"}
        mock_packb.return_value = b"packed_result"

        with patch("fastdeploy.inter_communicator.zmq_server.llm_logger") as mock_logger:
            server.response_for_control_cmd("task_123", test_result)

            mock_logger.error.assert_called_once()
            # Request should still be popped even on error
            self.assertNotIn("task_123", server.req_dict)

    @patch("zmq.Context")
    def test_close_success(self, mock_context):
        """Test close method successful execution."""
        mock_socket = Mock()
        mock_socket.closed = False
        mock_context.return_value.socket.return_value = mock_socket
        mock_context.return_value.closed = False

        server = ZmqTcpServer(self.test_port, zmq.ROUTER)

        server.close()

        self.assertFalse(server.running)
        mock_socket.close.assert_called_once()
        mock_context.return_value.term.assert_called_once()

    @patch("zmq.Context")
    def test_close_already_closed(self, mock_context):
        """Test close method when already closed."""
        mock_context.return_value.socket.return_value = Mock()

        server = ZmqTcpServer(self.test_port, zmq.ROUTER)
        server.running = False

        server.close()

        # Should not attempt to close again
        mock_socket = mock_context.return_value.socket.return_value
        mock_socket.close.assert_not_called()

    @patch("zmq.Context")
    def test_close_exception_handling(self, mock_context):
        """Test close method handles exceptions gracefully."""
        mock_socket = Mock()
        mock_socket.closed = False
        mock_socket.close.side_effect = Exception("Test error")
        mock_context.return_value.socket.return_value = mock_socket
        mock_context.return_value.closed = False

        server = ZmqTcpServer(self.test_port, zmq.ROUTER)

        with patch("fastdeploy.inter_communicator.zmq_server.llm_logger") as mock_logger:
            server.close()

            mock_logger.warning.assert_called_once()


class TestIntegration(unittest.TestCase):
    """Integration tests for ZMQ server functionality."""

    @patch("zmq.Context")
    def test_inheritance_structure(self, mock_context):
        """Test that concrete classes inherit properly from base class."""
        mock_context.return_value.socket.return_value = Mock()

        ipc_server = ZmqIpcServer("test", zmq.PULL)
        tcp_server = ZmqTcpServer(5555, zmq.ROUTER)

        # Test that both are instances of ZmqServerBase
        self.assertIsInstance(ipc_server, ZmqServerBase)
        self.assertIsInstance(tcp_server, ZmqServerBase)

        # Test that they have required methods
        self.assertTrue(hasattr(ipc_server, "_create_socket"))
        self.assertTrue(hasattr(ipc_server, "close"))
        self.assertTrue(hasattr(tcp_server, "_create_socket"))
        self.assertTrue(hasattr(tcp_server, "close"))

    def test_abstract_methods(self):
        """Test that ZmqServerBase is properly abstract."""
        # Should not be able to instantiate ZmqServerBase directly
        with self.assertRaises(TypeError):
            ZmqServerBase()


if __name__ == "__main__":
    # Print current test mode for clarity
    print(f"Running tests in {TEST_MODE} mode")
    if TEST_MODE == "standalone":
        print("To run in normal mode, ensure fastdeploy is properly installed")
        print("Or set FD_TEST_MODE=normal environment variable")
    unittest.main(verbosity=2)
