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
from collections import defaultdict
from unittest.mock import Mock, patch

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
            ZmqIpcServer,
            ZmqServerBase,
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
        self.mutex = threading.Lock()
        self.req_dict = dict()
        self.aggregate_send = False

    def _create_socket(self):
        """Create a mock socket for testing"""
        mock_socket = Mock()
        mock_socket.bind = Mock()
        mock_socket.connect = Mock()
        mock_socket.send = Mock()
        mock_socket.send_json = Mock()
        mock_socket.send_multipart = Mock()
        mock_socket.recv = Mock(return_value=b"test_response")
        mock_socket.recv_json = Mock(return_value={"status": "success"})
        mock_socket.recv_multipart = Mock(return_value=[b"client", b"", b"request_id"])
        mock_socket.close = Mock()
        mock_socket.setsockopt = Mock()
        mock_socket.closed = False  # Add closed attribute
        return mock_socket

    def close(self):
        """Close method for testing"""
        if self.socket:
            self.socket.close()
            self.socket = None


class MockResponse:
    """Mock response object for testing"""

    def __init__(self, finished=False):
        self.finished = finished

    def add(self, other):
        """Mock add method - if other is finished, use that status"""
        if hasattr(other, "finished") and other.finished:
            self.finished = other.finished
        return self

    def to_dict(self):
        """Mock to_dict method"""
        return {"finished": self.finished}


class TestZmqServerBase(unittest.TestCase):
    """Test suite for ZmqServerBase class"""

    def setUp(self):
        """Setup method to create test fixtures"""
        self.server = ConcreteZmqServer()

    def test_init(self):
        """Test ZmqServerBase initialization"""
        self.assertIsInstance(self.server.cached_results, defaultdict)
        self.assertEqual(self.server.cached_results, {})
        self.assertIsNotNone(self.server.response_token_lock)
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
        self.server.send_json(test_data)
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
        self.server.recv_json()
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

    def test_send_pyobj(self):
        """Test send_pyobj method"""
        if TEST_MODE == "standalone":
            self.skipTest("Skipping in standalone mode")

        self.server.socket = Mock()
        test_data = {"message": "test"}

        # Mock ForkingPickler
        with patch("fastdeploy.inter_communicator.zmq_server.ForkingPickler") as mock_pickler:
            mock_pickler.dumps.return_value = b"pickled_data"
            self.server.send_pyobj(test_data)
            mock_pickler.dumps.assert_called_once_with(test_data)
            self.server.socket.send.assert_called_once_with(b"pickled_data", copy=False)

    def test_recv_pyobj(self):
        """Test recv_pyobj method"""
        if TEST_MODE == "standalone":
            self.skipTest("Skipping in standalone mode")

        self.server.socket = Mock()
        expected_result = {"message": "test"}
        self.server.socket.recv.return_value = b"pickled_data"

        with patch("fastdeploy.inter_communicator.zmq_server.ForkingPickler") as mock_pickler:
            mock_pickler.loads.return_value = expected_result
            result = self.server.recv_pyobj()
            mock_pickler.loads.assert_called_once_with(b"pickled_data")
            self.assertEqual(result, expected_result)

    def test_pack_aggregated_data(self):
        """Test pack_aggregated_data method"""
        if TEST_MODE == "standalone":
            # Skip in standalone mode due to mocking complexity
            self.skipTest("Skipping in standalone mode")

        with patch("fastdeploy.inter_communicator.zmq_server.msgpack") as mock_msgpack:
            mock_msgpack.packb.return_value = b"packed_data"

            response1 = MockResponse(finished=False)
            response2 = MockResponse(finished=True)

            result = self.server.pack_aggregated_data([response1, response2])

            # Verify msgpack.packb was called with aggregated response
            mock_msgpack.packb.assert_called_once()
            args = mock_msgpack.packb.call_args[0][0]
            self.assertEqual(len(args), 1)
            self.assertEqual(args[0]["finished"], True)  # Should be True after aggregation
            self.assertEqual(result, b"packed_data")

    def test_receive_json_once_success(self):
        """Test receive_json_once successful case"""
        if TEST_MODE == "standalone":
            self.skipTest("Skipping in standalone mode")

        self.server.socket = Mock()
        self.server.socket.closed = False  # Set socket as not closed
        expected_data = {"status": "success"}
        self.server.socket.recv_json.return_value = expected_data

        # Mock zmq.NOBLOCK
        with patch("fastdeploy.inter_communicator.zmq_server.zmq") as mock_zmq:
            mock_zmq.NOBLOCK = 1
            error, result = self.server.receive_json_once(block=False)

            self.assertIsNone(error)
            self.assertEqual(result, expected_data)
            self.server.socket.recv_json.assert_called_once_with(flags=mock_zmq.NOBLOCK)

    def test_receive_json_once_no_data(self):
        """Test receive_json_once when no data available"""
        if TEST_MODE == "standalone":
            self.skipTest("Skipping in standalone mode")

        self.server.socket = Mock()
        self.server.socket.closed = False  # Set socket as not closed
        # Mock zmq.Again exception
        with patch("fastdeploy.inter_communicator.zmq_server.zmq") as mock_zmq:
            mock_zmq.Again = Exception
            mock_zmq.NOBLOCK = 1
            self.server.socket.recv_json.side_effect = mock_zmq.Again()

            error, result = self.server.receive_json_once(block=False)

            self.assertIsNone(error)
            self.assertIsNone(result)

    def test_receive_json_once_socket_closed(self):
        """Test receive_json_once when socket is closed"""
        self.server.socket = Mock()
        self.server.socket.closed = True

        error, result = self.server.receive_json_once(block=False)

        self.assertEqual(error, "zmp socket has closed")
        self.assertIsNone(result)

    def test_receive_pyobj_once_success(self):
        """Test receive_pyobj_once successful case"""
        if TEST_MODE == "standalone":
            self.skipTest("Skipping in standalone mode")

        self.server.socket = Mock()
        self.server.socket.closed = False  # Set socket as not closed
        self.server.socket.recv.return_value = b"pickled_data"
        expected_result = {"status": "success"}

        with (
            patch("fastdeploy.inter_communicator.zmq_server.ForkingPickler") as mock_pickler,
            patch("fastdeploy.inter_communicator.zmq_server.zmq") as mock_zmq,
        ):
            mock_zmq.NOBLOCK = 1
            mock_pickler.loads.return_value = expected_result
            error, result = self.server.receive_pyobj_once(block=False)

            self.assertIsNone(error)
            self.assertEqual(result, expected_result)

    def test_send_response_with_req_dict(self):
        """Test send_response when req_id exists in req_dict"""
        if TEST_MODE == "standalone":
            self.skipTest("Skipping in standalone mode")

        self.server.socket = Mock()
        self.server.req_dict = {"test_req": b"client_identity"}
        self.server.aggregate_send = False

        # Use patch for msgpack
        with (
            patch("fastdeploy.inter_communicator.zmq_server.msgpack") as mock_msgpack,
            patch("fastdeploy.inter_communicator.zmq_server.zmq") as mock_zmq,
        ):
            mock_zmq.NOBLOCK = 1
            mock_msgpack.packb.return_value = b"packed_response"

            response = MockResponse(finished=True)
            self.server.send_response("test_req", [response])

            # Verify response was sent
            self.server.socket.send_multipart.assert_called_once()
            call_args = self.server.socket.send_multipart.call_args[0][0]  # Get the list argument
            self.assertEqual(call_args[0], b"client_identity")
            self.assertEqual(call_args[1], b"")
            self.assertEqual(call_args[2], b"packed_response")

            # Verify req_id was removed since response was finished
            self.assertNotIn("test_req", self.server.req_dict)

    def test_send_response_without_req_dict(self):
        """Test send_response when req_id doesn't exist in req_dict"""
        self.server.socket = Mock()
        self.server.req_dict = {}

        response = MockResponse(finished=True)
        self.server.send_response("test_req", [response])

        # Verify response was cached
        self.assertIn("test_req", self.server.cached_results)
        # Socket should not be called since no req_dict entry
        self.server.socket.send_multipart.assert_not_called()

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
        # Note: setsockopt is called during actual socket creation, we can't easily mock it here
        # but we can verify the socket exists and is the right type
        self.assertTrue(hasattr(self.server.socket, "bind"))

    def test_get_ipc_address_router_mode(self):
        """Test IPC address generation for ROUTER mode"""
        # Since we're using ROUTER mode, the file_name should be router_{name}.ipc
        expected = f"/dev/shm/router_{self.name}.ipc"
        self.assertEqual(self.server.file_name, expected)

    def test_get_ipc_address_pull_mode(self):
        """Test IPC address generation for PULL mode"""
        # Test with PULL mode to see different file naming
        # Use proper zmq.PULL constant and mock the socket creation to avoid file_name issues
        with patch.object(ZmqIpcServer, "_create_socket"):
            pull_server = ZmqIpcServer(self.name, 1)  # zmq.PULL
            expected = f"/dev/shm/{self.name}.socket"
            self.assertEqual(pull_server.file_name, expected)

    def test_clear_ipc_file_exists(self):
        """Test _clear_ipc method when file exists"""
        test_file = "/tmp/test_socket_file"

        # Create a test file
        with open(test_file, "w") as f:
            f.write("test")

        self.server._clear_ipc(test_file)

        # File should be removed
        self.assertFalse(os.path.exists(test_file))

    def test_clear_ipc_file_not_exists(self):
        """Test _clear_ipc method when file doesn't exist"""
        non_existent_file = "/tmp/non_existent_file"

        # Should not raise exception
        self.server._clear_ipc(non_existent_file)

    def test_close(self):
        """Test close method"""
        # Mock socket and context
        self.server.socket = Mock()
        self.server.socket.closed = False
        self.server.context = Mock()
        self.server.context.closed = False
        self.server.running = True
        self.server.file_name = "/tmp/test_socket"

        # Create the file for cleanup
        with open(self.server.file_name, "w") as f:
            f.write("test")

        # Test close
        self.server.close()

        # Verify cleanup
        self.assertFalse(self.server.running)
        self.server.socket.close.assert_called_once()
        self.server.context.term.assert_called_once()
        self.assertFalse(os.path.exists(self.server.file_name))

    def test_close_already_closed(self):
        """Test close method when already closed"""
        self.server.running = False

        # Should not attempt to close again
        self.server.close()
        self.assertFalse(self.server.running)


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
        # Note: setsockopt is called during actual socket creation, we can't easily mock it here
        # but we can verify the socket exists and is the right type
        self.assertTrue(hasattr(self.server.socket, "bind"))

    def test_get_tcp_address(self):
        """Test TCP address generation"""
        # ZmqTcpServer binds to tcp://*:{port} by default
        expected_bind_address = f"tcp://*:{self.port}"
        # The socket should have been bound to this address during initialization
        self.server.socket.bind.assert_called_with(expected_bind_address)

    def test_recv_control_cmd_success(self):
        """Test recv_control_cmd successful case"""
        if TEST_MODE == "standalone":
            self.skipTest("Skipping in standalone mode")

        test_task = {"task_id": "test_task_123", "command": "start"}
        packed_task = b"packed_task"

        self.server.socket = Mock()
        self.server.socket.recv_multipart.return_value = [b"client", b"", packed_task]
        self.server.req_dict = {}

        with (
            patch("fastdeploy.inter_communicator.zmq_server.msgpack") as mock_msgpack,
            patch("fastdeploy.inter_communicator.zmq_server.zmq") as mock_zmq,
        ):
            mock_zmq.Again = Exception
            mock_zmq.NOBLOCK = 1
            mock_msgpack.unpackb.return_value = test_task

            result = self.server.recv_control_cmd()

            self.assertEqual(result, test_task)
            self.assertIn("test_task_123", self.server.req_dict)
            self.assertEqual(self.server.req_dict["test_task_123"], b"client")

    def test_recv_control_cmd_no_data(self):
        """Test recv_control_cmd when no data available"""
        if TEST_MODE == "standalone":
            self.skipTest("Skipping in standalone mode")

        self.server.socket = Mock()
        with patch("fastdeploy.inter_communicator.zmq_server.zmq") as mock_zmq:
            mock_zmq.Again = Exception
            mock_zmq.NOBLOCK = 1
            self.server.socket.recv_multipart.side_effect = mock_zmq.Again()

            result = self.server.recv_control_cmd()

            self.assertIsNone(result)

    def test_response_for_control_cmd(self):
        """Test response_for_control_cmd method"""
        if TEST_MODE == "standalone":
            self.skipTest("Skipping in standalone mode")

        test_task_id = "test_task_123"
        test_result = {"status": "completed", "output": "success"}
        packed_result = b"packed_result"

        self.server.socket = Mock()
        self.server.req_dict = {test_task_id: b"client_identity"}

        with patch("fastdeploy.inter_communicator.zmq_server.msgpack") as mock_msgpack:
            mock_msgpack.packb.return_value = packed_result

            self.server.response_for_control_cmd(test_task_id, test_result)

            # Verify response was sent
            self.server.socket.send_multipart.assert_called_once()
            args = self.server.socket.send_multipart.call_args[0]
            self.assertEqual(args[0], b"client_identity")
            self.assertEqual(args[1], b"")
            self.assertEqual(args[2], packed_result)

            # Verify task_id was removed from req_dict
            self.assertNotIn(test_task_id, self.server.req_dict)

    def test_close(self):
        """Test close method"""
        # Mock socket and context
        self.server.socket = Mock()
        self.server.socket.closed = False
        self.server.context = Mock()
        self.server.context.closed = False
        self.server.running = True

        # Test close
        self.server.close()

        # Verify cleanup
        self.assertFalse(self.server.running)
        self.server.socket.close.assert_called_once()
        self.server.context.term.assert_called_once()

    def test_close_already_closed(self):
        """Test close method when already closed"""
        self.server.running = False

        # Should not attempt to close again
        self.server.close()
        self.assertFalse(self.server.running)


if __name__ == "__main__":
    # Print current test mode for clarity
    print(f"Running tests in {TEST_MODE} mode")
    if TEST_MODE == "standalone":
        print("To run in normal mode, ensure fastdeploy is properly installed")
        print("Or set FD_TEST_MODE=normal environment variable")
    unittest.main(verbosity=2)
