"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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

import pickle
import time
import unittest
from multiprocessing.reduction import ForkingPickler
from unittest.mock import MagicMock, patch

import zmq
from zmq.utils import jsonapi

from fastdeploy.inter_communicator.zmq_client import ZmqClientBase, ZmqIpcClient


class ConcreteZmqClient(ZmqClientBase):
    """Concrete subclass for testing ZmqClientBase."""

    def __init__(self):
        super().__init__()
        self.socket = MagicMock()

    def _create_socket(self):
        return MagicMock()

    def connect(self):
        pass

    def close(self):
        pass


class TestZmqClientBaseInit(unittest.TestCase):
    """Test ZmqClientBase.__init__."""

    def test_init_sets_address_none(self):
        """__init__ sets address to None."""
        client = ConcreteZmqClient()
        self.assertIsNone(client.address)


class TestZmqClientBaseEnsureSocket(unittest.TestCase):
    """Test ZmqClientBase._ensure_socket."""

    def test_creates_socket_when_none(self):
        """_ensure_socket creates socket when it is None."""
        client = ConcreteZmqClient()
        client.socket = None

        client._ensure_socket()

        self.assertIsNotNone(client.socket)

    def test_does_not_recreate_existing_socket(self):
        """_ensure_socket does not recreate existing socket."""
        client = ConcreteZmqClient()
        original_socket = client.socket

        client._ensure_socket()

        self.assertIs(client.socket, original_socket)


class TestZmqClientBaseSendJson(unittest.TestCase):
    """Test ZmqClientBase.send_json."""

    @patch("fastdeploy.inter_communicator.zmq_client.main_process_metrics")
    def test_send_json_success(self, mock_metrics):
        """send_json sends JSON-serialized data with metadata."""
        client = ConcreteZmqClient()
        client.address = "ipc:///test.socket"
        client.socket.send.return_value = None

        client.send_json({"key": "value"})

        client.socket.send.assert_called_once()
        sent_data = client.socket.send.call_args[0][0]
        parsed = jsonapi.loads(sent_data)
        self.assertEqual(parsed["data"], {"key": "value"})
        self.assertIn("__meta", parsed)
        self.assertIn("send_ts", parsed["__meta"])

    @patch("fastdeploy.inter_communicator.zmq_client.main_process_metrics")
    def test_send_json_exception_records_failure(self, mock_metrics):
        """send_json increments failure counter on exception."""
        client = ConcreteZmqClient()
        client.address = "ipc:///test.socket"
        client.socket.send.side_effect = zmq.ZMQError("send failed")

        with self.assertRaises(zmq.ZMQError):
            client.send_json({"key": "value"})

        mock_metrics.record_zmq_stats.assert_called_once()
        stats = mock_metrics.record_zmq_stats.call_args[0][0]
        self.assertEqual(stats.msg_send_failed_total, 1)
        self.assertEqual(stats.msg_send_total, 1)


class TestZmqClientBaseRecvJson(unittest.TestCase):
    """Test ZmqClientBase.recv_json."""

    @patch("fastdeploy.inter_communicator.zmq_client.main_process_metrics")
    def test_recv_json_with_meta(self, mock_metrics):
        """recv_json extracts data from envelope with __meta."""
        client = ConcreteZmqClient()
        client.address = "ipc:///test.socket"

        envelope = {"__meta": {"send_ts": time.perf_counter()}, "data": {"result": 42}}
        msg = jsonapi.dumps(envelope)
        client.socket.recv.return_value = msg
        client.socket._deserialize.return_value = envelope

        result = client.recv_json()

        self.assertEqual(result, {"result": 42})
        mock_metrics.record_zmq_stats.assert_called_once()
        stats = mock_metrics.record_zmq_stats.call_args[0][0]
        self.assertEqual(stats.msg_recv_total, 1)
        self.assertGreater(stats.msg_bytes_recv_total, 0)
        self.assertGreater(stats.zmq_latency, 0)

    @patch("fastdeploy.inter_communicator.zmq_client.main_process_metrics")
    def test_recv_json_without_meta(self, mock_metrics):
        """recv_json returns raw data when no __meta present."""
        client = ConcreteZmqClient()
        client.address = "ipc:///test.socket"

        raw_data = {"plain": "data"}
        msg = jsonapi.dumps(raw_data)
        client.socket.recv.return_value = msg
        client.socket._deserialize.return_value = raw_data

        result = client.recv_json()

        self.assertEqual(result, {"plain": "data"})

    @patch("fastdeploy.inter_communicator.zmq_client.main_process_metrics")
    def test_recv_json_non_dict(self, mock_metrics):
        """recv_json returns raw value when response is not a dict."""
        client = ConcreteZmqClient()
        client.address = "ipc:///test.socket"

        msg = jsonapi.dumps([1, 2, 3])
        client.socket.recv.return_value = msg
        client.socket._deserialize.return_value = [1, 2, 3]

        result = client.recv_json()

        self.assertEqual(result, [1, 2, 3])


class TestZmqClientBaseSendPyobj(unittest.TestCase):
    """Test ZmqClientBase.send_pyobj."""

    @patch("fastdeploy.inter_communicator.zmq_client.main_process_metrics")
    def test_send_pyobj_success(self, mock_metrics):
        """send_pyobj serializes and sends data with metadata."""
        client = ConcreteZmqClient()
        client.address = "ipc:///test.socket"

        client.send_pyobj({"key": "value"})

        client.socket.send.assert_called_once()
        sent_bytes = client.socket.send.call_args[0][0]
        envelope = pickle.loads(sent_bytes.data if hasattr(sent_bytes, "data") else bytes(sent_bytes))
        self.assertEqual(envelope["data"], {"key": "value"})
        self.assertIn("__meta", envelope)
        self.assertIn("send_ts", envelope["__meta"])

    @patch("fastdeploy.inter_communicator.zmq_client.main_process_metrics")
    def test_send_pyobj_exception_records_failure(self, mock_metrics):
        """send_pyobj increments failure counter on exception."""
        client = ConcreteZmqClient()
        client.address = "ipc:///test.socket"
        client.socket.send.side_effect = zmq.ZMQError("send failed")

        with self.assertRaises(zmq.ZMQError):
            client.send_pyobj({"key": "value"})

        mock_metrics.record_zmq_stats.assert_called_once()
        stats = mock_metrics.record_zmq_stats.call_args[0][0]
        self.assertEqual(stats.msg_send_failed_total, 1)
        self.assertEqual(stats.msg_send_total, 1)


class TestZmqClientBaseRecvPyobj(unittest.TestCase):
    """Test ZmqClientBase.recv_pyobj."""

    @patch("fastdeploy.inter_communicator.zmq_client.main_process_metrics")
    def test_recv_pyobj_with_meta(self, mock_metrics):
        """recv_pyobj extracts data from envelope with __meta."""
        client = ConcreteZmqClient()
        client.address = "ipc:///test.socket"

        envelope = {"__meta": {"send_ts": time.perf_counter()}, "data": {"result": 99}}
        data_bytes = ForkingPickler.dumps(envelope)
        client.socket.recv.return_value = data_bytes

        result = client.recv_pyobj()

        self.assertEqual(result, {"result": 99})
        mock_metrics.record_zmq_stats.assert_called_once()
        stats = mock_metrics.record_zmq_stats.call_args[0][0]
        self.assertEqual(stats.msg_recv_total, 1)
        self.assertGreater(stats.msg_bytes_recv_total, 0)

    @patch("fastdeploy.inter_communicator.zmq_client.main_process_metrics")
    def test_recv_pyobj_without_meta(self, mock_metrics):
        """recv_pyobj returns raw envelope when no __meta present."""
        client = ConcreteZmqClient()
        client.address = "ipc:///test.socket"

        envelope = {"plain": "data"}
        data_bytes = ForkingPickler.dumps(envelope)
        client.socket.recv.return_value = data_bytes

        result = client.recv_pyobj()

        self.assertEqual(result, {"plain": "data"})

    @patch("fastdeploy.inter_communicator.zmq_client.main_process_metrics")
    def test_recv_pyobj_non_dict(self, mock_metrics):
        """recv_pyobj returns raw value when response is not a dict."""
        client = ConcreteZmqClient()
        client.address = "ipc:///test.socket"

        data_bytes = ForkingPickler.dumps([1, 2, 3])
        client.socket.recv.return_value = data_bytes

        result = client.recv_pyobj()

        self.assertEqual(result, [1, 2, 3])


class TestZmqIpcClientInit(unittest.TestCase):
    """Test ZmqIpcClient.__init__."""

    @patch("fastdeploy.inter_communicator.zmq_client.zmq.Context")
    def test_init_sets_attributes(self, mock_ctx_cls):
        """__init__ sets name, mode, file_name, context, and socket."""
        mock_ctx = MagicMock()
        mock_socket = MagicMock()
        mock_ctx.socket.return_value = mock_socket
        mock_ctx_cls.return_value = mock_ctx

        client = ZmqIpcClient("test_queue", zmq.PUSH)

        self.assertEqual(client.name, "test_queue")
        self.assertEqual(client.mode, zmq.PUSH)
        self.assertEqual(client.file_name, "/dev/shm/test_queue.socket")
        self.assertIs(client.context, mock_ctx)
        self.assertIs(client.socket, mock_socket)
        mock_ctx.socket.assert_called_once_with(zmq.PUSH)


class TestZmqIpcClientConnect(unittest.TestCase):
    """Test ZmqIpcClient.connect."""

    @patch("fastdeploy.inter_communicator.zmq_client.zmq.Context")
    def test_connect_sets_address_and_connects(self, mock_ctx_cls):
        """connect() sets address and connects socket."""
        mock_ctx = MagicMock()
        mock_socket = MagicMock()
        mock_ctx.socket.return_value = mock_socket
        mock_ctx_cls.return_value = mock_ctx

        client = ZmqIpcClient("my_queue", zmq.PULL)
        client.connect()

        self.assertEqual(client.address, "ipc:///dev/shm/my_queue.socket")
        mock_socket.connect.assert_called_once_with("ipc:///dev/shm/my_queue.socket")


class TestZmqIpcClientCreateSocket(unittest.TestCase):
    """Test ZmqIpcClient._create_socket."""

    @patch("fastdeploy.inter_communicator.zmq_client.zmq.Context")
    def test_create_socket_creates_new_context_and_socket(self, mock_ctx_cls):
        """_create_socket creates a new context and returns socket."""
        mock_ctx = MagicMock()
        mock_socket = MagicMock()
        mock_ctx.socket.return_value = mock_socket
        mock_ctx_cls.return_value = mock_ctx

        client = ZmqIpcClient.__new__(ZmqIpcClient)
        client.mode = zmq.PUSH

        result = client._create_socket()

        mock_ctx_cls.assert_called_once()
        mock_ctx.socket.assert_called_once_with(zmq.PUSH)
        self.assertIs(result, mock_socket)


class TestZmqIpcClientClose(unittest.TestCase):
    """Test ZmqIpcClient.close."""

    @patch("fastdeploy.inter_communicator.zmq_client.llm_logger")
    def test_close_exception_logs_warning(self, mock_logger):
        """close() logs warning when exception occurs."""
        client = ZmqIpcClient.__new__(ZmqIpcClient)
        client.socket = MagicMock()
        client.socket.closed = False
        client.socket.setsockopt.side_effect = Exception("socket error")
        client.context = MagicMock()

        client.close()

        mock_logger.warning.assert_called_once()
        self.assertIn("failed to close", mock_logger.warning.call_args[0][0])

    @patch("fastdeploy.inter_communicator.zmq_client.llm_logger")
    def test_close_success(self, mock_logger):
        """close() closes socket and terminates context."""
        client = ZmqIpcClient.__new__(ZmqIpcClient)
        client.socket = MagicMock()
        client.socket.closed = False
        client.context = MagicMock()

        client.close()

        client.socket.setsockopt.assert_called_once_with(zmq.LINGER, 0)
        client.socket.close.assert_called_once()
        client.context.term.assert_called_once()


if __name__ == "__main__":
    unittest.main()
