"""
Tests for ZmqServerBase and derived helpers covering response/recv paths.
"""

import os
import tempfile
import threading
import time
import types
import unittest
from collections import defaultdict
from multiprocessing.reduction import ForkingPickler
from unittest import mock

import msgpack
import paddle
import zmq
from zmq.utils import jsonapi

if not hasattr(paddle, "compat"):
    paddle.compat = types.SimpleNamespace(enable_torch_proxy=lambda **kwargs: None)

from fastdeploy import envs
from fastdeploy.inter_communicator.zmq_server import (
    ZmqIpcServer,
    ZmqServerBase,
    ZmqTcpServer,
)


class _DummyResponse:
    def __init__(self, value, finished=False):
        self.value = value
        self.finished = finished
        self.tensor = paddle.to_tensor([value])

    def add(self, other):
        self.value += other.value
        self.tensor = self.tensor + other.tensor
        self.finished = self.finished or other.finished

    def to_dict(self):
        return {
            "value": int(self.value),
            "finished": bool(self.finished),
            "tensor_sum": int(self.tensor.sum()),
        }

    def __getstate__(self):
        return {"value": self.value, "finished": self.finished}

    def __setstate__(self, state):
        self.value = state["value"]
        self.finished = state["finished"]
        self.tensor = paddle.to_tensor([self.value])


class _FakeSocket:
    def __init__(self):
        self.closed = False
        self.sent = []
        self.recv_payload = None
        self.recv_multipart_payload = None
        self.options = {}

    def setsockopt(self, option, value):
        self.options[option] = value

    def bind(self, address):
        self.address = address

    def send(self, msg, flags=0, **kwargs):
        self.sent.append(("send", msg, flags, kwargs))
        return msg

    def send_multipart(self, parts, copy=True):
        self.sent.append(("send_multipart", parts, copy))

    def recv(self, flags=0):
        if isinstance(self.recv_payload, Exception):
            raise self.recv_payload
        return self.recv_payload

    def recv_multipart(self, flags=0):
        if isinstance(self.recv_multipart_payload, Exception):
            raise self.recv_multipart_payload
        return self.recv_multipart_payload

    def _deserialize(self, msg, loader):
        return loader(msg)

    def close(self):
        self.closed = True


class _FakeContext:
    def __init__(self):
        self.closed = False
        self.socket_instance = _FakeSocket()

    def socket(self, mode):
        self.socket_instance.mode = mode
        return self.socket_instance

    def term(self):
        self.closed = True


class _DummyServer(ZmqServerBase):
    def __init__(self, socket=None):
        super().__init__()
        self.socket = socket
        self.running = False
        self.mutex = threading.Lock()
        self.req_dict = {}
        self.aggregate_send = False

    def _create_socket(self):
        return self.socket

    def close(self):
        self.closed = True


class TestZmqServerBase(unittest.TestCase):
    def test_send_and_recv_json_roundtrip(self):
        fake_socket = _FakeSocket()
        server = _DummyServer(socket=fake_socket)
        payload = {"hello": "world"}
        server.send_json(payload)
        sent_msg = fake_socket.sent[-1][1]
        envelope = jsonapi.loads(sent_msg)
        self.assertIn("__meta", envelope)
        self.assertEqual(envelope["data"], payload)

        recv_envelope = {"__meta": {"send_ts": time.perf_counter() - 0.01}, "data": {"ok": True}}
        fake_socket.recv_payload = jsonapi.dumps(recv_envelope)
        self.assertEqual(server.recv_json(), {"ok": True})

    def test_ensure_socket_creates_socket(self):
        server = _DummyServer(socket=None)
        server._create_socket = lambda: _FakeSocket()
        server._ensure_socket()
        self.assertIsNotNone(server.socket)

    def test_recv_json_returns_raw_payload(self):
        fake_socket = _FakeSocket()
        server = _DummyServer(socket=fake_socket)
        fake_socket.recv_payload = jsonapi.dumps(["plain", "list"])
        self.assertEqual(server.recv_json(), ["plain", "list"])

    def test_send_json_raises_on_socket_error(self):
        class _ErrorSocket(_FakeSocket):
            def send(self, msg, flags=0):
                raise RuntimeError("send failed")

        server = _DummyServer(socket=_ErrorSocket())
        with self.assertRaises(RuntimeError):
            server.send_json({"boom": True})

    def test_recv_pyobj_meta_envelope(self):
        fake_socket = _FakeSocket()
        server = _DummyServer(socket=fake_socket)
        data = {"token": 1}
        envelope = {"__meta": {"send_ts": time.perf_counter() - 0.05}, "data": data}
        fake_socket.recv_payload = b"payload"
        with mock.patch("fastdeploy.inter_communicator.zmq_server.ForkingPickler.loads", return_value=envelope):
            self.assertEqual(server.recv_pyobj(), data)

    def test_send_pyobj_and_recv_pyobj_fallback(self):
        fake_socket = _FakeSocket()
        server = _DummyServer(socket=fake_socket)
        server.send_pyobj({"hello": "world"})
        self.assertEqual(fake_socket.sent[-1][0], "send")

        envelope = {"payload": "raw"}
        fake_socket.recv_payload = b"payload"
        with mock.patch("fastdeploy.inter_communicator.zmq_server.ForkingPickler.loads", return_value=envelope):
            self.assertEqual(server.recv_pyobj(), envelope)

    def test_send_pyobj_raises_on_socket_error(self):
        class _ErrorSocket(_FakeSocket):
            def send(self, msg, flags=0, **kwargs):
                raise RuntimeError("send failed")

        server = _DummyServer(socket=_ErrorSocket())
        with self.assertRaises(RuntimeError):
            server.send_pyobj({"boom": True})

    def test_pack_aggregated_data_respects_env_flag(self):
        server = _DummyServer()
        responses = [_DummyResponse(1), _DummyResponse(2, finished=True)]
        with mock.patch.object(envs, "ENABLE_V1_DATA_PROCESSOR", False):
            packed = server.pack_aggregated_data(responses)
            unpacked = ForkingPickler.loads(packed)
            self.assertEqual(unpacked[0]["tensor_sum"], 3)

        with mock.patch.object(envs, "ENABLE_V1_DATA_PROCESSOR", True):
            packed = server.pack_aggregated_data(responses)
            unpacked = ForkingPickler.loads(packed)
            self.assertIsInstance(unpacked[0], _DummyResponse)

    def test_receive_json_once_paths(self):
        fake_socket = _FakeSocket()
        fake_socket.closed = True
        server = _DummyServer(socket=fake_socket)
        error, data = server.receive_json_once()
        self.assertEqual(error, "zmp socket has closed")
        self.assertIsNone(data)

        server = _DummyServer(socket=_FakeSocket())
        server.recv_json = mock.Mock(side_effect=zmq.Again())
        error, data = server.receive_json_once()
        self.assertIsNone(error)
        self.assertIsNone(data)

        server = _DummyServer(socket=_FakeSocket())
        server.recv_json = mock.Mock(side_effect=ValueError("boom"))
        error, data = server.receive_json_once()
        self.assertEqual(error, "boom")
        self.assertIsNone(data)
        self.assertTrue(server.closed)

    def test_receive_pyobj_once_paths(self):
        fake_socket = _FakeSocket()
        fake_socket.closed = True
        server = _DummyServer(socket=fake_socket)
        error, data = server.receive_pyobj_once()
        self.assertEqual(error, "zmp socket has closed")
        self.assertIsNone(data)

        server = _DummyServer(socket=_FakeSocket())
        server.recv_pyobj = mock.Mock(side_effect=zmq.Again())
        error, data = server.receive_pyobj_once()
        self.assertIsNone(error)
        self.assertIsNone(data)

        server = _DummyServer(socket=_FakeSocket())
        server.recv_pyobj = mock.Mock(side_effect=ValueError("boom"))
        error, data = server.receive_pyobj_once()
        self.assertEqual(error, "boom")
        self.assertIsNone(data)
        self.assertTrue(server.closed)

    def test_send_response_per_step_caches_and_sends(self):
        fake_socket = _FakeSocket()
        server = _DummyServer(socket=fake_socket)
        server.response_handle_per_step = None
        server.cached_results = {"data": []}
        server.batch_id_per_step = 0
        server._send_response_per_step(0, [[_DummyResponse(1)]])
        self.assertEqual(len(server.cached_results["data"]), 1)

        server.response_handle_per_step = b"client"
        server._send_response_per_step(0, [[_DummyResponse(2)]])
        self.assertEqual(server.batch_id_per_step, 1)
        self.assertEqual(fake_socket.sent[-1][0], "send_multipart")

    def test_send_response_per_step_raises_without_socket(self):
        server = _DummyServer(socket=None)
        with self.assertRaises(RuntimeError):
            server._send_response_per_step(0, [[_DummyResponse(1)]])

    def test_send_response_per_step_handles_send_error(self):
        class _ErrorSocket(_FakeSocket):
            def send_multipart(self, parts, copy=True):
                raise RuntimeError("send failed")

        server = _DummyServer(socket=_ErrorSocket())
        server.response_handle_per_step = b"client"
        server.cached_results = {"data": []}
        server._send_response_per_step(0, [[_DummyResponse(1)]])
        self.assertEqual(server.batch_id_per_step, 0)

    def test_send_response_per_query_cache_and_flush(self):
        fake_socket = _FakeSocket()
        server = _DummyServer(socket=fake_socket)
        server.cached_results = defaultdict(list)
        server.req_dict = {}
        server.aggregate_send = False
        req_id = "req-1"
        server._send_response_per_query(req_id, [_DummyResponse(3)])
        self.assertIn(req_id, server.cached_results)

        server.req_dict[req_id] = b"client"
        with mock.patch.object(envs, "ENABLE_V1_DATA_PROCESSOR", False):
            server._send_response_per_query(req_id, [_DummyResponse(4, finished=True)])
        self.assertNotIn(req_id, server.req_dict)
        self.assertEqual(fake_socket.sent[-1][0], "send_multipart")

    def test_send_response_per_query_aggregate(self):
        fake_socket = _FakeSocket()
        server = _DummyServer(socket=fake_socket)
        server.req_dict["req-agg"] = b"client"
        server.aggregate_send = True
        with mock.patch.object(envs, "ENABLE_V1_DATA_PROCESSOR", False):
            server._send_response_per_query("req-agg", [_DummyResponse(5, finished=True)])
        self.assertEqual(fake_socket.sent[-1][0], "send_multipart")

    def test_send_response_per_query_v1_processor(self):
        fake_socket = _FakeSocket()
        server = _DummyServer(socket=fake_socket)
        server.req_dict["req-v1"] = b"client"
        server.aggregate_send = False
        with mock.patch.object(envs, "ENABLE_V1_DATA_PROCESSOR", True):
            server._send_response_per_query("req-v1", [_DummyResponse(6, finished=True)])
        self.assertEqual(fake_socket.sent[-1][0], "send_multipart")

    def test_send_response_per_query_send_failure(self):
        class _ErrorSocket(_FakeSocket):
            def send_multipart(self, parts, copy=True):
                raise RuntimeError("send failed")

        server = _DummyServer(socket=_ErrorSocket())
        server.req_dict["req-error"] = b"client"
        server.aggregate_send = False
        with mock.patch.object(envs, "ENABLE_V1_DATA_PROCESSOR", False):
            server._send_response_per_query("req-error", [_DummyResponse(7, finished=True)])
        self.assertEqual(server.req_dict, {})

    def test_send_response_per_query_raises_without_socket(self):
        server = _DummyServer(socket=None)
        with self.assertRaises(RuntimeError):
            server._send_response_per_query("req-missing", [_DummyResponse(1)])

    def test_send_response_dispatches_by_env(self):
        server = _DummyServer(socket=_FakeSocket())
        server._send_response_per_step = mock.Mock()
        server._send_response_per_query = mock.Mock()
        with mock.patch.object(envs, "FD_ENABLE_INTERNAL_ADAPTER", True):
            server.send_response("req", [_DummyResponse(1)])
            server._send_response_per_step.assert_called_once()
        with mock.patch.object(envs, "FD_ENABLE_INTERNAL_ADAPTER", False):
            server.send_response("req", [_DummyResponse(1)])
            server._send_response_per_query.assert_called_once()

    def test_recv_result_handle_paths(self):
        fake_socket = _FakeSocket()
        server = _DummyServer(socket=fake_socket)
        server.running = True
        server.cached_results = defaultdict(list)
        server.req_dict = {}
        client_id = b"client"
        req_id = b"req-1"

        def _recv_once(*args, **kwargs):
            server.running = False
            return client_id, b"", req_id

        fake_socket.recv_multipart = _recv_once
        with mock.patch.object(envs, "FD_ENABLE_INTERNAL_ADAPTER", True):
            server.recv_result_handle()
        self.assertEqual(server.response_handle_per_step, client_id)

        server.running = True
        server.response_handle_per_step = None
        server.cached_results = defaultdict(list)
        server.cached_results["req-1"].append([_DummyResponse(1, finished=True)])
        fake_socket.recv_multipart = _recv_once
        server.send_response = mock.Mock()
        with mock.patch.object(envs, "FD_ENABLE_INTERNAL_ADAPTER", False):
            server.recv_result_handle()
        server.send_response.assert_called_once_with("req-1", [])

    def test_exit_calls_close(self):
        server = _DummyServer(socket=_FakeSocket())
        server.close = mock.Mock()
        server.__exit__(None, None, None)
        server.close.assert_called_once()


class TestZmqServers(unittest.TestCase):
    def test_zmq_ipc_server_file_name_and_clear_ipc_error(self):
        fake_context = _FakeContext()
        with mock.patch("fastdeploy.inter_communicator.zmq_server.zmq.Context", return_value=fake_context):
            server = ZmqIpcServer("test", zmq.ROUTER)
        self.assertIn("router_test.ipc", server.file_name)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            server.file_name = tmp.name
        with mock.patch("fastdeploy.inter_communicator.zmq_server.os.remove", side_effect=OSError("fail")):
            server._clear_ipc(server.file_name)

    def test_zmq_ipc_server_close_cleans_ipc(self):
        fake_context = _FakeContext()
        with mock.patch("fastdeploy.inter_communicator.zmq_server.zmq.Context", return_value=fake_context):
            server = ZmqIpcServer("test", zmq.PULL)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            server.file_name = tmp.name
        server.close()
        self.assertFalse(os.path.exists(server.file_name))
        self.assertTrue(fake_context.closed)

        server.close()
        self.assertFalse(server.running)

    def test_zmq_ipc_server_close_exception(self):
        fake_context = _FakeContext()
        with mock.patch("fastdeploy.inter_communicator.zmq_server.zmq.Context", return_value=fake_context):
            server = ZmqIpcServer("test", zmq.PULL)

        class _BadSocket(_FakeSocket):
            def close(self):
                raise RuntimeError("close failed")

        server.socket = _BadSocket()
        server.context = _FakeContext()
        server.close()
        self.assertFalse(server.running)

    def test_zmq_tcp_server_control_cmd_flow(self):
        fake_context = _FakeContext()
        task = {"task_id": "task-1", "payload": "ok"}
        fake_context.socket_instance.recv_multipart_payload = [
            b"client",
            b"",
            msgpack.packb(task),
        ]
        with mock.patch("fastdeploy.inter_communicator.zmq_server.zmq.Context", return_value=fake_context):
            server = ZmqTcpServer(12345, zmq.ROUTER)
        received = server.recv_control_cmd()
        self.assertEqual(received["task_id"], "task-1")
        self.assertIn("task-1", server.req_dict)

        server.response_for_control_cmd("task-1", {"status": "done"})
        self.assertNotIn("task-1", server.req_dict)
        self.assertEqual(fake_context.socket_instance.sent[-1][0], "send_multipart")

    def test_zmq_tcp_server_control_cmd_empty(self):
        fake_context = _FakeContext()
        fake_context.socket_instance.recv_multipart_payload = zmq.Again()
        with mock.patch("fastdeploy.inter_communicator.zmq_server.zmq.Context", return_value=fake_context):
            server = ZmqTcpServer(12345, zmq.ROUTER)
        self.assertIsNone(server.recv_control_cmd())

    def test_zmq_tcp_server_response_errors_and_close(self):
        fake_context = _FakeContext()
        with mock.patch("fastdeploy.inter_communicator.zmq_server.zmq.Context", return_value=fake_context):
            server = ZmqTcpServer(12345, zmq.ROUTER)
        server.socket = None
        server._create_socket = lambda: None
        with self.assertRaises(RuntimeError):
            server.response_for_control_cmd("task", {"status": "fail"})

        class _ErrorSocket(_FakeSocket):
            def send_multipart(self, parts, copy=True):
                raise RuntimeError("send failed")

        server.socket = _ErrorSocket()
        server.req_dict["task"] = b"client"
        server.response_for_control_cmd("task", {"status": "fail"})
        self.assertEqual(server.req_dict, {})

        server.running = False
        server.close()


if __name__ == "__main__":
    unittest.main()
