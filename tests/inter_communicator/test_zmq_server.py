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
import socket
import tempfile
import threading
import time
import unittest
from multiprocessing.reduction import ForkingPickler
from unittest import mock

import msgpack
import paddle
import zmq
from zmq.utils import jsonapi

from fastdeploy import envs
from fastdeploy.inter_communicator.zmq_server import (
    ZmqIpcServer,
    ZmqServerBase,
    ZmqTcpServer,
)


class _EnvGuard:
    def __init__(self, **values):
        self._values = values
        self._originals = {}

    def __enter__(self):
        for key, value in self._values.items():
            self._originals[key] = getattr(envs, key)
            setattr(envs, key, value)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self._originals.items():
            setattr(envs, key, value)


class DummyResponse:
    def __init__(self, value, finished=False):
        self.value = value
        self.finished = finished
        self.add_calls = 0

    def add(self, other):
        self.value += other.value
        self.add_calls += 1

    def to_dict(self):
        return {"value": self.value, "finished": self.finished}


class _PairServer(ZmqServerBase):
    def __init__(self, context, address):
        super().__init__()
        self.context = context
        self.address = address
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.bind(self.address)

    def _create_socket(self):
        return self.socket

    def close(self):
        if self.socket is not None and not self.socket.closed:
            self.socket.close()


class _NoSocketServer(ZmqServerBase):
    def __init__(self):
        super().__init__()
        self.socket = None
        self.mutex = threading.Lock()
        self.req_dict = {}
        self.aggregate_send = False

    def _create_socket(self):
        return None

    def close(self):
        pass


def _unique_inproc(name):
    return f"inproc://{name}_{os.getpid()}_{time.time_ns()}"


def _get_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class TestZmqServerBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")

    def test_send_recv_json_with_meta_and_fallback(self):
        context = zmq.Context()
        address = _unique_inproc("json")
        server = _PairServer(context, address)
        client = context.socket(zmq.PAIR)
        client.connect(address)
        try:
            tensor = paddle.to_tensor([1, 2], dtype="int64")
            payload = {"tokens": tensor.numpy().tolist()}

            server.send_json(payload)
            envelope = jsonapi.loads(client.recv())
            self.assertIn("__meta", envelope)
            self.assertEqual(envelope["data"], payload)

            client.send(jsonapi.dumps({"__meta": {"send_ts": time.perf_counter()}, "data": payload}))
            received = server.recv_json()
            self.assertEqual(received, payload)

            plain = ["plain", 1]
            client.send(jsonapi.dumps(plain))
            received_plain = server.recv_json()
            self.assertEqual(received_plain, plain)
        finally:
            client.close()
            server.close()
            context.term()

    def test_send_recv_pyobj_with_meta_and_fallback(self):
        context = zmq.Context()
        address = _unique_inproc("pyobj")
        server = _PairServer(context, address)
        client = context.socket(zmq.PAIR)
        client.connect(address)
        try:
            tensor = paddle.to_tensor([3, 4], dtype="int64")
            payload = {"tokens": tensor.numpy().tolist()}

            server.send_pyobj(payload)
            envelope = ForkingPickler.loads(client.recv())
            self.assertEqual(envelope["data"], payload)

            client.send(ForkingPickler.dumps({"__meta": {"send_ts": time.perf_counter()}, "data": payload}))
            received = server.recv_pyobj()
            self.assertEqual(received, payload)

            raw_payload = ["raw", 5]
            client.send(ForkingPickler.dumps(raw_payload))
            received_raw = server.recv_pyobj()
            self.assertEqual(received_raw, raw_payload)
        finally:
            client.close()
            server.close()
            context.term()

    def test_receive_once_handles_empty_and_closed(self):
        context = zmq.Context()
        address = _unique_inproc("once")
        server = _PairServer(context, address)
        client = context.socket(zmq.PAIR)
        client.connect(address)
        try:
            err, data = server.receive_json_once(block=False)
            self.assertIsNone(err)
            self.assertIsNone(data)

            err, data = server.receive_pyobj_once(block=False)
            self.assertIsNone(err)
            self.assertIsNone(data)

            server.socket.close()
            err, data = server.receive_json_once(block=False)
            self.assertEqual(err, "zmp socket has closed")
            self.assertIsNone(data)

            err, data = server.receive_pyobj_once(block=False)
            self.assertEqual(err, "zmp socket has closed")
            self.assertIsNone(data)
        finally:
            client.close()
            server.close()
            context.term()

    def test_pack_aggregated_data_respects_processor_flag(self):
        server = _NoSocketServer()
        response_a = DummyResponse(1)
        response_b = DummyResponse(2)

        with _EnvGuard(ENABLE_V1_DATA_PROCESSOR=0):
            packed = server.pack_aggregated_data([response_a, response_b])
            unpacked = ForkingPickler.loads(packed)
            self.assertEqual(unpacked, [{"value": 3, "finished": False}])
            self.assertEqual(response_a.add_calls, 1)

        response_c = DummyResponse(4)
        response_d = DummyResponse(5)
        with _EnvGuard(ENABLE_V1_DATA_PROCESSOR=1):
            packed = server.pack_aggregated_data([response_c, response_d])
            unpacked = ForkingPickler.loads(packed)
            self.assertIsInstance(unpacked[0], DummyResponse)
            self.assertEqual(unpacked[0].value, 9)
            self.assertEqual(response_c.add_calls, 1)

    def test_recv_result_handle_registers_request(self):
        port = _get_free_port()
        server = ZmqTcpServer(port=port, mode=zmq.ROUTER)
        client = server.context.socket(zmq.DEALER)
        client.setsockopt(zmq.IDENTITY, b"client-A")
        client.connect(server.address.replace("*", "127.0.0.1"))

        thread = threading.Thread(target=server.recv_result_handle, daemon=True)
        thread.start()
        client.send_multipart([b"", b"req-1"])

        for _ in range(100):
            if "req-1" in server.req_dict:
                break
            time.sleep(0.01)
        self.assertIn("req-1", server.req_dict)

        server.running = False
        thread.join(timeout=1)
        server.socket.close(0)
        client.close()

    def test_recv_result_handle_internal_adapter_sets_handle(self):
        port = _get_free_port()
        with _EnvGuard(FD_ENABLE_INTERNAL_ADAPTER=1):
            server = ZmqTcpServer(port=port, mode=zmq.ROUTER)
            client = server.context.socket(zmq.DEALER)
            client.setsockopt(zmq.IDENTITY, b"client-B")
            client.connect(server.address.replace("*", "127.0.0.1"))

            thread = threading.Thread(target=server.recv_result_handle, daemon=True)
            thread.start()
            client.send_multipart([b"", b"req-2"])

            for _ in range(100):
                if server.response_handle_per_step is not None:
                    break
                time.sleep(0.01)
            self.assertEqual(server.response_handle_per_step, b"client-B")

            server.running = False
            thread.join(timeout=1)
            server.socket.close(0)
            client.close()

    def test_recv_result_handle_cached_finished_sends_response(self):
        port = _get_free_port()
        server = ZmqTcpServer(port=port, mode=zmq.ROUTER)
        client = server.context.socket(zmq.DEALER)
        client.setsockopt(zmq.IDENTITY, b"client-C")
        client.setsockopt(zmq.RCVTIMEO, 2000)
        client.connect(server.address.replace("*", "127.0.0.1"))

        req_id = "req-finished"
        server.cached_results[req_id] = [[DummyResponse(7, finished=True)]]

        thread = threading.Thread(target=server.recv_result_handle, daemon=True)
        thread.start()
        client.send_multipart([b"", req_id.encode("utf-8")])

        frames = client.recv_multipart()
        payload = ForkingPickler.loads(frames[-1])
        self.assertEqual(payload, [{"value": 7, "finished": True}])
        self.assertNotIn(req_id, server.req_dict)

        server.running = False
        thread.join(timeout=1)
        server.socket.close(0)
        client.close()

    def test_recv_result_handle_zmq_error_breaks(self):
        class _ZmqErrorSocket:
            def recv_multipart(self, flags=0):
                raise zmq.error.ZMQError("boom")

        server = _NoSocketServer()
        server.socket = _ZmqErrorSocket()
        server.running = True
        server.recv_result_handle()
        self.assertTrue(server.running)

    def test_recv_result_handle_generic_error_continues_once(self):
        class _GenericErrorSocket:
            def __init__(self, owner):
                self.owner = owner

            def recv_multipart(self, flags=0):
                self.owner.running = False
                raise RuntimeError("generic")

        server = _NoSocketServer()
        server.socket = _GenericErrorSocket(server)
        server.running = True
        server.recv_result_handle()
        self.assertFalse(server.running)


class TestZmqIpcServer(unittest.TestCase):
    def test_create_socket_and_close_removes_file(self):
        tmp_dir = tempfile.mkdtemp()
        file_path = os.path.join(tmp_dir, f"ipc_{os.getpid()}_{time.time_ns()}.socket")
        server = ZmqIpcServer.__new__(ZmqIpcServer)
        ZmqServerBase.__init__(server)
        server.mode = zmq.PULL
        server.file_name = file_path
        server.ZMQ_SNDHWM = 0
        server.context = zmq.Context()
        server.running = True
        try:
            ZmqIpcServer._create_socket(server)
            self.assertTrue(os.path.exists(file_path))
            server.close()
            self.assertFalse(os.path.exists(file_path))
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.isdir(tmp_dir):
                os.rmdir(tmp_dir)

    def test_ipc_init_sets_paths_with_patched_socket(self):
        class _DummySocket:
            closed = True

            def close(self):
                pass

        def _fake_create_socket(self):
            self.socket = _DummySocket()
            self.address = f"ipc://{self.file_name}"
            return self.socket

        with mock.patch.object(ZmqIpcServer, "_create_socket", _fake_create_socket):
            server = ZmqIpcServer(name="patched", mode=zmq.PULL)
            self.assertIn("patched", server.file_name)
            server.context.term()

    def test_clear_ipc_handles_remove_error(self):
        server = ZmqIpcServer.__new__(ZmqIpcServer)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "fail.ipc")
        with open(temp_path, "w", encoding="utf-8") as handle:
            handle.write("x")

        try:
            with mock.patch("os.remove", side_effect=OSError("nope")):
                server._clear_ipc(temp_path)
        finally:
            os.remove(temp_path)
            os.rmdir(temp_dir)

    def test_close_early_return_when_not_running(self):
        server = ZmqIpcServer.__new__(ZmqIpcServer)
        server.running = False
        self.assertIsNone(server.close())

    def test_close_logs_exception(self):
        class _FailingCloseSocket:
            closed = False

            def close(self):
                raise RuntimeError("close boom")

        server = ZmqIpcServer.__new__(ZmqIpcServer)
        server.running = True
        server.socket = _FailingCloseSocket()
        server.context = mock.Mock()
        server.context.closed = False
        server.file_name = "unused"
        server.close()
        self.assertFalse(server.running)

    def test_send_response_per_query_caches_and_sends(self):
        req_id = "req-42"
        client_id = b"client-42"

        with _EnvGuard(FD_USE_AGGREGATE_SEND=False, ENABLE_V1_DATA_PROCESSOR=0):
            port = _get_free_port()
            server = ZmqTcpServer(port=port, mode=zmq.ROUTER)
            client = server.context.socket(zmq.DEALER)
            client.setsockopt(zmq.IDENTITY, client_id)
            client.setsockopt(zmq.RCVTIMEO, 2000)
            client.connect(server.address.replace("*", "127.0.0.1"))

            try:
                client.send_multipart([b"", b"hello"])
                for _ in range(50):
                    try:
                        server.socket.recv_multipart(flags=zmq.NOBLOCK)
                        break
                    except zmq.Again:
                        time.sleep(0.01)

                first = DummyResponse(1, finished=False)
                server._send_response_per_query(req_id, [first])
                self.assertIn(req_id, server.cached_results)
                self.assertEqual(server.cached_results[req_id][0], [first])

                server.req_dict[req_id] = client_id
                second = DummyResponse(2, finished=True)
                server._send_response_per_query(req_id, [second])
                frames = client.recv_multipart()
                payload = ForkingPickler.loads(frames[-1])
                self.assertEqual(payload, [{"value": 1, "finished": False}, {"value": 2, "finished": True}])
                self.assertNotIn(req_id, server.cached_results)
                self.assertNotIn(req_id, server.req_dict)
            finally:
                client.close()
                server.close()


class TestZmqTcpServer(unittest.TestCase):
    def test_send_response_per_step_flushes_cache(self):
        port = _get_free_port()
        server = ZmqTcpServer(port=port, mode=zmq.ROUTER)
        client = server.context.socket(zmq.DEALER)
        client.setsockopt(zmq.IDENTITY, b"step-client")
        client.setsockopt(zmq.RCVTIMEO, 2000)
        client.connect(server.address.replace("*", "127.0.0.1"))

        try:
            client.send_multipart([b"", b"hello"])
            for _ in range(50):
                try:
                    server.socket.recv_multipart(flags=zmq.NOBLOCK)
                    break
                except zmq.Again:
                    time.sleep(0.01)

            cached_data = [[DummyResponse(1)], [DummyResponse(2)]]
            server.response_handle_per_step = None
            server._send_response_per_step(0, cached_data)
            self.assertEqual(len(server.cached_results["data"]), 2)

            server.response_handle_per_step = b"step-client"
            server._send_response_per_step(1, [[DummyResponse(3)]])
            frames = client.recv_multipart()
            decoded = msgpack.unpackb(frames[-1], raw=False)
            self.assertEqual(len(decoded), 3)
            self.assertEqual(decoded[0][0]["value"], 1)
            self.assertEqual(decoded[2][0]["value"], 3)
            self.assertEqual(server.batch_id_per_step, 1)
            self.assertEqual(server.cached_results["data"], [])
        finally:
            client.close()
            server.close()

    def test_recv_control_cmd_and_response(self):
        port = _get_free_port()
        server = ZmqTcpServer(port=port, mode=zmq.ROUTER)
        client = server.context.socket(zmq.DEALER)
        client.setsockopt(zmq.IDENTITY, b"cmd-client")
        client.setsockopt(zmq.RCVTIMEO, 2000)
        client.connect(server.address.replace("*", "127.0.0.1"))

        try:
            self.assertIsNone(server.recv_control_cmd())
            client.send_multipart([b"", msgpack.packb({"task_id": "task-1"})])
            task = None
            for _ in range(50):
                task = server.recv_control_cmd()
                if task is not None:
                    break
                time.sleep(0.01)
            self.assertIsNotNone(task)
            self.assertEqual(task["task_id"], "task-1")
            self.assertIn("task-1", server.req_dict)

            server.response_for_control_cmd("task-1", {"ok": True})
            frames = client.recv_multipart()
            result = msgpack.unpackb(frames[-1], raw=False)
            self.assertEqual(result, {"ok": True})
            self.assertNotIn("task-1", server.req_dict)
        finally:
            client.close()
            server.close()

    def test_response_for_control_cmd_socket_none_raises(self):
        server = ZmqTcpServer.__new__(ZmqTcpServer)
        ZmqServerBase.__init__(server)
        server.socket = None
        server._create_socket = lambda: None
        with self.assertRaises(RuntimeError):
            server.response_for_control_cmd("task-x", {"ok": True})

    def test_response_for_control_cmd_send_error(self):
        class _FailingSendSocket:
            closed = False

            def send_multipart(self, *args, **kwargs):
                raise RuntimeError("send boom")

        server = ZmqTcpServer.__new__(ZmqTcpServer)
        ZmqServerBase.__init__(server)
        server.socket = _FailingSendSocket()
        server.mutex = threading.Lock()
        server.req_dict = {"task-y": b"client-y"}
        server.response_for_control_cmd("task-y", {"ok": True})
        self.assertNotIn("task-y", server.req_dict)

    def test_close_early_return_when_not_running(self):
        server = ZmqTcpServer.__new__(ZmqTcpServer)
        server.running = False
        self.assertIsNone(server.close())

    def test_close_logs_exception(self):
        class _FailingCloseSocket:
            closed = False

            def close(self):
                raise RuntimeError("close boom")

        server = ZmqTcpServer.__new__(ZmqTcpServer)
        server.running = True
        server.socket = _FailingCloseSocket()
        server.context = mock.Mock()
        server.context.closed = False
        server.close()
        self.assertFalse(server.running)


class TestZmqServerErrorPaths(unittest.TestCase):
    def test_ensure_socket_creates_socket(self):
        class _EnsureSocketServer(_NoSocketServer):
            def __init__(self, socket_to_return):
                super().__init__()
                self.socket = None
                self._socket_to_return = socket_to_return

            def _create_socket(self):
                return self._socket_to_return

        sentinel = object()
        server = _EnsureSocketServer(sentinel)
        server._ensure_socket()
        self.assertIs(server.socket, sentinel)

    def test_send_json_error_raises(self):
        class _FailingSendSocket:
            closed = False

            def send(self, *args, **kwargs):
                raise RuntimeError("send json boom")

        server = _NoSocketServer()
        server.socket = _FailingSendSocket()
        with self.assertRaises(RuntimeError):
            server.send_json({"a": 1})

    def test_send_pyobj_error_raises(self):
        class _FailingSendSocket:
            closed = False

            def send(self, *args, **kwargs):
                raise RuntimeError("send pyobj boom")

        server = _NoSocketServer()
        server.socket = _FailingSendSocket()
        with self.assertRaises(RuntimeError):
            server.send_pyobj({"b": 2})

    def test_receive_json_once_error_calls_close(self):
        class _ErrorRecvServer(_NoSocketServer):
            def __init__(self):
                super().__init__()
                self.socket = mock.Mock()
                self.socket.closed = False
                self.close_called = False

            def recv_json(self, flags=0):
                raise ValueError("json error")

            def close(self):
                self.close_called = True

        server = _ErrorRecvServer()
        err, data = server.receive_json_once(block=False)
        self.assertEqual(err, "json error")
        self.assertIsNone(data)
        self.assertTrue(server.close_called)

    def test_receive_pyobj_once_error_calls_close(self):
        class _ErrorRecvServer(_NoSocketServer):
            def __init__(self):
                super().__init__()
                self.socket = mock.Mock()
                self.socket.closed = False
                self.close_called = False

            def recv_pyobj(self, flags=0):
                raise ValueError("pyobj error")

            def close(self):
                self.close_called = True

        server = _ErrorRecvServer()
        err, data = server.receive_pyobj_once(block=False)
        self.assertEqual(err, "pyobj error")
        self.assertIsNone(data)
        self.assertTrue(server.close_called)

    def test_send_response_socket_none_raises(self):
        server = _NoSocketServer()
        with self.assertRaises(RuntimeError):
            server._send_response_per_step(0, [])
        with self.assertRaises(RuntimeError):
            server._send_response_per_query("req", [])

    def test_send_response_per_step_send_error(self):
        class _FailingMultipartSocket:
            closed = False

            def send_multipart(self, *args, **kwargs):
                raise RuntimeError("send multipart boom")

        server = _NoSocketServer()
        server.socket = _FailingMultipartSocket()
        server.response_handle_per_step = b"client"
        server._send_response_per_step(0, [[DummyResponse(1)]])
        self.assertEqual(server.batch_id_per_step, 0)

    def test_send_response_per_query_aggregate_send(self):
        class _CaptureSocket:
            closed = False

            def __init__(self):
                self.frames = None

            def send_multipart(self, frames, copy=False):
                self.frames = frames

        server = _NoSocketServer()
        server.socket = _CaptureSocket()
        server.aggregate_send = True
        server.req_dict["req-agg"] = b"client"
        server._send_response_per_query("req-agg", [DummyResponse(3)])
        payload = ForkingPickler.loads(server.socket.frames[-1])
        self.assertEqual(payload, [{"value": 3, "finished": False}])

    def test_send_response_per_query_enable_v1(self):
        class _CaptureSocket:
            closed = False

            def __init__(self):
                self.frames = None

            def send_multipart(self, frames, copy=False):
                self.frames = frames

        server = _NoSocketServer()
        server.socket = _CaptureSocket()
        server.req_dict["req-v1"] = b"client"
        with _EnvGuard(ENABLE_V1_DATA_PROCESSOR=1):
            server._send_response_per_query("req-v1", [DummyResponse(4)])
        payload = ForkingPickler.loads(server.socket.frames[-1])
        self.assertIsInstance(payload[0], DummyResponse)

    def test_send_response_per_query_send_error(self):
        class _FailingMultipartSocket:
            closed = False

            def send_multipart(self, *args, **kwargs):
                raise RuntimeError("send multipart boom")

        server = _NoSocketServer()
        server.socket = _FailingMultipartSocket()
        server.req_dict["req-fail"] = b"client"
        server._send_response_per_query("req-fail", [DummyResponse(5)])
        self.assertIn("req-fail", server.req_dict)

    def test_send_response_branching(self):
        class _SendResponseServer(_NoSocketServer):
            def __init__(self):
                super().__init__()
                self.step_called = False
                self.query_called = False

            def _send_response_per_step(self, req_id, data):
                self.step_called = True

            def _send_response_per_query(self, req_id, data):
                self.query_called = True

        server = _SendResponseServer()
        with _EnvGuard(FD_ENABLE_INTERNAL_ADAPTER=1):
            server.send_response("req", [])
        self.assertTrue(server.step_called)
        with _EnvGuard(FD_ENABLE_INTERNAL_ADAPTER=0):
            server.send_response("req", [])
        self.assertTrue(server.query_called)

    def test_exit_calls_close(self):
        class _ExitServer(_NoSocketServer):
            def __init__(self):
                super().__init__()
                self.closed = False

            def close(self):
                self.closed = True

        server = _ExitServer()
        server.__exit__(None, None, None)
        self.assertTrue(server.closed)


if __name__ == "__main__":
    unittest.main()
