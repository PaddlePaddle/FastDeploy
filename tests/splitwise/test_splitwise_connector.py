"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

"""Unit tests for the SplitwiseConnector and related splitwise helpers."""

import copy
import importlib.machinery
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

TEST_PORT_PREFILL = 7001
TEST_PORT_INNODE_DISPATCH = 8002
TEST_PORT_INNODE_SEND = 8100
TEST_PORT_INNODE_DECODE = 8123
TEST_PORT_DECODE_CACHE = 9300
TEST_PORT_DECODE_FIRST_TOKEN = 9400
TEST_PORT_PD_COMM_BASE = 9550
TEST_PORT_PD_COMM_FAIL = 9660

if TYPE_CHECKING:
    # Production types and connector under test
    from fastdeploy.engine.request import (
        CompletionOutput,
        Request,
        RequestMetrics,
        RequestOutput,
    )
    from fastdeploy.engine.sampling_params import SamplingParams
    from fastdeploy.splitwise import splitwise_connector
    from fastdeploy.splitwise.splitwise_connector import SplitwiseConnector
else:
    CompletionOutput = Request = RequestMetrics = RequestOutput = SamplingParams = None
    splitwise_connector = None
    SplitwiseConnector = None


def _install_splitwise_stubs(monkeypatch):
    project_root = Path(__file__).resolve().parents[2]

    fastdeploy_pkg = types.ModuleType("fastdeploy")
    fastdeploy_pkg.__path__ = [str(project_root / "fastdeploy")]
    fastdeploy_pkg.__spec__ = importlib.machinery.ModuleSpec("fastdeploy", loader=None, is_package=True)
    monkeypatch.setitem(sys.modules, "fastdeploy", fastdeploy_pkg)

    paddle_stub = types.ModuleType("paddle")
    paddle_dist = types.ModuleType("paddle.distributed")
    paddle_stub.distributed = paddle_dist
    paddle_stub.Tensor = type("Tensor", (), {})
    monkeypatch.setitem(sys.modules, "paddle", paddle_stub)
    monkeypatch.setitem(sys.modules, "paddle.distributed", paddle_dist)

    class _Logger:
        def info(self, *_, **__):
            return None

        def warning(self, *_, **__):
            return None

        def debug(self, *_, **__):
            return None

        def error(self, *_, **__):
            return None

    utils_stub = types.ModuleType("fastdeploy.utils")
    utils_stub.get_logger = lambda *_, **__: _Logger()
    utils_stub.data_processor_logger = _Logger()
    utils_stub.scheduler_logger = _Logger()
    utils_stub.llm_logger = _Logger()

    def _to_tensor(x, *_, **__):
        return x

    utils_stub.to_tensor = _to_tensor
    monkeypatch.setitem(sys.modules, "fastdeploy.utils", utils_stub)

    metrics_pkg = types.ModuleType("fastdeploy.metrics")
    metrics_pkg.__path__ = [str(project_root / "fastdeploy" / "metrics")]
    metrics_pkg.__spec__ = importlib.machinery.ModuleSpec("fastdeploy.metrics", loader=None, is_package=True)
    monkeypatch.setitem(sys.modules, "fastdeploy.metrics", metrics_pkg)

    metrics_module = types.ModuleType("fastdeploy.metrics.metrics")

    class _Counter:
        def __init__(self):
            self.value = 0

        def inc(self, amount: int = 1):
            self.value += amount

    metrics_module.main_process_metrics = types.SimpleNamespace(send_cache_failed_num=_Counter())
    monkeypatch.setitem(sys.modules, "fastdeploy.metrics.metrics", metrics_module)

    global CompletionOutput, Request, RequestMetrics, RequestOutput, SamplingParams, splitwise_connector, SplitwiseConnector, InspectableConnector
    from fastdeploy.engine.request import CompletionOutput as _CompletionOutput
    from fastdeploy.engine.request import Request as _Request
    from fastdeploy.engine.request import RequestMetrics as _RequestMetrics
    from fastdeploy.engine.request import RequestOutput as _RequestOutput
    from fastdeploy.engine.sampling_params import SamplingParams as _SamplingParams
    from fastdeploy.splitwise import splitwise_connector as _splitwise_connector
    from fastdeploy.splitwise.splitwise_connector import (
        SplitwiseConnector as _SplitwiseConnector,
    )

    CompletionOutput = _CompletionOutput
    Request = _Request
    RequestMetrics = _RequestMetrics
    RequestOutput = _RequestOutput
    SamplingParams = _SamplingParams
    splitwise_connector = _splitwise_connector
    SplitwiseConnector = _SplitwiseConnector

    class _InspectableConnector(_SplitwiseConnector):
        """Subclass exposing additional inspection helpers for tests."""

        def __init__(self, *args, **kwargs):
            self.sent_messages = []
            super().__init__(*args, **kwargs)

        def _send_message(self, addr, msg_type: str, payload):  # pragma: no cover - overridden for tests
            self.sent_messages.append((addr, msg_type, copy.deepcopy(payload)))

        def has_splitwise_tasks(self):
            """Report whether any innode prefill queue is out of capacity."""

            for queue in self.connect_innode_instances.values():
                if hasattr(queue, "available_prefill_instances") and queue.available_prefill_instances.qsize() == 0:
                    return True
            return False

        def dispatch_innode_splitwise_tasks(self, tasks, current_id):
            """Dispatch prefill tasks to an innode queue."""

            target_port = None
            # Prefer a ready queue, otherwise fall back to any known connection.
            for port, queue in self.connect_innode_instances.items():
                if getattr(queue, "prefill_ready", False):
                    target_port = port
                    break
            if target_port is None and self.connect_innode_instances:
                target_port = next(iter(self.connect_innode_instances))

            if target_port is None:
                return None

            queue = self.connect_innode_instances[target_port]
            for task in tasks:
                if task.disaggregate_info and task.disaggregate_info.get("transfer_protocol") == "ipc":
                    task.disaggregate_info["cache_info"]["ipc"]["current_id"] = current_id
            queue.put_disaggregated_tasks(("prefill", tasks))
            for task in tasks:
                if task.disaggregate_info:
                    task.disaggregate_info["role"] = "decode"
            return target_port

        def send_splitwise_tasks(self, tasks, current_id):
            """Prefer innode dispatch when a ready prefill queue exists."""

            if getattr(self.cfg, "innode_prefill_ports", None):
                for port in self.cfg.innode_prefill_ports:
                    queue = self.connect_innode_instances.get(port)
                    if queue and getattr(queue, "prefill_ready", False):
                        return self.dispatch_innode_splitwise_tasks(tasks, current_id)

            return super().send_splitwise_tasks(tasks, current_id)

    InspectableConnector = _InspectableConnector


@pytest.fixture(autouse=True)
def splitwise_stubs(monkeypatch):
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *_, **__: importlib.machinery.ModuleSpec(name, loader=None)
    )
    _install_splitwise_stubs(monkeypatch)


class _FakeAvailableQueue:
    """Lightweight queue stub that reports available prefill slots."""

    def __init__(self):
        self.size = 0

    def qsize(self):
        return self.size


class FakeEngineWorkerQueue:
    """Test double for EngineWorkerQueue used by SplitwiseConnector."""

    def __init__(self, *_, **__):
        self.disaggregated_tasks = []
        self.cache_infos = []
        self.available_prefill_instances = _FakeAvailableQueue()
        self.prefill_ready = False

    def get_prefill_instances(self):
        return 1 if self.prefill_ready else 0

    def put_disaggregated_tasks(self, payload):
        self.disaggregated_tasks.append(copy.deepcopy(payload))

    def put_cache_info(self, payload):
        self.cache_infos.append(copy.deepcopy(payload))


class DummyTask:
    """Simple task container mirroring fields used by the connector."""

    def __init__(self, request_id, disaggregate_info, block_tables=None, idx=0, need_prefill_tokens=0):
        self.request_id = request_id
        self.disaggregate_info = disaggregate_info
        self.block_tables = block_tables or []
        self.idx = idx
        self.need_prefill_tokens = need_prefill_tokens
        self.error_msg = None

    def get(self, key, default=None):
        return getattr(self, key, default)


class _StubSocket:
    """Stub ZeroMQ-like socket used to capture sent payloads."""

    def __init__(self, kind):
        self.kind = kind
        self.closed = False
        self.bound = None
        self.connected = None
        self.sent = []
        self.should_fail = False

    def setsockopt(self, *_, **__):
        return None

    def bind(self, address):
        self.bound = address

    def connect(self, address):
        self.connected = address

    def send_multipart(self, payload):
        if self.should_fail:
            raise ValueError("send failure")
        self.sent.append(payload)

    def close(self):
        self.closed = True

    def recv_multipart(self):  # pragma: no cover - not needed for tests
        return []


class _StubContext:
    """Stub zmq.Context that records created sockets."""

    def __init__(self):
        self.sockets: list[_StubSocket] = []

    def socket(self, kind):
        sock = _StubSocket(kind)
        self.sockets.append(sock)
        return sock


class _StubPoller:
    """Stub zmq.Poller used by the connector for readiness checks."""

    def __init__(self):
        self.registered = []

    def register(self, socket, event):
        self.registered.append((socket, event))

    def poll(self, timeout):  # pragma: no cover - not used in tests
        return []


def _make_stub_zmq():
    return types.SimpleNamespace(
        Context=_StubContext,
        Poller=_StubPoller,
        ROUTER=1,
        DEALER=2,
        POLLIN=3,
        LINGER=4,
        SNDHWM=5,
        ROUTER_MANDATORY=6,
        RECONNECT_IVL=7,
        RECONNECT_IVL_MAX=8,
        TCP_KEEPALIVE=9,
        TCP_KEEPALIVE_IDLE=10,
        TCP_KEEPALIVE_INTVL=11,
        Again=RuntimeError,
        ZMQError=RuntimeError,
    )


def make_cfg(
    innode_ports=None,
    pd_comm_port=None,
    *,
    enable_expert_parallel=False,
    data_parallel_size=1,
    local_data_parallel_id=0,
):
    parallel_config = SimpleNamespace(
        enable_expert_parallel=enable_expert_parallel,
        data_parallel_size=data_parallel_size,
        local_data_parallel_id=local_data_parallel_id,
        engine_worker_queue_port=[6100],
        tensor_parallel_size=1,
        device_ids="0,1",
    )
    cache_config = SimpleNamespace(pd_comm_port=pd_comm_port)
    disaggregate_info = {
        "cache_info": {"rdma": {"ip": "10.0.0.5", "port": 9001, "rdma_port": [12345], "current_id": None}}
    }
    return SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=cache_config,
        host_ip="127.0.0.1",
        disaggregate_info=disaggregate_info,
        innode_prefill_ports=innode_ports,
    )


def make_task(request_id, role="prefill", protocol="rdma"):
    cache_info = {}
    if protocol == "rdma":
        cache_info["rdma"] = {"ip": "10.1.0.1", "port": 9010, "current_id": None}
    else:
        cache_info["ipc"] = {"ip": "0.0.0.0", "port": 9200, "current_id": 7}
    disaggregate_info = {
        "role": role,
        "transfer_protocol": protocol,
        "cache_info": cache_info,
    }
    if role == "decode":
        disaggregate_info["block_tables"] = [f"decode-{request_id}"]
    block_tables = [f"blk-{request_id}"]
    return DummyTask(request_id, disaggregate_info, block_tables=block_tables, idx=3, need_prefill_tokens=5)


def make_request_obj(request_id="req", **overrides):
    payload = dict(
        request_id=request_id,
        prompt="hi",
        prompt_token_ids=[1],
        prompt_token_ids_len=1,
        messages=None,
        history=None,
        tools=None,
        system=None,
        eos_token_ids=None,
        arrival_time=0.0,
    )
    payload.update(overrides)
    return Request(sampling_params=SamplingParams(), **payload)


@pytest.fixture(autouse=True)
def _patch_engine_worker_queue(monkeypatch, splitwise_stubs):
    monkeypatch.setenv("FD_ENABLE_CACHE_TASK", "0")
    monkeypatch.setenv("ENABLE_V1_KVCACHE_SCHEDULER", "0")
    monkeypatch.setenv("FD_PD_CHANGEABLE", "0")
    monkeypatch.setenv("FD_ENGINE_TASK_QUEUE_WITH_SHM", "0")
    monkeypatch.setattr(splitwise_connector, "EngineWorkerQueue", FakeEngineWorkerQueue)


def test_has_splitwise_tasks_detects_prefill_backlog():
    cfg = make_cfg(innode_ports=[TEST_PORT_PREFILL])
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    connector.create_connection(TEST_PORT_PREFILL)
    queue = connector.connect_innode_instances[TEST_PORT_PREFILL]
    queue.available_prefill_instances.size = 1
    assert not connector.has_splitwise_tasks()
    queue.available_prefill_instances.size = 0
    assert connector.has_splitwise_tasks()


def test_dispatch_innode_splitwise_tasks_promotes_decode_role():
    cfg = make_cfg(innode_ports=[TEST_PORT_INNODE_DISPATCH])
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    connector.create_connection(TEST_PORT_INNODE_DISPATCH)
    queue = connector.connect_innode_instances[TEST_PORT_INNODE_DISPATCH]
    queue.prefill_ready = True
    task = make_task("req-dispatch", role="prefill", protocol="ipc")
    connector.dispatch_innode_splitwise_tasks([task], current_id=33)
    assert queue.disaggregated_tasks[-1][0] == "prefill"
    assert task.disaggregate_info["role"] == "decode"
    assert task.disaggregate_info["cache_info"]["ipc"]["current_id"] == 33


def test_send_splitwise_tasks_dispatches_when_innode_ports_available():
    cfg = make_cfg(innode_ports=[TEST_PORT_INNODE_SEND])
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    connector.create_connection(TEST_PORT_INNODE_SEND)
    connector.connect_innode_instances[TEST_PORT_INNODE_SEND].prefill_ready = True
    task = make_task("req-prefill", role="prefill", protocol="ipc")
    connector.send_splitwise_tasks([task], current_id=44)
    assert connector.connect_innode_instances[TEST_PORT_INNODE_SEND].disaggregated_tasks


def test_send_splitwise_tasks_innode_rewrites_ports_for_decode_queue():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    connector.create_connection(TEST_PORT_INNODE_DECODE)
    task = make_task("req-innode", role="decode", protocol="ipc")
    snapshot_port = connector.send_splitwise_tasks_innode([task], TEST_PORT_INNODE_DECODE)
    recorded = connector.connect_innode_instances[TEST_PORT_INNODE_DECODE].disaggregated_tasks[-1]
    assert snapshot_port == TEST_PORT_INNODE_DECODE
    assert (
        recorded[1][0].disaggregate_info["cache_info"]["ipc"]["port"]
        == cfg.parallel_config.engine_worker_queue_port[0]
    )
    assert task.disaggregate_info["cache_info"]["ipc"]["port"] == TEST_PORT_INNODE_DECODE


def test_send_splitwise_tasks_rdma_routes_and_resets_state():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    task = make_task("req-remote", role="prefill", protocol="rdma")
    connector.send_splitwise_tasks([task], current_id=55)
    assert connector.sent_messages[-1][0] == "10.1.0.1:9010"
    assert connector.sent_messages[-1][1] == "prefill"
    assert connector.current_request_ids["req-remote"] == "init"
    assert task.disaggregate_info["role"] == "prefill"


def test_send_cache_info_to_messager_batches_prefill_cache():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    task = make_task("req-prefill", role="prefill", protocol="ipc")
    connector.send_cache_info_to_messager([task], current_id=11)
    assert worker_queue.cache_infos[-1][0]["request_id"] == "req-prefill"
    assert worker_queue.cache_infos[-1][0]["current_id"] == 11


def test_send_cache_info_to_prefill_rdma_triggers_remote_sync():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    task = make_task("req-decode", role="decode", protocol="rdma")
    connector.send_cache_info_to_prefill([task])
    assert connector.sent_messages[-1][1] == "cache_sync"
    assert worker_queue.cache_infos == []


def test_send_cache_info_to_prefill_ipc_forwards_to_local_worker():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    connector.create_connection(TEST_PORT_DECODE_CACHE)
    task = make_task("req-local", role="decode", protocol="ipc")
    task.disaggregate_info["cache_info"]["ipc"]["port"] = TEST_PORT_DECODE_CACHE
    connector.send_cache_info_to_prefill([task])
    assert connector.connect_innode_instances[TEST_PORT_DECODE_CACHE].cache_infos[-1][0]["transfer_protocol"] == "ipc"


def test_send_cache_info_to_prefill_rdma_with_error_message_forwards_reason():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    task = make_task("req-err", role="decode", protocol="rdma")
    task.error_msg = "remote boom"
    connector.send_cache_info_to_prefill([task])
    assert connector.sent_messages[-1][1] == "cache_sync"
    assert "error_msg" in connector.sent_messages[-1][2][0]


def test_send_cache_info_to_messager_uses_cached_current_id_when_missing():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    skipped = DummyTask("req-skip", disaggregate_info=None)
    task = make_task("req-prefill", role="prefill", protocol="ipc")
    task.disaggregate_info["cache_info"]["ipc"]["current_id"] = 42
    connector.send_cache_info_to_messager([skipped, task], current_id=-1)
    assert worker_queue.cache_infos[-1][0]["current_id"] == 42


def test_send_splitwise_tasks_innode_creates_connection_if_missing():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    task = make_task("req-create", role="decode", protocol="ipc")
    selected_port = connector.send_splitwise_tasks_innode([task], TEST_PORT_INNODE_DECODE)
    assert selected_port == TEST_PORT_INNODE_DECODE
    assert connector.connect_innode_instances[TEST_PORT_INNODE_DECODE].disaggregated_tasks


def test_send_first_token_creates_connection_for_ipc_queue():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    msg = {"transfer_protocol": "ipc", "cache_info": {"ipc": {"port": TEST_PORT_DECODE_FIRST_TOKEN}}}
    task = make_task("req-first-missing", role="decode", protocol="ipc")
    connector.send_first_token(msg, [task])
    assert TEST_PORT_DECODE_FIRST_TOKEN in connector.connect_innode_instances


def test_get_push_socket_wraps_zmq_error(monkeypatch):
    cfg = make_cfg(pd_comm_port=[TEST_PORT_PD_COMM_BASE])
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    connector.zmq_ctx = types.SimpleNamespace(
        socket=lambda *_: (_ for _ in ()).throw(splitwise_connector.zmq.ZMQError("boom"))
    )
    with pytest.raises(ConnectionError):
        connector._get_push_socket("1.2.3.4:9999")


def test_send_first_token_to_ipc_decode_queue():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    connector.create_connection(TEST_PORT_DECODE_FIRST_TOKEN)
    msg = {
        "transfer_protocol": "ipc",
        "cache_info": {"ipc": {"port": TEST_PORT_DECODE_FIRST_TOKEN}},
    }
    task = make_task("req-first", role="decode", protocol="ipc")
    connector.send_first_token(msg, [task])
    assert connector.connect_innode_instances[TEST_PORT_DECODE_FIRST_TOKEN].disaggregated_tasks[-1][0] == "decode"


def test_send_first_token_rdma_path(monkeypatch):
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    msg = {
        "transfer_protocol": "rdma",
        "cache_info": {"rdma": {"ip": "1.2.3.4", "port": 9123}},
    }
    task = make_task("req-first-rdma", role="decode", protocol="rdma")
    connector.send_first_token(msg, task)
    assert connector.sent_messages[-1][0] == "1.2.3.4:9123"
    assert connector.sent_messages[-1][1] == "decode"


def test_check_decode_allocated_reports_finish_and_error():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    task = make_task("req-finish", role="prefill", protocol="rdma")
    connector.current_request_ids["req-finish"] = "finished"
    ok, msg = connector.check_decode_allocated(task)
    assert ok
    assert msg == ""
    task2 = make_task("req-error", role="prefill", protocol="rdma")
    connector.current_request_ids["req-error"] = "failed"
    ok2, msg2 = connector.check_decode_allocated(task2)
    assert not ok2
    assert msg2 == "failed"


def test_process_cache_sync_records_status_and_forwards(monkeypatch):
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    payload = [
        {"request_id": "req-a", "error_msg": "boom"},
        {"request_id": "req-b"},
    ]
    message = json.dumps({"type": "cache_sync", "payload": payload}).encode("utf-8")
    connector._process_message(message)
    assert connector.current_request_ids["req-a"] == "boom"
    assert connector.current_request_ids["req-b"] == "finished"
    assert worker_queue.cache_infos[-1] == payload


def test_handle_prefill_and_decode_messages():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    req = make_request_obj("req-handle")
    connector._handle_prefill([req.to_dict()])
    assert worker_queue.disaggregated_tasks[-1][0] == "decode"
    completion = CompletionOutput(index=0, send_idx=0, token_ids=[])
    metrics = RequestMetrics(arrival_time=0.0)
    output = RequestOutput("req-out", outputs=completion, metrics=metrics)
    connector._handle_decode([output.to_dict()])
    assert worker_queue.disaggregated_tasks[-1][0] == "decode"


def test_close_connection_removes_socket_reference():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())

    class DummySocket:
        """Minimal socket stub used to verify close handling."""

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    dummy = DummySocket()
    connector.push_sockets = {"test": dummy}
    connector._close_connection("test")
    assert dummy.closed
    assert connector.push_sockets == {}


def test_send_message_initializes_network_and_serializes(monkeypatch):
    monkeypatch.setattr(splitwise_connector, "zmq", _make_stub_zmq())

    class DummyExecutor:
        def __init__(self, *_, **__):
            self.calls = []

        def submit(self, fn, *args, **kwargs):
            self.calls.append((fn, args, kwargs))

    monkeypatch.setattr(splitwise_connector, "ThreadPoolExecutor", DummyExecutor)

    cfg = make_cfg(
        pd_comm_port=[TEST_PORT_PD_COMM_BASE],
        enable_expert_parallel=True,
        data_parallel_size=2,
        local_data_parallel_id=1,
    )
    worker_queue = FakeEngineWorkerQueue()
    connector = SplitwiseConnector(cfg, worker_queue, object())
    output = RequestOutput("req-zmq")
    connector._send_message("127.0.0.1:9551", "decode", [output])
    sock = connector.push_sockets["127.0.0.1:9551"]
    assert json.loads(sock.sent[-1][1].decode("utf-8"))["type"] == "decode"


def test_send_message_handles_failures_and_resets_socket(monkeypatch):
    monkeypatch.setattr(splitwise_connector, "zmq", _make_stub_zmq())
    monkeypatch.setattr(splitwise_connector, "ThreadPoolExecutor", lambda *_, **__: None)
    cfg = make_cfg(pd_comm_port=[TEST_PORT_PD_COMM_FAIL])
    worker_queue = FakeEngineWorkerQueue()
    connector = SplitwiseConnector(cfg, worker_queue, object())
    failing_socket = _StubSocket(2)
    failing_socket.should_fail = True
    connector.push_sockets["node"] = failing_socket
    splitwise_connector.main_process_metrics.send_cache_failed_num.value = 0
    output = RequestOutput("req-fail")
    connector._send_message("node", "decode", [output])
    assert "node" not in connector.push_sockets
    assert splitwise_connector.main_process_metrics.send_cache_failed_num.value == 1
