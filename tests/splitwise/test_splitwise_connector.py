import copy
import importlib.machinery
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "fastdeploy" not in sys.modules:
    fastdeploy_pkg = types.ModuleType("fastdeploy")
    fastdeploy_pkg.__path__ = [str(PROJECT_ROOT / "fastdeploy")]
    fastdeploy_pkg.__spec__ = importlib.machinery.ModuleSpec("fastdeploy", loader=None, is_package=True)
    sys.modules["fastdeploy"] = fastdeploy_pkg

if "paddle" not in sys.modules:
    paddle_stub = types.ModuleType("paddle")
    paddle_dist = types.ModuleType("paddle.distributed")
    paddle_stub.distributed = paddle_dist
    paddle_stub.Tensor = type("Tensor", (), {})
    sys.modules["paddle"] = paddle_stub
    sys.modules["paddle.distributed"] = paddle_dist

if "fastdeploy.utils" not in sys.modules:

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
    sys.modules["fastdeploy.utils"] = utils_stub

if "fastdeploy.metrics" not in sys.modules:
    metrics_pkg = types.ModuleType("fastdeploy.metrics")
    metrics_pkg.__path__ = [str(PROJECT_ROOT / "fastdeploy" / "metrics")]
    metrics_pkg.__spec__ = importlib.machinery.ModuleSpec("fastdeploy.metrics", loader=None, is_package=True)
    sys.modules["fastdeploy.metrics"] = metrics_pkg

if "fastdeploy.metrics.metrics" not in sys.modules:
    metrics_module = types.ModuleType("fastdeploy.metrics.metrics")

    class _Counter:
        def __init__(self):
            self.value = 0

        def inc(self, amount: int = 1):
            self.value += amount

    metrics_module.main_process_metrics = types.SimpleNamespace(send_cache_failed_num=_Counter())
    sys.modules["fastdeploy.metrics.metrics"] = metrics_module

from fastdeploy.engine.request import (
    CompletionOutput,
    Request,
    RequestMetrics,
    RequestOutput,
)
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.splitwise import splitwise_connector
from fastdeploy.splitwise.splitwise_connector import SplitwiseConnector


class _FakeAvailableQueue:
    def __init__(self):
        self.size = 0

    def qsize(self):
        return self.size


class FakeEngineWorkerQueue:
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


class InspectableConnector(SplitwiseConnector):
    def __init__(self, *args, **kwargs):
        self.sent_messages = []
        super().__init__(*args, **kwargs)

    def _send_message(self, addr, msg_type: str, payload):  # pragma: no cover - overridden for tests
        self.sent_messages.append((addr, msg_type, copy.deepcopy(payload)))


class DummyTask:
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
    def __init__(self):
        self.sockets: list[_StubSocket] = []

    def socket(self, kind):
        sock = _StubSocket(kind)
        self.sockets.append(sock)
        return sock


class _StubPoller:
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
def _patch_engine_worker_queue(monkeypatch):
    monkeypatch.setenv("FD_ENABLE_CACHE_TASK", "0")
    monkeypatch.setenv("ENABLE_V1_KVCACHE_SCHEDULER", "0")
    monkeypatch.setenv("FD_PD_CHANGEABLE", "0")
    monkeypatch.setenv("FD_ENGINE_TASK_QUEUE_WITH_SHM", "0")
    monkeypatch.setattr(splitwise_connector, "EngineWorkerQueue", FakeEngineWorkerQueue)


def test_has_splitwise_tasks_detects_prefill_backlog():
    cfg = make_cfg(innode_ports=[7001])
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    connector.create_connection(7001)
    queue = connector.connect_innode_instances[7001]
    queue.available_prefill_instances.size = 1
    assert connector.has_splitwise_tasks() is False
    queue.available_prefill_instances.size = 0
    assert connector.has_splitwise_tasks() is True


def test_dispatch_innode_splitwise_tasks_promotes_decode_role():
    cfg = make_cfg(innode_ports=[8002])
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    connector.create_connection(8002)
    queue = connector.connect_innode_instances[8002]
    queue.prefill_ready = True
    task = make_task("req-dispatch", role="prefill", protocol="ipc")
    connector.dispatch_innode_splitwise_tasks([task], current_id=33)
    assert queue.disaggregated_tasks[-1][0] == "prefill"
    assert task.disaggregate_info["role"] == "decode"
    assert task.disaggregate_info["cache_info"]["ipc"]["current_id"] == 33


def test_send_splitwise_tasks_dispatches_when_innode_ports_available():
    cfg = make_cfg(innode_ports=[8100])
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    connector.create_connection(8100)
    connector.connect_innode_instances[8100].prefill_ready = True
    task = make_task("req-prefill", role="prefill", protocol="ipc")
    connector.send_splitwise_tasks([task], current_id=44)
    assert connector.connect_innode_instances[8100].disaggregated_tasks


def test_send_splitwise_tasks_innode_rewrites_ports_for_decode_queue():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    connector.create_connection(8123)
    task = make_task("req-innode", role="decode", protocol="ipc")
    snapshot_port = connector.send_splitwise_tasks_innode([task], 8123)
    recorded = connector.connect_innode_instances[8123].disaggregated_tasks[-1]
    assert snapshot_port == 8123
    assert (
        recorded[1][0].disaggregate_info["cache_info"]["ipc"]["port"]
        == cfg.parallel_config.engine_worker_queue_port[0]
    )
    assert task.disaggregate_info["cache_info"]["ipc"]["port"] == 8123


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


def test_send_cache_infos_prefill_batches_into_worker_queue():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    task = make_task("req-prefill", role="prefill", protocol="ipc")
    was_decode = connector.send_cache_infos([task], current_id=11)
    assert was_decode is False
    assert worker_queue.cache_infos[-1][0]["request_id"] == "req-prefill"
    assert worker_queue.cache_infos[-1][0]["current_id"] == 11


def test_send_cache_infos_decode_rdma_triggers_remote_sync():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    task = make_task("req-decode", role="decode", protocol="rdma")
    result = connector.send_cache_infos([task], current_id=22)
    assert result is True
    assert connector.sent_messages[-1][1] == "cache_sync"
    assert worker_queue.cache_infos == []


def test_send_cache_infos_decode_ipc_forwards_to_local_worker():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    connector.create_connection(9300)
    task = make_task("req-local", role="decode", protocol="ipc")
    task.disaggregate_info["cache_info"]["ipc"]["port"] = 9300
    connector.send_cache_infos([task], current_id=7)
    assert connector.connect_innode_instances[9300].cache_infos[-1][0]["transfer_protocol"] == "ipc"


def test_send_cache_infos_rdma_with_error_message_forwards_reason():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    task = make_task("req-err", role="decode", protocol="rdma")
    task.error_msg = "remote boom"
    connector.send_cache_infos([task], current_id=0)
    assert connector.sent_messages[-1][1] == "cache_sync"
    assert "error_msg" in connector.sent_messages[-1][2][0]


def test_send_first_token_to_ipc_decode_queue():
    cfg = make_cfg()
    worker_queue = FakeEngineWorkerQueue()
    connector = InspectableConnector(cfg, worker_queue, object())
    connector.create_connection(9400)
    msg = {"transfer_protocol": "ipc", "cache_info": {"ipc": {"port": 9400}}}
    task = make_task("req-first", role="decode", protocol="ipc")
    connector.send_first_token(msg, [task])
    assert connector.connect_innode_instances[9400].disaggregated_tasks[-1][0] == "decode"


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
    assert ok and msg == ""
    task2 = make_task("req-error", role="prefill", protocol="rdma")
    connector.current_request_ids["req-error"] = "failed"
    ok2, msg2 = connector.check_decode_allocated(task2)
    assert ok2 is False and msg2 == "failed"


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
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    dummy = DummySocket()
    connector.push_sockets = {"test": dummy}
    connector._close_connection("test")
    assert dummy.closed is True
    assert connector.push_sockets == {}


def test_send_message_initializes_network_and_serializes(monkeypatch):
    monkeypatch.setattr(splitwise_connector, "zmq", _make_stub_zmq())

    class DummyExecutor:
        def __init__(self, *_, **__):
            self.calls = []

        def submit(self, fn, *args, **kwargs):
            self.calls.append((fn, args, kwargs))

    monkeypatch.setattr(splitwise_connector, "ThreadPoolExecutor", DummyExecutor)

    cfg = make_cfg(pd_comm_port=[9550], enable_expert_parallel=True, data_parallel_size=2, local_data_parallel_id=1)
    worker_queue = FakeEngineWorkerQueue()
    connector = SplitwiseConnector(cfg, worker_queue, object())
    output = RequestOutput("req-zmq")
    connector._send_message("127.0.0.1:9551", "decode", [output])
    sock = connector.push_sockets["127.0.0.1:9551"]
    assert json.loads(sock.sent[-1][1].decode("utf-8"))["type"] == "decode"


def test_send_message_handles_failures_and_resets_socket(monkeypatch):
    monkeypatch.setattr(splitwise_connector, "zmq", _make_stub_zmq())
    monkeypatch.setattr(splitwise_connector, "ThreadPoolExecutor", lambda *_, **__: None)
    cfg = make_cfg(pd_comm_port=[9660])
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
