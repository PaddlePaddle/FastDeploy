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

import sys
import types

import numpy as np
import paddle
import pytest

if not hasattr(paddle, "compat"):
    paddle.compat = types.SimpleNamespace(enable_torch_proxy=lambda *args, **kwargs: None)

from fastdeploy.cache_manager import cache_messager


class _DummyBarrier:
    def __init__(self):
        self.calls = 0

    def wait(self):
        self.calls += 1


class _DummyEngineWorkerQueue:
    def __init__(self, cache_info_sequence=None, connect_task_sequence=None, **kwargs):
        self.cache_info_sequence = list(cache_info_sequence or [])
        self.connect_task_sequence = list(connect_task_sequence or [])
        self.cache_info_calls = 0
        self.connect_task_calls = 0
        self.cache_info_barrier = _DummyBarrier()
        self.finish_add_cache_task_barrier = _DummyBarrier()
        self.finish_send_cache_barrier = _DummyBarrier()
        self.connect_task_barrier = _DummyBarrier()
        self.connect_task_response_barrier = _DummyBarrier()
        self.begin_send_cache_barrier = _DummyBarrier()
        self.finished_add_cache_task_req_ids = []
        self.finished_req_payloads = []
        self.connect_task_responses = []

    def get_cache_info(self):
        if self.cache_info_calls >= len(self.cache_info_sequence):
            raise SystemExit
        info = self.cache_info_sequence[self.cache_info_calls]
        self.cache_info_calls += 1
        return info

    def put_finished_add_cache_task_req(self, req_ids):
        self.finished_add_cache_task_req_ids.append(req_ids)

    def put_finished_req(self, payload):
        self.finished_req_payloads.append(payload)

    def get_connect_rdma_task(self):
        if self.connect_task_calls >= len(self.connect_task_sequence):
            raise SystemExit
        task = self.connect_task_sequence[self.connect_task_calls]
        self.connect_task_calls += 1
        return task, None

    def put_connect_rdma_task_response(self, response):
        self.connect_task_responses.append(response)


class _DummyRDMACommManager:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.connect_calls = []

    def connect(self, *args):
        self.connect_calls.append(args)
        return True


class _DummyIPCCommManager:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.write_cache_calls = []
        self.sync_calls = []

    def write_cache(self, *args):
        self.write_cache_calls.append(args)
        return 0

    def write_block_by_sync(self, decode_idx):
        self.sync_calls.append(decode_idx)


class _DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(("info", msg))

    def debug(self, msg):
        self.messages.append(("debug", msg))

    def error(self, msg):
        self.messages.append(("error", msg))


class _DummySignalValue:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.set_calls = []

    def __getitem__(self, index):
        if not self.sequence:
            return 0
        value = self.sequence.pop(0)
        return value

    def __setitem__(self, index, value):
        self.set_calls.append((index, value))


class _DummyIPCSignal:
    instances = []

    def __init__(self, name, array, **kwargs):
        self.name = name
        self.dtype = kwargs.get("dtype", np.array(array).dtype)
        self.value = _DummySignalValue(array)
        _DummyIPCSignal.instances.append(self)


class _DummyTensor:
    def __init__(self, shape, dtype, place):
        self.shape = shape
        self.dtype = dtype
        self.place = place

    def data_ptr(self):
        return 123


class _DummyPlace:
    def __str__(self):
        return "Place(gpu:0)"


def _build_cache_kvs(dtype="float16", include_value_cache=True, num_layers=1):
    gpu_cache_kvs = {}
    for layer_idx in range(num_layers):
        key_cache = paddle.zeros([2, 3], dtype=dtype)
        gpu_cache_kvs[f"key_caches_{layer_idx}_rank0_device0"] = key_cache
        if include_value_cache:
            gpu_cache_kvs[f"value_caches_{layer_idx}_rank0_device0"] = paddle.zeros([2, 3], dtype=dtype)
    return gpu_cache_kvs


def _build_dummy_cache_kvs(include_value_cache=True, num_layers=1):
    gpu_cache_kvs = {}
    for layer_idx in range(num_layers):
        gpu_cache_kvs[f"key_caches_{layer_idx}_rank0_device0"] = _DummyTensor(
            shape=[2, 3], dtype=paddle.float16, place=_DummyPlace()
        )
        if include_value_cache:
            gpu_cache_kvs[f"value_caches_{layer_idx}_rank0_device0"] = _DummyTensor(
                shape=[2, 3], dtype=paddle.float16, place=_DummyPlace()
            )
    return gpu_cache_kvs


def test_parse_args_and_get_decode_ip_idx(monkeypatch):
    args = [
        "prog",
        "--splitwise_role",
        "prefill",
        "--rank",
        "1",
        "--device_id",
        "2",
        "--num_layers",
        "3",
        "--key_cache_shape",
        "2,3,4,5",
        "--value_cache_shape",
        "2,3,4,5",
        "--rdma_port",
        "1234",
        "--mp_num",
        "2",
        "--ipc_suffix",
        "suffix",
        "--protocol",
        "rdma",
        "--pod_ip",
        "127.0.0.1",
        "--cache_queue_port",
        "9911",
        "--engine_worker_queue_port",
        "9912",
        "--cache_dtype",
        "uint8",
        "--speculative_config",
        "{}",
        "--local_data_parallel_id",
        "1",
    ]
    monkeypatch.setattr(cache_messager, "logger", _DummyLogger(), raising=False)
    monkeypatch.setattr(sys, "argv", args, raising=False)
    parsed = cache_messager.parse_args()
    assert parsed.splitwise_role == "prefill"
    assert parsed.rank == 1
    assert parsed.device_id == 2
    assert parsed.cache_dtype == "uint8"

    decode_ip, decode_ports = cache_messager.get_decode_ip_idx({"ip": "1.1.1.1", "rdma_ports": [3]})
    assert decode_ip == "1.1.1.1"
    assert decode_ports == [3]
    decode_ip, decode_ports = cache_messager.get_decode_ip_idx({"decode_ip": "2.2.2.2", "decode_rdma_ports": [5]})
    assert decode_ip == "2.2.2.2"
    assert decode_ports == [5]


def test_cache_messager_init_rdma_block_bytes(monkeypatch):
    monkeypatch.setattr(cache_messager, "EngineWorkerQueue", _DummyEngineWorkerQueue)
    monkeypatch.setattr(cache_messager, "RDMACommManager", _DummyRDMACommManager)
    monkeypatch.setattr(cache_messager, "logger", _DummyLogger(), raising=False)
    gpu_cache_kvs = _build_cache_kvs(dtype="float16", include_value_cache=True, num_layers=1)
    messager = cache_messager.CacheMessager(
        splitwise_role="mixed",
        transfer_protocol="rdma",
        pod_ip="0.0.0.0",
        engine_worker_queue_port=9000,
        local_data_parallel_id=1,
        gpu_cache_kvs=gpu_cache_kvs,
        rank=0,
        nranks=2,
        num_layers=1,
        gpu_id=0,
        rdma_port="1111",
    )
    assert messager.block_bytes == 6
    assert messager.rank_id == 2
    assert "rdma" in messager.messager


def test_cache_messager_init_ipc_uses_local_device_id(monkeypatch):
    monkeypatch.setattr(cache_messager, "EngineWorkerQueue", _DummyEngineWorkerQueue)
    monkeypatch.setattr(cache_messager, "IPCCommManager", _DummyIPCCommManager)
    monkeypatch.setattr(cache_messager, "logger", _DummyLogger(), raising=False)
    dummy_key = _DummyTensor(shape=[2, 3], dtype=paddle.float16, place=_DummyPlace())
    dummy_val = _DummyTensor(shape=[2, 3], dtype=paddle.float16, place=_DummyPlace())
    gpu_cache_kvs = {
        "key_caches_0_rank0_device0": dummy_key,
        "value_caches_0_rank0_device0": dummy_val,
    }
    messager = cache_messager.CacheMessager(
        splitwise_role="mixed",
        transfer_protocol="ipc",
        pod_ip="0.0.0.0",
        engine_worker_queue_port=9000,
        local_data_parallel_id=0,
        gpu_cache_kvs=gpu_cache_kvs,
        rank=0,
        nranks=1,
        num_layers=1,
        gpu_id=0,
        rdma_port=None,
    )
    assert messager.block_bytes == 6
    assert "ipc" in messager.messager


def test_cache_messager_prefill_layerwise_send_cache_thread(monkeypatch):
    class _PrefillIPCSignal:
        def __init__(self, name, array, **kwargs):
            if "step" in name:
                sequence = [0, 1, 1]
            else:
                sequence = [0, 0, 0]
            self.name = name
            self.dtype = kwargs.get("dtype", np.array(array).dtype)
            self.value = _DummySignalValue(sequence)

    dummy_queue = _DummyEngineWorkerQueue(
        cache_info_sequence=[
            [
                {
                    "request_id": "req-1",
                    "src_block_ids": [0],
                    "dest_block_ids": [1],
                    "status": "init",
                    "current_id": 0,
                    "transfer_protocol": "ipc",
                    "device_ids": [0],
                }
            ],
            None,
            None,
        ]
    )
    monkeypatch.setattr(cache_messager, "EngineWorkerQueue", lambda *args, **kwargs: dummy_queue)
    monkeypatch.setattr(cache_messager, "IPCCommManager", _DummyIPCCommManager)
    monkeypatch.setattr(cache_messager, "shared_memory_exists", lambda name: False)
    monkeypatch.setattr(cache_messager, "IPCSignal", _PrefillIPCSignal)
    monkeypatch.setattr(cache_messager, "logger", _DummyLogger(), raising=False)
    gpu_cache_kvs = _build_dummy_cache_kvs(include_value_cache=True, num_layers=2)
    messager = cache_messager.CacheMessager(
        splitwise_role="mixed",
        transfer_protocol="ipc",
        pod_ip="0.0.0.0",
        engine_worker_queue_port=9000,
        local_data_parallel_id=0,
        gpu_cache_kvs=gpu_cache_kvs,
        rank=0,
        nranks=1,
        num_layers=2,
        gpu_id=0,
        rdma_port=None,
    )
    _DummyIPCSignal.instances.clear()
    with pytest.raises(SystemExit):
        messager.prefill_layerwise_send_cache_thread()
    assert dummy_queue.finished_req_payloads
    assert dummy_queue.finished_req_payloads[0][0][0] == "req-1"


def test_cache_messager_v1_add_cache_task_thread(monkeypatch):
    dummy_queue = _DummyEngineWorkerQueue(
        cache_info_sequence=[
            [
                {
                    "request_id": "req-2",
                    "src_block_ids": [0, 1, 2],
                    "dest_block_ids": [3],
                    "current_id": 7,
                    "need_prefill_tokens": 128,
                    "transfer_protocol": "rdma",
                },
                {"request_id": "req-new"},
            ]
        ]
    )
    monkeypatch.setattr(cache_messager, "EngineWorkerQueue", lambda *args, **kwargs: dummy_queue)
    monkeypatch.setattr(cache_messager, "RDMACommManager", _DummyRDMACommManager)
    monkeypatch.setattr(cache_messager, "logger", _DummyLogger(), raising=False)
    gpu_cache_kvs = _build_cache_kvs(dtype="float16", include_value_cache=True, num_layers=1)
    messager = cache_messager.CacheMessagerV1(
        splitwise_role="mixed",
        transfer_protocol="rdma",
        pod_ip="0.0.0.0",
        engine_worker_queue_port=9000,
        local_data_parallel_id=0,
        gpu_cache_kvs=gpu_cache_kvs,
        rank=0,
        nranks=1,
        num_layers=1,
        gpu_id=0,
        block_size=64,
        rdma_port="2222",
    )
    messager.cache_info["req-2"] = {
        "request_id": "req-2",
        "src_block_ids": [0, 1, 2],
        "dest_block_ids": [3],
        "current_id": 7,
        "need_prefill_tokens": 128,
        "transfer_protocol": "rdma",
    }
    with pytest.raises(SystemExit):
        messager._add_cache_task_thread()
    assert dummy_queue.finished_add_cache_task_req_ids == [["req-2"]]
    assert messager.cache_info["req-2"]["status"] == "init"


def test_cache_messager_v1_prefill_layerwise_send_cache_thread(monkeypatch):
    class _OneShotQueue:
        def __init__(self):
            self.called = False

        def get(self):
            if self.called:
                raise SystemExit
            self.called = True
            return [(0, 64)]

    dummy_queue = _DummyEngineWorkerQueue()
    monkeypatch.setattr(cache_messager, "EngineWorkerQueue", lambda *args, **kwargs: dummy_queue)
    monkeypatch.setattr(cache_messager, "IPCCommManager", _DummyIPCCommManager)
    monkeypatch.setattr(cache_messager, "logger", _DummyLogger(), raising=False)
    gpu_cache_kvs = _build_dummy_cache_kvs(include_value_cache=True, num_layers=2)
    messager = cache_messager.CacheMessagerV1(
        splitwise_role="mixed",
        transfer_protocol="ipc",
        pod_ip="0.0.0.0",
        engine_worker_queue_port=9000,
        local_data_parallel_id=0,
        gpu_cache_kvs=gpu_cache_kvs,
        rank=0,
        nranks=1,
        num_layers=2,
        gpu_id=0,
        block_size=64,
        rdma_port=None,
    )
    messager.cache_prefilled_engine_ids_queue = _OneShotQueue()
    messager.idx_cache_task_dict[0] = {
        "request_id": "req-3",
        "src_block_ids": [0],
        "dest_block_ids": [1],
        "transfer_protocol": "ipc",
        "device_ids": [0],
        "need_prefill_tokens": 64,
        "sended_layer_id": -1,
        "sended_block_num": 0,
        "status": "init",
        "current_id": 0,
    }
    messager.engine_cache_tasks[0] = {"prefilled_layer_idx": 1, "prefilled_token_num": 64}
    messager.cache_info["req-3"] = messager.idx_cache_task_dict[0]
    with pytest.raises(SystemExit):
        messager.prefill_layerwise_send_cache_thread()
    assert dummy_queue.finished_req_payloads
    assert dummy_queue.finished_req_payloads[0][0][0] == "req-3"


def test_cache_messager_v1_handle_connect_task(monkeypatch):
    dummy_queue = _DummyEngineWorkerQueue(
        connect_task_sequence=[
            {"task_id": 1, "decode_ip": "1.1.1.1", "decode_rdma_ports": [1234, 5678]},
            {"task_id": 2, "decode_ip": "2.2.2.2", "decode_rdma_ports": [4321]},
        ]
    )
    monkeypatch.setattr(cache_messager, "EngineWorkerQueue", lambda *args, **kwargs: dummy_queue)
    monkeypatch.setattr(cache_messager, "RDMACommManager", _DummyRDMACommManager)
    monkeypatch.setattr(cache_messager, "logger", _DummyLogger(), raising=False)
    gpu_cache_kvs = _build_cache_kvs(dtype="float16", include_value_cache=False, num_layers=1)
    messager = cache_messager.CacheMessagerV1(
        splitwise_role="mixed",
        transfer_protocol="rdma",
        pod_ip="0.0.0.0",
        engine_worker_queue_port=9000,
        local_data_parallel_id=0,
        gpu_cache_kvs=gpu_cache_kvs,
        rank=0,
        nranks=2,
        num_layers=1,
        gpu_id=0,
        block_size=64,
        rdma_port="2222",
    )
    with pytest.raises(SystemExit):
        messager._handle_connect_task()
    assert dummy_queue.connect_task_responses[0]["success"] is True
    assert dummy_queue.connect_task_responses[1]["success"] is True


def test_cache_messager_init_shm_and_xpu_paths(monkeypatch):
    monkeypatch.setattr(cache_messager.envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", True)
    monkeypatch.setattr(cache_messager, "EngineWorkerQueue", _DummyEngineWorkerQueue)
    monkeypatch.setattr(cache_messager, "RDMACommManager", _DummyRDMACommManager)
    monkeypatch.setattr(cache_messager, "get_peer_mem_addr", lambda ptr: ptr + 1)
    monkeypatch.setattr(cache_messager.paddle, "is_compiled_with_xpu", lambda: True)
    monkeypatch.setattr(cache_messager, "logger", _DummyLogger(), raising=False)
    gpu_cache_kvs = _build_dummy_cache_kvs(include_value_cache=True, num_layers=1)
    messager = cache_messager.CacheMessager(
        splitwise_role="mixed",
        transfer_protocol="rdma",
        pod_ip="0.0.0.0",
        engine_worker_queue_port=9000,
        local_data_parallel_id=0,
        gpu_cache_kvs=gpu_cache_kvs,
        rank=0,
        nranks=1,
        num_layers=1,
        gpu_id=0,
        rdma_port="1111",
    )
    assert messager.engine_worker_queue is not None


def test_cache_messager_handle_connect_task_error(monkeypatch):
    dummy_queue = _DummyEngineWorkerQueue(
        connect_task_sequence=[
            None,
            {"task_id": 1, "decode_ip": "1.1.1.1", "decode_rdma_ports": [1234]},
        ]
    )

    class _FailingRDMACommManager(_DummyRDMACommManager):
        def connect(self, *args):
            self.connect_calls.append(args)
            return False

    monkeypatch.setattr(cache_messager, "EngineWorkerQueue", lambda *args, **kwargs: dummy_queue)
    monkeypatch.setattr(cache_messager, "RDMACommManager", _FailingRDMACommManager)
    monkeypatch.setattr(cache_messager, "logger", _DummyLogger(), raising=False)
    gpu_cache_kvs = _build_cache_kvs(dtype="float16", include_value_cache=False, num_layers=1)
    messager = cache_messager.CacheMessager(
        splitwise_role="mixed",
        transfer_protocol="rdma",
        pod_ip="0.0.0.0",
        engine_worker_queue_port=9000,
        local_data_parallel_id=0,
        gpu_cache_kvs=gpu_cache_kvs,
        rank=0,
        nranks=1,
        num_layers=1,
        gpu_id=0,
        rdma_port="1111",
    )
    with pytest.raises(SystemExit):
        messager._handle_connect_task()
    assert dummy_queue.connect_task_responses[0]["success"] is False


def test_cache_messager_v1_shm_xpu_and_bfloat16(monkeypatch):
    monkeypatch.setattr(cache_messager.envs, "FD_ENGINE_TASK_QUEUE_WITH_SHM", True)
    monkeypatch.setattr(cache_messager, "EngineWorkerQueue", _DummyEngineWorkerQueue)
    monkeypatch.setattr(cache_messager, "RDMACommManager", _DummyRDMACommManager)
    monkeypatch.setattr(cache_messager, "get_peer_mem_addr", lambda ptr: ptr + 1)
    monkeypatch.setattr(cache_messager.paddle, "is_compiled_with_xpu", lambda: True)
    monkeypatch.setattr(cache_messager, "logger", _DummyLogger(), raising=False)
    gpu_cache_kvs = _build_dummy_cache_kvs(include_value_cache=True, num_layers=1)
    gpu_cache_kvs["key_caches_0_rank0_device0"].dtype = paddle.bfloat16
    messager = cache_messager.CacheMessagerV1(
        splitwise_role="mixed",
        transfer_protocol="rdma",
        pod_ip="0.0.0.0",
        engine_worker_queue_port=9000,
        local_data_parallel_id=0,
        gpu_cache_kvs=gpu_cache_kvs,
        rank=0,
        nranks=1,
        num_layers=1,
        gpu_id=0,
        block_size=64,
        rdma_port="2222",
    )
    assert messager.block_bytes == 6


def test_cache_messager_v1_consume_signals(monkeypatch):
    monkeypatch.setattr(cache_messager, "EngineWorkerQueue", _DummyEngineWorkerQueue)
    monkeypatch.setattr(cache_messager, "RDMACommManager", _DummyRDMACommManager)
    monkeypatch.setattr(cache_messager, "logger", _DummyLogger(), raising=False)

    class _QueueRecorder:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    counter = {"calls": 0}

    def _fake_get_output_kv_signal(kv_signal_data, rank_id, wait_flag):
        if counter["calls"] > 0:
            raise SystemExit
        counter["calls"] += 1
        data = np.full(kv_signal_data.shape, -1, dtype="int32")
        data[0] = 1
        data[1] = 0
        data[2] = 2
        data[3] = 4
        data[4] = 5
        kv_signal_data.set_value(data)

    monkeypatch.setattr(cache_messager, "get_output_kv_signal", _fake_get_output_kv_signal)
    gpu_cache_kvs = _build_cache_kvs(dtype="float16", include_value_cache=False, num_layers=1)
    messager = cache_messager.CacheMessagerV1(
        splitwise_role="mixed",
        transfer_protocol="rdma",
        pod_ip="0.0.0.0",
        engine_worker_queue_port=9000,
        local_data_parallel_id=0,
        gpu_cache_kvs=gpu_cache_kvs,
        rank=0,
        nranks=1,
        num_layers=1,
        gpu_id=0,
        block_size=64,
        rdma_port="2222",
    )
    messager.cache_info["req-4"] = {"request_id": "req-4"}
    messager.cache_prefilled_engine_ids_queue = _QueueRecorder()
    with pytest.raises(SystemExit):
        messager.consume_signals()
    assert messager.cache_prefilled_engine_ids_queue.items == [[(2, 9)]]


def test_main_initializes_cache_and_exits(monkeypatch):
    monkeypatch.setattr(cache_messager, "set_device", lambda device: None)
    monkeypatch.setattr(cache_messager, "set_data_ipc", lambda tensor, name: None)
    monkeypatch.setattr(cache_messager, "EngineWorkerQueue", _DummyEngineWorkerQueue)
    monkeypatch.setattr(cache_messager, "IPCSignal", _DummyIPCSignal)
    monkeypatch.setattr(cache_messager, "RDMACommManager", _DummyRDMACommManager)
    monkeypatch.setattr(cache_messager, "logger", _DummyLogger(), raising=False)
    monkeypatch.setattr(cache_messager.envs, "ENABLE_V1_KVCACHE_SCHEDULER", True)

    class _DummySpeculativeConfig:
        def __init__(self, *args, **kwargs):
            self.num_extra_cache_layer = 0
            self.num_gpu_block_expand_ratio = 0

    monkeypatch.setattr(cache_messager, "SpeculativeConfig", _DummySpeculativeConfig)
    monkeypatch.setattr(cache_messager.CacheMessagerV1, "_handle_connect_task", lambda self: None)

    args = types.SimpleNamespace(
        device_id=0,
        rank=0,
        default_dtype="float16",
        cache_dtype="float16",
        key_cache_shape="1,1,1,1",
        value_cache_shape="",
        mp_num=1,
        num_layers=1,
        splitwise_role="decode",
        protocol="rdma",
        pod_ip="0.0.0.0",
        engine_worker_queue_port=9000,
        cache_queue_port=9001,
        ipc_suffix=None,
        rdma_port="2222",
        speculative_config={},
        local_data_parallel_id=0,
    )
    monkeypatch.setattr(cache_messager, "args", args, raising=False)
    monkeypatch.setattr(
        cache_messager.CacheMessagerV1,
        "prefill_layerwise_send_cache_thread",
        lambda self: (_ for _ in ()).throw(SystemExit),
    )

    with pytest.raises(SystemExit):
        cache_messager.main()
