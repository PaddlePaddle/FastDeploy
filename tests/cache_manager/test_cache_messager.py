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

from __future__ import annotations

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


class _FakePlace:
    def __init__(self, device: str):
        self._device = device

    def __str__(self):  # pragma: no cover - representation helper
        return f"Place({self._device})"


class _FakeTensor:
    def __init__(self, array, dtype="float32", device="gpu:0"):
        self._array = np.array(array)
        self.shape = tuple(self._array.shape)
        self.dtype = dtype
        self.place = _FakePlace(device)

    def data_ptr(self):
        return int(self._array.__array_interface__["data"][0])

    def numel(self):
        return int(self._array.size)

    def numpy(self):
        return self._array

    def tolist(self):  # pragma: no cover - convenience helper
        return self.numpy().tolist()

    def __len__(self):
        return len(self._array)

    def __iter__(self):  # pragma: no cover - container helper
        return iter(self._array)

    def __getitem__(self, idx):
        value = self._array[idx]
        if isinstance(value, np.ndarray):
            return _FakeTensor(value, dtype=self.dtype)
        return _FakeScalar(value)

    def __setitem__(self, idx, value):
        self._array[idx] = value


class _FakeScalar:
    def __init__(self, value):
        self._value = value.item() if hasattr(value, "item") else value

    def numpy(self):
        return np.array(self._value)

    def tolist(self):  # pragma: no cover - compatibility helper
        return self.numpy().tolist()

    def __int__(self):
        return int(self._value)

    def __index__(self):  # pragma: no cover - required for range()
        return int(self._value)

    def __eq__(self, other):  # pragma: no cover - comparison helper
        return int(self._value) == other


class ParseArgsTest(unittest.TestCase):
    def test_parse_args_reads_cli_values(self):
        module = _load_cache_messager()
        argv = [
            "prog",
            "--splitwise_role",
            "decode",
            "--rank",
            "3",
            "--device_id",
            "5",
            "--num_layers",
            "4",
            "--key_cache_shape",
            "2,3,4,5",
            "--value_cache_shape",
            "2,3,4,5",
            "--rdma_port",
            "1234",
            "--mp_num",
            "2",
            "--engine_pid",
            "abc",
            "--protocol",
            "ipc,rdma",
            "--pod_ip",
            "1.2.3.4",
            "--cache_queue_port",
            "9100",
            "--engine_worker_queue_port",
            "9101",
            "--cache_dtype",
            "uint8",
            "--speculative_config",
            '{"num_extra_cache_layer":1}',
            "--local_data_parallel_id",
            "7",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = module.parse_args()

        self.assertEqual(args.splitwise_role, "decode")
        self.assertEqual(args.rank, 3)
        self.assertEqual(args.device_id, 5)
        self.assertEqual(args.num_layers, 4)
        self.assertEqual(args.protocol, "ipc,rdma")
        self.assertEqual(args.cache_dtype, "uint8")
        self.assertEqual(args.local_data_parallel_id, 7)
        self.assertEqual(args.speculative_config["num_extra_cache_layer"], 1)


class _Barrier:
    def __init__(self):
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1


class _IPCCommManager:
    def __init__(self, rank, gpu_id, cache_k, cache_v):  # pylint: disable=unused-argument
        self.rank = rank
        self.gpu_id = gpu_id
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.write_calls = []
        self.sync_targets = []

    def write_cache(self, target_ip, target_id, src_block_ids, dest_block_ids, layer_idx):
        self.write_calls.append((target_ip, target_id, tuple(src_block_ids), tuple(dest_block_ids), layer_idx))
        return 0

    def write_block_by_sync(self, target_id):
        self.sync_targets.append(target_id)


class _RDMACommManager:
    def __init__(
        self,
        splitwise_role,
        rank,
        gpu_id,
        cache_k_ptr_list,
        cache_v_ptr_list,
        max_block_num,
        block_bytes,
        rdma_port,
    ):  # pylint: disable=unused-argument
        self.rank = rank
        self.calls = []
        self.connect_results = []

    def connect(self, target_ip, target_id):
        result = True if not self.connect_results else self.connect_results.pop(0)
        self.calls.append((target_ip, target_id, result))
        return result

    def write_cache(self, *args, **kwargs):  # pragma: no cover - compatibility helper
        return 0


class _IPCSignal:
    instances: dict[str, "_IPCSignal"] = {}

    def __init__(self, name, array, dtype=None, suffix=None, create=False):  # noqa: D401
        # pylint: disable=unused-argument
        self.name = name
        self.value = np.array(array)
        _IPCSignal.instances[name if suffix is None else f"{name}_{suffix}"] = self


class _EngineWorkerQueue:
    def __init__(
        self,
        address,
        is_server,
        num_client,
        client_id,
        local_data_parallel_id,
    ):
        self.address = address
        self.is_server = is_server
        self.num_client = num_client
        self.client_id = client_id
        self.local_data_parallel_id = local_data_parallel_id
        self.cache_info_barrier = _Barrier()
        self.finish_send_cache_barrier = _Barrier()
        self.finish_add_cache_task_barrier = _Barrier()
        self.begin_send_cache_barrier = _Barrier()
        self.connect_task_barrier = _Barrier()
        self.connect_task_response_barrier = _Barrier()
        self.cache_info_sequence = []
        self.cache_info_calls = 0
        self.stop_after_cache_info = False
        self.signal_initializer = None
        self.connect_tasks = []
        self.connect_task_calls = 0
        self.stop_after_connect_tasks = False
        self.finished_requests = []
        self.connect_responses = []
        self.finished_add_cache_task_req = []

    def get_cache_info(self):
        if self.cache_info_calls == 0 and self.signal_initializer:
            self.signal_initializer()
        if self.cache_info_calls < len(self.cache_info_sequence):
            info = self.cache_info_sequence[self.cache_info_calls]
            self.cache_info_calls += 1
            return info
        if self.stop_after_cache_info:
            raise SystemExit("stop cache info")
        return []

    def put_finished_req(self, request_payload):
        self.finished_requests.append(request_payload)

    def put_finished_add_cache_task_req(self, req_ids):
        self.finished_add_cache_task_req.append(req_ids)

    def get_connect_rdma_task(self):
        if self.connect_task_calls < len(self.connect_tasks):
            task = self.connect_tasks[self.connect_task_calls]
            self.connect_task_calls += 1
            return task, None
        if self.stop_after_connect_tasks:
            raise SystemExit("stop connect task")
        return None, None

    def put_connect_rdma_task_response(self, response):
        self.connect_responses.append(response)


def _install_paddle_stubs():
    paddle = _ensure_module("paddle")
    paddle.Tensor = _FakeTensor
    paddle.bfloat16 = "bfloat16"

    def _full(shape, fill_value=0, dtype="float32"):
        dtype_str = dtype if isinstance(dtype, str) else str(dtype)
        return _FakeTensor(np.full(shape, fill_value), dtype=dtype_str)

    def _to_tensor(data, dtype="float32", place=None):  # pylint: disable=unused-argument
        dtype_str = dtype if isinstance(dtype, str) else str(dtype)
        return _FakeTensor(np.array(data), dtype=dtype_str)

    paddle.full = _full
    paddle.to_tensor = _to_tensor
    paddle.is_compiled_with_xpu = lambda: False
    paddle.float16 = "float16"
    paddle.set_device = lambda _name: None

    device_mod = types.ModuleType("paddle.device")
    device_mod.set_device = lambda _name: None
    cuda_mod = types.ModuleType("paddle.device.cuda")
    cuda_mod.memory_allocated = lambda: 0
    device_mod.cuda = cuda_mod
    paddle.device = device_mod
    sys.modules["paddle.device"] = device_mod
    sys.modules["paddle.device.cuda"] = cuda_mod


def _install_fastdeploy_core_stubs():
    fastdeploy_pkg = _ensure_module("fastdeploy")
    fastdeploy_pkg.__path__ = [str(PROJECT_ROOT / "fastdeploy")]

    utils_module = types.ModuleType("fastdeploy.utils")
    envs_module = types.ModuleType("fastdeploy.utils.envs")
    envs_module.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
    envs_module.ENABLE_V1_KVCACHE_SCHEDULER = False

    class _Logger:
        def __init__(self):
            self.messages = {"info": [], "debug": [], "error": []}

        def info(self, msg):
            self.messages["info"].append(msg)

        def debug(self, msg):
            self.messages["debug"].append(msg)

        def error(self, msg):
            self.messages["error"].append(msg)

    def _get_logger(_name, _filename=None):  # pylint: disable=unused-argument
        return _Logger()

    utils_module.envs = envs_module
    utils_module.get_logger = _get_logger

    def console_logger(_name, _filename=None):  # pylint: disable=unused-argument
        """Stub console_logger returning a logger instance for tests."""
        return _Logger()

    utils_module.console_logger = console_logger
    sys.modules["fastdeploy.utils"] = utils_module
    sys.modules["fastdeploy.utils.envs"] = envs_module
    fastdeploy_pkg.utils = utils_module

    platforms_module = types.ModuleType("fastdeploy.platforms")

    class _Platform:
        def is_cuda(self):
            return True

        def is_xpu(self):  # pragma: no cover - alternate platform helper
            return False

    platforms_module.current_platform = _Platform()
    sys.modules["fastdeploy.platforms"] = platforms_module
    fastdeploy_pkg.platforms = platforms_module


def _install_transfer_factory_stubs():
    transfer_factory = types.ModuleType("fastdeploy.cache_manager.transfer_factory")
    transfer_factory.IPCCommManager = _IPCCommManager
    transfer_factory.RDMACommManager = _RDMACommManager
    sys.modules["fastdeploy.cache_manager.transfer_factory"] = transfer_factory


def _install_config_stubs():
    fastdeploy_pkg = _ensure_module("fastdeploy")
    config_module = types.ModuleType("fastdeploy.config")

    class _SpeculativeConfig:
        def __init__(self, config_dict):
            self.num_extra_cache_layer = config_dict.get("num_extra_cache_layer", 0)
            self.num_gpu_block_expand_ratio = config_dict.get("num_gpu_block_expand_ratio", 0)

    config_module.SpeculativeConfig = _SpeculativeConfig
    sys.modules["fastdeploy.config"] = config_module
    fastdeploy_pkg.config = config_module


def _install_inter_comm_stubs():
    inter_comm_module = types.ModuleType("fastdeploy.inter_communicator")
    inter_comm_module.EngineWorkerQueue = _EngineWorkerQueue
    inter_comm_module.IPCSignal = _IPCSignal
    inter_comm_module.shared_memory_exists = lambda _name: False
    sys.modules["fastdeploy.inter_communicator"] = inter_comm_module


def _install_ops_gpu_stubs():
    ops_gpu_module = types.ModuleType("fastdeploy.model_executor.ops.gpu")

    def _get_output_kv_signal(buffer, rank_id, flag):  # pylint: disable=unused-argument
        sequence = getattr(_get_output_kv_signal, "sequence", None)
        if not sequence:
            raise SystemExit("kv signal stop")

        step = sequence.pop(0)
        if step.get("stop"):
            raise SystemExit("kv signal stop")

        data = buffer.numpy()
        data.fill(-1)
        tasks = step.get("tasks", -1)
        data[0] = tasks
        if tasks == -1:
            return
        data[1] = step.get("layer", 0)
        data[2] = step.get("engine", 0)
        data[3] = step.get("offset", 0)
        data[4] = step.get("current", 0)

    ops_gpu_module.get_output_kv_signal = _get_output_kv_signal
    ops_gpu_module.set_data_ipc = lambda *args, **kwargs: None
    ops_gpu_module.unset_data_ipc = lambda *args, **kwargs: None
    ops_gpu_module.share_external_data = lambda cache, *args, **kwargs: cache
    ops_gpu_module.swap_cache_all_layers = lambda *args, **kwargs: None
    ops_gpu_module.cuda_host_alloc = lambda *args, **kwargs: None
    ops_gpu_module.cuda_host_free = lambda *args, **kwargs: None
    ops_gpu_module.get_data_ptr_ipc = lambda *args, **kwargs: 0
    ops_gpu_module.ipc_sent_key_value_cache_by_remote_ptr = lambda *args, **kwargs: 0
    ops_gpu_module.ipc_sent_key_value_cache_by_remote_ptr_block_sync = lambda *args, **kwargs: 0
    sys.modules["fastdeploy.model_executor.ops.gpu"] = ops_gpu_module


def _install_dependency_stubs():
    _install_paddle_stubs()
    _install_fastdeploy_core_stubs()
    _install_transfer_factory_stubs()
    _install_config_stubs()
    _install_inter_comm_stubs()
    _install_ops_gpu_stubs()


def _load_cache_messager():
    module_name = "fastdeploy.cache_manager.cache_messager"
    if module_name in sys.modules:
        return sys.modules[module_name]

    _install_dependency_stubs()

    spec = importlib.util.spec_from_file_location(
        module_name, PROJECT_ROOT / "fastdeploy" / "cache_manager" / "cache_messager.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


def _make_cache_tensors(num_layers, dtype="bfloat16"):
    cache = {}
    for layer in range(num_layers):
        cache[f"key_caches_{layer}_rank0_device0"] = _FakeTensor(np.zeros((2, 3, 4, 5)), dtype=dtype)
        cache[f"value_caches_{layer}_rank0_device0"] = _FakeTensor(np.zeros((2, 3, 4, 5)), dtype=dtype)
    return cache


class CacheMessagerInitTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_cache_messager()
        envs = sys.modules["fastdeploy.utils.envs"]
        envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        _IPCSignal.instances.clear()
        ops_gpu = sys.modules["fastdeploy.model_executor.ops.gpu"]
        ops_gpu.get_output_kv_signal.sequence = []

    def tearDown(self):
        _IPCSignal.instances.clear()

    def test_initializes_with_ipc_and_rdma(self):
        cache = _make_cache_tensors(num_layers=2)
        messager = self.module.CacheMessager(
            splitwise_role="mixed",
            transfer_protocol="ipc,rdma",
            pod_ip="127.0.0.1",
            engine_worker_queue_port=9000,
            local_data_parallel_id=0,
            gpu_cache_kvs=cache,
            rank=0,
            nranks=1,
            num_layers=2,
            gpu_id=0,
            rdma_port=55,
        )

        self.assertIsInstance(messager.engine_worker_queue, _EngineWorkerQueue)
        self.assertEqual(messager.engine_worker_queue.address, ("127.0.0.1", 9000))
        self.assertIn("ipc", messager.messager)
        self.assertIn("rdma", messager.messager)
        expected_block_bytes = math.prod(cache["key_caches_0_rank0_device0"].shape[1:]) * 2
        self.assertEqual(messager.block_bytes, expected_block_bytes)

    def test_shm_socket_address_and_uint8_dtype(self):
        envs = sys.modules["fastdeploy.utils.envs"]
        envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = True
        cache = _make_cache_tensors(num_layers=1, dtype="uint8")
        messager = self.module.CacheMessager(
            splitwise_role="mixed",
            transfer_protocol="ipc",
            pod_ip="127.0.0.1",
            engine_worker_queue_port=9010,
            local_data_parallel_id=0,
            gpu_cache_kvs=cache,
            rank=0,
            nranks=1,
            num_layers=1,
            gpu_id=0,
        )

        self.assertTrue(str(messager.engine_worker_queue.address).startswith("/dev/shm/fd_task_queue_"))
        expected_block_bytes = math.prod(cache["key_caches_0_rank0_device0"].shape[1:])
        self.assertEqual(messager.block_bytes, expected_block_bytes)


class PrefillThreadTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_cache_messager()
        envs = sys.modules["fastdeploy.utils.envs"]
        envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        _IPCSignal.instances.clear()
        ops_gpu = sys.modules["fastdeploy.model_executor.ops.gpu"]
        ops_gpu.get_output_kv_signal.sequence = []

    def tearDown(self):
        _IPCSignal.instances.clear()

    def test_prefill_thread_transfers_and_marks_finished(self):
        cache = _make_cache_tensors(num_layers=1)
        messager = self.module.CacheMessager(
            splitwise_role="mixed",
            transfer_protocol="ipc",
            pod_ip="127.0.0.1",
            engine_worker_queue_port=9001,
            local_data_parallel_id=0,
            gpu_cache_kvs=cache,
            rank=0,
            nranks=1,
            num_layers=1,
            gpu_id=0,
        )

        queue = messager.engine_worker_queue
        queue.cache_info_sequence = [
            [
                {
                    "request_id": "req-1",
                    "transfer_protocol": "ipc",
                    "src_block_ids": [0, 1],
                    "dest_block_ids": [2, 3],
                    "current_id": 0,
                    "status": "init",
                    "layer_idx": 0,
                    "device_ids": {0: 0},
                }
            ]
        ]
        queue.stop_after_cache_info = True

        def _set_signals(instance):
            step_key = f"splitwise_complete_prefilled_step_{instance.rank_id}_{instance.gpu_id}"
            layer_key = f"splitwise_complete_prefilled_layer_{instance.rank_id}_{instance.gpu_id}"
            _IPCSignal.instances[step_key].value[0] = 0
            _IPCSignal.instances[layer_key].value[0] = 0

        queue.signal_initializer = lambda: _set_signals(messager)

        with self.assertRaises(SystemExit):
            messager.prefill_layerwise_send_cache_thread()

        self.assertEqual(queue.finish_send_cache_barrier.wait_calls, 1)
        self.assertEqual(queue.finished_requests, [[["req-1", "finished"]]])
        self.assertEqual(
            messager.messager["ipc"].write_calls,
            [("0.0.0.0", 0, (0, 1), (2, 3), 0)],
        )


class HandleConnectTaskTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_cache_messager()
        envs = sys.modules["fastdeploy.utils.envs"]
        envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        _IPCSignal.instances.clear()
        ops_gpu = sys.modules["fastdeploy.model_executor.ops.gpu"]
        ops_gpu.get_output_kv_signal.sequence = []

    def tearDown(self):
        _IPCSignal.instances.clear()

    def test_handle_connect_task_success_and_failure(self):
        cache = _make_cache_tensors(num_layers=1)
        messager = self.module.CacheMessager(
            splitwise_role="decode",
            transfer_protocol="rdma",
            pod_ip="127.0.0.1",
            engine_worker_queue_port=9002,
            local_data_parallel_id=0,
            gpu_cache_kvs=cache,
            rank=0,
            nranks=1,
            num_layers=1,
            gpu_id=0,
            rdma_port=88,
        )

        rdma_manager = messager.messager["rdma"]
        rdma_manager.connect_results = [True, False]

        queue = messager.engine_worker_queue
        queue.connect_tasks = [
            {
                "task_id": 1,
                "ip": "10.0.0.1",
                "rdma_ports": {0: 7},
            },
            {
                "task_id": 2,
                "ip": "10.0.0.2",
                "rdma_ports": {0: 9},
            },
        ]
        queue.stop_after_connect_tasks = True

        with self.assertRaises(SystemExit):
            messager._handle_connect_task()

        self.assertEqual(
            queue.connect_responses,
            [
                {"task_id": 1, "success": True},
                {"task_id": 2, "success": False},
            ],
        )


class CacheMessagerV1Test(unittest.TestCase):
    def setUp(self):
        self.module = _load_cache_messager()
        envs = sys.modules["fastdeploy.utils.envs"]
        envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        _IPCSignal.instances.clear()
        ops_gpu = sys.modules["fastdeploy.model_executor.ops.gpu"]
        ops_gpu.get_output_kv_signal.sequence = []

    def tearDown(self):
        _IPCSignal.instances.clear()

    def test_consume_signals_populates_queue(self):
        cache = _make_cache_tensors(num_layers=1)
        envs = sys.modules["fastdeploy.utils.envs"]
        envs.ENABLE_V1_KVCACHE_SCHEDULER = True

        with mock.patch("threading.Thread") as thread_cls:

            def _fake_thread(*_args, **_kwargs):
                return types.SimpleNamespace(start=lambda: None)

            thread_cls.side_effect = _fake_thread
            messager = self.module.CacheMessagerV1(
                splitwise_role="prefill",
                transfer_protocol="ipc",
                pod_ip="127.0.0.1",
                engine_worker_queue_port=9003,
                local_data_parallel_id=0,
                gpu_cache_kvs=cache,
                rank=0,
                nranks=1,
                num_layers=1,
                gpu_id=0,
            )

        ops_gpu = sys.modules["fastdeploy.model_executor.ops.gpu"]
        ops_gpu.get_output_kv_signal.sequence = [
            {"tasks": -1},
            {"tasks": 1, "layer": 0, "engine": 0, "offset": 0, "current": 4},
            {"stop": True},
        ]
        messager.cache_info = {"req": {"status": "init"}}

        with self.assertRaises(SystemExit):
            messager.consume_signals()

        queued = messager.cache_prefilled_engine_ids_queue.get_nowait()
        self.assertEqual(queued, [(0, 4)])

    def test_add_cache_task_thread_updates_state(self):
        cache = _make_cache_tensors(num_layers=1)
        envs = sys.modules["fastdeploy.utils.envs"]
        envs.ENABLE_V1_KVCACHE_SCHEDULER = True

        with mock.patch("threading.Thread") as thread_cls:

            def _fake_thread(*_args, **_kwargs):
                return types.SimpleNamespace(start=lambda: None)

            thread_cls.side_effect = _fake_thread
            messager = self.module.CacheMessagerV1(
                splitwise_role="prefill",
                transfer_protocol="ipc",
                pod_ip="127.0.0.1",
                engine_worker_queue_port=9006,
                local_data_parallel_id=0,
                gpu_cache_kvs=cache,
                rank=0,
                nranks=1,
                num_layers=1,
                gpu_id=0,
            )

        messager.cache_info = {
            "req-existing": {
                "request_id": "req-existing",
                "src_block_ids": [0, 1, 2, 3],
                "dest_block_ids": [0, 1],
                "current_id": 5,
                "transfer_protocol": "ipc",
                "status": "pending",
                "rdma_ports": {0: 0},
            }
        }

        queue = messager.engine_worker_queue
        queue.cache_info_sequence = [
            [
                {
                    "request_id": "req-existing",
                    "src_block_ids": [0, 1, 2, 3],
                    "dest_block_ids": [0, 1],
                    "current_id": 5,
                    "transfer_protocol": "ipc",
                },
                {
                    "request_id": "req-new",
                    "src_block_ids": [10, 11],
                    "dest_block_ids": [12, 13],
                    "current_id": 7,
                    "transfer_protocol": "rdma",
                    "status": "pending",
                    "ip": "10.0.0.5",
                    "rdma_ports": {0: 4},
                    "device_ids": {0: 1},
                },
            ]
        ]
        queue.stop_after_cache_info = True

        with self.assertRaises(SystemExit):
            messager._add_cache_task_thread()

        self.assertEqual(queue.cache_info_barrier.wait_calls, 1)
        self.assertEqual(queue.finish_add_cache_task_barrier.wait_calls, 1)
        self.assertEqual(queue.finished_add_cache_task_req, [["req-existing"]])
        updated = messager.cache_info["req-existing"]
        self.assertEqual(updated["decode_cached_tokens"], 2 * messager.block_size)
        self.assertEqual(updated["sended_block_num"], 2)
        self.assertIn(5, messager.idx_cache_task_dict)
        self.assertIn("req-new", messager.cache_info)

    def test_prefill_layerwise_send_cache_thread_finishes_request(self):
        cache = _make_cache_tensors(num_layers=1)
        envs = sys.modules["fastdeploy.utils.envs"]
        envs.ENABLE_V1_KVCACHE_SCHEDULER = True

        with mock.patch("threading.Thread") as thread_cls:

            def _fake_thread(*_args, **_kwargs):
                return types.SimpleNamespace(start=lambda: None)

            thread_cls.side_effect = _fake_thread
            messager = self.module.CacheMessagerV1(
                splitwise_role="prefill",
                transfer_protocol="ipc",
                pod_ip="127.0.0.1",
                engine_worker_queue_port=9007,
                local_data_parallel_id=0,
                gpu_cache_kvs=cache,
                rank=0,
                nranks=1,
                num_layers=1,
                gpu_id=0,
            )

        class _QueueStub:
            def __init__(self, payloads):
                self._payloads = list(payloads)

            def get(self):
                if not self._payloads:
                    raise SystemExit("stop prefill v1")
                return self._payloads.pop(0)

        task = {
            "request_id": "req-1",
            "transfer_protocol": "ipc",
            "device_ids": {0: 0},
            "rdma_ports": {0: 0},
            "src_block_ids": [0, 1],
            "dest_block_ids": [2, 3],
            "status": "init",
            "sended_layer_id": -1,
            "sended_block_num": 0,
            "current_id": 0,
            "need_prefill_tokens": 4,
        }

        messager.idx_cache_task_dict = {0: task}
        messager.cache_info = {"req-1": task}
        messager.engine_cache_tasks[0] = {"prefilled_layer_idx": 0, "prefilled_token_num": 4}
        messager.cache_prefilled_engine_ids_queue = _QueueStub([[(0, 4)]])

        with self.assertRaises(SystemExit):
            messager.prefill_layerwise_send_cache_thread()

        queue = messager.engine_worker_queue
        self.assertEqual(queue.begin_send_cache_barrier.wait_calls, 1)
        self.assertEqual(queue.finish_send_cache_barrier.wait_calls, 1)
        self.assertEqual(queue.finished_requests, [[["req-1", "finished"]]])
        self.assertEqual(messager.messager["ipc"].sync_targets, [0])
        self.assertNotIn("req-1", messager.cache_info)


class CacheMessagerV1ConnectTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_cache_messager()
        envs = sys.modules["fastdeploy.utils.envs"]
        envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        _IPCSignal.instances.clear()
        ops_gpu = sys.modules["fastdeploy.model_executor.ops.gpu"]
        ops_gpu.get_output_kv_signal.sequence = []

    def tearDown(self):
        _IPCSignal.instances.clear()

    def test_handle_connect_task_rdma_paths(self):
        cache = _make_cache_tensors(num_layers=1)
        with mock.patch("threading.Thread") as thread_cls:

            def _fake_thread(*_args, **_kwargs):
                return types.SimpleNamespace(start=lambda: None)

            thread_cls.side_effect = _fake_thread
            messager = self.module.CacheMessagerV1(
                splitwise_role="decode",
                transfer_protocol="ipc,rdma",
                pod_ip="127.0.0.1",
                engine_worker_queue_port=9008,
                local_data_parallel_id=0,
                gpu_cache_kvs=cache,
                rank=0,
                nranks=1,
                num_layers=1,
                gpu_id=0,
            )

        rdma_manager = messager.messager["rdma"]
        rdma_manager.connect_results = [True, False]

        queue = messager.engine_worker_queue
        queue.connect_tasks = [
            {
                "task_id": 11,
                "ip": "10.0.0.1",
                "rdma_ports": {0: 5},
            },
            {
                "task_id": 12,
                "ip": "10.0.0.2",
                "rdma_ports": {0: 6},
            },
        ]
        queue.stop_after_connect_tasks = True

        with self.assertRaises(SystemExit):
            messager._handle_connect_task()

        self.assertEqual(
            queue.connect_responses,
            [
                {"task_id": 11, "success": True},
                {"task_id": 12, "success": False},
            ],
        )


class MainEntryTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_cache_messager()
        envs = sys.modules["fastdeploy.utils.envs"]
        envs.FD_ENGINE_TASK_QUEUE_WITH_SHM = False
        envs.ENABLE_V1_KVCACHE_SCHEDULER = False
        _IPCSignal.instances.clear()
        ops_gpu = sys.modules["fastdeploy.model_executor.ops.gpu"]
        ops_gpu.get_output_kv_signal.sequence = []

    def tearDown(self):
        _IPCSignal.instances.clear()

    def test_main_initializes_and_triggers_prefill(self):
        args = types.SimpleNamespace(
            splitwise_role="prefill",
            device_id=0,
            rank=0,
            num_layers=1,
            key_cache_shape="2,3,4,5",
            value_cache_shape="2,3,4,5",
            rdma_port=None,
            mp_num=1,
            pod_ip="127.0.0.1",
            cache_queue_port=9004,
            engine_worker_queue_port=9005,
            cache_dtype="bfloat16",
            speculative_config={"num_extra_cache_layer": 1, "num_gpu_block_expand_ratio": 0},
            protocol="ipc",
            engine_pid="42",
            local_data_parallel_id=0,
        )
        self.module.args = args

        with mock.patch.object(
            self.module.CacheMessager,
            "prefill_layerwise_send_cache_thread",
            side_effect=SystemExit("stop prefill"),
        ) as prefill_mock:
            with self.assertRaises(SystemExit):
                self.module.main()

        prefill_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
