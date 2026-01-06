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
import threading
import types
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Module under test: PrefixCacheManager and related cache primitives.
from fastdeploy.cache_manager.cache_data import BlockNode, CacheStatus
from fastdeploy.cache_manager.prefix_cache_manager import PrefixCacheManager
from fastdeploy.inter_communicator.ipc_signal_const import PrefixTreeStatus
from fastdeploy.utils import get_hash_str


# Metric test double used to track metric updates.
class _DummyMetric:
    """Minimal metric stub that records the last values it receives."""

    def __init__(self):
        self.values = []

    def set(self, value):
        self.values.append(value)

    def inc(self, value=1):
        self.values.append(("inc", value))

    def dec(self, value=1):
        self.values.append(("dec", value))

    def observe(self, value):
        self.values.append(("observe", value))


# Metric registry that lazily creates metrics referenced in tests.
class _DummyMainMetrics:
    """Creates metric objects on demand so code can freely reference metrics."""

    def __init__(self):
        self.metrics = {}

    def __getattr__(self, name):
        if name not in self.metrics:
            self.metrics[name] = _DummyMetric()
        return self.metrics[name]


# IPC signal stub that mirrors the real object's surface area.
class _DummyIPCSignal:
    def __init__(self, name, array, **kwargs):
        self.name = name
        self.dtype = kwargs.get("dtype", np.array(array).dtype)
        self.value = np.ones_like(array, dtype=self.dtype)


# Mock engine cache queue used to capture issued tasks.
class _DummyEngineCacheQueue:
    def __init__(self, *args, **kwargs):
        self.tasks = []

    def put_transfer_task(self, payload):
        self.tasks.append(payload)


# Test double for process objects spawned by PrefixCacheManager.
class _DummyProcess:
    def __init__(self, *args, poll_value=None, **kwargs):
        self.args = args
        self._poll_value = poll_value

    def poll(self):
        return self._poll_value


class _TrackingThread:
    """Thread double that records whether start was called."""

    instances = []

    def __init__(self, target=None, **kwargs):
        self.target = target
        self.kwargs = kwargs
        self.started = False
        _TrackingThread.instances.append(self)

    def start(self):
        self.started = True


@pytest.fixture(autouse=True)
def _clear_tracking_thread_instances():
    _TrackingThread.instances.clear()
    yield
    _TrackingThread.instances.clear()


# Immediate future used to synchronously invoke submitted functions.
class _ImmediateFuture:
    def __init__(self, fn=None, *args):
        self._result = fn(*args) if fn is not None else None

    def result(self):
        return self._result

    def done(self):
        return True


class _PendingFuture:
    def done(self):
        return False


class _CompletedFuture:
    def __init__(self, result=None):
        self.result_called = False
        self._result = result

    def done(self):
        return True

    def result(self):
        self.result_called = True
        return self._result


# Fake transfer queue returning preset payloads then raising SystemExit.
class _FakeTransferQueue:
    def __init__(self, payloads, include_none=False):
        self.payloads = payloads
        self.include_none = include_none
        self.returned_none = False

    def get_transfer_done_signal(self):
        if self.include_none and not self.returned_none:
            self.returned_none = True
            return None
        if self.payloads:
            return self.payloads.pop(0)
        raise SystemExit


def _create_manager(
    *,
    enable_prefix_caching=True,
    num_gpu_blocks=6,
    num_cpu_blocks=0,
    quant_config=None,
    splitwise_role="mixed",
):
    cache_config = SimpleNamespace(
        total_block_num=num_gpu_blocks,
        prefill_kvcache_block_num=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        bytes_per_layer_per_block=1,
        enable_prefix_caching=enable_prefix_caching,
        enable_hierarchical_cache=False,
        cache_dtype="float16",
        model_cfg=SimpleNamespace(num_hidden_layers=1),
        cache_queue_port=9000,
        cache_transfer_protocol="zmq",
        rdma_comm_ports=None,
        local_cache_queue_port=9000,
        local_rdma_comm_ports=None,
        kvcache_storage_backend=None,
        write_policy="write_through",
        swap_space=4,
    )
    model_config = SimpleNamespace(
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=1,
        _architecture="",
        dtype="float16",
        max_model_len=128,
    )
    config = SimpleNamespace(
        cache_config=cache_config,
        speculative_config=SimpleNamespace(to_json_string=lambda: "{}"),
        model_config=model_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        quant_config=quant_config,
    )
    return PrefixCacheManager(config, tensor_parallel_size=1, splitwise_role=splitwise_role)


def _make_block_node(manager, node_id, input_ids, *, block_size=2, parent=None, cache_status=CacheStatus.GPU):
    parent = parent or manager.radix_tree_root
    block_hash = get_hash_str(input_ids)
    node = BlockNode(
        node_id,
        input_ids,
        block_hash,
        parent.depth + 1,
        len(parent.children),
        block_size,
        block_hash,
        0,
        parent=parent,
        cache_status=cache_status,
    )
    parent.children[block_hash] = node
    return node


# Core behavior validation tests. These cases focus on black-box behavior
# instead of binding to internal implementation details.
class PrefixCacheManagerTest(unittest.TestCase):
    def setUp(self):
        self.metrics = _DummyMainMetrics()
        self.prefix_patch = patch(
            "fastdeploy.cache_manager.prefix_cache_manager.main_process_metrics",
            self.metrics,
        )
        self.cache_metrics_patch = patch(
            "fastdeploy.cache_manager.cache_metrics.main_process_metrics",
            self.metrics,
        )
        self.prefix_patch.start()
        self.cache_metrics_patch.start()
        self.addCleanup(self.prefix_patch.stop)
        self.addCleanup(self.cache_metrics_patch.stop)

    def test_allocate_and_recycle_gpu_blocks_update_metrics(self):
        manager = _create_manager(num_gpu_blocks=4)

        allocated = manager.allocate_gpu_blocks(2)

        self.assertEqual(allocated, [0, 1])
        self.assertAlmostEqual(manager.available_gpu_resource, 0.5)

        manager.recycle_gpu_blocks(allocated)

        self.assertEqual(len(manager.gpu_free_block_list), 4)
        self.assertEqual(self.metrics.metrics["free_gpu_block_num"].values[-1], 4)
        self.assertAlmostEqual(self.metrics.metrics["available_gpu_resource"].values[-1], 1.0)

    def test_init_uses_prefill_blocks_when_scheduler_disabled(self):
        with patch(
            "fastdeploy.cache_manager.prefix_cache_manager.envs.ENABLE_V1_KVCACHE_SCHEDULER",
            0,
        ):
            manager = _create_manager(num_gpu_blocks=3)
        self.assertEqual(manager.num_gpu_blocks, manager.cache_config.prefill_kvcache_block_num)

    def test_can_allocate_gpu_blocks_triggers_free_when_prefix_enabled(self):
        manager = _create_manager(enable_prefix_caching=True, num_gpu_blocks=2)
        manager.gpu_free_block_list.clear()

        with patch.object(manager, "free_block_ids") as mock_free:

            def _free(blocks):
                manager.gpu_free_block_list.append(0)

            mock_free.side_effect = _free
            self.assertTrue(manager.can_allocate_gpu_blocks(1))
            mock_free.assert_called_once_with(1)

    def test_check_validity_raises_when_memory_is_insufficient(self):
        manager = _create_manager(num_gpu_blocks=2)

        with self.assertRaises(Exception):
            manager._check_validity("req-1", match_gpu_blocks_num=0, expected_block_num=3)

    def test_prepare_cache_allocates_for_cpu_matches(self):
        manager = _create_manager(num_gpu_blocks=6)
        match_gpu_block_ids = [100]
        match_cpu_block_ids = [200, 201]
        swap_node_ids = [1]

        with patch.object(manager, "_prepare_cpu_cache") as mock_prepare_cpu:
            gpu_recv, gpu_extra = manager._prepare_cache(
                req_id="req-prepare",
                input_ids=[1, 2, 3, 4],
                block_size=2,
                expected_block_num=4,
                match_gpu_block_ids=match_gpu_block_ids,
                match_cpu_block_ids=match_cpu_block_ids,
                match_node_ids=swap_node_ids,
            )

        self.assertEqual(len(gpu_recv), len(match_cpu_block_ids))
        self.assertEqual(len(gpu_extra), 1)
        mock_prepare_cpu.assert_called_once()

    def test_request_block_ids_combines_matched_and_unique_blocks(self):
        manager = _create_manager(num_gpu_blocks=6)
        block_size = 2
        task = SimpleNamespace(prompt_token_ids=[1, 2, 3, 4], request_id="req-2")
        match_node = BlockNode(
            node_id=999,
            input_ids=task.prompt_token_ids,
            input_hash_value=0,
            depth=1,
            block_id=10,
            token_num=block_size,
            hash_value=123,
            last_used_time=0,
            parent=manager.radix_tree_root,
        )

        with (
            patch.object(
                manager,
                "match_block",
                return_value=([5], [7], [8], match_node, 4, 2),
            ),
            patch.object(
                manager,
                "_prepare_cache",
                return_value=([9], [11]),
            ),
            patch.object(
                manager,
                "build_path",
                return_value=match_node,
            ),
        ):
            common, unique, hit_info = manager.request_block_ids(task, block_size, dec_token_num=2)

        self.assertEqual(common, [5, 9])
        self.assertEqual(unique, [11])
        self.assertIn("req-2", manager.req_leaf_map)
        self.assertIs(manager.req_leaf_map["req-2"], match_node)
        self.assertEqual(hit_info["gpu_cache_blocks"], 2)
        self.assertEqual(hit_info["cpu_cache_blocks"], 1)
        self.assertEqual(manager.metrics.hit_req_count, 1)

    def test_request_match_blocks_raises_when_gpu_unavailable(self):
        manager = _create_manager()
        task = SimpleNamespace(prompt_token_ids=[1, 2], output_token_ids=[], request_id="fail")
        with (
            patch.object(
                manager,
                "mm_match_block",
                return_value=([], [9], [10], manager.radix_tree_root, 0, 2),
            ),
            patch.object(manager, "can_allocate_gpu_blocks", return_value=False),
        ):
            with self.assertRaises(Exception):
                manager.request_match_blocks(task, block_size=2)

    def test_request_match_blocks_with_numpy_prompt_and_metric_reset(self):
        manager = _create_manager()
        manager.metrics.reset_metrics = MagicMock()
        manager.metrics.req_count = 9999

        task = SimpleNamespace(
            prompt_token_ids=np.array([1, 2, 3]),
            output_token_ids=[4],
            request_id="np",
        )
        with patch.object(
            manager,
            "mm_match_block",
            return_value=([], [], [], manager.radix_tree_root, 0, 0),
        ):
            common, matched_tokens, hit_info = manager.request_match_blocks(task, block_size=2)

        self.assertEqual(common, [])
        self.assertEqual(matched_tokens, 0)
        self.assertEqual(hit_info["gpu_match_token_num"], 0)
        manager.metrics.reset_metrics.assert_called_once()

    def test_get_required_block_num_rounds_up(self):
        manager = _create_manager()
        self.assertEqual(manager.get_required_block_num(0, 4), 0)
        self.assertEqual(manager.get_required_block_num(7, 4), 2)
        self.assertEqual(manager.get_required_block_num(8, 4), 2)

    def test_launch_cache_manager_initializes_processes(self):
        manager = _create_manager()
        manager.cache_config.enable_hierarchical_cache = False

        with (
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.IPCSignal",
                side_effect=_DummyIPCSignal,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.EngineCacheQueue",
                _DummyEngineCacheQueue,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.get_all_visible_devices",
                return_value="CUDA_VISIBLE_DEVICES=0",
                create=True,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.subprocess.Popen",
                _DummyProcess,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.threading.Thread",
                _TrackingThread,
            ),
            patch.object(
                manager,
                "_get_kv_cache_shape",
                return_value=([1], [1]),
            ),
        ):
            processes = manager.launch_cache_manager(
                cache_config=manager.cache_config,
                tensor_parallel_size=1,
                device_ids=[0],
                pod_ip="127.0.0.1",
                engine_worker_queue_port=8000,
                ipc_suffix="pid",
                create_cache_tensor=True,
            )

        self.assertEqual(len(processes), 1)

    def test_launch_cache_manager_invokes_splitwise_messager(self):
        manager = _create_manager(splitwise_role="decode")
        manager.cache_config.enable_hierarchical_cache = False
        with (
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.IPCSignal",
                side_effect=_DummyIPCSignal,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.EngineCacheQueue",
                _DummyEngineCacheQueue,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.get_all_visible_devices",
                return_value="CUDA_VISIBLE_DEVICES=0",
                create=True,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.subprocess.Popen",
                _DummyProcess,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.threading.Thread",
                _TrackingThread,
            ),
            patch.object(
                manager,
                "_get_kv_cache_shape",
                return_value=([1], [1]),
            ),
            patch.object(
                manager,
                "launch_cache_messager",
                return_value=[_DummyProcess()],
            ) as mock_launch,
        ):
            manager.launch_cache_manager(
                cache_config=manager.cache_config,
                tensor_parallel_size=1,
                device_ids=[0],
                pod_ip="127.0.0.1",
                engine_worker_queue_port=8000,
                ipc_suffix="pid",
                create_cache_tensor=False,
            )

        mock_launch.assert_called_once()

    def test_launch_cache_manager_errors_when_messager_fails(self):
        manager = _create_manager(splitwise_role="decode")
        manager.cache_config.enable_hierarchical_cache = False
        with (
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.IPCSignal",
                side_effect=_DummyIPCSignal,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.EngineCacheQueue",
                _DummyEngineCacheQueue,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.subprocess.Popen",
                _DummyProcess,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.threading.Thread",
                _TrackingThread,
            ),
            patch.object(manager, "_get_kv_cache_shape", return_value=([1], [1])),
            patch.object(manager, "launch_cache_messager", return_value=None),
        ):
            with self.assertRaises(RuntimeError):
                manager.launch_cache_manager(
                    cache_config=manager.cache_config,
                    tensor_parallel_size=1,
                    device_ids=[0],
                    pod_ip="127.0.0.1",
                    engine_worker_queue_port=8000,
                    ipc_suffix="pid",
                    create_cache_tensor=False,
                )

    def test_launch_cache_manager_waits_for_signals_with_hierarchical_cache(self):
        manager = _create_manager(num_cpu_blocks=2)
        manager.cache_config.enable_hierarchical_cache = True

        created_signals = {}

        def _signal_factory(name=None, array=None, **kwargs):
            dtype = kwargs.get("dtype", np.array(array).dtype)
            signal = SimpleNamespace(name=name, value=np.array(array, copy=True, dtype=dtype))
            signal.dtype = dtype
            created_signals[name] = signal
            return signal

        def _fake_sleep(_):
            ready_signal = created_signals.get("cache_ready_signal")
            if ready_signal is not None and np.sum(ready_signal.value) == 0:
                ready_signal.value[:] = 1
                return
            swap_signal = created_signals.get("swap_space_ready_signal")
            if swap_signal is not None and np.sum(swap_signal.value) == 0:
                swap_signal.value[:] = 1
                return

        with (
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.IPCSignal",
                side_effect=_signal_factory,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.EngineCacheQueue",
                _DummyEngineCacheQueue,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.get_all_visible_devices",
                return_value="CUDA_VISIBLE_DEVICES=0",
                create=True,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.subprocess.Popen",
                partial(_DummyProcess, poll_value=1),
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.threading.Thread",
                _TrackingThread,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.time.sleep",
                side_effect=_fake_sleep,
            ),
            patch.object(manager, "_get_kv_cache_shape", return_value=([1], [1])),
        ):
            processes = manager.launch_cache_manager(
                cache_config=manager.cache_config,
                tensor_parallel_size=1,
                device_ids=[0],
                pod_ip="127.0.0.1",
                engine_worker_queue_port=8000,
                ipc_suffix="pid",
                create_cache_tensor=False,
            )

        self.assertEqual(len(processes), 1)
        started_targets = {thread.target for thread in _TrackingThread.instances if thread.started}
        self.assertIn(manager.recv_data_transfer_result, started_targets)
        self.assertIn(manager.clear_prefix_cache, started_targets)

    def test_launch_cache_messager_waits_for_ready_signal(self):
        manager = _create_manager()
        ready_snapshots = {}

        def _signal_factory(name=None, array=None, **kwargs):
            dtype = kwargs.get("dtype", np.array(array).dtype)
            signal = SimpleNamespace(name=name, value=np.array(array, copy=True, dtype=dtype))
            signal.dtype = dtype
            if name == "cache_ready_signal":
                ready_snapshots["initial"] = signal.value.copy()
            return signal

        def _fake_sleep(_):
            signal = manager.cache_ready_signal
            # Simulate messager process marking readiness.
            signal.value[:] = 1
            ready_snapshots["after_ready"] = signal.value.copy()

        with (
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.IPCSignal",
                side_effect=_signal_factory,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.get_all_visible_devices",
                return_value="CUDA_VISIBLE_DEVICES=0",
                create=True,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.subprocess.Popen",
                _DummyProcess,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.time.sleep",
                side_effect=_fake_sleep,
            ),
        ):
            processes = manager.launch_cache_messager(
                cache_config=manager.cache_config,
                tensor_parallel_size=1,
                device_ids=[0],
                key_cache_shape="1",
                value_cache_shape="1",
                pod_ip="127.0.0.1",
                engine_worker_queue_port=8000,
                ipc_suffix="pid",
            )

        self.assertEqual(len(processes), 1)
        self.assertTrue(np.all(ready_snapshots["initial"] == 0))
        self.assertTrue(np.all(ready_snapshots["after_ready"] == 1))
        self.assertTrue(np.all(manager.cache_ready_signal.value == 1))

    def test_launch_cache_messager_returns_none_when_process_fails(self):
        manager = _create_manager()

        with (
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.IPCSignal",
                side_effect=_DummyIPCSignal,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.get_all_visible_devices",
                return_value="CUDA_VISIBLE_DEVICES=0",
                create=True,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.subprocess.Popen",
                partial(_DummyProcess, poll_value=2),
            ),
        ):
            processes = manager.launch_cache_messager(
                cache_config=manager.cache_config,
                tensor_parallel_size=1,
                device_ids=[0],
                key_cache_shape="1",
                value_cache_shape="1",
                pod_ip="127.0.0.1",
                engine_worker_queue_port=8000,
                ipc_suffix="pid",
            )

        self.assertIsNone(processes)

    def test_launch_cache_manager_formats_value_cache_shape(self):
        manager = _create_manager()

        captured = {}

        class _CmdProcess:
            def __init__(self, cmd):
                captured["cmd"] = cmd

            def poll(self):
                return None

        with (
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.IPCSignal",
                side_effect=_DummyIPCSignal,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.EngineCacheQueue",
                _DummyEngineCacheQueue,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.get_all_visible_devices",
                return_value="CUDA_VISIBLE_DEVICES=0",
                create=True,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.subprocess.Popen",
                side_effect=lambda cmd, **_: _CmdProcess(cmd),
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.threading.Thread",
                _TrackingThread,
            ),
            patch.object(
                manager,
                "_get_kv_cache_shape",
                return_value=([1], [2, 3]),
            ),
        ):
            manager.launch_cache_manager(
                cache_config=manager.cache_config,
                tensor_parallel_size=1,
                device_ids=[0],
                pod_ip="127.0.0.1",
                engine_worker_queue_port=8000,
                ipc_suffix="pid",
                create_cache_tensor=True,
            )

        self.assertIn("--value_cache_shape 2,3", captured["cmd"])

    def test_update_cache_config_adjusts_gpu_pool_based_on_scheduler_flag(self):
        manager = _create_manager()
        cache_config = SimpleNamespace(
            total_block_num=5,
            prefill_kvcache_block_num=3,
            model_cfg=SimpleNamespace(num_hidden_layers=1),
            cache_queue_port=9000,
            rdma_comm_ports=None,
            local_cache_queue_port=9000,
            local_rdma_comm_ports=None,
        )

        with patch(
            "fastdeploy.cache_manager.prefix_cache_manager.envs.ENABLE_V1_KVCACHE_SCHEDULER",
            1,
        ):
            manager.update_cache_config(cache_config)
            self.assertEqual(manager.num_gpu_blocks, cache_config.total_block_num)
            self.assertEqual(len(manager.gpu_free_block_list), cache_config.total_block_num)

        with patch(
            "fastdeploy.cache_manager.prefix_cache_manager.envs.ENABLE_V1_KVCACHE_SCHEDULER",
            0,
        ):
            manager.update_cache_config(cache_config)
            self.assertEqual(manager.num_gpu_blocks, cache_config.prefill_kvcache_block_num)
            self.assertEqual(len(manager.gpu_free_block_list), cache_config.prefill_kvcache_block_num)

    def test_allocate_and_recycle_cpu_blocks(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=3)
        allocated = manager.allocate_cpu_blocks(2)
        self.assertEqual(allocated, [0, 1])
        self.assertEqual(len(manager.cpu_free_block_list), 1)

        manager.recycle_cpu_blocks(allocated)
        self.assertEqual(len(manager.cpu_free_block_list), 3)

    def test_issue_swap_task_sync_path(self):
        manager = _create_manager()
        manager.cache_task_queue = _DummyEngineCacheQueue()

        class _NoWaitEvent:
            instances = []

            def __init__(self, *_, **__):
                self.wait_called = False
                _NoWaitEvent.instances.append(self)

            def wait(self):
                self.wait_called = True

        with patch("fastdeploy.cache_manager.prefix_cache_manager.Event", _NoWaitEvent):
            manager.issue_swap_task(
                transfer_task_id="sync-task",
                swap_node_ids=[1],
                gpu_block_ids=[2],
                cpu_block_ids=[3],
                event_type=CacheStatus.SWAP2GPU,
                is_sync=True,
            )

        self.assertEqual(len(_NoWaitEvent.instances), 1)
        self.assertTrue(_NoWaitEvent.instances[0].wait_called)
        self.assertNotIn("sync-task", manager.task_swapping_event)
        self.assertEqual(len(manager.cache_task_queue.tasks), 1)

    def test_prepare_cpu_cache_dispatches_swap(self):
        manager = _create_manager()
        issued = {}

        def _capture_issue(task_id, swap_node_ids, gpu_ids, cpu_ids, event_type, is_sync):
            issued["args"] = (task_id, swap_node_ids, gpu_ids, cpu_ids, event_type, is_sync)

        manager.issue_swap_task = _capture_issue
        manager._prepare_cpu_cache(
            req_id="req-id",
            swap_node_ids=[10],
            gpu_recv_block_ids=[1, 2],
            cpu_recv_block_ids=[3, 4],
            match_cpu_block_ids=[3, 4],
        )

        self.assertIn("args", issued)
        task_id, swap_nodes, gpu_ids, cpu_ids, event_type, is_sync = issued["args"]
        self.assertEqual(task_id, "req-id")
        self.assertEqual(swap_nodes, [10])
        self.assertEqual(gpu_ids, [1, 2])
        self.assertEqual(cpu_ids, [3, 4])
        self.assertEqual(event_type, CacheStatus.SWAP2GPU)
        self.assertTrue(is_sync)

    def test_update_cache_blocks_refreshes_mappings(self):
        manager = _create_manager(num_gpu_blocks=2)
        req_id = "update-req"
        last_node = BlockNode(1, [], 0, 1, 0, 2, 0, 0, parent=manager.radix_tree_root)
        manager.req_to_radix_tree_info[req_id] = (last_node, 0)
        manager.leaf_req_map[last_node].add(req_id)

        new_leaf = BlockNode(2, [], 0, 1, 0, 2, 1, 0, parent=last_node)
        with patch.object(manager, "mm_build_path", return_value=new_leaf):
            task = SimpleNamespace(request_id=req_id, output_token_ids=[1, 2], block_tables=[0])
            manager.update_cache_blocks(task, block_size=2, num_computed_tokens=4)

        self.assertIs(manager.req_leaf_map[req_id], new_leaf)
        self.assertIn(req_id, manager.leaf_req_map[new_leaf])
        self.assertEqual(task.num_cached_blocks, 2)

    def test_is_chunked_mm_input_detects_overlap(self):
        manager = _create_manager()
        mm_inputs = {
            "mm_positions": [SimpleNamespace(offset=2, length=3)],
            "mm_hashes": ["img"],
        }

        chunked, idx = manager.is_chunked_mm_input(mm_inputs, matched_token_num=3)
        self.assertTrue(chunked)
        self.assertEqual(idx, 0)

    def test_issue_and_sync_swap_tasks(self):
        manager = _create_manager()
        manager.cache_task_queue = _DummyEngineCacheQueue()
        manager.issue_swap_task(
            transfer_task_id="task-1",
            swap_node_ids=[1],
            gpu_block_ids=[2],
            cpu_block_ids=[3],
            event_type=CacheStatus.SWAP2GPU,
            is_sync=False,
        )
        self.assertEqual(len(manager.cache_task_queue.tasks), 1)

        manager.task_swapping_event["sync-task"] = threading.Event()
        manager.task_swapping_event["sync-task"].set()
        manager.sync_swap_task("sync-task")

    def test_release_block_ids_recycles_unfilled_blocks_for_root(self):
        manager = _create_manager()
        req_id = "root-release"
        manager.req_leaf_map[req_id] = manager.radix_tree_root
        manager.unfilled_req_block_map[req_id] = [5]

        manager.release_block_ids(SimpleNamespace(request_id=req_id))
        self.assertNotIn(req_id, manager.unfilled_req_block_map)

    def test_free_nodes_directly_handles_gpu_leafs(self):
        manager = _create_manager()
        node = _make_block_node(manager, node_id=200, input_ids=[7, 8])
        node.shared_count = 0
        node.reverved_dec_block_ids = [9]
        manager.node_map[node.node_id] = node
        manager.gpu_lru_leaf_heap.append(node)
        manager.gpu_lru_leaf_set.add(node)

        recycled = []

        def _record(block_ids):
            recycled.append(block_ids)

        manager.recycle_gpu_blocks = _record

        manager.free_nodes_directly(node)

        self.assertTrue(any(9 in entry if isinstance(entry, list) else entry == 9 for entry in recycled))

    def test_match_block_moves_cpu_nodes_to_swap(self):
        manager = _create_manager(num_gpu_blocks=4)
        block_size = 2
        root = manager.radix_tree_root
        gpu_hash = get_hash_str([1, 2])
        gpu_node = BlockNode(1, [], 0, 1, 0, block_size, gpu_hash, 0, parent=root)
        root.children[gpu_hash] = gpu_node
        cpu_hash = get_hash_str([3, 4], extra_keys=[gpu_hash])
        cpu_node = BlockNode(2, [], 0, 2, 1, block_size, cpu_hash, 0, parent=gpu_node, cache_status=CacheStatus.CPU)
        gpu_node.children[cpu_hash] = cpu_node
        manager.gpu_lru_leaf_set.add(gpu_node)
        manager.gpu_lru_leaf_heap.append(gpu_node)

        result = manager.match_block("req", [1, 2, 3, 4], block_size)
        match_gpu, match_cpu, swap_node_ids, last_node, *_ = result

        self.assertEqual(match_gpu, [0])
        self.assertEqual(match_cpu, [1])
        self.assertEqual(swap_node_ids, [cpu_node.node_id])
        self.assertEqual(last_node, cpu_node)
        self.assertEqual(cpu_node.cache_status, CacheStatus.SWAP2GPU)

    def test_build_path_extends_tree(self):
        manager = _create_manager(num_gpu_blocks=4)
        block_size = 2
        req_id = "req"
        gpu_node = BlockNode(1, [1, 2], 0, 1, 0, block_size, 111, 0, parent=manager.radix_tree_root)
        manager.radix_tree_root.children[111] = gpu_node
        leaf = manager.build_path(
            req_id=req_id,
            current_time=0.0,
            input_ids=[1, 2, 3, 4],
            left_input_ids=[3, 4],
            gpu_block_ids=[0],
            block_size=block_size,
            last_node=gpu_node,
            reverved_dec_block_num=0,
        )
        self.assertEqual(leaf.block_id, 0)
        self.assertEqual(leaf.parent, gpu_node)

    def test_free_block_ids_async_recycles_gpu_nodes(self):
        manager = _create_manager(num_gpu_blocks=4)
        node_hash = get_hash_str([1, 2])
        node = BlockNode(10, [1, 2], node_hash, 1, 0, 2, node_hash, 0, parent=manager.radix_tree_root)
        node.shared_count = 0
        manager.radix_tree_root.children[node_hash] = node
        manager.gpu_lru_leaf_heap.append(node)
        manager.gpu_lru_leaf_set.add(node)

        manager.free_block_ids_async(1)

        self.assertIn(0, manager.gpu_free_block_list)

    def test_free_block_ids_async_swaps_to_cpu(self):
        manager = _create_manager(num_gpu_blocks=4, num_cpu_blocks=2)
        manager.cache_config.enable_hierarchical_cache = True
        manager.cache_task_queue = _DummyEngineCacheQueue()
        manager.free_cpu_executor_pool = types.SimpleNamespace(submit=_ImmediateFuture)
        manager.free_gpu_executor_pool = types.SimpleNamespace(submit=_ImmediateFuture)
        issued = {}

        def _fake_issue(task_id, swap_node_ids, gpu_ids, cpu_ids, event_type, is_sync):
            issued["payload"] = (swap_node_ids, gpu_ids, cpu_ids, event_type, is_sync)

        manager.issue_swap_task = _fake_issue

        node_hash = get_hash_str([3, 4])
        node = BlockNode(11, [3, 4], node_hash, 1, 1, 2, node_hash, 0, parent=manager.radix_tree_root)
        node.shared_count = 0
        manager.radix_tree_root.children[node_hash] = node
        manager.gpu_lru_leaf_heap.append(node)
        manager.gpu_lru_leaf_set.add(node)

        manager.free_block_ids_async(1)

        self.assertIn("payload", issued)

    def test_mm_match_block_handles_multimodal_inputs(self):
        manager = _create_manager(num_gpu_blocks=4)
        block_size = 2
        manager.cache_config.disable_chunked_mm_input = False
        input_ids = [1, 2, 3, 4]
        hash_input = get_hash_str(input_ids)
        hash_first = get_hash_str([1, 2])
        hash_second = get_hash_str([3, 4], [hash_first, "img"])

        node1 = BlockNode(30, input_ids, hash_input, 1, 0, block_size, hash_first, 0, parent=manager.radix_tree_root)
        manager.radix_tree_root.children[hash_first] = node1
        node2 = BlockNode(
            31,
            input_ids,
            hash_input,
            2,
            1,
            block_size,
            hash_second,
            0,
            parent=node1,
            cache_status=CacheStatus.CPU,
        )
        node1.children[hash_second] = node2

        request = SimpleNamespace(
            prompt_token_ids=input_ids,
            output_token_ids=[],
            request_id="mm-req",
            multimodal_inputs={
                "mm_positions": [SimpleNamespace(offset=2, length=2)],
                "mm_hashes": ["img"],
            },
            num_total_tokens=4,
        )

        match_gpu, match_cpu, swap_nodes, last_node, gpu_tokens, cpu_tokens = manager.mm_match_block(
            request, block_size
        )

        self.assertEqual(match_gpu, [0])
        self.assertEqual(match_cpu, [1])
        self.assertEqual(swap_nodes, [node2.node_id])
        self.assertEqual(last_node, node2)
        self.assertEqual(gpu_tokens, 2)
        self.assertEqual(cpu_tokens, 2)

    def test_request_match_blocks_updates_metrics(self):
        manager = _create_manager(num_gpu_blocks=6)
        manager.cache_config.disable_chunked_mm_input = False
        block_size = 2
        input_ids = [1, 2, 3, 4]
        hash_input = get_hash_str(input_ids)
        hash_first = get_hash_str([1, 2])
        hash_second = get_hash_str([3, 4], [hash_first, "img"])
        node1 = BlockNode(40, input_ids, hash_input, 1, 0, block_size, hash_first, 0, parent=manager.radix_tree_root)
        node2 = BlockNode(
            41,
            input_ids,
            hash_input,
            2,
            1,
            block_size,
            hash_second,
            0,
            parent=node1,
            cache_status=CacheStatus.CPU,
        )
        manager.radix_tree_root.children[hash_first] = node1
        node1.children[hash_second] = node2
        task = SimpleNamespace(
            prompt_token_ids=input_ids,
            output_token_ids=[],
            request_id="match-req",
            multimodal_inputs={
                "mm_positions": [SimpleNamespace(offset=2, length=2)],
                "mm_hashes": ["img"],
            },
            num_total_tokens=4,
        )

        manager.cache_task_queue = _DummyEngineCacheQueue()
        with patch.object(manager, "_prepare_cpu_cache") as mock_prepare_cpu:
            common_blocks, matched_tokens, hit_info = manager.request_match_blocks(task, block_size)

        self.assertEqual(common_blocks[0], 0)
        self.assertGreaterEqual(matched_tokens, 4)
        mock_prepare_cpu.assert_called()
        self.assertEqual(hit_info["gpu_match_token_num"], block_size)
        self.assertEqual(hit_info["cpu_match_token_num"], block_size)

    def test_release_block_ids_cleans_request_state(self):
        manager = _create_manager(num_gpu_blocks=4)
        node = BlockNode(50, [1, 2], 0, 1, 0, 2, get_hash_str([1, 2]), 0, parent=manager.radix_tree_root)
        node.cache_status = CacheStatus.GPU
        manager.radix_tree_root.children[node.hash_value] = node
        req_id = "release-req"
        manager.req_leaf_map[req_id] = node
        manager.leaf_req_map[node].add(req_id)
        node.req_id_set.add(req_id)
        node.shared_count = 1
        task = SimpleNamespace(request_id=req_id)

        manager.release_block_ids(task)

        self.assertNotIn(req_id, manager.req_leaf_map)

    def test_free_cpu_block_ids_eviction(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=2)
        cpu_node = BlockNode(60, [3, 4], 0, 1, 0, 2, get_hash_str([3, 4]), 0, parent=manager.radix_tree_root)
        cpu_node.cache_status = CacheStatus.CPU
        manager.cpu_lru_leaf_heap.append(cpu_node)
        manager.cpu_lru_leaf_set.add(cpu_node)
        freed = manager.free_cpu_block_ids(1)
        self.assertGreaterEqual(freed, 0)

    def test_free_nodes_directly_recovers_chain(self):
        manager = _create_manager(num_gpu_blocks=4)
        parent = BlockNode(70, [1, 2], 0, 1, 0, 2, get_hash_str([1, 2]), 0, parent=manager.radix_tree_root)
        child_hash = get_hash_str([3, 4])
        child = BlockNode(71, [1, 2, 3, 4], 0, 2, 1, 2, child_hash, 0, parent=parent)
        parent.children[child_hash] = child
        parent.shared_count = 0
        child.shared_count = 0
        manager.free_nodes_directly(child)
        self.assertIn(parent.block_id, manager.gpu_free_block_list)

    def test_free_block_ids_async_returns_for_pending_future(self):
        manager = _create_manager()
        manager.gpu_free_task_future = _PendingFuture()

        manager.free_block_ids_async(need_block_num=1)

        self.assertIsInstance(manager.gpu_free_task_future, _PendingFuture)

    def test_free_block_ids_async_consumes_finished_future(self):
        manager = _create_manager()
        finished = _CompletedFuture(result="done")
        manager.gpu_free_task_future = finished

        manager.free_block_ids_async(need_block_num=1)

        self.assertIsNone(manager.gpu_free_task_future)
        self.assertTrue(finished.result_called)

    def test_mm_match_block_reverts_chunked_inputs(self):
        manager = _create_manager(num_gpu_blocks=4)
        manager.cache_config.disable_chunked_mm_input = True
        block_size = 2
        input_ids = [1, 2, 3, 4]
        hash_input = get_hash_str(input_ids)
        hash_first = get_hash_str([1, 2])
        hash_second = get_hash_str([3, 4], ["img"])
        node1 = BlockNode(80, input_ids, hash_input, 1, 0, block_size, hash_first, 0, parent=manager.radix_tree_root)
        node2 = BlockNode(81, input_ids, hash_input, 2, 1, block_size, hash_second, 0, parent=node1)
        manager.radix_tree_root.children[hash_first] = node1
        node1.children[hash_second] = node2

        request = SimpleNamespace(
            prompt_token_ids=input_ids,
            output_token_ids=[],
            request_id="chunk-req",
            multimodal_inputs={
                "mm_positions": [SimpleNamespace(offset=1, length=3)],
                "mm_hashes": ["img"],
            },
            num_total_tokens=4,
        )

        match_gpu, *_ = manager.mm_match_block(request, block_size)
        self.assertEqual(match_gpu, [])

    def test_mm_build_path_creates_new_nodes(self):
        manager = _create_manager(num_gpu_blocks=6)
        request = SimpleNamespace(
            prompt_token_ids=[1, 2],
            output_token_ids=[3, 4],
            block_tables=[0, 1, 2],
            request_id="mm-build",
            multimodal_inputs={"mm_positions": [], "mm_hashes": []},
        )
        leaf = manager.mm_build_path(
            request=request,
            num_computed_tokens=4,
            block_size=2,
            last_node=manager.radix_tree_root,
            num_cached_tokens=0,
        )
        self.assertNotEqual(leaf, manager.radix_tree_root)

    def test_handle_swap_result_updates_status(self):
        manager = _create_manager(num_gpu_blocks=4, num_cpu_blocks=2)
        node = BlockNode(90, [1], 0, 1, 0, 1, get_hash_str([1]), 0, parent=manager.radix_tree_root)
        node.cache_status = CacheStatus.SWAP2CPU
        manager.node_map[node.node_id] = node
        manager._handle_swap_result(node.node_id, 2, 3, CacheStatus.SWAP2CPU)
        self.assertEqual(node.cache_status, CacheStatus.CPU)
        manager._handle_swap_result(node.node_id, 4, 5, CacheStatus.SWAP2GPU)
        self.assertEqual(node.cache_status, CacheStatus.GPU)
        node.cache_status = CacheStatus.GPU
        manager._handle_swap_result(node.node_id, 6, 7, CacheStatus.SWAP2CPU)

    def test_reset_clears_internal_state(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=1)
        node = BlockNode(100, [1], 0, 1, 0, 1, get_hash_str([1]), 0, parent=manager.radix_tree_root)
        manager.node_map[node.node_id] = node
        manager.task_swapping_event["evt"] = threading.Event()
        manager.task_swapping_event["evt"].set()
        manager.gpu_free_task_future = _ImmediateFuture(lambda: None)
        manager.reset()
        self.assertEqual(len(manager.node_map), 0)

    def test_recv_data_transfer_result_processes_queue(self):
        manager = _create_manager(num_gpu_blocks=4, num_cpu_blocks=1)
        node = BlockNode(110, [1], 0, 1, 0, 1, get_hash_str([1]), 0, parent=manager.radix_tree_root)
        manager.node_map[node.node_id] = node
        payload = [(CacheStatus.SWAP2GPU, "task", [node.node_id], [2], [3])]
        manager.cache_task_queue = _FakeTransferQueue(payload, include_none=True)
        manager.task_swapping_event["task"] = threading.Event()
        with self.assertRaises(SystemExit):
            manager.recv_data_transfer_result()
        self.assertTrue(manager.task_swapping_event["task"].is_set())

    def test_clear_prefix_cache_resets_on_signal(self):
        manager = _create_manager()
        manager.prefix_tree_status_signal = SimpleNamespace(
            value=np.array([PrefixTreeStatus.CLEARING], dtype=np.int32)
        )
        manager.reset = MagicMock()
        with patch("fastdeploy.cache_manager.prefix_cache_manager.time.sleep", side_effect=SystemExit):
            with self.assertRaises(SystemExit):
                manager.clear_prefix_cache()
        manager.reset.assert_called_once()
        manager.prefix_tree_status_signal.value[0] = PrefixTreeStatus.UPDATING
        with patch("fastdeploy.cache_manager.prefix_cache_manager.time.sleep", side_effect=SystemExit):
            with self.assertRaises(SystemExit):
                manager.clear_prefix_cache()

    @unittest.skip("Skip TestRevertMatchBlocks")
    def test_revert_match_blocks_adjusts_lists(self):
        manager = _create_manager()
        request = SimpleNamespace(
            request_id="revert",
            multimodal_inputs={"mm_positions": [SimpleNamespace(offset=2, length=2)]},
        )
        node = BlockNode(120, [1, 2], 0, 1, 0, 2, get_hash_str([1, 2]), 0, parent=manager.radix_tree_root)
        matche_nodes = [node]
        match_gpu = [0]
        match_node_ids = [node.node_id]
        swap_nodes = [node.block_id]
        gpu_tokens, cpu_tokens, current = manager._revert_match_blocks(
            request=request,
            matched_token_num=4,
            block_size=2,
            chunk_idx=0,
            match_node_ids=match_node_ids,
            matche_nodes=matche_nodes,
            match_gpu_block_ids=match_gpu,
            match_cpu_block_ids=[],
            gpu_match_token_num=4,
            cpu_match_token_num=0,
            swap_node_ids=swap_nodes,
        )
        self.assertEqual(gpu_tokens, 2)
        self.assertEqual(current, manager.radix_tree_root)


# Coverage-oriented tests. These are used to lightly exercise specific
# implementation details without constraining core behavior.
class TestPrefixCacheManagerCoverage(unittest.TestCase):
    def test_get_kv_cache_shape_returns_shape_from_backend(self):
        quant = SimpleNamespace(kv_cache_quant_type="int8")
        manager = _create_manager(quant_config=quant)

        class _Backend:
            def __call__(self, *args, **kwargs):
                return self

            def get_kv_cache_shape(self, max_num_blocks, kv_cache_quant_type=None):
                return ([max_num_blocks, 2], [3, kv_cache_quant_type])

        backend = _Backend()
        attention_module = types.ModuleType("fastdeploy.model_executor.layers.attention")
        attention_module.get_attention_backend = lambda: backend

        with patch.dict(
            sys.modules,
            {"fastdeploy.model_executor.layers.attention": attention_module},
        ):
            key_shape, value_shape = manager._get_kv_cache_shape(5)

        self.assertIsInstance(key_shape, list)
        self.assertIsInstance(value_shape, list)
        self.assertEqual(key_shape, [5, 2])
        self.assertEqual(value_shape, [3, "int8"])
        self.assertTrue(all(dim >= 0 for dim in key_shape))
        self.assertTrue(all(dim is not None for dim in value_shape))


if __name__ == "__main__":
    unittest.main()
