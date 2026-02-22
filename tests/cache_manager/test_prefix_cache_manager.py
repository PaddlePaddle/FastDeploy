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
from contextlib import ExitStack, contextmanager
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import paddle
import pytest

if not hasattr(paddle, "compat"):
    paddle.compat = SimpleNamespace(enable_torch_proxy=lambda **_: None)

# Module under test: PrefixCacheManager and related cache primitives.
from fastdeploy.cache_manager.cache_data import BlockNode, CacheStatus
from fastdeploy.cache_manager.cache_tasks import ReadStorageTask, WriteStorageTask
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

    def result_queue_empty(self):
        return True


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
        bytes_per_token_per_layer=2048,
        swap_space=4,
    )
    model_config = SimpleNamespace(
        model="test_model",
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
    manager = PrefixCacheManager(config, tensor_parallel_size=1, splitwise_role=splitwise_role)
    # Newer manager code initializes these IPC attributes in launch_cache_manager,
    # while many unit tests exercise methods directly on a fresh manager.
    manager.cache_task_inflight_signal = SimpleNamespace(value=np.zeros([1], dtype=np.int32))
    manager.prefix_tree_status_signal = SimpleNamespace(value=np.array([PrefixTreeStatus.NORMAL], dtype=np.int32))
    manager.cache_task_queue = _DummyEngineCacheQueue()
    return manager


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


@contextmanager
def _launch_cache_manager_patches(manager, *, popen, kv_shape=([1], [1]), signal=_DummyIPCSignal):
    with ExitStack() as stack:
        stack.enter_context(patch("fastdeploy.cache_manager.prefix_cache_manager.IPCSignal", side_effect=signal))
        stack.enter_context(
            patch("fastdeploy.cache_manager.prefix_cache_manager.EngineCacheQueue", _DummyEngineCacheQueue)
        )
        stack.enter_context(
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.get_all_visible_devices",
                return_value="CUDA_VISIBLE_DEVICES=0",
                create=True,
            )
        )
        stack.enter_context(patch("fastdeploy.cache_manager.prefix_cache_manager.subprocess.Popen", popen))
        stack.enter_context(patch("fastdeploy.cache_manager.prefix_cache_manager.threading.Thread", _TrackingThread))
        stack.enter_context(patch.object(manager, "_get_kv_cache_shape", return_value=kv_shape))
        yield stack


@contextmanager
def _launch_cache_messager_patches(*, popen, signal=_DummyIPCSignal, sleep_side_effect=None):
    with ExitStack() as stack:
        stack.enter_context(patch("fastdeploy.cache_manager.prefix_cache_manager.IPCSignal", side_effect=signal))
        stack.enter_context(
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.get_all_visible_devices",
                return_value="CUDA_VISIBLE_DEVICES=0",
                create=True,
            )
        )
        stack.enter_context(patch("fastdeploy.cache_manager.prefix_cache_manager.subprocess.Popen", popen))
        if sleep_side_effect is not None:
            stack.enter_context(
                patch("fastdeploy.cache_manager.prefix_cache_manager.time.sleep", side_effect=sleep_side_effect)
            )
        yield stack


def _make_parent_child_nodes(manager, *, parent_id, child_id, cache_status=CacheStatus.GPU):
    parent = _make_block_node(manager, node_id=parent_id, input_ids=[1, 2], cache_status=cache_status)
    child = _make_block_node(
        manager, node_id=child_id, input_ids=[1, 2, 3, 4], parent=parent, cache_status=cache_status
    )
    return parent, child


def _set_pending_executors(manager):
    manager.free_gpu_executor_pool = types.SimpleNamespace(submit=lambda *_: _PendingFuture())
    manager.free_cpu_executor_pool = types.SimpleNamespace(submit=lambda *_: _PendingFuture())


def _make_mm_build_request(
    request_id, prompt_token_ids, *, output_token_ids=None, block_tables=None, multimodal_inputs=None
):
    return SimpleNamespace(
        prompt_token_ids=prompt_token_ids,
        output_token_ids=[] if output_token_ids is None else output_token_ids,
        block_tables=[0] if block_tables is None else block_tables,
        request_id=request_id,
        multimodal_inputs={"mm_positions": [], "mm_hashes": []} if multimodal_inputs is None else multimodal_inputs,
    )


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

    def test_sync_swap_task_breaks_when_tree_status_changes(self):
        manager = _create_manager()
        manager.prefix_tree_status_signal.value[0] = PrefixTreeStatus.UPDATING

        class _NeverSetEvent:
            def wait(self, timeout=None):
                return False

        manager.task_swapping_event["timeout-task"] = _NeverSetEvent()

        manager.sync_swap_task("timeout-task")

        self.assertNotIn("timeout-task", manager.task_swapping_event)

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

        with _launch_cache_manager_patches(manager, popen=_DummyProcess):
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
        with _launch_cache_manager_patches(manager, popen=_DummyProcess) as stack:
            mock_launch = stack.enter_context(
                patch.object(manager, "launch_cache_messager", return_value=[_DummyProcess()])
            )
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
        with _launch_cache_manager_patches(manager, popen=_DummyProcess) as stack:
            stack.enter_context(patch.object(manager, "launch_cache_messager", return_value=None))
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

        with _launch_cache_messager_patches(
            popen=_DummyProcess, signal=_signal_factory, sleep_side_effect=_fake_sleep
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

        with _launch_cache_messager_patches(popen=partial(_DummyProcess, poll_value=2)):
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

        with _launch_cache_manager_patches(
            manager,
            popen=lambda cmd, **_: _CmdProcess(cmd),
            kv_shape=([1], [2, 3]),
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
        prefix_tree_status_data = np.zeros([manager.config.parallel_config.tensor_parallel_size], dtype=np.int32)
        manager.prefix_tree_status_signal = _DummyIPCSignal("prefix_tree_status", prefix_tree_status_data)
        manager.prefix_tree_status_signal.value[:] = 0

        class _NoWaitEvent:
            instances = []

            def __init__(self, *_, **__):
                self.wait_called = False
                _NoWaitEvent.instances.append(self)

            def wait(self, timeout=None):
                self.wait_called = True
                self.timeout = timeout
                return True

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

    def test_issue_and_sync_swap_tasks(self):
        manager = _create_manager()
        prefix_tree_status_data = np.zeros([manager.config.parallel_config.tensor_parallel_size], dtype=np.int32)
        manager.prefix_tree_status_signal = _DummyIPCSignal("prefix_tree_status", prefix_tree_status_data)
        manager.prefix_tree_status_signal.value[:] = 0
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

    def test_mm_build_path_full_blocks_no_unfilled(self):
        manager = _create_manager(num_gpu_blocks=4)
        request = SimpleNamespace(
            prompt_token_ids=[1, 2, 3, 4],
            output_token_ids=[],
            block_tables=[0, 1],
            request_id="mm-full",
            multimodal_inputs={"mm_positions": [], "mm_hashes": []},
        )

        leaf = manager.mm_build_path(
            request=request,
            num_computed_tokens=4,
            block_size=2,
            last_node=manager.radix_tree_root,
            num_cached_tokens=0,
        )

        self.assertIsNot(leaf, manager.radix_tree_root)
        self.assertEqual(leaf.reverved_dec_block_ids, [])
        self.assertEqual(manager.unfilled_req_block_map, {})

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
        cache_task_inflight_data = np.zeros([manager.config.parallel_config.tensor_parallel_size], dtype=np.int32)
        manager.cache_task_inflight_signal = _DummyIPCSignal("cache_task_inflight", cache_task_inflight_data)
        manager.cache_task_inflight_signal.value[:] = 0
        manager.cache_task_queue = _DummyEngineCacheQueue()

        node = BlockNode(100, [1], 0, 1, 0, 1, get_hash_str([1]), 0, parent=manager.radix_tree_root)
        manager.node_map[node.node_id] = node
        manager.task_swapping_event["evt"] = threading.Event()
        manager.task_swapping_event["evt"].set()
        manager.gpu_free_task_future = _ImmediateFuture(lambda: None)
        manager.reset()
        self.assertEqual(len(manager.node_map), 0)

    def test_reset_without_gpu_free_future(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=1)
        node = BlockNode(101, [1], 0, 1, 0, 1, get_hash_str([1]), 0, parent=manager.radix_tree_root)
        manager.node_map[node.node_id] = node
        manager.task_swapping_event["evt"] = threading.Event()
        manager.task_swapping_event["evt"].set()

        manager.reset()

        self.assertIsNone(manager.gpu_free_task_future)
        self.assertEqual(manager.task_swapping_event, {})

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

    def test_clear_prefix_cache_noop_for_normal_status(self):
        manager = _create_manager()
        manager.prefix_tree_status_signal = SimpleNamespace(value=np.array([PrefixTreeStatus.NORMAL], dtype=np.int32))
        manager.reset = MagicMock()
        with patch("fastdeploy.cache_manager.prefix_cache_manager.time.sleep", side_effect=SystemExit):
            with self.assertRaises(SystemExit):
                manager.clear_prefix_cache()
        manager.reset.assert_not_called()
        self.assertEqual(manager.prefix_tree_status_signal.value[0], PrefixTreeStatus.NORMAL)


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

    def test_get_block_hash_extra_keys_handles_ranges(self):
        manager = _create_manager()
        request = SimpleNamespace(
            multimodal_inputs={
                "mm_positions": [SimpleNamespace(offset=1, length=3)],
                "mm_hashes": ["img-a"],
            },
            num_total_tokens=8,
        )

        mm_idx, hash_keys = manager.get_block_hash_extra_keys(request, start_idx=0, end_idx=2, mm_idx=0)

        self.assertEqual(mm_idx, 0)
        self.assertEqual(hash_keys, ["img-a"])

        mm_idx, hash_keys = manager.get_block_hash_extra_keys(request, start_idx=6, end_idx=8, mm_idx=0)
        self.assertEqual(hash_keys, [])

    def test_mm_build_path_handles_unfilled_block_with_paddle_prompt(self):
        manager = _create_manager(num_gpu_blocks=4)
        paddle_prompt = paddle.to_tensor([1, 2, 3], dtype="int64")
        request = SimpleNamespace(
            prompt_token_ids=paddle_prompt.numpy(),
            output_token_ids=[],
            block_tables=[0, 1],
            request_id="mm-unfilled",
            multimodal_inputs={"mm_positions": [], "mm_hashes": []},
        )

        leaf = manager.mm_build_path(
            request=request,
            num_computed_tokens=4,
            block_size=2,
            last_node=manager.radix_tree_root,
            num_cached_tokens=0,
        )

        self.assertIsNot(leaf, manager.radix_tree_root)
        self.assertEqual(leaf.reverved_dec_block_ids, [1])

    def test_mm_build_path_handles_multimodal_partial_block(self):
        manager = _create_manager(num_gpu_blocks=4)
        request = SimpleNamespace(
            prompt_token_ids=np.array([1, 2, 3]),
            output_token_ids=[],
            block_tables=[0, 1],
            request_id="mm-partial",
            multimodal_inputs={
                "mm_positions": [SimpleNamespace(offset=1, length=1)],
                "mm_hashes": ["img"],
            },
            num_total_tokens=3,
        )

        leaf = manager.mm_build_path(
            request=request,
            num_computed_tokens=4,
            block_size=2,
            last_node=manager.radix_tree_root,
            num_cached_tokens=0,
        )

        self.assertIsNot(leaf, manager.radix_tree_root)
        self.assertEqual(leaf.reverved_dec_block_ids, [1])

    def test_launch_cache_manager_handles_storage_and_threads(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=1)
        manager.cache_config.kvcache_storage_backend = "backend"
        manager.cache_config.swap_space = 1
        manager.cache_config.local_rdma_comm_ports = [0]
        manager.cache_config.enable_prefix_caching = True

        class _Signal:
            def __init__(self, name, array, **_):
                self.name = name
                self.value = np.array(array, copy=True)

        def _fake_sleep(_):
            if hasattr(manager, "cache_transfer_inited_signal"):
                manager.cache_transfer_inited_signal.value[:] = 1
            if hasattr(manager, "cache_ready_signal"):
                manager.cache_ready_signal.value[:] = 1
            if hasattr(manager, "swap_space_ready_signal"):
                manager.swap_space_ready_signal.value[:] = 1

        captured = {}

        def _capture_popen(cmd, **_):
            captured["cmd"] = cmd
            return _DummyProcess(poll_value=1)

        with (
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.IPCSignal",
                side_effect=_Signal,
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
                side_effect=_capture_popen,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.threading.Thread",
                _TrackingThread,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.time.sleep",
                side_effect=_fake_sleep,
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

        self.assertIn("--kvcache_storage_backend backend", captured["cmd"])
        self.assertIn("--value_cache_shape 2,3", captured["cmd"])
        self.assertEqual(len(_TrackingThread.instances), 2)

    def test_launch_cache_messager_formats_value_cache_list(self):
        manager = _create_manager()
        manager.cache_config.local_rdma_comm_ports = [0]
        captured = {}

        def _capture_popen(cmd, **_):
            captured["cmd"] = cmd
            return _DummyProcess()

        def _fake_sleep(_):
            manager.cache_ready_signal.value[:] = 1

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
                side_effect=_capture_popen,
            ),
            patch(
                "fastdeploy.cache_manager.prefix_cache_manager.time.sleep",
                side_effect=_fake_sleep,
            ),
        ):
            manager.launch_cache_messager(
                cache_config=manager.cache_config,
                tensor_parallel_size=1,
                device_ids=[0],
                key_cache_shape="1",
                value_cache_shape=[1, 2],
                pod_ip="127.0.0.1",
                engine_worker_queue_port=8000,
                ipc_suffix="pid",
            )

        self.assertIn("--value_cache_shape 1,2", captured["cmd"])

    def test_request_match_blocks_prefetches_storage_cache(self):
        manager = _create_manager(num_gpu_blocks=4)
        manager.kvcache_storage_backend = "backend"
        task = SimpleNamespace(prompt_token_ids=[1, 2, 3, 4], output_token_ids=[], request_id="storage")
        captured = {}

        def _capture_prefetch(read_task):
            captured["read_task"] = read_task
            return [10]

        with (
            patch.object(manager, "mm_match_block", return_value=([], [], [], manager.radix_tree_root, 0, 0)),
            patch.object(manager, "can_allocate_gpu_blocks", return_value=True),
            patch.object(manager, "allocate_gpu_blocks", return_value=[10, 11]),
            patch.object(manager, "issue_prefetch_storage_task", side_effect=_capture_prefetch),
            patch.object(manager, "recycle_gpu_blocks") as mock_recycle,
        ):
            common_blocks, match_tokens, metrics = manager.request_match_blocks(task, block_size=2)

        self.assertEqual(common_blocks, [10])
        self.assertEqual(match_tokens, 2)
        self.assertEqual(metrics["storage_match_token_num"], 2)
        self.assertIsInstance(captured["read_task"], ReadStorageTask)
        mock_recycle.assert_called_once_with([11])

    def test_request_match_blocks_raises_for_storage_allocation_failure(self):
        manager = _create_manager(num_gpu_blocks=2)
        manager.kvcache_storage_backend = "backend"
        task = SimpleNamespace(prompt_token_ids=[1, 2], output_token_ids=[3, 4], request_id="storage-fail")

        with (
            patch.object(manager, "mm_match_block", return_value=([], [], [], manager.radix_tree_root, 0, 0)),
            patch.object(manager, "can_allocate_gpu_blocks", side_effect=[True, False]),
        ):
            with self.assertRaises(Exception):
                manager.request_match_blocks(task, block_size=2)

    def test_request_block_ids_resets_metrics_at_threshold(self):
        manager = _create_manager(num_gpu_blocks=4)
        manager.metrics.reset_metrics = MagicMock()
        manager.metrics.req_count = 9999
        task = SimpleNamespace(prompt_token_ids=[1, 2], request_id="block")
        node = BlockNode(160, [1, 2], 0, 1, 0, 2, get_hash_str([1, 2]), 0, parent=manager.radix_tree_root)

        with (
            patch.object(
                manager,
                "match_block",
                return_value=([], [], [], node, 0, 0),
            ),
            patch.object(manager, "_check_validity"),
            patch.object(manager, "_prepare_cache", return_value=([], [0])),
            patch.object(manager, "build_path", return_value=node),
        ):
            manager.request_block_ids(task, block_size=2, dec_token_num=0)

        manager.metrics.reset_metrics.assert_called_once()

    def test_request_block_ids_raises_on_match_error(self):
        manager = _create_manager()

        with patch.object(manager, "match_block", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                manager.request_block_ids(
                    SimpleNamespace(prompt_token_ids=[1], request_id="bad"), block_size=2, dec_token_num=0
                )

    def test_write_cache_to_storage_builds_task_from_leaf(self):
        manager = _create_manager()
        manager.kvcache_storage_backend = "backend"
        manager.cache_config.enable_output_caching = True
        node = BlockNode(130, [1, 2], 0, 1, 0, 2, get_hash_str([1, 2]), 0, parent=manager.radix_tree_root)
        manager.req_leaf_map["write-req"] = node
        manager.radix_tree_root.children[node.hash_value] = node
        request = SimpleNamespace(
            prompt_token_ids=np.array([1, 2]),
            output_token_ids=[3],
            request_id="write-req",
            block_tables=[7],
        )
        captured = {}

        def _capture_write(task, is_sync=True):
            captured["task"] = task
            captured["is_sync"] = is_sync

        manager.issue_write_back_storage_task = _capture_write

        manager.write_cache_to_storage(request)

        self.assertEqual(captured["task"].keys, [node.hash_value])
        self.assertEqual(captured["task"].gpu_block_ids, [7])
        self.assertEqual(captured["task"].token_ids, [1, 2, 3])

    def test_write_cache_to_storage_returns_when_no_keys(self):
        manager = _create_manager()
        manager.kvcache_storage_backend = "backend"
        manager.cache_config.enable_output_caching = False
        manager.req_leaf_map["root"] = manager.radix_tree_root
        request = SimpleNamespace(
            prompt_token_ids=[1, 2],
            output_token_ids=[3],
            request_id="root",
            block_tables=[7],
        )

        manager.write_cache_to_storage(request)

    def test_write_cache_to_storage_returns_when_backend_missing(self):
        manager = _create_manager()
        request = SimpleNamespace(
            prompt_token_ids=np.array([1, 2]),
            output_token_ids=[3],
            request_id="no-backend",
            block_tables=[7],
        )

        manager.write_cache_to_storage(request)

    def test_issue_write_back_storage_task_enqueues_when_valid(self):
        manager = _create_manager()
        manager.kvcache_storage_backend = "backend"
        manager.cache_task_queue = _DummyEngineCacheQueue()
        task = WriteStorageTask(task_id="write-ok", keys=["hash"], token_ids=[1], gpu_block_ids=[0])

        manager.issue_write_back_storage_task(task, is_sync=False)

        self.assertIn("write-ok", manager.task_write_back_event)
        self.assertEqual(len(manager.cache_task_queue.tasks), 1)

    def test_issue_write_back_storage_task_no_backend_returns(self):
        manager = _create_manager()
        task = WriteStorageTask(task_id="write-none", keys=["hash"], token_ids=[1], gpu_block_ids=[0])

        manager.issue_write_back_storage_task(task, is_sync=False)

    def test_wait_write_storage_task_clears_event(self):
        manager = _create_manager()
        manager.task_write_back_event["write"] = threading.Event()
        manager.task_write_back_event["write"].set()

        manager.wait_write_storage_task("write")

        self.assertNotIn("write", manager.task_write_back_event)

    def test_wait_write_storage_task_noop_when_missing(self):
        manager = _create_manager()

        manager.wait_write_storage_task("missing")

    def test_issue_write_back_storage_task_sync_waits(self):
        manager = _create_manager()
        manager.kvcache_storage_backend = "backend"
        manager.cache_task_queue = _DummyEngineCacheQueue()
        task = WriteStorageTask(task_id="sync", keys=["hash"], token_ids=[1], gpu_block_ids=[0])

        with patch.object(manager, "wait_write_storage_task") as mock_wait:
            manager.issue_write_back_storage_task(task, is_sync=True)

        mock_wait.assert_called_once_with("sync")

    def test_issue_prefetch_storage_task_returns_empty_when_disabled(self):
        manager = _create_manager()

        result = manager.issue_prefetch_storage_task(
            ReadStorageTask(task_id="task", keys=[], token_ids=[], gpu_block_ids=[], start_read_block_idx=0)
        )

        self.assertEqual(result, [])

    def test_issue_prefetch_storage_task_sync_waits(self):
        manager = _create_manager()
        manager.kvcache_storage_backend = "backend"
        manager.cache_task_queue = _DummyEngineCacheQueue()
        task = ReadStorageTask(
            task_id="prefetch", keys=["k"], token_ids=[1], gpu_block_ids=[0], start_read_block_idx=0
        )

        with patch.object(manager, "wait_prefetch_storage_task", return_value=[0]) as mock_wait:
            result = manager.issue_prefetch_storage_task(task, is_sync=True)

        self.assertEqual(result, [0])
        self.assertEqual(len(manager.cache_task_queue.tasks), 1)
        mock_wait.assert_called_once_with("prefetch")

    def test_wait_prefetch_storage_task_returns_ids_and_clears_state(self):
        manager = _create_manager()
        manager.task_prefetch_event["prefetch"] = threading.Event()
        manager.storage_prefetch_block_ids["prefetch"] = [1, 2]
        manager.task_prefetch_event["prefetch"].set()

        result = manager.wait_prefetch_storage_task("prefetch")

        self.assertEqual(result, [1, 2])
        self.assertNotIn("prefetch", manager.task_prefetch_event)
        self.assertNotIn("prefetch", manager.storage_prefetch_block_ids)

    def test_issue_write_back_storage_task_rejects_mismatched_lengths(self):
        manager = _create_manager()
        manager.kvcache_storage_backend = "dummy"
        manager.cache_task_queue = _DummyEngineCacheQueue()
        task = WriteStorageTask(
            task_id="task",
            keys=["hash-a"],
            token_ids=[1, 2],
            gpu_block_ids=[0, 1],
        )

        with self.assertRaises(ValueError):
            manager.issue_write_back_storage_task(task, is_sync=False)

    def test_wait_prefetch_storage_task_returns_none_when_missing(self):
        manager = _create_manager()

        result = manager.wait_prefetch_storage_task("missing")

        self.assertIsNone(result)

    def test_can_allocate_gpu_blocks_returns_false_after_free_attempt(self):
        manager = _create_manager()
        manager.gpu_free_block_list.clear()

        with patch.object(manager, "free_block_ids"):
            result = manager.can_allocate_gpu_blocks(1)

        self.assertFalse(result)

    def test_release_block_ids_removes_radix_tree_info(self):
        manager = _create_manager()
        req_id = "release"
        node = BlockNode(123, [1, 2], 0, 1, 0, 2, get_hash_str([1, 2]), 0, parent=manager.radix_tree_root)
        node.req_id_set.add(req_id)
        manager.req_leaf_map[req_id] = node
        manager.leaf_req_map[node].add(req_id)
        manager.req_to_radix_tree_info[req_id] = [node, 2]

        manager.release_block_ids(SimpleNamespace(request_id=req_id))

        self.assertNotIn(req_id, manager.req_to_radix_tree_info)

    def test_release_block_ids_raises_on_missing_leaf(self):
        manager = _create_manager()

        with self.assertRaises(KeyError):
            manager.release_block_ids(SimpleNamespace(request_id="missing"))

    def test_is_chunked_mm_input_handles_missing_positions(self):
        manager = _create_manager()

        result = manager.is_chunked_mm_input(None, matched_token_num=0)

        self.assertEqual(result, (False, 0))

    def test_handle_swap_result_recycles_cpu_block_when_reused(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=1)
        manager.cpu_free_block_list.clear()
        node = BlockNode(120, [1], 0, 1, 0, 1, get_hash_str([1]), 0, parent=manager.radix_tree_root)
        node.cache_status = CacheStatus.GPU
        manager.node_map[node.node_id] = node

        manager._handle_swap_result(
            node.node_id,
            task_gpu_block_id=0,
            task_cpu_block_id=5,
            event_type=CacheStatus.SWAP2CPU,
        )

        self.assertIn(5, manager.cpu_free_block_list)

    def test_free_nodes_directly_continues_when_parent_in_lru(self):
        manager = _create_manager(num_gpu_blocks=2)
        parent = BlockNode(124, [1, 2], 0, 1, 0, 2, get_hash_str([1, 2]), 0, parent=manager.radix_tree_root)
        child_hash = get_hash_str([3, 4])
        child = BlockNode(125, [1, 2, 3, 4], 0, 2, 1, 2, child_hash, 0, parent=parent)
        parent.children[child_hash] = child
        child.shared_count = 0
        manager.gpu_lru_leaf_set.add(child)
        manager.gpu_lru_leaf_heap.append(child)
        manager.gpu_lru_leaf_set.add(parent)

        manager.free_nodes_directly(child)

        self.assertIn(parent, manager.gpu_lru_leaf_set)

    def test_free_nodes_directly_raises_on_error(self):
        manager = _create_manager()
        node = _make_block_node(manager, node_id=126, input_ids=[1, 2])
        node.shared_count = 0

        with patch.object(manager, "_handle_free_gpu_node_without_cpu", side_effect=ValueError("boom")):
            with self.assertRaises(ValueError):
                manager.free_nodes_directly(node)

    def test_handle_swap_result_adds_cpu_lru_when_swapped(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=1)
        node = BlockNode(121, [1], 0, 1, 0, 1, get_hash_str([1]), 0, parent=manager.radix_tree_root)
        node.cache_status = CacheStatus.SWAP2CPU
        node.shared_count = 0
        manager.node_map[node.node_id] = node

        manager._handle_swap_result(
            node.node_id, task_gpu_block_id=0, task_cpu_block_id=6, event_type=CacheStatus.SWAP2CPU
        )

        self.assertEqual(node.cache_status, CacheStatus.CPU)
        self.assertIn(node, manager.cpu_lru_leaf_set)

    def test_handle_swap_result_handles_none_id(self):
        manager = _create_manager()

        manager._handle_swap_result(None, task_gpu_block_id=0, task_cpu_block_id=0, event_type=CacheStatus.SWAP2CPU)

    def test_handle_swap_result_logs_unexpected_event(self):
        manager = _create_manager()
        node = BlockNode(122, [1], 0, 1, 0, 1, get_hash_str([1]), 0, parent=manager.radix_tree_root)
        manager.node_map[node.node_id] = node
        unexpected_event = SimpleNamespace(value=999)

        manager._handle_swap_result(
            node.node_id, task_gpu_block_id=0, task_cpu_block_id=0, event_type=unexpected_event
        )

    def test_free_cpu_block_ids_evicts_cpu_nodes(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=2)
        node = BlockNode(140, [1, 2], 0, 1, 0, 2, get_hash_str([1, 2]), 0, parent=manager.radix_tree_root)
        node.cache_status = CacheStatus.CPU
        node.shared_count = 0
        manager.node_map[node.node_id] = node
        manager.radix_tree_root.children[node.hash_value] = node
        manager.cpu_lru_leaf_heap.append(node)
        manager.cpu_lru_leaf_set.add(node)

        freed = manager.free_cpu_block_ids(1)

        self.assertEqual(freed, 1)
        self.assertIn(node.block_id, manager.cpu_free_block_list)

    def test_free_cpu_block_ids_breaks_when_enough(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=2)
        parent = BlockNode(141, [1, 2], 0, 1, 0, 2, get_hash_str([1, 2]), 0, parent=manager.radix_tree_root)
        child_hash = get_hash_str([3, 4])
        child = BlockNode(142, [1, 2, 3, 4], 0, 2, 1, 2, child_hash, 0, parent=parent)
        child.cache_status = CacheStatus.CPU
        child.shared_count = 0
        parent.children[child_hash] = child
        sibling_hash = get_hash_str([5, 6])
        sibling = BlockNode(143, [5, 6], 0, 1, 0, 2, sibling_hash, 0, parent=manager.radix_tree_root)
        sibling.cache_status = CacheStatus.CPU
        sibling.shared_count = 0
        manager.cpu_lru_leaf_heap.extend([child, sibling])
        manager.cpu_lru_leaf_set.update([child, sibling])

        freed = manager.free_cpu_block_ids(1)

        self.assertEqual(freed, 1)

    def test_free_cpu_block_ids_continues_when_parent_in_lru(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=2)
        parent, child = _make_parent_child_nodes(manager, parent_id=143, child_id=144, cache_status=CacheStatus.CPU)
        child.shared_count = 0
        manager.cpu_lru_leaf_heap.append(child)
        manager.cpu_lru_leaf_set.add(child)
        manager.cpu_lru_leaf_set.add(parent)

        manager.free_cpu_block_ids(1)

        self.assertIn(parent, manager.cpu_lru_leaf_set)

    def test_free_cpu_block_ids_pushes_parent_when_eligible(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=2)
        parent, child = _make_parent_child_nodes(manager, parent_id=145, child_id=146, cache_status=CacheStatus.CPU)
        child.shared_count = 0
        parent.shared_count = 0
        manager.cpu_lru_leaf_heap.append(child)
        manager.cpu_lru_leaf_set.add(child)

        manager.free_cpu_block_ids(1)

        self.assertIn(parent, manager.cpu_lru_leaf_set)

    def test_free_nodes_directly_breaks_when_parent_has_children(self):
        manager = _create_manager(num_gpu_blocks=2)
        parent = BlockNode(170, [1, 2], 0, 1, 0, 2, get_hash_str([1, 2]), 0, parent=manager.radix_tree_root)
        child_hash = get_hash_str([3, 4])
        child = BlockNode(171, [1, 2, 3, 4], 0, 2, 1, 2, child_hash, 0, parent=parent)
        sibling_hash = get_hash_str([5, 6])
        sibling = BlockNode(172, [1, 2, 5, 6], 0, 2, 2, 2, sibling_hash, 0, parent=parent)
        parent.children[child_hash] = child
        parent.children[sibling_hash] = sibling
        child.shared_count = 0
        manager.gpu_lru_leaf_set.add(child)
        manager.gpu_lru_leaf_heap.append(child)

        manager.free_nodes_directly(child)

        self.assertIn(sibling_hash, parent.children)

    def test_free_block_ids_async_swaps_gpu_nodes_to_cpu(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=2)
        manager.cache_config.num_cpu_blocks = 2
        manager.cache_task_queue = _DummyEngineCacheQueue()
        manager.cpu_free_block_list.clear()
        manager.allocate_cpu_blocks = MagicMock(return_value=[5])

        class _DeferredFuture:
            def result(self):
                return None

        manager.free_cpu_executor_pool = types.SimpleNamespace(submit=lambda *_: _DeferredFuture())
        manager.free_gpu_executor_pool = types.SimpleNamespace(submit=_ImmediateFuture)
        manager.issue_swap_task = MagicMock()

        node_hash = get_hash_str([1, 2])
        node = BlockNode(180, [1, 2], node_hash, 1, 0, 2, node_hash, 0, parent=manager.radix_tree_root)
        node.shared_count = 0
        manager.radix_tree_root.children[node_hash] = node
        manager.gpu_lru_leaf_heap.append(node)
        manager.gpu_lru_leaf_set.add(node)

        manager.free_block_ids_async(need_block_num=1)

        manager.issue_swap_task.assert_called_once()

    def test_free_block_ids_async_breaks_when_enough_blocks_freed(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=0)
        parent, child = _make_parent_child_nodes(manager, parent_id=181, child_id=182)
        parent.shared_count = 0
        child.shared_count = 0
        manager.gpu_lru_leaf_heap.append(child)
        manager.gpu_lru_leaf_set.add(child)

        manager.free_block_ids_async(need_block_num=1)

        self.assertIn(parent, manager.gpu_lru_leaf_set)

    def test_free_block_ids_async_continues_when_parent_in_lru(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=0)
        parent, child = _make_parent_child_nodes(manager, parent_id=183, child_id=184)
        child.shared_count = 0
        manager.gpu_lru_leaf_heap.append(child)
        manager.gpu_lru_leaf_set.update([child, parent])

        manager.free_block_ids_async(need_block_num=1)

        self.assertIn(parent, manager.gpu_lru_leaf_set)

    def test_free_block_ids_async_skips_in_use_node(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=0)
        node = _make_block_node(manager, node_id=185, input_ids=[1, 2])
        node.shared_count = 1
        manager.gpu_lru_leaf_heap.append(node)
        manager.gpu_lru_leaf_set.add(node)

        manager.free_block_ids_async(need_block_num=1)

        self.assertNotIn(node, manager.gpu_lru_leaf_set)

    def test_free_block_ids_async_handles_parent_in_lru_for_swap(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=2)
        manager.cache_config.num_cpu_blocks = 2
        parent, child = _make_parent_child_nodes(manager, parent_id=186, child_id=187)
        child.shared_count = 0
        manager.gpu_lru_leaf_heap.append(child)
        manager.gpu_lru_leaf_set.update([child, parent])

        _set_pending_executors(manager)

        manager.free_block_ids_async(need_block_num=1)

        self.assertIn(parent, manager.gpu_lru_leaf_set)

    def test_free_block_ids_async_skips_in_use_node_with_swap(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=2)
        manager.cache_config.num_cpu_blocks = 2
        node = _make_block_node(manager, node_id=187, input_ids=[1, 2])
        node.shared_count = 1
        manager.gpu_lru_leaf_heap.append(node)
        manager.gpu_lru_leaf_set.add(node)

        manager.free_block_ids_async(need_block_num=1)

        self.assertNotIn(node, manager.gpu_lru_leaf_set)

    def test_free_block_ids_async_pushes_parent_when_eligible_for_swap(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=2)
        manager.cache_config.num_cpu_blocks = 2
        parent, child = _make_parent_child_nodes(manager, parent_id=188, child_id=189)
        child.shared_count = 0
        parent.shared_count = 0
        manager.gpu_lru_leaf_heap.append(child)
        manager.gpu_lru_leaf_set.add(child)

        _set_pending_executors(manager)

        manager.free_block_ids_async(need_block_num=1)

        self.assertIn(parent, manager.gpu_lru_leaf_set)

    def test_free_block_ids_async_adjusts_cpu_free_count(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=4)
        manager.cache_config.num_cpu_blocks = 4
        node_hash = get_hash_str([1, 2])
        node = BlockNode(190, [1, 2], node_hash, 1, 0, 2, node_hash, 0, parent=manager.radix_tree_root)
        node.shared_count = 0
        manager.gpu_lru_leaf_heap.append(node)
        manager.gpu_lru_leaf_set.add(node)
        manager.cpu_free_block_list.clear()

        manager.free_cpu_executor_pool = types.SimpleNamespace(submit=lambda *_: _PendingFuture())
        manager.free_gpu_executor_pool = types.SimpleNamespace(submit=lambda *_: _PendingFuture())

        manager.free_block_ids_async(need_block_num=3)

        self.assertIsNotNone(manager.gpu_free_task_future)

    def test_free_block_ids_async_raises_on_error(self):
        manager = _create_manager()
        node = _make_block_node(manager, node_id=191, input_ids=[1, 2])
        manager.gpu_lru_leaf_heap.append(node)
        manager.gpu_lru_leaf_set.add(node)

        with patch("fastdeploy.cache_manager.prefix_cache_manager.heapq.heappop", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                manager.free_block_ids_async(need_block_num=1)

    def test_is_chunked_mm_input_detects_positions(self):
        manager = _create_manager()
        mm_inputs = {"mm_positions": [SimpleNamespace(offset=2, length=2)], "mm_hashes": ["h"]}

        is_chunked, idx = manager.is_chunked_mm_input(mm_inputs, matched_token_num=3)
        self.assertTrue(is_chunked)
        self.assertEqual(idx, 0)

        is_chunked, idx = manager.is_chunked_mm_input(mm_inputs, matched_token_num=1)
        self.assertFalse(is_chunked)
        self.assertEqual(idx, 0)

    def test_get_block_hash_extra_keys_handles_more_branches(self):
        manager = _create_manager()
        request = SimpleNamespace(
            multimodal_inputs={
                "mm_positions": [
                    SimpleNamespace(offset=0, length=1),
                    SimpleNamespace(offset=4, length=1),
                ],
                "mm_hashes": ["img-0", "img-1"],
            },
            num_total_tokens=6,
        )

        mm_idx, hash_keys = manager.get_block_hash_extra_keys(request, start_idx=2, end_idx=4, mm_idx=0)
        self.assertEqual(hash_keys, [])
        self.assertEqual(mm_idx, 1)

        mm_idx, hash_keys = manager.get_block_hash_extra_keys(request, start_idx=0, end_idx=3, mm_idx=0)
        self.assertEqual(hash_keys, ["img-0"])

    def test_mm_build_path_returns_last_node_when_already_cached(self):
        manager = _create_manager()
        request = _make_mm_build_request("cached", np.array([1, 2]))

        leaf = manager.mm_build_path(
            request=request,
            num_computed_tokens=2,
            block_size=2,
            last_node=manager.radix_tree_root,
            num_cached_tokens=2,
        )

        self.assertIs(leaf, manager.radix_tree_root)

    def test_mm_build_path_records_unfilled_block_on_root(self):
        manager = _create_manager(num_gpu_blocks=2)
        request = _make_mm_build_request("unfilled", [1])

        leaf = manager.mm_build_path(
            request=request,
            num_computed_tokens=2,
            block_size=2,
            last_node=manager.radix_tree_root,
            num_cached_tokens=0,
        )

        self.assertIs(leaf, manager.radix_tree_root)
        self.assertIn("unfilled", manager.unfilled_req_block_map)

    def test_build_path_handles_reserved_blocks_and_unfilled(self):
        manager = _create_manager(num_gpu_blocks=4)
        leaf = manager.build_path(
            req_id="build",
            current_time=0.0,
            input_ids=[1, 2, 3],
            left_input_ids=[3],
            gpu_block_ids=[0, 1],
            block_size=2,
            last_node=manager.radix_tree_root,
            reverved_dec_block_num=1,
        )

        self.assertIs(leaf, manager.radix_tree_root)
        self.assertEqual(manager.unfilled_req_block_map["build"], [0, 1])

    def test_mm_match_block_handles_lru_removals(self):
        manager = _create_manager(num_gpu_blocks=2)
        block_size = 2
        node = _make_block_node(manager, node_id=190, input_ids=[1, 2], block_size=block_size)
        manager.gpu_lru_leaf_heap.append(node)
        manager.gpu_lru_leaf_set.add(node)

        request = SimpleNamespace(
            prompt_token_ids=np.array([1, 2, 3]),
            output_token_ids=[],
            request_id="mm",
            multimodal_inputs=None,
        )
        match_gpu, match_cpu, *_ = manager.mm_match_block(request, block_size)

        self.assertEqual(match_gpu, [0])
        self.assertEqual(match_cpu, [])

    def test_mm_match_block_swaps_from_swap2cpu(self):
        manager = _create_manager(num_gpu_blocks=2)
        block_size = 2
        node = _make_block_node(manager, node_id=192, input_ids=[1, 2], block_size=block_size)
        node.cache_status = CacheStatus.SWAP2CPU

        request = SimpleNamespace(
            prompt_token_ids=[1, 2],
            output_token_ids=[],
            request_id="swap-mm",
            multimodal_inputs=None,
        )
        match_gpu, match_cpu, *_ = manager.mm_match_block(request, block_size)

        self.assertEqual(match_gpu, [0])
        self.assertEqual(match_cpu, [])
        self.assertEqual(node.cache_status, CacheStatus.GPU)

    def test_mm_match_block_breaks_when_no_match(self):
        manager = _create_manager(num_gpu_blocks=2)
        request = SimpleNamespace(
            prompt_token_ids=[9, 10],
            output_token_ids=[],
            request_id="no-match",
            multimodal_inputs=None,
        )

        match_gpu, match_cpu, swap_nodes, last_node, gpu_tokens, cpu_tokens = manager.mm_match_block(request, 2)

        self.assertEqual(match_gpu, [])
        self.assertEqual(match_cpu, [])
        self.assertEqual(swap_nodes, [])
        self.assertEqual(last_node, manager.radix_tree_root)
        self.assertEqual(gpu_tokens, 0)
        self.assertEqual(cpu_tokens, 0)

    def test_mm_match_block_swaps_from_cpu_lru(self):
        manager = _create_manager(num_gpu_blocks=2)
        block_size = 2
        node = _make_block_node(manager, node_id=191, input_ids=[1, 2], block_size=block_size)
        node.cache_status = CacheStatus.CPU
        manager.cpu_lru_leaf_heap.append(node)
        manager.cpu_lru_leaf_set.add(node)

        request = SimpleNamespace(
            prompt_token_ids=[1, 2],
            output_token_ids=[],
            request_id="mm2",
            multimodal_inputs=None,
        )
        match_gpu, match_cpu, swap_nodes, *_ = manager.mm_match_block(request, block_size)

        self.assertEqual(match_gpu, [])
        self.assertEqual(match_cpu, [0])
        self.assertEqual(swap_nodes, [node.node_id])

    def test_match_block_breaks_on_partial_block(self):
        manager = _create_manager(num_gpu_blocks=2)

        match_gpu, match_cpu, swap_nodes, last_node, gpu_tokens, cpu_tokens = manager.match_block(
            "req", [1], block_size=2
        )

        self.assertEqual(match_gpu, [])
        self.assertEqual(match_cpu, [])
        self.assertEqual(swap_nodes, [])
        self.assertEqual(last_node, manager.radix_tree_root)
        self.assertEqual(gpu_tokens, 0)
        self.assertEqual(cpu_tokens, 0)

    def test_match_block_breaks_when_no_match(self):
        manager = _create_manager(num_gpu_blocks=2)

        match_gpu, match_cpu, swap_nodes, last_node, gpu_tokens, cpu_tokens = manager.match_block(
            "req", [9, 10], block_size=2
        )

        self.assertEqual(match_gpu, [])
        self.assertEqual(match_cpu, [])
        self.assertEqual(swap_nodes, [])
        self.assertEqual(last_node, manager.radix_tree_root)
        self.assertEqual(gpu_tokens, 0)
        self.assertEqual(cpu_tokens, 0)

    def test_match_block_swaps_node_from_swap2cpu(self):
        manager = _create_manager(num_gpu_blocks=2)
        block_size = 2
        node = _make_block_node(manager, node_id=150, input_ids=[1, 2], block_size=block_size)
        node.cache_status = CacheStatus.SWAP2CPU
        manager.gpu_lru_leaf_heap.append(node)
        manager.gpu_lru_leaf_set.add(node)

        match_gpu, match_cpu, swap_nodes, last_node, gpu_tokens, cpu_tokens = manager.match_block(
            "swap", [1, 2], block_size
        )

        self.assertEqual(match_gpu, [0])
        self.assertEqual(match_cpu, [])
        self.assertEqual(swap_nodes, [])
        self.assertEqual(last_node, node)
        self.assertEqual(gpu_tokens, block_size)
        self.assertEqual(cpu_tokens, 0)
        self.assertNotIn(node, manager.gpu_lru_leaf_set)
        self.assertEqual(node.cache_status, CacheStatus.GPU)

    def test_match_block_handles_cpu_lru_leaf(self):
        manager = _create_manager(num_gpu_blocks=2)
        block_size = 2
        node = _make_block_node(manager, node_id=155, input_ids=[1, 2], block_size=block_size)
        node.cache_status = CacheStatus.CPU
        manager.cpu_lru_leaf_heap.append(node)
        manager.cpu_lru_leaf_set.add(node)

        match_gpu, match_cpu, swap_nodes, *_ = manager.match_block("cpu", [1, 2], block_size)

        self.assertEqual(match_gpu, [])
        self.assertEqual(match_cpu, [0])
        self.assertEqual(swap_nodes, [node.node_id])

    def test_build_path_handles_reserved_only(self):
        manager = _create_manager(num_gpu_blocks=4)
        node = manager.build_path(
            req_id="reserved",
            current_time=0.0,
            input_ids=[1, 2],
            left_input_ids=[],
            gpu_block_ids=[0, 1],
            block_size=2,
            last_node=manager.radix_tree_root,
            reverved_dec_block_num=2,
        )

        self.assertEqual(node, manager.radix_tree_root)
        self.assertEqual(manager.radix_tree_root.reverved_dec_block_ids, [0, 1])

    def test_recv_data_transfer_result_handles_storage_events(self):
        manager = _create_manager()
        manager.cache_task_queue = _FakeTransferQueue(
            [
                (CacheStatus.STORAGE2GPU, "prefetch", ["hash"], [1, 2]),
                (CacheStatus.GPU2STORAGE, "write", ["hash"], [3]),
            ]
        )
        manager.task_prefetch_event["prefetch"] = threading.Event()
        manager.task_write_back_event["write"] = threading.Event()

        with self.assertRaises(SystemExit):
            manager.recv_data_transfer_result()

        self.assertTrue(manager.task_prefetch_event["prefetch"].is_set())
        self.assertTrue(manager.task_write_back_event["write"].is_set())

    def test_recv_data_transfer_result_logs_error_for_bad_payload(self):
        manager = _create_manager()

        class _BadQueue:
            def get_transfer_done_signal(self):
                return ("bad",)

        manager.cache_task_queue = _BadQueue()

        with self.assertRaises(AttributeError):
            manager.recv_data_transfer_result()

    def test_cache_output_blocks_handles_cpu_hit_and_updates_leaf(self):
        manager = _create_manager(num_gpu_blocks=4, num_cpu_blocks=1)
        req_id = "cache-out"
        child = _make_block_node(manager, node_id=230, input_ids=[1, 2], cache_status=CacheStatus.CPU)
        child.block_id = 9
        manager.cpu_lru_leaf_heap.append(child)
        manager.cpu_lru_leaf_set.add(child)
        manager.req_to_radix_tree_info[req_id] = (manager.radix_tree_root, 0)
        manager.leaf_req_map[manager.radix_tree_root].add(req_id)
        task = SimpleNamespace(
            request_id=req_id,
            prompt_token_ids=[1, 2],
            output_token_ids=[3, 4],
            block_tables=[7, 8],
        )

        with (
            patch.object(manager, "mm_build_path", return_value=child) as mock_build,
            patch.object(manager, "recycle_gpu_blocks") as mock_recycle_gpu,
            patch.object(manager, "recycle_cpu_blocks") as mock_recycle_cpu,
        ):
            manager.cache_output_blocks(task, block_size=2)

        mock_build.assert_called_once()
        mock_recycle_gpu.assert_called_once_with([])
        mock_recycle_cpu.assert_called_once_with([9])
        self.assertEqual(child.cache_status, CacheStatus.GPU)
        self.assertEqual(child.block_id, 7)
        self.assertEqual(task.num_cached_blocks, 2)

    def test_non_normal_prefix_tree_status_swallows_expected_errors(self):
        manager = _create_manager()
        manager.prefix_tree_status_signal.value[0] = PrefixTreeStatus.UPDATING

        with patch.object(manager, "match_block", side_effect=RuntimeError("boom")):
            manager.request_block_ids(
                SimpleNamespace(prompt_token_ids=[1], request_id="req"), block_size=2, dec_token_num=0
            )

        with patch.object(manager, "mm_match_block", side_effect=RuntimeError("boom")):
            manager.request_match_blocks(
                SimpleNamespace(prompt_token_ids=[1], output_token_ids=[], request_id="req"),
                block_size=2,
            )

        manager.update_cache_blocks(
            SimpleNamespace(request_id="missing", output_token_ids=[1], block_tables=[0]),
            block_size=2,
            num_computed_tokens=2,
        )
        manager.release_block_ids(SimpleNamespace(request_id="missing"))

        node = _make_block_node(manager, node_id=231, input_ids=[1, 2])
        node.shared_count = 0
        with patch.object(manager, "_handle_free_gpu_node_without_cpu", side_effect=ValueError("boom")):
            manager.free_nodes_directly(node)

        manager.gpu_lru_leaf_heap.append(node)
        manager.gpu_lru_leaf_set.add(node)
        with patch("fastdeploy.cache_manager.prefix_cache_manager.heapq.heappop", side_effect=RuntimeError("boom")):
            manager.free_block_ids_async(need_block_num=1)

        class _OneBadThenExitQueue:
            def __init__(self):
                self.calls = 0

            def get_transfer_done_signal(self):
                self.calls += 1
                if self.calls == 1:
                    return ("bad",)
                raise SystemExit

        manager.cache_task_queue = _OneBadThenExitQueue()
        with self.assertRaises(SystemExit):
            manager.recv_data_transfer_result()

    def test_release_block_ids_async_submits(self):
        manager = _create_manager()

        with patch.object(manager.executor_pool, "submit", return_value="future") as mock_submit:
            result = manager.release_block_ids_async(SimpleNamespace(request_id="req"))

        self.assertEqual(result, "future")
        mock_submit.assert_called_once()

    def test_free_block_ids_waits_for_future_once(self):
        manager = _create_manager()

        class _FlippingFuture:
            def __init__(self):
                self.calls = 0

            def done(self):
                self.calls += 1
                return self.calls > 1

        manager.gpu_free_task_future = _FlippingFuture()
        with (
            patch.object(manager, "free_block_ids_async"),
            patch("fastdeploy.cache_manager.prefix_cache_manager.time.sleep"),
        ):
            manager.free_block_ids(need_block_num=1)

    def test_release_block_ids_returns_when_leaf_in_lru(self):
        manager = _create_manager()
        node = BlockNode(200, [1, 2], 0, 1, 0, 2, get_hash_str([1, 2]), 0, parent=manager.radix_tree_root)
        manager.req_leaf_map["req"] = node
        manager.leaf_req_map[node].add("req")
        manager.gpu_lru_leaf_set.add(node)

        manager.release_block_ids(SimpleNamespace(request_id="req"))

        self.assertNotIn("req", manager.req_leaf_map)

    def test_update_cache_blocks_raises_on_missing_request(self):
        manager = _create_manager()
        task = SimpleNamespace(request_id="missing", output_token_ids=[1], block_tables=[0])

        with self.assertRaises(KeyError):
            manager.update_cache_blocks(task, block_size=2, num_computed_tokens=2)

    def test_reset_returns_when_empty(self):
        manager = _create_manager()

        manager.reset()

    def test_reset_handles_no_cpu_blocks(self):
        manager = _create_manager(num_gpu_blocks=2, num_cpu_blocks=0)
        node = BlockNode(210, [1], 0, 1, 0, 1, get_hash_str([1]), 0, parent=manager.radix_tree_root)
        manager.node_map[node.node_id] = node

        manager.reset()

        self.assertEqual(manager.cpu_free_block_list, [])


if __name__ == "__main__":
    unittest.main()
