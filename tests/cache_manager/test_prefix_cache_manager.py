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
import time
import types
import unittest
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import asdict
from functools import partial
from threading import Event
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import paddle
import pytest

# Module under test: PrefixCacheManager and related cache primitives.
from fastdeploy.cache_manager.cache_data import BlockNode, CacheStatus
from fastdeploy.cache_manager.prefix_cache_manager import PrefixCacheManager
from fastdeploy.config import CacheConfig, FDConfig, ParallelConfig
from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.request import ImagePosition
from fastdeploy.inter_communicator import PrefixTreeStatus
from fastdeploy.scheduler import SchedulerConfig
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


# Dummy classes for additional tests
class DummySpeculativeConfig:
    method = None

    def to_json_string(self):
        return "{}"


class DummyRequest:
    def __init__(
        self,
        request_id,
        prompt_token_ids,
        output_token_ids=None,
        multimodal_inputs=None,
        block_tables=None,
    ):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids = output_token_ids or []
        self.multimodal_inputs = multimodal_inputs
        self.block_tables = block_tables or []
        self.num_total_tokens = len(self.prompt_token_ids) + len(self.output_token_ids)


def make_prefix_cache_manager(
    max_num_seqs=2,
    num_gpu_blocks_override=12,
    num_cpu_blocks=0,
    block_size=4,
    enable_mm=False,
    disable_chunked_mm_input=False,
    splitwise_role="mixed",
):
    max_model_len = max(16, block_size * 4)
    engine_args = EngineArgs(
        max_num_seqs=max_num_seqs,
        num_gpu_blocks_override=num_gpu_blocks_override,
        block_size=block_size,
        max_model_len=max_model_len,
    )
    args = asdict(engine_args)
    cache_cfg = CacheConfig(args)
    cache_cfg.bytes_per_layer_per_block = 1
    cache_cfg.cache_queue_port = 12345
    cache_cfg.cache_transfer_protocol = "shm"
    cache_cfg.rdma_comm_ports = None
    cache_cfg.num_cpu_blocks = num_cpu_blocks
    cache_cfg.enable_hierarchical_cache = num_cpu_blocks > 0
    cache_cfg.enable_prefix_caching = False
    cache_cfg.disable_chunked_mm_input = disable_chunked_mm_input

    model_cfg = SimpleNamespace(
        enable_mm=enable_mm,
        max_model_len=max_model_len,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=16,
    )
    model_cfg.print = lambda *args, **kwargs: None
    cache_cfg.model_cfg = model_cfg

    parallel_cfg = ParallelConfig(args)
    scheduler_cfg = SchedulerConfig(args)
    graph_opt_cfg = engine_args.create_graph_optimization_config()
    speculative_cfg = DummySpeculativeConfig()
    fd_config = FDConfig(
        model_config=model_cfg,
        cache_config=cache_cfg,
        parallel_config=parallel_cfg,
        graph_opt_config=graph_opt_cfg,
        speculative_config=speculative_cfg,
        scheduler_config=scheduler_cfg,
    )
    fd_config.postprocess()
    return PrefixCacheManager(
        config=fd_config,
        tensor_parallel_size=parallel_cfg.tensor_parallel_size,
        splitwise_role=splitwise_role,
    )


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


# Additional test cases from test1.py
class TestPrefixCacheManagerBasics(unittest.TestCase):
    def test_update_config_and_allocate_blocks(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=6, num_cpu_blocks=2)
        new_cache_cfg = manager.cache_config
        new_cache_cfg.total_block_num = 4
        manager.update_cache_config(new_cache_cfg)

        allocated = manager.allocate_gpu_blocks(2)
        self.assertEqual(len(allocated), 2)
        self.assertEqual(len(manager.gpu_free_block_list), 2)

        manager.recycle_gpu_blocks(allocated)
        self.assertEqual(len(manager.gpu_free_block_list), 4)

        cpu_allocated = manager.allocate_cpu_blocks(1)
        self.assertEqual(len(cpu_allocated), 1)
        manager.recycle_cpu_blocks(cpu_allocated)
        self.assertEqual(len(manager.cpu_free_block_list), 2)

    def test_can_allocate_and_check_validity(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=6)
        manager.gpu_free_block_list = [0]
        manager.cache_config.enable_prefix_caching = False
        self.assertFalse(manager.can_allocate_gpu_blocks(2))
        with self.assertRaises(Exception):
            manager._check_validity("req", match_gpu_blocks_num=0, expected_block_num=3)

    def test_issue_and_sync_swap_task(self):
        manager = make_prefix_cache_manager()

        class DummyQueue:
            def __init__(self):
                self.payload = None

            def put_transfer_task(self, payload):
                self.payload = payload

        manager.cache_task_queue = DummyQueue()
        transfer_task_id = "transfer-1"
        manager.issue_swap_task(
            transfer_task_id,
            swap_node_ids=[1],
            gpu_block_ids=[2],
            cpu_block_ids=[3],
            event_type=CacheStatus.SWAP2CPU,
            is_sync=False,
        )
        self.assertIn(transfer_task_id, manager.task_swapping_event)
        manager.task_swapping_event[transfer_task_id].set()
        manager.sync_swap_task(transfer_task_id)
        self.assertNotIn(transfer_task_id, manager.task_swapping_event)

    def test_prepare_cache_and_required_blocks(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=6)
        self.assertEqual(manager.get_required_block_num(5, 4), 2)
        manager.cache_task_queue = mock.Mock()
        with mock.patch.object(manager, "issue_swap_task") as issue_swap_task:
            gpu_recv, gpu_extra = manager._prepare_cache(
                req_id="req",
                input_ids=[1, 2, 3, 4],
                block_size=4,
                expected_block_num=2,
                match_gpu_block_ids=[],
                match_cpu_block_ids=[0],
                match_node_ids=[1],
            )
        issue_swap_task.assert_called_once()
        self.assertEqual(len(gpu_recv), 1)
        self.assertEqual(len(gpu_extra), 1)

    def test_get_block_hash_extra_keys_and_hash(self):
        manager = make_prefix_cache_manager(enable_mm=True)
        paddle_tensor = paddle.to_tensor([1, 2, 3, 4], dtype="int64")
        input_ids = paddle_tensor.numpy().tolist()

        mm_inputs = {
            "mm_positions": [ImagePosition(offset=6, length=3)],
            "mm_hashes": ["img-1"],
        }
        request = DummyRequest("req-1", input_ids, multimodal_inputs=mm_inputs)

        mm_idx, hash_keys = manager.get_block_hash_extra_keys(request, 0, 4, 0)
        self.assertEqual(mm_idx, 0)
        self.assertEqual(hash_keys, [])

        request.multimodal_inputs["mm_positions"][0] = ImagePosition(offset=2, length=3)
        mm_idx, hash_keys = manager.get_block_hash_extra_keys(request, 0, 4, 0)
        self.assertEqual(hash_keys, ["img-1"])

        hash_a = manager.hash_block_features(input_ids, hash_keys)
        hash_b = manager.hash_block_features(input_ids, hash_keys)
        self.assertEqual(hash_a, hash_b)

    def test_get_block_hash_extra_keys_edge_cases(self):
        manager = make_prefix_cache_manager(enable_mm=True)
        request = DummyRequest("req", [1, 2, 3, 4], multimodal_inputs=None)
        mm_idx, hash_keys = manager.get_block_hash_extra_keys(request, 0, 4, 0)
        self.assertEqual(mm_idx, 0)
        self.assertEqual(hash_keys, [])

        mm_inputs = {
            "mm_positions": [ImagePosition(offset=0, length=2), ImagePosition(offset=6, length=2)],
            "mm_hashes": ["h1", "h2"],
        }
        request.multimodal_inputs = mm_inputs
        request.num_total_tokens = 12
        mm_idx, hash_keys = manager.get_block_hash_extra_keys(request, 9, 12, 0)
        self.assertEqual(mm_idx, 0)
        self.assertEqual(hash_keys, [])

        mm_idx, hash_keys = manager.get_block_hash_extra_keys(request, 4, 6, 0)
        self.assertEqual(mm_idx, 1)
        self.assertEqual(hash_keys, [])

        mm_idx, hash_keys = manager.get_block_hash_extra_keys(request, 0, 10, 0)
        self.assertEqual(mm_idx, 1)
        self.assertEqual(hash_keys, ["h1", "h2"])

    def test_build_path_for_full_and_empty_left_input(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=8, block_size=4)
        gpu_block_ids = [0, 1, 2, 3]
        leaf_node = manager.build_path(
            req_id="req-empty",
            current_time=time.time(),
            input_ids=[1, 2],
            left_input_ids=[],
            gpu_block_ids=gpu_block_ids,
            block_size=4,
            last_node=manager.radix_tree_root,
            reverved_dec_block_num=2,
        )
        self.assertEqual(leaf_node, manager.radix_tree_root)
        self.assertEqual(leaf_node.reverved_dec_block_ids, [0, 1])

        gpu_block_ids = [4, 5, 6, 7]
        leaf_node = manager.build_path(
            req_id="req-partial",
            current_time=time.time(),
            input_ids=[1, 2, 3, 4, 5, 6],
            left_input_ids=[1, 2, 3, 4, 5, 6],
            gpu_block_ids=gpu_block_ids,
            block_size=4,
            last_node=manager.radix_tree_root,
            reverved_dec_block_num=1,
        )
        self.assertEqual(leaf_node.reverved_dec_block_ids, [5, 6])
        self.assertIn(leaf_node.node_id, manager.node_map)

        gpu_block_ids = [8, 9]
        leaf_node = manager.build_path(
            req_id="req-unfilled",
            current_time=time.time(),
            input_ids=[1, 2],
            left_input_ids=[1, 2],
            gpu_block_ids=gpu_block_ids,
            block_size=4,
            last_node=manager.radix_tree_root,
            reverved_dec_block_num=0,
        )
        self.assertEqual(leaf_node, manager.radix_tree_root)
        self.assertEqual(manager.unfilled_req_block_map["req-unfilled"], [8])

    def test_match_block_cache_status_transitions(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=6, block_size=4)
        input_ids = [1, 2, 3, 4]
        hash_value = manager.cal_block_hash(input_ids)

        node = BlockNode(1, input_ids, hash_value, 1, 0, 4, hash_value, time.time(), parent=manager.radix_tree_root)
        manager.node_map[node.node_id] = node
        manager.radix_tree_root.children[hash_value] = node
        manager.gpu_lru_leaf_set.add(node)
        manager.gpu_lru_leaf_heap.append(node)

        match_gpu, match_cpu, swap_ids, _, gpu_tokens, cpu_tokens = manager.match_block("req-gpu", input_ids, 4)
        self.assertEqual(match_gpu, [0])
        self.assertEqual(match_cpu, [])
        self.assertEqual(swap_ids, [])
        self.assertEqual(gpu_tokens, 4)
        self.assertEqual(cpu_tokens, 0)

        node.cache_status = CacheStatus.SWAP2CPU
        match_gpu, match_cpu, swap_ids, _, gpu_tokens, cpu_tokens = manager.match_block("req-swap", input_ids, 4)
        self.assertEqual(match_gpu, [0])
        self.assertEqual(match_cpu, [])
        self.assertEqual(swap_ids, [])
        self.assertEqual(node.cache_status, CacheStatus.GPU)

        node.cache_status = CacheStatus.CPU
        manager.cpu_lru_leaf_set.add(node)
        manager.cpu_lru_leaf_heap.append(node)
        match_gpu, match_cpu, swap_ids, _, gpu_tokens, cpu_tokens = manager.match_block("req-cpu", input_ids, 4)
        self.assertEqual(match_gpu, [])
        self.assertEqual(match_cpu, [0])
        self.assertEqual(swap_ids, [node.node_id])
        self.assertEqual(node.cache_status, CacheStatus.SWAP2GPU)

        match_gpu, match_cpu, swap_ids, _, gpu_tokens, cpu_tokens = manager.match_block("req-partial", [1, 2], 4)
        self.assertEqual(match_gpu, [])
        self.assertEqual(match_cpu, [])
        self.assertEqual(swap_ids, [])

    def test_mm_match_block_with_chunk_revert(self):
        manager = make_prefix_cache_manager(
            num_gpu_blocks_override=8,
            block_size=4,
            enable_mm=True,
            disable_chunked_mm_input=True,
        )
        prompt_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        mm_inputs = {
            "mm_positions": [ImagePosition(offset=6, length=4)],
            "mm_hashes": ["img-1"],
        }
        request = DummyRequest("req-mm", prompt_ids, multimodal_inputs=mm_inputs)

        hash_first = manager.hash_block_features(prompt_ids[:4].tolist(), [])
        hash_second = manager.hash_block_features(prompt_ids[4:8].tolist(), ["img-1"])

        first_node = BlockNode(
            1, prompt_ids, hash_first, 1, 0, 4, hash_first, time.time(), parent=manager.radix_tree_root
        )
        first_node.cache_status = CacheStatus.SWAP2CPU
        second_node = BlockNode(2, prompt_ids, hash_second, 2, 1, 4, hash_second, time.time(), parent=first_node)
        second_node.cache_status = CacheStatus.CPU
        manager.node_map[first_node.node_id] = first_node
        manager.node_map[second_node.node_id] = second_node
        manager.radix_tree_root.children[hash_first] = first_node
        first_node.children[hash_second] = second_node
        manager.gpu_lru_leaf_set.add(first_node)
        manager.gpu_lru_leaf_heap.append(first_node)
        manager.cpu_lru_leaf_set.add(second_node)
        manager.cpu_lru_leaf_heap.append(second_node)

        match_gpu, match_cpu, swap_ids, current_node, gpu_tokens, cpu_tokens = manager.mm_match_block(request, 4)
        self.assertEqual(match_gpu, [0])
        self.assertEqual(match_cpu, [1])
        self.assertEqual(swap_ids, [second_node.node_id])
        self.assertEqual(current_node, second_node)
        self.assertEqual(gpu_tokens, 4)
        self.assertEqual(cpu_tokens, 2)

    def test_mm_match_block_partial_input(self):
        manager = make_prefix_cache_manager(enable_mm=True)
        request = DummyRequest("req-mm-short", [1, 2], multimodal_inputs=None)
        match_gpu, match_cpu, swap_ids, current_node, gpu_tokens, cpu_tokens = manager.mm_match_block(request, 4)
        self.assertEqual(match_gpu, [])
        self.assertEqual(match_cpu, [])
        self.assertEqual(swap_ids, [])
        self.assertEqual(current_node, manager.radix_tree_root)
        self.assertEqual(gpu_tokens, 0)
        self.assertEqual(cpu_tokens, 0)

    def test_is_chunked_mm_input_and_revert_blocks(self):
        manager = make_prefix_cache_manager(enable_mm=True)
        mm_inputs = {"mm_positions": [ImagePosition(offset=5, length=4)]}
        is_chunked, idx = manager.is_chunked_mm_input(mm_inputs, 3)
        self.assertFalse(is_chunked)
        self.assertEqual(idx, 0)

        mm_inputs = {
            "mm_positions": [ImagePosition(offset=0, length=12)],
            "mm_hashes": ["img"],
        }
        request = DummyRequest("req-revert", [1] * 8, multimodal_inputs=mm_inputs)
        match_node_ids = [1, 2]
        matche_nodes = [
            BlockNode(1, [], 0, 1, 0, 4, "h1", time.time(), parent=manager.radix_tree_root),
            BlockNode(2, [], 0, 2, 1, 4, "h2", time.time(), parent=manager.radix_tree_root),
        ]
        match_gpu_block_ids = [0, 1]
        match_cpu_block_ids = []
        gpu_match_token_num = 8
        cpu_match_token_num = 0
        swap_node_ids = [1, 2]
        gpu_match_token_num, cpu_match_token_num, current_node = manager._revert_match_blocks(
            request=request,
            matched_token_num=8,
            block_size=4,
            chunk_idx=0,
            match_node_ids=match_node_ids,
            matche_nodes=matche_nodes,
            match_gpu_block_ids=match_gpu_block_ids,
            match_cpu_block_ids=match_cpu_block_ids,
            gpu_match_token_num=gpu_match_token_num,
            cpu_match_token_num=cpu_match_token_num,
            swap_node_ids=swap_node_ids,
        )
        self.assertEqual(match_gpu_block_ids, [])
        self.assertEqual(match_node_ids, [])
        self.assertEqual(gpu_match_token_num, 0)
        self.assertEqual(current_node, manager.radix_tree_root)

    def test_mm_build_path_variants(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=8, block_size=4, enable_mm=True)
        prompt_ids = paddle.to_tensor([1, 2, 3, 4], dtype="int64").numpy()
        request_full = DummyRequest("req-mm-full", prompt_ids, block_tables=[0, 1])
        leaf_node = manager.mm_build_path(
            request=request_full,
            num_computed_tokens=4,
            block_size=4,
            last_node=manager.radix_tree_root,
            num_cached_tokens=0,
        )
        self.assertNotEqual(leaf_node, manager.radix_tree_root)
        self.assertIn(leaf_node.node_id, manager.node_map)

        prompt_ids = paddle.to_tensor([5, 6], dtype="int64").numpy()
        request_unfilled = DummyRequest("req-mm-unfilled", prompt_ids, block_tables=[2])
        leaf_node = manager.mm_build_path(
            request=request_unfilled,
            num_computed_tokens=4,
            block_size=4,
            last_node=manager.radix_tree_root,
            num_cached_tokens=0,
        )
        self.assertEqual(leaf_node, manager.radix_tree_root)
        self.assertEqual(manager.unfilled_req_block_map["req-mm-unfilled"], [2])

    def test_request_block_ids_flow(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=10, block_size=4)
        task = SimpleNamespace(prompt_token_ids=[1, 2, 3, 4, 5, 6], request_id="req-flow")

        common_blocks, unique_blocks, hit_info = manager.request_block_ids(task, block_size=4, dec_token_num=4)
        self.assertEqual(common_blocks, [])
        self.assertGreater(len(unique_blocks), 0)
        self.assertEqual(hit_info["gpu_cache_blocks"], 0)

    def test_update_cache_blocks_and_request_match(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=8, block_size=4, enable_mm=True)
        prompt_ids = paddle.to_tensor([1, 2, 3, 4], dtype="int64").numpy()
        task = DummyRequest("req-update", prompt_ids, block_tables=[0, 1])
        manager.cache_info[task.request_id] = (manager.radix_tree_root, 0)
        manager.req_leaf_map[task.request_id] = manager.radix_tree_root
        manager.leaf_req_map[manager.radix_tree_root].add(task.request_id)

        manager.update_cache_blocks(task, block_size=4, num_computed_tokens=4)
        self.assertEqual(task.cached_block_num, 1)
        self.assertIn(task.request_id, manager.req_leaf_map)

        match_task = DummyRequest("req-match", np.array([1, 2, 3, 4], dtype=np.int64))
        match_task.output_token_ids = []
        common_blocks, matched_token_num, hit_info = manager.request_match_blocks(match_task, block_size=4)
        self.assertEqual(common_blocks, [0])
        self.assertEqual(matched_token_num, 4)
        self.assertEqual(hit_info["gpu_cache_blocks"], 1)

    def test_release_block_ids_paths(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=6)
        node = BlockNode(
            node_id=1,
            input_ids=[1, 2, 3, 4],
            input_hash_value=0,
            depth=1,
            block_id=0,
            token_num=4,
            hash_value="hash-release",
            last_used_time=time.time(),
            parent=manager.radix_tree_root,
            shared_count=1,
        )
        node.req_id_set.add("req-release")
        manager.node_map[node.node_id] = node
        manager.radix_tree_root.children[node.hash_value] = node
        manager.req_leaf_map["req-release"] = node
        manager.leaf_req_map[node].add("req-release")
        manager.cache_info["req-release"] = (node, 0)

        task = SimpleNamespace(request_id="req-release")
        manager.release_block_ids(task)
        self.assertIn(node, manager.gpu_lru_leaf_set)

        root_task = SimpleNamespace(request_id="req-root")
        manager.req_leaf_map["req-root"] = manager.radix_tree_root
        manager.unfilled_req_block_map["req-root"] = [1]
        manager.release_block_ids(root_task)
        self.assertNotIn("req-root", manager.unfilled_req_block_map)

        async_manager = make_prefix_cache_manager(num_gpu_blocks_override=6)
        async_node = BlockNode(
            node_id=2,
            input_ids=[1, 2, 3, 4],
            input_hash_value=0,
            depth=1,
            block_id=1,
            token_num=4,
            hash_value="hash-async",
            last_used_time=time.time(),
            parent=async_manager.radix_tree_root,
            shared_count=1,
        )
        async_node.req_id_set.add("req-async")
        async_manager.node_map[async_node.node_id] = async_node
        async_manager.radix_tree_root.children[async_node.hash_value] = async_node
        async_manager.req_leaf_map["req-async"] = async_node
        async_manager.leaf_req_map[async_node].add("req-async")
        async_manager.cache_info["req-async"] = (async_node, 0)
        future = async_manager.release_block_ids_async(SimpleNamespace(request_id="req-async"))
        future.result(timeout=2)

    def test_free_block_ids_and_cpu_free(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=6)
        node = BlockNode(
            node_id=2,
            input_ids=[1, 2, 3, 4],
            input_hash_value=0,
            depth=1,
            block_id=2,
            token_num=4,
            hash_value="hash-free",
            last_used_time=time.time(),
            parent=manager.radix_tree_root,
            shared_count=0,
        )
        manager.node_map[node.node_id] = node
        manager.radix_tree_root.children[node.hash_value] = node
        manager.gpu_lru_leaf_set.add(node)
        manager.gpu_lru_leaf_heap.append(node)
        manager.free_block_ids_async(need_block_num=1)
        manager.free_block_ids(need_block_num=1)
        self.assertIsNone(manager.gpu_free_task_future)

        cpu_manager = make_prefix_cache_manager(num_gpu_blocks_override=6, num_cpu_blocks=2)
        cpu_node = BlockNode(
            node_id=3,
            input_ids=[1, 2, 3, 4],
            input_hash_value=0,
            depth=1,
            block_id=0,
            token_num=4,
            hash_value="hash-cpu",
            last_used_time=time.time(),
            parent=cpu_manager.radix_tree_root,
            shared_count=0,
            cache_status=CacheStatus.CPU,
        )
        cpu_manager.node_map[cpu_node.node_id] = cpu_node
        cpu_manager.radix_tree_root.children[cpu_node.hash_value] = cpu_node
        cpu_manager.cpu_lru_leaf_set.add(cpu_node)
        cpu_manager.cpu_lru_leaf_heap.append(cpu_node)
        freed = cpu_manager.free_cpu_block_ids(1)
        self.assertEqual(freed, 1)

    def test_evict_cache_async_and_update_matched_info(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=6, num_cpu_blocks=2)
        manager.cpu_free_block_list = [0, 1]
        hash_value_gpu_block_ids_map = {"hash": [2]}
        hash_value_block_ids_map = {"hash": [2]}
        hash_value_swap_node_ids_map = {"hash": [10]}
        hash_value_input_ids_map = {"hash": [1, 2]}
        hash_value_depth_map = {"hash": 1}
        node = BlockNode(
            node_id=7,
            input_ids=[1, 2, 3, 4],
            input_hash_value=0,
            depth=1,
            block_id=2,
            token_num=4,
            hash_value="hash",
            last_used_time=time.time(),
            parent=manager.radix_tree_root,
            shared_count=0,
            reverved_dec_block_ids=[3],
        )
        need_recycle_gpu_block_ids = []
        hash_value_gpu_block_ids_map = defaultdict(list)
        hash_value_swap_node_ids_map = defaultdict(list)
        manager._handle_free_gpu_node_with_cpu(
            node,
            hash_value_input_ids_map,
            hash_value_depth_map,
            need_recycle_gpu_block_ids,
            hash_value_gpu_block_ids_map,
            hash_value_swap_node_ids_map,
        )
        self.assertEqual(node.reverved_dec_block_ids, [])
        self.assertEqual(need_recycle_gpu_block_ids, [2])

        hash_value_gpu_block_ids_map = {"hash": [2]}
        hash_value_swap_node_ids_map = {"hash": [10]}
        with mock.patch.object(manager, "issue_swap_task") as issue_swap_task:
            manager._evict_cache_async(
                None,
                1,
                hash_value_gpu_block_ids_map,
                hash_value_block_ids_map,
                hash_value_swap_node_ids_map,
                hash_value_input_ids_map,
                hash_value_depth_map,
            )
        issue_swap_task.assert_called_once()

        node = BlockNode(
            node_id=4,
            input_ids=[1, 2, 3, 4],
            input_hash_value=0,
            depth=1,
            block_id=1,
            token_num=4,
            hash_value="hash-info",
            last_used_time=time.time(),
            parent=manager.radix_tree_root,
            shared_count=0,
        )
        manager._update_matched_node_info("req-info", node, current_time=time.time())
        self.assertIn("req-info", node.req_id_set)

    def test_handle_swap_result_edge_cases(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=6)
        manager._handle_swap_result(None, 0, 0, CacheStatus.SWAP2CPU)
        node = BlockNode(
            node_id=6,
            input_ids=[1, 2, 3, 4],
            input_hash_value=0,
            depth=1,
            block_id=1,
            token_num=4,
            hash_value="hash-unexpected",
            last_used_time=time.time(),
            parent=manager.radix_tree_root,
            shared_count=0,
        )
        manager.node_map[node.node_id] = node
        manager._handle_swap_result(
            node.node_id,
            0,
            0,
            SimpleNamespace(value=999),
        )

    def test_recv_data_transfer_result(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=6, num_cpu_blocks=2)
        node = BlockNode(
            node_id=5,
            input_ids=[1, 2, 3, 4],
            input_hash_value=0,
            depth=1,
            block_id=1,
            token_num=4,
            hash_value="hash-recv",
            last_used_time=time.time(),
            parent=manager.radix_tree_root,
            shared_count=0,
            cache_status=CacheStatus.SWAP2GPU,
        )
        manager.node_map[node.node_id] = node
        transfer_id = "transfer-1"
        manager.task_swapping_event[transfer_id] = Event()

        class DummyQueue:
            def __init__(self):
                self.calls = 0

            def get_transfer_done_signal(self):
                self.calls += 1
                if self.calls == 1:
                    return ([node.node_id], [1], [0], CacheStatus.SWAP2GPU, transfer_id)
                raise RuntimeError("stop")

        manager.cache_task_queue = DummyQueue()
        with self.assertRaises(RuntimeError):
            manager.recv_data_transfer_result()
        self.assertTrue(manager.task_swapping_event[transfer_id].is_set())

    def test_reset_noop_with_empty_state(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=6, num_cpu_blocks=2)
        manager.reset()
        self.assertEqual(manager.node_map, {})

    def test_clear_prefix_cache_branches(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=6)

        class DummySignal:
            def __init__(self, value):
                self.value = value

        manager.prefix_tree_status_signal = DummySignal([PrefixTreeStatus.CLEARING])
        with (
            mock.patch.object(manager, "reset") as reset_mock,
            mock.patch("fastdeploy.cache_manager.prefix_cache_manager.time.sleep", side_effect=RuntimeError("stop")),
        ):
            with self.assertRaises(RuntimeError):
                manager.clear_prefix_cache()
        reset_mock.assert_called_once()
        self.assertEqual(manager.prefix_tree_status_signal.value[0], PrefixTreeStatus.CLEARED)

        manager.prefix_tree_status_signal = DummySignal([PrefixTreeStatus.UPDATING])
        with mock.patch("fastdeploy.cache_manager.prefix_cache_manager.time.sleep", side_effect=RuntimeError("stop")):
            with self.assertRaises(RuntimeError):
                manager.clear_prefix_cache()
        self.assertEqual(manager.prefix_tree_status_signal.value[0], PrefixTreeStatus.NORMAL)

    def test_free_nodes_directly_and_handle_swap_result(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=6, num_cpu_blocks=2)
        node = BlockNode(
            node_id=1,
            input_ids=[1, 2, 3, 4],
            input_hash_value=0,
            depth=1,
            block_id=1,
            token_num=4,
            hash_value="hash",
            last_used_time=time.time(),
            parent=manager.radix_tree_root,
            shared_count=0,
            reverved_dec_block_ids=[2],
        )
        manager.node_map[node.node_id] = node
        manager.radix_tree_root.children[node.hash_value] = node
        manager.gpu_lru_leaf_set.add(node)
        manager.gpu_lru_leaf_heap.append(node)

        manager.free_nodes_directly(node)
        self.assertNotIn(node.node_id, manager.node_map)
        self.assertNotIn(node.hash_value, manager.radix_tree_root.children)
        self.assertIn(1, manager.gpu_free_block_list)
        self.assertIn(2, manager.gpu_free_block_list)

        parent_node = BlockNode(
            node_id=2,
            input_ids=[1, 2, 3, 4],
            input_hash_value=0,
            depth=1,
            block_id=3,
            token_num=4,
            hash_value="hash-parent",
            last_used_time=time.time(),
            parent=manager.radix_tree_root,
            shared_count=0,
        )
        leaf_node = BlockNode(
            node_id=3,
            input_ids=[1, 2, 3, 4],
            input_hash_value=0,
            depth=2,
            block_id=4,
            token_num=4,
            hash_value="hash-child",
            last_used_time=time.time(),
            parent=parent_node,
            shared_count=0,
        )
        manager.node_map[parent_node.node_id] = parent_node
        manager.node_map[leaf_node.node_id] = leaf_node
        manager.radix_tree_root.children[parent_node.hash_value] = parent_node
        parent_node.children[leaf_node.hash_value] = leaf_node
        manager.gpu_lru_leaf_set.add(leaf_node)
        manager.gpu_lru_leaf_heap.append(leaf_node)
        manager.free_nodes_directly(leaf_node)
        self.assertNotIn(parent_node.node_id, manager.node_map)
        self.assertNotIn(parent_node.hash_value, manager.radix_tree_root.children)

        gpu_node = BlockNode(
            node_id=10,
            input_ids=[1, 2, 3, 4],
            input_hash_value=0,
            depth=1,
            block_id=3,
            token_num=4,
            hash_value="hash-3",
            last_used_time=time.time(),
            parent=manager.radix_tree_root,
            shared_count=0,
        )
        manager.node_map[gpu_node.node_id] = gpu_node
        manager._handle_swap_result(gpu_node.node_id, 3, 0, CacheStatus.SWAP2CPU)
        self.assertIn(0, manager.cpu_free_block_list)
        self.assertEqual(gpu_node.cache_status, CacheStatus.GPU)

        cpu_node = BlockNode(
            node_id=11,
            input_ids=[1, 2, 3, 4],
            input_hash_value=0,
            depth=1,
            block_id=4,
            token_num=4,
            hash_value="hash-4",
            last_used_time=time.time(),
            parent=manager.radix_tree_root,
            shared_count=0,
            cache_status=CacheStatus.SWAP2CPU,
        )
        manager.node_map[cpu_node.node_id] = cpu_node
        manager._handle_swap_result(cpu_node.node_id, 4, 1, CacheStatus.SWAP2CPU)
        self.assertEqual(cpu_node.cache_status, CacheStatus.CPU)
        self.assertEqual(cpu_node.block_id, 1)
        self.assertIn(cpu_node, manager.cpu_lru_leaf_set)

        manager._handle_swap_result(cpu_node.node_id, 2, 0, CacheStatus.SWAP2GPU)
        self.assertEqual(cpu_node.cache_status, CacheStatus.GPU)
        self.assertIn(0, manager.cpu_free_block_list)

    def test_reset_clears_cache_state(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=6, num_cpu_blocks=2)
        node = BlockNode(1, [], 0, 0, 1, 0, None, time.time(), parent=manager.radix_tree_root, shared_count=0)
        manager.node_map[node.node_id] = node
        manager.req_leaf_map["req"] = node
        manager.leaf_req_map[node].add("req")
        manager.unfilled_req_block_map["req"] = [1]
        manager.cache_info["req"] = (node, 0)
        manager.gpu_lru_leaf_heap.append(node)
        manager.gpu_lru_leaf_set.add(node)
        manager.cpu_lru_leaf_heap.append(node)
        manager.cpu_lru_leaf_set.add(node)

        future = Future()
        future.set_result(None)
        manager.gpu_free_task_future = future
        event = manager.task_swapping_event["task"] = mock.Mock()
        event.wait.return_value = None

        manager.reset()
        self.assertEqual(manager.node_map, {})
        self.assertEqual(manager.req_leaf_map, {})
        self.assertEqual(manager.leaf_req_map, {})
        self.assertEqual(manager.unfilled_req_block_map, {})
        self.assertEqual(manager.cache_info, {})
        self.assertEqual(len(manager.gpu_free_block_list), manager.num_gpu_blocks)
        self.assertEqual(manager.radix_tree_root.node_id, -1)

    def test_launch_cache_manager_with_mocks(self):
        manager = make_prefix_cache_manager(num_gpu_blocks_override=6, splitwise_role="cache")
        cache_cfg = manager.cache_config

        class DummySignal:
            def __init__(self, name=None, array=None, dtype=None, suffix=None, create=None, **kwargs):
                self.name = name
                self.value = array
                if name in {"cache_ready_signal", "swap_space_ready_signal"}:
                    self.value[:] = 1

        class DummyQueue:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class DummyProcess:
            def poll(self):
                return None

        with (
            mock.patch("fastdeploy.cache_manager.prefix_cache_manager.IPCSignal", DummySignal),
            mock.patch("fastdeploy.cache_manager.prefix_cache_manager.EngineCacheQueue", DummyQueue),
            mock.patch("fastdeploy.cache_manager.prefix_cache_manager.get_all_visible_devices", return_value=""),
            mock.patch("fastdeploy.cache_manager.prefix_cache_manager.subprocess.Popen", return_value=DummyProcess()),
            mock.patch("fastdeploy.cache_manager.prefix_cache_manager.time.sleep", return_value=None),
            mock.patch.object(PrefixCacheManager, "_get_kv_cache_shape", return_value=([1, 2], [3, 4])),
        ):
            processes = manager.launch_cache_manager(
                cache_cfg,
                tensor_parallel_size=2,
                device_ids=[0, 1],
                pod_ip="127.0.0.1",
                engine_worker_queue_port=1234,
                pid_suffix="pid",
                create_cache_tensor=False,
            )
        self.assertEqual(len(processes), 4)


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
