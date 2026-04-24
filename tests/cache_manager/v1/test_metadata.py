"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

Unit tests for data classes and enums in metadata.py.

Tests cover:
- BlockNode: add_child, remove_child, update_access, is_leaf, is_root,
             is_on_device, is_on_host, is_swapping, increment_ref, decrement_ref, touch
- RadixTreeStats: evictable_count property, to_dict
- MatchResult: device_block_ids, total_matched_blocks, matched_*_nums
- CacheSwapMetadata: is_success, mapping property
- AsyncTaskHandler: wait, cancel, get_result, set_result, set_error
"""

import threading
import time
import unittest

from fastdeploy.cache_manager.v1.metadata import (
    AsyncTaskHandler,
    BlockNode,
    CacheLevel,
    CacheStatus,
    CacheSwapMetadata,
    MatchResult,
    RadixTreeStats,
)

# ---------------------------------------------------------------------------
# BlockNode
# ---------------------------------------------------------------------------


class TestBlockNodeChildManagement(unittest.TestCase):
    def test_add_child_appends_id(self):
        node = BlockNode()
        node.add_child(5)
        self.assertIn(5, node.children_ids)

    def test_add_child_deduplicates(self):
        node = BlockNode()
        node.add_child(5)
        node.add_child(5)
        self.assertEqual(node.children_ids.count(5), 1)

    def test_remove_child_returns_true_when_found(self):
        node = BlockNode()
        node.add_child(7)
        result = node.remove_child(7)
        self.assertTrue(result)
        self.assertNotIn(7, node.children_ids)

    def test_remove_child_returns_false_when_not_found(self):
        node = BlockNode()
        result = node.remove_child(99)
        self.assertFalse(result)

    def test_add_multiple_children(self):
        node = BlockNode()
        for i in range(5):
            node.add_child(i)
        self.assertEqual(len(node.children_ids), 5)


class TestBlockNodeRefCount(unittest.TestCase):
    def test_increment_ref_increases_count(self):
        node = BlockNode(ref_count=0)
        new_count = node.increment_ref()
        self.assertEqual(new_count, 1)
        self.assertEqual(node.ref_count, 1)

    def test_decrement_ref_decreases_count(self):
        node = BlockNode(ref_count=2)
        new_count = node.decrement_ref()
        self.assertEqual(new_count, 1)

    def test_decrement_ref_does_not_go_below_zero(self):
        node = BlockNode(ref_count=0)
        new_count = node.decrement_ref()
        self.assertEqual(new_count, 0)


class TestBlockNodeUpdateAccess(unittest.TestCase):
    def test_update_access_positive_delta_increments(self):
        node = BlockNode(ref_count=1)
        node.update_access(delta_ref=2)
        self.assertEqual(node.ref_count, 3)

    def test_update_access_negative_delta_decrements(self):
        node = BlockNode(ref_count=5)
        node.update_access(delta_ref=-3)
        self.assertEqual(node.ref_count, 2)

    def test_update_access_clamps_at_zero(self):
        node = BlockNode(ref_count=1)
        node.update_access(delta_ref=-10)
        self.assertEqual(node.ref_count, 0)

    def test_update_access_updates_last_access_time(self):
        node = BlockNode()
        old_time = node.last_access_time
        time.sleep(0.01)
        node.update_access(delta_ref=0)
        self.assertGreaterEqual(node.last_access_time, old_time)

    def test_update_access_zero_delta_only_touches(self):
        node = BlockNode(ref_count=3)
        node.update_access(delta_ref=0)
        self.assertEqual(node.ref_count, 3)


class TestBlockNodeStatusChecks(unittest.TestCase):
    def test_is_leaf_no_children(self):
        node = BlockNode()
        self.assertTrue(node.is_leaf())

    def test_is_leaf_with_children_ids(self):
        node = BlockNode()
        node.add_child(1)
        self.assertFalse(node.is_leaf())

    def test_is_leaf_with_children_dict(self):
        node = BlockNode()
        child = BlockNode()
        node.children["key"] = child
        self.assertFalse(node.is_leaf())

    def test_is_root_no_parent(self):
        node = BlockNode()
        self.assertTrue(node.is_root())

    def test_is_root_with_parent(self):
        parent = BlockNode()
        child = BlockNode(parent=parent)
        self.assertFalse(child.is_root())

    def test_is_on_device_default(self):
        node = BlockNode(cache_status=CacheStatus.DEVICE)
        self.assertTrue(node.is_on_device())
        self.assertFalse(node.is_on_host())
        self.assertFalse(node.is_swapping())

    def test_is_on_host(self):
        node = BlockNode(cache_status=CacheStatus.HOST)
        self.assertTrue(node.is_on_host())
        self.assertFalse(node.is_on_device())
        self.assertFalse(node.is_swapping())

    def test_is_swapping_swap_to_host(self):
        node = BlockNode(cache_status=CacheStatus.SWAP_TO_HOST)
        self.assertTrue(node.is_swapping())

    def test_is_swapping_swap_to_device(self):
        node = BlockNode(cache_status=CacheStatus.SWAP_TO_DEVICE)
        self.assertTrue(node.is_swapping())

    def test_is_swapping_deleting(self):
        node = BlockNode(cache_status=CacheStatus.DELETING)
        self.assertTrue(node.is_swapping())


class TestBlockNodeTouch(unittest.TestCase):
    def test_touch_updates_last_access_time(self):
        node = BlockNode()
        old_time = node.last_access_time
        time.sleep(0.01)
        node.touch()
        self.assertGreater(node.last_access_time, old_time)


# ---------------------------------------------------------------------------
# RadixTreeStats
# ---------------------------------------------------------------------------


class TestRadixTreeStats(unittest.TestCase):
    def test_evictable_count_is_sum(self):
        stats = RadixTreeStats(
            node_count=10,
            evictable_device_count=3,
            evictable_host_count=4,
        )
        self.assertEqual(stats.evictable_count, 7)

    def test_evictable_count_zero_when_both_zero(self):
        stats = RadixTreeStats()
        self.assertEqual(stats.evictable_count, 0)

    def test_to_dict_keys(self):
        stats = RadixTreeStats(node_count=5, evictable_device_count=2, evictable_host_count=1)
        d = stats.to_dict()
        self.assertIn("node_count", d)
        self.assertIn("evictable_device_count", d)
        self.assertIn("evictable_host_count", d)
        self.assertIn("evictable_count", d)

    def test_to_dict_values(self):
        stats = RadixTreeStats(node_count=5, evictable_device_count=2, evictable_host_count=3)
        d = stats.to_dict()
        self.assertEqual(d["node_count"], 5)
        self.assertEqual(d["evictable_device_count"], 2)
        self.assertEqual(d["evictable_host_count"], 3)
        self.assertEqual(d["evictable_count"], 5)


# ---------------------------------------------------------------------------
# MatchResult
# ---------------------------------------------------------------------------


class TestMatchResult(unittest.TestCase):
    def _make_node(self, block_id: int) -> BlockNode:
        return BlockNode(block_id=block_id)

    def test_device_block_ids_extracts_ids(self):
        nodes = [self._make_node(1), self._make_node(2), self._make_node(3)]
        result = MatchResult(device_nodes=nodes)
        self.assertEqual(result.device_block_ids, [1, 2, 3])

    def test_matched_device_nums(self):
        result = MatchResult(device_nodes=[self._make_node(0)] * 4)
        self.assertEqual(result.matched_device_nums, 4)

    def test_matched_host_nums(self):
        result = MatchResult(host_nodes=[self._make_node(0)] * 3)
        self.assertEqual(result.matched_host_nums, 3)

    def test_matched_storage_nums(self):
        result = MatchResult(storage_nodes=[self._make_node(0)] * 2)
        self.assertEqual(result.matched_storage_nums, 2)

    def test_total_matched_blocks(self):
        result = MatchResult(
            device_nodes=[self._make_node(0)] * 2,
            host_nodes=[self._make_node(0)] * 3,
            storage_nodes=[self._make_node(0)] * 1,
        )
        self.assertEqual(result.total_matched_blocks, 6)

    def test_empty_match_result(self):
        result = MatchResult()
        self.assertEqual(result.device_block_ids, [])
        self.assertEqual(result.total_matched_blocks, 0)


# ---------------------------------------------------------------------------
# CacheSwapMetadata
# ---------------------------------------------------------------------------


class TestCacheSwapMetadata(unittest.TestCase):
    def test_is_success_true(self):
        meta = CacheSwapMetadata(
            src_block_ids=[0, 1],
            dst_block_ids=[10, 11],
            success=True,
        )
        self.assertTrue(meta.is_success())

    def test_is_success_false(self):
        meta = CacheSwapMetadata(success=False)
        self.assertFalse(meta.is_success())

    def test_mapping_returns_dict_when_success(self):
        meta = CacheSwapMetadata(
            src_block_ids=[0, 1, 2],
            dst_block_ids=[10, 11, 12],
            success=True,
        )
        self.assertEqual(meta.mapping, {0: 10, 1: 11, 2: 12})

    def test_mapping_returns_empty_when_not_success(self):
        meta = CacheSwapMetadata(
            src_block_ids=[0, 1],
            dst_block_ids=[10, 11],
            success=False,
        )
        self.assertEqual(meta.mapping, {})

    def test_mapping_empty_ids_success_true(self):
        meta = CacheSwapMetadata(src_block_ids=[], dst_block_ids=[], success=True)
        self.assertEqual(meta.mapping, {})

    def test_cache_level_fields(self):
        meta = CacheSwapMetadata(
            src_type=CacheLevel.DEVICE,
            dst_type=CacheLevel.HOST,
            success=True,
        )
        self.assertEqual(meta.src_type, CacheLevel.DEVICE)
        self.assertEqual(meta.dst_type, CacheLevel.HOST)


# ---------------------------------------------------------------------------
# AsyncTaskHandler
# ---------------------------------------------------------------------------


class TestAsyncTaskHandler(unittest.TestCase):
    def test_set_result_marks_completed(self):
        handler = AsyncTaskHandler()
        handler.set_result(42)
        self.assertTrue(handler.is_completed)
        self.assertEqual(handler.result, 42)
        self.assertIsNone(handler.error)

    def test_set_error_marks_completed(self):
        handler = AsyncTaskHandler()
        handler.set_error("something went wrong")
        self.assertTrue(handler.is_completed)
        self.assertEqual(handler.error, "something went wrong")

    def test_get_result_returns_result(self):
        handler = AsyncTaskHandler()
        handler.set_result("hello")
        self.assertEqual(handler.get_result(), "hello")

    def test_get_result_raises_on_error(self):
        handler = AsyncTaskHandler()
        handler.set_error("failed")
        with self.assertRaises(RuntimeError) as ctx:
            handler.get_result()
        self.assertIn("failed", str(ctx.exception))

    def test_cancel_before_completion(self):
        handler = AsyncTaskHandler()
        result = handler.cancel()
        self.assertTrue(result)
        self.assertTrue(handler.is_completed)
        self.assertEqual(handler.error, "Task cancelled")

    def test_cancel_after_completion_returns_false(self):
        handler = AsyncTaskHandler()
        handler.set_result(1)
        result = handler.cancel()
        self.assertFalse(result)

    def test_wait_returns_true_when_already_done(self):
        handler = AsyncTaskHandler()
        handler.set_result(True)
        result = handler.wait(timeout=1.0)
        self.assertTrue(result)

    def test_wait_timeout_returns_false_when_not_done(self):
        handler = AsyncTaskHandler()
        # Do not call set_result – wait should time out
        result = handler.wait(timeout=0.05)
        self.assertFalse(result)

    def test_wait_unblocks_after_set_result(self):
        handler = AsyncTaskHandler()

        def _complete():
            time.sleep(0.05)
            handler.set_result("done")

        t = threading.Thread(target=_complete)
        t.start()
        result = handler.wait(timeout=2.0)
        t.join()
        self.assertTrue(result)

    def test_get_result_blocks_until_ready(self):
        handler = AsyncTaskHandler()

        def _complete():
            time.sleep(0.05)
            handler.set_result(999)

        t = threading.Thread(target=_complete)
        t.start()
        val = handler.get_result()
        t.join()
        self.assertEqual(val, 999)

    def test_task_id_is_unique(self):
        ids = {AsyncTaskHandler().task_id for _ in range(20)}
        self.assertEqual(len(ids), 20)


if __name__ == "__main__":
    unittest.main()
