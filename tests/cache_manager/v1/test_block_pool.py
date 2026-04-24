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

Unit tests for BlockPool, DeviceBlockPool, and HostBlockPool.

Tests cover:
- allocate / release basic operations
- get_metadata / set_metadata
- resize (expand, shrink, fail when used > new_size)
- available_blocks / used_blocks / reset / get_stats
- DeviceBlockPool and HostBlockPool subclass-specific behavior
"""

import unittest

from fastdeploy.cache_manager.v1.block_pool import DeviceBlockPool, HostBlockPool
from fastdeploy.cache_manager.v1.metadata import CacheBlockMetadata


def _make_device_pool(num_blocks: int = 10, block_size: int = 64) -> DeviceBlockPool:
    return DeviceBlockPool(num_blocks=num_blocks, block_size=block_size)


def _make_host_pool(
    num_blocks: int = 10,
    block_size: int = 64,
    use_pinned_memory: bool = True,
) -> HostBlockPool:
    return HostBlockPool(num_blocks=num_blocks, block_size=block_size, use_pinned_memory=use_pinned_memory)


def _make_metadata(block_id: int = 0) -> CacheBlockMetadata:
    return CacheBlockMetadata(block_id=block_id, device_id=0, block_size=64)


# ---------------------------------------------------------------------------
# BlockPool – metadata
# ---------------------------------------------------------------------------


class TestBlockPoolMetadata(unittest.TestCase):
    """Tests for get_metadata / set_metadata."""

    def test_get_metadata_returns_none_by_default(self):
        pool = _make_device_pool()
        self.assertIsNone(pool.get_metadata(0))

    def test_set_then_get_metadata(self):
        pool = _make_device_pool()
        meta = _make_metadata(block_id=3)
        pool.set_metadata(3, meta)
        result = pool.get_metadata(3)
        self.assertIs(result, meta)

    def test_set_metadata_overwrites_previous(self):
        pool = _make_device_pool()
        meta1 = _make_metadata(block_id=5)
        meta2 = _make_metadata(block_id=5)
        meta2.ref_count = 99
        pool.set_metadata(5, meta1)
        pool.set_metadata(5, meta2)
        self.assertEqual(pool.get_metadata(5).ref_count, 99)

    def test_metadata_cleared_on_release(self):
        pool = _make_device_pool()
        block_ids = pool.allocate(1)
        block_id = block_ids[0]
        pool.set_metadata(block_id, _make_metadata(block_id))
        pool.release([block_id])
        self.assertIsNone(pool.get_metadata(block_id))

    def test_get_metadata_unknown_block_returns_none(self):
        pool = _make_device_pool()
        self.assertIsNone(pool.get_metadata(999))


# ---------------------------------------------------------------------------
# BlockPool – resize
# ---------------------------------------------------------------------------


class TestBlockPoolResize(unittest.TestCase):
    """Tests for resize (expand / shrink)."""

    def test_resize_expand_adds_free_blocks(self):
        pool = _make_device_pool(num_blocks=5)
        self.assertEqual(pool.available_blocks(), 5)
        result = pool.resize(10)
        self.assertTrue(result)
        self.assertEqual(pool.num_blocks, 10)
        self.assertEqual(pool.available_blocks(), 10)

    def test_resize_shrink_removes_free_blocks(self):
        pool = _make_device_pool(num_blocks=10)
        result = pool.resize(5)
        self.assertTrue(result)
        self.assertEqual(pool.num_blocks, 5)
        self.assertEqual(pool.available_blocks(), 5)

    def test_resize_shrink_fails_when_too_many_used(self):
        pool = _make_device_pool(num_blocks=10)
        pool.allocate(8)  # 8 used, 2 free
        result = pool.resize(5)  # cannot shrink below 8
        self.assertFalse(result)
        self.assertEqual(pool.num_blocks, 10)  # unchanged

    def test_resize_shrink_clears_metadata_for_removed_blocks(self):
        pool = _make_device_pool(num_blocks=10)
        pool.set_metadata(7, _make_metadata(block_id=7))
        pool.set_metadata(9, _make_metadata(block_id=9))
        pool.resize(6)
        self.assertIsNone(pool.get_metadata(7))
        self.assertIsNone(pool.get_metadata(9))

    def test_resize_to_same_size_is_noop(self):
        pool = _make_device_pool(num_blocks=8)
        result = pool.resize(8)
        self.assertTrue(result)
        self.assertEqual(pool.num_blocks, 8)
        self.assertEqual(pool.available_blocks(), 8)

    def test_resize_expand_keeps_existing_used_blocks(self):
        pool = _make_device_pool(num_blocks=5)
        pool.allocate(3)
        pool.resize(10)
        self.assertEqual(pool.used_blocks(), 3)
        self.assertEqual(pool.available_blocks(), 7)

    def test_resize_shrink_to_zero_when_no_used(self):
        pool = _make_device_pool(num_blocks=5)
        result = pool.resize(0)
        self.assertTrue(result)
        self.assertEqual(pool.num_blocks, 0)
        self.assertEqual(pool.available_blocks(), 0)

    def test_resize_shrink_fails_below_used(self):
        pool = _make_device_pool(num_blocks=10)
        pool.allocate(6)
        # Shrink to 4 is impossible (6 used)
        result = pool.resize(4)
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# BlockPool – basic ops already indirectly tested; add direct coverage
# ---------------------------------------------------------------------------


class TestBlockPoolBasicOps(unittest.TestCase):
    def test_allocate_zero_returns_empty_list(self):
        pool = _make_device_pool()
        result = pool.allocate(0)
        self.assertEqual(result, [])

    def test_allocate_more_than_available_returns_none(self):
        pool = _make_device_pool(num_blocks=3)
        result = pool.allocate(5)
        self.assertIsNone(result)

    def test_release_updates_free_and_used_counts(self):
        pool = _make_device_pool(num_blocks=10)
        blocks = pool.allocate(4)
        self.assertEqual(pool.used_blocks(), 4)
        pool.release(blocks)
        self.assertEqual(pool.used_blocks(), 0)
        self.assertEqual(pool.available_blocks(), 10)

    def test_reset_restores_all_blocks(self):
        pool = _make_device_pool(num_blocks=10)
        pool.allocate(7)
        pool.set_metadata(0, _make_metadata())
        pool.reset()
        self.assertEqual(pool.available_blocks(), 10)
        self.assertEqual(pool.used_blocks(), 0)
        self.assertIsNone(pool.get_metadata(0))


# ---------------------------------------------------------------------------
# DeviceBlockPool – get_stats
# ---------------------------------------------------------------------------


class TestDeviceBlockPoolStats(unittest.TestCase):
    def test_get_stats_returns_expected_keys(self):
        pool = _make_device_pool(num_blocks=20, block_size=128)
        stats = pool.get_stats()
        self.assertEqual(stats["num_blocks"], 20)
        self.assertEqual(stats["block_size"], 128)
        self.assertEqual(stats["available"], 20)
        self.assertEqual(stats["used"], 0)

    def test_get_stats_reflects_allocation(self):
        pool = _make_device_pool(num_blocks=10)
        pool.allocate(4)
        stats = pool.get_stats()
        self.assertEqual(stats["available"], 6)
        self.assertEqual(stats["used"], 4)


# ---------------------------------------------------------------------------
# HostBlockPool – __init__ and get_stats
# ---------------------------------------------------------------------------


class TestHostBlockPoolInit(unittest.TestCase):
    def test_default_use_pinned_memory_is_true(self):
        pool = _make_host_pool()
        self.assertTrue(pool.use_pinned_memory)

    def test_use_pinned_memory_false(self):
        pool = _make_host_pool(use_pinned_memory=False)
        self.assertFalse(pool.use_pinned_memory)


class TestHostBlockPoolStats(unittest.TestCase):
    def test_get_stats_includes_use_pinned_memory_true(self):
        pool = _make_host_pool(use_pinned_memory=True)
        stats = pool.get_stats()
        self.assertIn("use_pinned_memory", stats)
        self.assertTrue(stats["use_pinned_memory"])

    def test_get_stats_includes_use_pinned_memory_false(self):
        pool = _make_host_pool(use_pinned_memory=False)
        stats = pool.get_stats()
        self.assertFalse(stats["use_pinned_memory"])

    def test_get_stats_base_fields_present(self):
        pool = _make_host_pool(num_blocks=8, block_size=32)
        stats = pool.get_stats()
        self.assertEqual(stats["num_blocks"], 8)
        self.assertEqual(stats["block_size"], 32)
        self.assertIn("available", stats)
        self.assertIn("used", stats)


if __name__ == "__main__":
    unittest.main()
