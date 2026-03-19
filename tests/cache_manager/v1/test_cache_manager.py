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

Unit tests for CacheManager class.

Tests cover:
- Block allocation (device/host)
- Block release (device/host)
- Resource checking (can_allocate_*)
- Free block counting (num_free_*_blocks)
- Reset functionality
- Request lifecycle management
- Prefix matching
"""

import unittest

from utils import get_default_test_fd_config


def create_cache_manager(
    total_block_num: int = 100,
    num_cpu_blocks: int = 50,
    block_size: int = 64,
    enable_prefix_caching: bool = True,
):
    """Helper to create CacheManager with test config."""
    from fastdeploy.cache_manager.v1.cache_manager import CacheManager

    config = get_default_test_fd_config()
    # Set cache_config attributes needed by CacheManager
    config.cache_config.total_block_num = total_block_num
    config.cache_config.num_cpu_blocks = num_cpu_blocks
    config.cache_config.block_size = block_size
    config.cache_config.enable_prefix_caching = enable_prefix_caching

    return CacheManager(config)


class TestCacheManagerAllocation(unittest.TestCase):
    """Test CacheManager block allocation functionality."""

    # ============ Device Block Allocation Tests ============

    def test_allocate_device_blocks_success(self):
        """Test successful device block allocation."""
        cache_manager = create_cache_manager()
        allocated = cache_manager.allocate_device_blocks(10)

        self.assertIsNotNone(allocated)
        self.assertEqual(len(allocated), 10)
        self.assertEqual(len(set(allocated)), 10)  # All unique

    def test_allocate_device_blocks_insufficient(self):
        """Test device block allocation returns None when not enough blocks."""
        cache_manager = create_cache_manager()
        cache_manager.allocate_device_blocks(95)
        allocated = cache_manager.allocate_device_blocks(10)

        self.assertIsNone(allocated)

    def test_allocate_device_blocks_exhausted(self):
        """Test device block allocation returns None when no blocks available."""
        cache_manager = create_cache_manager()
        cache_manager.allocate_device_blocks(100)
        allocated = cache_manager.allocate_device_blocks(1)

        self.assertIsNone(allocated)

    # ============ Host Block Allocation Tests ============

    def test_allocate_host_blocks_success(self):
        """Test successful host block allocation."""
        cache_manager = create_cache_manager()
        allocated = cache_manager.allocate_host_blocks(10)

        self.assertIsNotNone(allocated)
        self.assertEqual(len(allocated), 10)
        self.assertEqual(len(set(allocated)), 10)

    def test_allocate_host_blocks_insufficient(self):
        """Test host block allocation returns None when not enough blocks."""
        cache_manager = create_cache_manager()
        cache_manager.allocate_host_blocks(45)
        allocated = cache_manager.allocate_host_blocks(10)

        self.assertIsNone(allocated)

    # ============ Free Block Count Tests ============

    def test_num_free_device_blocks_initial(self):
        """Test initial free device blocks count."""
        cache_manager = create_cache_manager()
        self.assertEqual(cache_manager.num_free_device_blocks, 100)

    def test_num_free_device_blocks_after_allocation(self):
        """Test free device blocks count after allocation."""
        cache_manager = create_cache_manager()
        cache_manager.allocate_device_blocks(30)
        self.assertEqual(cache_manager.num_free_device_blocks, 70)

    def test_num_free_host_blocks_initial(self):
        """Test initial free host blocks count."""
        cache_manager = create_cache_manager()
        self.assertEqual(cache_manager.num_free_host_blocks, 50)

    def test_num_free_host_blocks_after_allocation(self):
        """Test free host blocks count after allocation."""
        cache_manager = create_cache_manager()
        cache_manager.allocate_host_blocks(20)
        self.assertEqual(cache_manager.num_free_host_blocks, 30)

    # ============ Resource Checking Tests ============

    def test_can_allocate_device_blocks_true(self):
        """Test can_allocate_device_blocks returns True when enough blocks."""
        cache_manager = create_cache_manager()
        self.assertTrue(cache_manager.can_allocate_device_blocks(50))

    def test_can_allocate_device_blocks_false(self):
        """Test can_allocate_device_blocks returns False when not enough blocks."""
        cache_manager = create_cache_manager()
        cache_manager.allocate_device_blocks(95)
        self.assertFalse(cache_manager.can_allocate_device_blocks(10))

    def test_can_allocate_device_blocks_exact(self):
        """Test can_allocate_device_blocks with exact available blocks."""
        cache_manager = create_cache_manager()
        self.assertTrue(cache_manager.can_allocate_device_blocks(100))

    def test_can_allocate_host_blocks_true(self):
        """Test can_allocate_host_blocks returns True when enough blocks."""
        cache_manager = create_cache_manager()
        self.assertTrue(cache_manager.can_allocate_host_blocks(25))

    def test_can_allocate_host_blocks_false(self):
        """Test can_allocate_host_blocks returns False when not enough blocks."""
        cache_manager = create_cache_manager()
        cache_manager.allocate_host_blocks(45)
        self.assertFalse(cache_manager.can_allocate_host_blocks(10))


class TestCacheManagerRelease(unittest.TestCase):
    """Test CacheManager block release functionality."""

    def test_free_device_blocks(self):
        """Test freeing device blocks."""
        cache_manager = create_cache_manager()
        allocated = cache_manager.allocate_device_blocks(10)
        initial_free = cache_manager.num_free_device_blocks

        cache_manager.free_device_blocks(allocated)

        self.assertEqual(cache_manager.num_free_device_blocks, initial_free + 10)

    def test_free_host_blocks(self):
        """Test freeing host blocks."""
        cache_manager = create_cache_manager()
        allocated = cache_manager.allocate_host_blocks(10)
        initial_free = cache_manager.num_free_host_blocks

        cache_manager.free_host_blocks(allocated)

        self.assertEqual(cache_manager.num_free_host_blocks, initial_free + 10)

    def test_free_all_device_blocks(self):
        """Test freeing all device blocks."""
        cache_manager = create_cache_manager()
        cache_manager.allocate_device_blocks(50)

        freed = cache_manager.free_all_device_blocks()

        self.assertEqual(freed, 50)
        self.assertEqual(cache_manager.num_free_device_blocks, 100)

    def test_free_all_host_blocks(self):
        """Test freeing all host blocks."""
        cache_manager = create_cache_manager()
        cache_manager.allocate_host_blocks(25)

        freed = cache_manager.free_all_host_blocks()

        self.assertEqual(freed, 25)
        self.assertEqual(cache_manager.num_free_host_blocks, 50)


class TestCacheManagerReset(unittest.TestCase):
    """Test CacheManager reset functionality."""

    def test_reset_cache(self):
        """Test cache reset functionality."""
        cache_manager = create_cache_manager()
        # Allocate some blocks
        cache_manager.allocate_device_blocks(50)
        cache_manager.allocate_host_blocks(25)

        result = cache_manager.reset_cache()

        self.assertTrue(result)
        self.assertEqual(cache_manager.num_free_device_blocks, 100)
        self.assertEqual(cache_manager.num_free_host_blocks, 50)


class TestCacheManagerResize(unittest.TestCase):
    """Test CacheManager resize functionality."""

    def test_resize_device_pool_expand(self):
        """Test expanding device pool."""
        cache_manager = create_cache_manager(total_block_num=100)

        result = cache_manager.resize_device_pool(150)

        self.assertTrue(result)
        self.assertEqual(cache_manager.num_gpu_blocks, 150)
        self.assertEqual(cache_manager.num_free_device_blocks, 150)

    def test_resize_device_pool_shrink(self):
        """Test shrinking device pool when no blocks are used."""
        cache_manager = create_cache_manager(total_block_num=100)

        result = cache_manager.resize_device_pool(50)

        self.assertTrue(result)
        self.assertEqual(cache_manager.num_gpu_blocks, 50)
        self.assertEqual(cache_manager.num_free_device_blocks, 50)

    def test_resize_device_pool_shrink_with_used_blocks(self):
        """Test shrinking device pool fails when used blocks exceed new size."""
        cache_manager = create_cache_manager(total_block_num=100)
        # Allocate 60 blocks
        cache_manager.allocate_device_blocks(60)

        # Try to shrink to 50 - should fail since 60 blocks are used
        result = cache_manager.resize_device_pool(50)

        self.assertFalse(result)
        # Original state should be preserved
        self.assertEqual(cache_manager.num_gpu_blocks, 100)
        self.assertEqual(cache_manager.num_free_device_blocks, 40)

    def test_resize_device_pool_shrink_to_exact_used(self):
        """Test shrinking device pool to exact number of used blocks."""
        cache_manager = create_cache_manager(total_block_num=100)
        # Allocate 50 blocks
        cache_manager.allocate_device_blocks(50)

        # Shrink to exactly 50 - should succeed
        result = cache_manager.resize_device_pool(50)

        self.assertTrue(result)
        self.assertEqual(cache_manager.num_gpu_blocks, 50)
        self.assertEqual(cache_manager.num_free_device_blocks, 0)

    def test_resize_device_pool_allocate_after_expand(self):
        """Test allocating blocks after expanding pool."""
        cache_manager = create_cache_manager(total_block_num=100)

        # Expand pool
        cache_manager.resize_device_pool(150)

        # Should be able to allocate 120 blocks now
        allocated = cache_manager.allocate_device_blocks(120)
        self.assertIsNotNone(allocated)
        self.assertEqual(len(allocated), 120)
        self.assertEqual(cache_manager.num_free_device_blocks, 30)


class TestCacheManagerProperties(unittest.TestCase):
    """Test CacheManager properties."""

    def test_device_pool_property(self):
        """Test device_pool property returns correct pool."""
        from fastdeploy.cache_manager.v1.block_pool import DeviceBlockPool

        cache_manager = create_cache_manager()
        self.assertIsInstance(cache_manager.device_pool, DeviceBlockPool)

    def test_host_pool_property(self):
        """Test host_pool property returns correct pool."""
        from fastdeploy.cache_manager.v1.block_pool import HostBlockPool

        cache_manager = create_cache_manager()
        self.assertIsInstance(cache_manager.host_pool, HostBlockPool)

    def test_radix_tree_property(self):
        """Test radix_tree property returns correct tree."""
        from fastdeploy.cache_manager.v1.radix_tree import RadixTree

        cache_manager = create_cache_manager()
        self.assertIsInstance(cache_manager.radix_tree, RadixTree)


class TestCacheManagerWithDisabledPrefixCaching(unittest.TestCase):
    """Test CacheManager with prefix caching disabled."""

    def test_radix_tree_none_when_disabled(self):
        """Test radix_tree is None when prefix caching disabled."""
        cache_manager = create_cache_manager(enable_prefix_caching=False)
        self.assertIsNone(cache_manager.radix_tree)

    def test_allocation_works_without_prefix_caching(self):
        """Test block allocation still works without prefix caching."""
        cache_manager = create_cache_manager(enable_prefix_caching=False)
        allocated = cache_manager.allocate_device_blocks(10)
        self.assertIsNotNone(allocated)
        self.assertEqual(len(allocated), 10)


class TestCacheManagerWithNoHostCache(unittest.TestCase):
    """Test CacheManager with no host cache."""

    def test_host_cache_disabled(self):
        """Test host cache is disabled."""
        cache_manager = create_cache_manager(num_cpu_blocks=0)
        self.assertFalse(cache_manager.enable_host_cache)

    def test_num_free_host_blocks_zero(self):
        """Test no free host blocks when disabled."""
        cache_manager = create_cache_manager(num_cpu_blocks=0)
        self.assertEqual(cache_manager.num_free_host_blocks, 0)

    def test_can_allocate_host_blocks_false(self):
        """Test cannot allocate host blocks when disabled."""
        cache_manager = create_cache_manager(num_cpu_blocks=0)
        self.assertFalse(cache_manager.can_allocate_host_blocks(1))


class TestCacheManagerRequestLifecycle(unittest.TestCase):
    """Test CacheManager request lifecycle management."""

    def test_update_on_request_finish(self):
        """Test updating cache state on request finish."""
        cache_manager = create_cache_manager()
        block_hashes = ["hash1", "hash2", "hash3"]
        device_block_ids = [1, 2, 3]

        cache_manager.update_on_request_finish(
            block_hashes=block_hashes, device_block_ids=device_block_ids, request_id="test_request"
        )

        # Verify blocks are tracked
        result = cache_manager.match_prefix(block_hashes)
        self.assertEqual(result.total_matched_blocks, 3)

    def test_release_request_blocks(self):
        """Test releasing blocks for a specific request."""
        cache_manager = create_cache_manager()
        # First allocate blocks from the pool
        allocated = cache_manager.allocate_device_blocks(2)
        self.assertIsNotNone(allocated)

        block_hashes = ["hash1", "hash2"]
        device_block_ids = allocated

        cache_manager.update_on_request_finish(
            block_hashes=block_hashes, device_block_ids=device_block_ids, request_id="test_request"
        )

        initial_free = cache_manager.num_free_device_blocks

        cache_manager.release_request_blocks("test_request")

        # Blocks should be freed
        self.assertEqual(cache_manager.num_free_device_blocks, initial_free + 2)


class TestCacheManagerStats(unittest.TestCase):
    """Test CacheManager statistics methods."""

    def test_get_stats(self):
        """Test get_stats returns correct structure."""
        cache_manager = create_cache_manager()
        stats = cache_manager.get_stats()

        self.assertIn("initialized", stats)
        self.assertIn("num_gpu_blocks", stats)
        self.assertIn("num_cpu_blocks", stats)
        self.assertIn("block_size", stats)
        self.assertIn("device_pool", stats)
        self.assertIn("host_pool", stats)
        self.assertIn("num_free_device_blocks", stats)
        self.assertIn("num_free_host_blocks", stats)

        self.assertTrue(stats["initialized"])
        self.assertEqual(stats["num_gpu_blocks"], 100)
        self.assertEqual(stats["num_cpu_blocks"], 50)

    def test_get_memory_usage(self):
        """Test get_memory_usage returns correct structure."""
        cache_manager = create_cache_manager()
        usage = cache_manager.get_memory_usage()

        self.assertIn("device", usage)
        self.assertIn("host", usage)

        self.assertIn("total_blocks", usage["device"])
        self.assertIn("used_blocks", usage["device"])
        self.assertIn("free_blocks", usage["device"])
        self.assertIn("usage_percent", usage["device"])


class TestCacheManagerMatchPrefix(unittest.TestCase):
    """Test CacheManager prefix matching."""

    def test_match_prefix_empty(self):
        """Test matching with empty hashes."""
        cache_manager = create_cache_manager()
        result = cache_manager.match_prefix([])

        self.assertEqual(result.total_matched_blocks, 0)
        self.assertEqual(len(result.device_block_ids), 0)

    def test_match_prefix_no_match(self):
        """Test matching with no existing blocks."""
        cache_manager = create_cache_manager()
        result = cache_manager.match_prefix(["hash1", "hash2"])

        self.assertEqual(result.total_matched_blocks, 0)
        self.assertEqual(len(result.device_block_ids), 0)

    def test_match_prefix_with_match(self):
        """Test matching with existing blocks."""
        cache_manager = create_cache_manager()
        # Insert blocks first
        block_hashes = ["hash1", "hash2", "hash3"]
        device_block_ids = [1, 2, 3]
        cache_manager.update_on_request_finish(
            block_hashes=block_hashes,
            device_block_ids=device_block_ids,
        )

        # Match the same hashes
        result = cache_manager.match_prefix(block_hashes)

        self.assertEqual(result.total_matched_blocks, 3)


if __name__ == "__main__":
    unittest.main()
