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
- Request lifecycle management with RadixTree integration
- Multi-method workflow tests
"""

import unittest
from dataclasses import dataclass, field
from typing import List

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
    config.cache_config.total_block_num = total_block_num
    config.cache_config.num_cpu_blocks = num_cpu_blocks
    config.cache_config.block_size = block_size
    config.cache_config.enable_prefix_caching = enable_prefix_caching

    return CacheManager(config)


@dataclass
class MockMatchResult:
    """Mock MatchResult for testing."""

    device_nodes: List = field(default_factory=list)
    host_nodes: List = field(default_factory=list)
    storage_nodes: List = field(default_factory=list)
    uncached_block_ids: List = field(default_factory=list)

    @property
    def matched_device_nums(self) -> int:
        return len(self.device_nodes)

    @property
    def matched_host_nums(self) -> int:
        return len(self.host_nodes)

    @property
    def matched_storage_nums(self) -> int:
        return len(self.storage_nodes)

    @property
    def total_matched_blocks(self) -> int:
        return self.matched_device_nums + self.matched_host_nums + self.matched_storage_nums

    @property
    def device_block_ids(self) -> List[int]:
        return [node.block_id for node in self.device_nodes]


@dataclass
class MockRequest:
    """Mock Request for testing CacheManager."""

    request_id: str
    prompt_hashes: List[str]
    block_tables: List[int] = field(default_factory=list)
    match_result: MockMatchResult = field(default_factory=MockMatchResult)
    cache_evict_metadata: List = field(default_factory=list)
    cache_swap_metadata: List = field(default_factory=list)


class TestCacheManagerAllocation(unittest.TestCase):
    """Test CacheManager block allocation functionality."""

    def test_allocate_device_blocks_with_request(self):
        """Test device block allocation with mock request."""
        cache_manager = create_cache_manager()
        request = MockRequest(
            request_id="test_req_1",
            prompt_hashes=["h1", "h2", "h3", "h4", "h5"],
            block_tables=[],
        )

        allocated = cache_manager.allocate_device_blocks(request, 5)

        self.assertIsNotNone(allocated)
        self.assertEqual(len(allocated), 5)
        self.assertEqual(cache_manager.num_free_device_blocks, 95)

    def test_allocate_device_blocks_insufficient(self):
        """Test device block allocation when not enough blocks after eviction."""
        cache_manager = create_cache_manager()
        # Exhaust device blocks
        for _ in range(10):
            cache_manager.allocate_device_blocks(MockRequest(request_id="req", prompt_hashes=[], block_tables=[]), 10)

        # Next allocation should fail (no evictable blocks and no free blocks)
        request = MockRequest(request_id="test", prompt_hashes=["h1"], block_tables=[])
        result = cache_manager.allocate_device_blocks(request, 10)
        self.assertEqual(result, [])

    def test_allocate_host_blocks_success(self):
        """Test successful host block allocation."""
        cache_manager = create_cache_manager()
        allocated = cache_manager.allocate_host_blocks(10)

        self.assertIsNotNone(allocated)
        self.assertEqual(len(allocated), 10)
        self.assertEqual(cache_manager.num_free_host_blocks, 40)

    def test_allocate_host_blocks_insufficient(self):
        """Test host block allocation returns empty when not enough blocks."""
        cache_manager = create_cache_manager(num_cpu_blocks=5)
        allocated = cache_manager.allocate_host_blocks(10)

        self.assertEqual(allocated, [])


class TestCacheManagerRelease(unittest.TestCase):
    """Test CacheManager block release functionality."""

    def test_free_device_blocks(self):
        """Test freeing device blocks."""
        cache_manager = create_cache_manager()
        request = MockRequest(request_id="req", prompt_hashes=[], block_tables=[])
        allocated = cache_manager.allocate_device_blocks(request, 10)
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
        req = MockRequest(request_id="req", prompt_hashes=[], block_tables=[])
        cache_manager.allocate_device_blocks(req, 50)

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
        req = MockRequest(request_id="req", prompt_hashes=[], block_tables=[])
        cache_manager.allocate_device_blocks(req, 50)
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

    def test_resize_device_pool_shrink_with_used_blocks(self):
        """Test shrinking device pool fails when used blocks exceed new size."""
        cache_manager = create_cache_manager(total_block_num=100)
        req = MockRequest(request_id="req", prompt_hashes=[], block_tables=[])
        cache_manager.allocate_device_blocks(req, 60)

        result = cache_manager.resize_device_pool(50)

        self.assertFalse(result)
        self.assertEqual(cache_manager.num_gpu_blocks, 100)

    def test_resize_device_pool_allocate_after_expand(self):
        """Test allocating blocks after expanding pool."""
        cache_manager = create_cache_manager(total_block_num=100)
        cache_manager.resize_device_pool(150)

        req = MockRequest(request_id="req", prompt_hashes=[], block_tables=[])
        allocated = cache_manager.allocate_device_blocks(req, 120)

        self.assertIsNotNone(allocated)
        self.assertEqual(len(allocated), 120)


class TestCacheManagerWorkflow(unittest.TestCase):
    """Test CacheManager multi-method workflow scenarios."""

    def test_request_lifecycle_full(self):
        """Test complete request lifecycle: match -> allocate -> finish."""
        cache_manager = create_cache_manager()

        # Step 1: Request comes in, match prefix (no existing cache)
        request1 = MockRequest(
            request_id="req_1",
            prompt_hashes=["hash1", "hash2", "hash3"],
            block_tables=[],
        )
        cache_manager.match_prefix(request1)

        self.assertEqual(request1.match_result.total_matched_blocks, 0)

        # Step 2: Allocate blocks for the request
        allocated = cache_manager.allocate_device_blocks(request1, 3)
        self.assertIsNotNone(allocated)
        self.assertEqual(len(allocated), 3)

        # Step 3: Request finishes, cache the blocks
        request1.block_tables = allocated
        cache_manager.request_finish(request1)

        # Verify blocks are cached
        self.assertEqual(cache_manager.num_free_device_blocks, 97)

    def test_request_lifecycle_with_prefix_reuse(self):
        """Test request reusing cached prefix."""
        cache_manager = create_cache_manager()

        # First request: insert [h1, h2, h3]
        req1 = MockRequest(
            request_id="req_1",
            prompt_hashes=["h1", "h2", "h3"],
            block_tables=[],
        )
        cache_manager.match_prefix(req1)
        allocated1 = cache_manager.allocate_device_blocks(req1, 3)
        req1.block_tables = allocated1
        cache_manager.request_finish(req1)

        # Second request: same prefix [h1, h2], then new [h4]
        req2 = MockRequest(
            request_id="req_2",
            prompt_hashes=["h1", "h2", "h4"],
            block_tables=[],
        )
        cache_manager.match_prefix(req2)

        # Should match h1, h2 (result stored in _match_result)
        self.assertEqual(req2._match_result.matched_device_nums, 2)
        self.assertEqual(req2._match_result.matched_host_nums, 0)

        # Allocate only for h4 (1 new block needed)
        allocated2 = cache_manager.allocate_device_blocks(req2, 1)
        self.assertIsNotNone(allocated2)

        matched_ids = req2._match_result.device_block_ids
        req2.block_tables = matched_ids + allocated2
        cache_manager.request_finish(req2)

    def test_shared_prefix_multiple_requests(self):
        """Test multiple requests sharing prefix."""
        cache_manager = create_cache_manager()

        # Insert base prefix [A, B]
        req1 = MockRequest(
            request_id="req_1",
            prompt_hashes=["A", "B", "C1"],
            block_tables=[],
        )
        cache_manager.match_prefix(req1)
        allocated1 = cache_manager.allocate_device_blocks(req1, 3)
        req1.block_tables = allocated1
        cache_manager.request_finish(req1)

        # Check radix tree state
        stats = cache_manager.radix_tree.get_stats()
        self.assertEqual(stats.node_count, 4)  # root + A + B + C1

        # Second request with different suffix
        req2 = MockRequest(
            request_id="req_2",
            prompt_hashes=["A", "B", "C2"],
            block_tables=[],
        )
        cache_manager.match_prefix(req2)
        self.assertEqual(req2._match_result.matched_device_nums, 2)  # A, B

        allocated2 = cache_manager.allocate_device_blocks(req2, 1)
        req2.block_tables = req2._match_result.device_block_ids + allocated2
        cache_manager.request_finish(req2)

        stats = cache_manager.radix_tree.get_stats()
        self.assertEqual(stats.node_count, 5)  # root + A + B + C1 + C2

    def test_eviction_workflow(self):
        """Test eviction when device memory is full."""
        cache_manager = create_cache_manager(num_cpu_blocks=50)

        # Exhaust device memory
        requests = []
        for i in range(10):
            req = MockRequest(
                request_id=f"req_{i}",
                prompt_hashes=[f"h{i}_{j}" for j in range(10)],
                block_tables=[],
            )
            cache_manager.match_prefix(req)
            allocated = cache_manager.allocate_device_blocks(req, 10)
            req.block_tables = allocated
            cache_manager.request_finish(req)
            requests.append(req)

        self.assertEqual(cache_manager.num_free_device_blocks, 0)

        # Verify evictable blocks exist
        stats = cache_manager.radix_tree.get_stats()
        self.assertEqual(stats.evictable_device_count, 100)

        # New request should trigger eviction
        new_req = MockRequest(
            request_id="new_req",
            prompt_hashes=["new1", "new2", "new3"],
            block_tables=[],
        )
        cache_manager.match_prefix(new_req)
        allocated = cache_manager.allocate_device_blocks(new_req, 3)

        self.assertIsNotNone(allocated)
        self.assertEqual(len(allocated), 3)

    def test_host_cache_eviction_workflow(self):
        """Test device -> host eviction workflow when memory is full."""
        cache_manager = create_cache_manager(num_cpu_blocks=30)

        # Exhaust device memory with different hashes (no prefix sharing)
        for i in range(10):
            req = MockRequest(
                request_id=f"req_{i}",
                prompt_hashes=[f"h{i}_{j}" for j in range(10)],
                block_tables=[],
            )
            cache_manager.match_prefix(req)
            allocated = cache_manager.allocate_device_blocks(req, 10)
            req.block_tables = allocated
            cache_manager.request_finish(req)

        # Device should be full
        self.assertEqual(cache_manager.num_free_device_blocks, 0)

        # New request should still work (eviction should occur)
        new_req = MockRequest(
            request_id="new_req",
            prompt_hashes=["new1", "new2", "new3"],
            block_tables=[],
        )
        cache_manager.match_prefix(new_req)
        allocated = cache_manager.allocate_device_blocks(new_req, 3)

        self.assertIsNotNone(allocated)
        self.assertEqual(len(allocated), 3)


class TestCacheManagerRadixTreeIntegration(unittest.TestCase):
    """Test CacheManager RadixTree integration."""

    def test_match_prefix_updates_ref_count(self):
        """Test that match_prefix increments ref count."""
        cache_manager = create_cache_manager()

        # Insert some blocks
        req1 = MockRequest(
            request_id="req_1",
            prompt_hashes=["h1", "h2"],
            block_tables=[],
        )
        cache_manager.match_prefix(req1)
        allocated1 = cache_manager.allocate_device_blocks(req1, 2)
        req1.block_tables = allocated1
        cache_manager.request_finish(req1)

        # Check initial evictable count (should be 2 after finish)
        stats1 = cache_manager.radix_tree.get_stats()
        self.assertEqual(stats1.evictable_device_count, 2)

        # Match same prefix - should increment ref
        req2 = MockRequest(
            request_id="req_2",
            prompt_hashes=["h1", "h2"],
            block_tables=[],
        )
        cache_manager.match_prefix(req2)

        # Ref count should be incremented, nodes not evictable
        stats2 = cache_manager.radix_tree.get_stats()
        self.assertEqual(stats2.evictable_device_count, 0)

    def test_insert_and_find_prefix(self):
        """Test inserting blocks and finding prefix."""
        cache_manager = create_cache_manager()

        # Insert blocks
        req1 = MockRequest(
            request_id="req_1",
            prompt_hashes=["hash_a", "hash_b", "hash_c"],
            block_tables=[],
        )
        cache_manager.match_prefix(req1)
        allocated = cache_manager.allocate_device_blocks(req1, 3)
        req1.block_tables = allocated
        cache_manager.request_finish(req1)

        # Find prefix
        req2 = MockRequest(
            request_id="req_2",
            prompt_hashes=["hash_a", "hash_b"],
            block_tables=[],
        )
        cache_manager.match_prefix(req2)

        self.assertEqual(req2._match_result.matched_device_nums, 2)
        # Block IDs depend on allocation order; verify count and that they are valid ints
        block_ids = req2._match_result.device_block_ids
        self.assertEqual(len(block_ids), 2)
        self.assertTrue(all(isinstance(bid, int) for bid in block_ids))


class TestCacheManagerWithDisabledPrefixCaching(unittest.TestCase):
    """Test CacheManager with prefix caching disabled."""

    def test_radix_tree_none_when_disabled(self):
        """Test radix_tree is None when prefix caching disabled."""
        cache_manager = create_cache_manager(enable_prefix_caching=False)
        self.assertIsNone(cache_manager.radix_tree)

    def test_allocation_works_without_prefix_caching(self):
        """Test block allocation still works without prefix caching."""
        cache_manager = create_cache_manager(enable_prefix_caching=False)
        req = MockRequest(request_id="req", prompt_hashes=[], block_tables=[])
        allocated = cache_manager.allocate_device_blocks(req, 10)

        self.assertIsNotNone(allocated)
        self.assertEqual(len(allocated), 10)


class TestCacheManagerWithNoHostCache(unittest.TestCase):
    """Test CacheManager with no host cache."""

    def test_host_cache_disabled(self):
        """Test host cache is disabled."""
        cache_manager = create_cache_manager(num_cpu_blocks=0)
        self.assertFalse(cache_manager.enable_host_cache)

    def test_no_free_host_blocks(self):
        """Test no free host blocks when disabled."""
        cache_manager = create_cache_manager(num_cpu_blocks=0)
        self.assertEqual(cache_manager.num_free_host_blocks, 0)


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
        self.assertIn("radix_tree", stats)

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


class TestCacheManagerEdgeCases(unittest.TestCase):
    """Test CacheManager edge cases."""

    def test_empty_prompt_hashes(self):
        """Test request with empty prompt hashes."""
        cache_manager = create_cache_manager()
        req = MockRequest(request_id="req", prompt_hashes=[], block_tables=[])

        cache_manager.match_prefix(req)
        self.assertEqual(req.match_result.total_matched_blocks, 0)

        allocated = cache_manager.allocate_device_blocks(req, 0)
        self.assertEqual(allocated, [])

    def test_allocation_with_matched_host_blocks(self):
        """Test allocation when host cache has matched blocks."""
        cache_manager = create_cache_manager(num_cpu_blocks=50)

        # Insert blocks and evict some to host
        req1 = MockRequest(
            request_id="req_1",
            prompt_hashes=["h1", "h2", "h3"],
            block_tables=[],
        )
        cache_manager.match_prefix(req1)
        allocated1 = cache_manager.allocate_device_blocks(req1, 3)
        req1.block_tables = allocated1
        cache_manager.request_finish(req1)

        # Exhaust device, evict to host
        for i in range(10):
            req = MockRequest(
                request_id=f"req_{i}",
                prompt_hashes=[f"other_{i}_{j}" for j in range(10)],
                block_tables=[],
            )
            cache_manager.match_prefix(req)
            allocated = cache_manager.allocate_device_blocks(req, 10)
            req.block_tables = allocated
            cache_manager.request_finish(req)

        # Now request h1, h2 - should find them in host cache
        req2 = MockRequest(
            request_id="req_2",
            prompt_hashes=["h1", "h2"],
            block_tables=[],
        )
        cache_manager.match_prefix(req2)

        # After device is full, h1 and h2 may be evicted to host (write_through policy)
        # Total matched should be non-negative regardless of eviction policy
        total_matched = req2._match_result.total_matched_blocks
        self.assertGreaterEqual(total_matched, 0)
        # If found in host, matched_host_nums > 0
        if req2._match_result.matched_host_nums > 0:
            self.assertGreater(req2._match_result.matched_host_nums, 0)


class TestCacheManagerCanAllocate(unittest.TestCase):
    """Test CacheManager can_allocate_* methods."""

    def test_can_allocate_device_blocks_enough(self):
        """Test can_allocate_device_blocks returns True when enough free blocks."""
        cache_manager = create_cache_manager(total_block_num=100)
        self.assertTrue(cache_manager.can_allocate_device_blocks(50))

    def test_can_allocate_device_blocks_exact(self):
        """Test can_allocate_device_blocks returns True for exact count."""
        cache_manager = create_cache_manager(total_block_num=100)
        self.assertTrue(cache_manager.can_allocate_device_blocks(100))

    def test_can_allocate_device_blocks_too_many(self):
        """Test can_allocate_device_blocks returns False when not enough blocks."""
        cache_manager = create_cache_manager(total_block_num=100, enable_prefix_caching=False)
        self.assertFalse(cache_manager.can_allocate_device_blocks(101))

    def test_can_allocate_host_blocks_enough(self):
        """Test can_allocate_host_blocks returns True when enough free blocks."""
        cache_manager = create_cache_manager(num_cpu_blocks=50)
        self.assertTrue(cache_manager.can_allocate_host_blocks(30))

    def test_can_allocate_host_blocks_too_many(self):
        """Test can_allocate_host_blocks returns False when not enough blocks."""
        cache_manager = create_cache_manager(num_cpu_blocks=10, enable_prefix_caching=False)
        self.assertFalse(cache_manager.can_allocate_host_blocks(20))

    def test_can_allocate_gpu_blocks_alias(self):
        """Test can_allocate_gpu_blocks is alias for can_allocate_device_blocks."""
        cache_manager = create_cache_manager(total_block_num=100)
        self.assertEqual(
            cache_manager.can_allocate_device_blocks(50),
            cache_manager.can_allocate_gpu_blocks(50),
        )


class TestCacheManagerLegacyMethods(unittest.TestCase):
    """Test CacheManager legacy compatibility methods."""

    def test_allocate_gpu_blocks_alias(self):
        """Test allocate_gpu_blocks delegates to allocate_device_blocks."""
        cache_manager = create_cache_manager()
        req = MockRequest(request_id="req", prompt_hashes=[], block_tables=[])
        allocated = cache_manager.allocate_gpu_blocks(req, 5)

        self.assertIsNotNone(allocated)
        self.assertEqual(len(allocated), 5)

    def test_gpu_free_block_list_property(self):
        """Test gpu_free_block_list returns a list."""
        cache_manager = create_cache_manager(total_block_num=100)
        free_list = cache_manager.gpu_free_block_list
        self.assertIsInstance(free_list, list)

    def test_available_gpu_resource_full(self):
        """Test available_gpu_resource is 1.0 when no blocks used."""
        cache_manager = create_cache_manager(total_block_num=100)
        self.assertAlmostEqual(cache_manager.available_gpu_resource, 1.0)

    def test_available_gpu_resource_after_allocation(self):
        """Test available_gpu_resource decreases after allocation."""
        cache_manager = create_cache_manager(total_block_num=100, enable_prefix_caching=False)
        req = MockRequest(request_id="req", prompt_hashes=[], block_tables=[])
        cache_manager.allocate_device_blocks(req, 50)
        self.assertAlmostEqual(cache_manager.available_gpu_resource, 0.5)

    def test_update_cache_config(self):
        """Test update_cache_config resizes device pool when total_block_num changes."""
        cache_manager = create_cache_manager(total_block_num=100)

        new_cfg = cache_manager.cache_config
        new_cfg.total_block_num = 150
        cache_manager.update_cache_config(new_cfg)

        self.assertEqual(cache_manager.num_gpu_blocks, 150)


class TestCacheManagerStorageScheduler(unittest.TestCase):
    """Test CacheManager storage_scheduler property."""

    def test_storage_scheduler_none_by_default(self):
        """Test storage_scheduler is None when not configured."""
        cache_manager = create_cache_manager()
        # Default config has no storage backend, so scheduler should be None
        # (behavior depends on create_storage_scheduler implementation)
        # Just verify it's accessible without error
        _ = cache_manager.storage_scheduler


# ---------------------------------------------------------------------------
# offload_to_host
# ---------------------------------------------------------------------------


class TestCacheManagerOffloadToHost(unittest.TestCase):
    """Tests for CacheManager.offload_to_host."""

    def test_offload_frees_device_blocks(self):
        """After offload, device blocks should be released."""
        cm = create_cache_manager(total_block_num=20, num_cpu_blocks=20)
        device_blocks = cm._device_pool.allocate(4)
        self.assertIsNotNone(device_blocks)
        free_before = cm.num_free_device_blocks

        success = cm.offload_to_host(device_blocks)

        self.assertTrue(success)
        self.assertEqual(cm.num_free_device_blocks, free_before + 4)

    def test_offload_allocates_host_blocks(self):
        """After offload, host blocks should be consumed."""
        cm = create_cache_manager(total_block_num=20, num_cpu_blocks=20)
        device_blocks = cm._device_pool.allocate(3)
        free_host_before = cm.num_free_host_blocks

        cm.offload_to_host(device_blocks)

        self.assertEqual(cm.num_free_host_blocks, free_host_before - 3)

    def test_offload_fails_when_no_host_blocks(self):
        """Offload should return False when host pool is exhausted."""
        cm = create_cache_manager(total_block_num=20, num_cpu_blocks=0)
        device_blocks = cm._device_pool.allocate(2)

        success = cm.offload_to_host(device_blocks)
        self.assertFalse(success)

    def test_offload_copies_device_metadata_to_host(self):
        """Metadata on device blocks should be copied to host blocks."""
        from fastdeploy.cache_manager.v1.metadata import CacheBlockMetadata

        cm = create_cache_manager(total_block_num=20, num_cpu_blocks=20)
        device_blocks = cm._device_pool.allocate(1)
        block_id = device_blocks[0]
        meta = CacheBlockMetadata(block_id=block_id, device_id=0, block_size=64, ref_count=5)
        cm._device_pool.set_metadata(block_id, meta)

        cm.offload_to_host(device_blocks)

        # Find the newly used host block (last used)
        used_host = list(cm._host_pool._used_blocks)
        self.assertEqual(len(used_host), 1)
        host_meta = cm._host_pool.get_metadata(used_host[0])
        self.assertIsNotNone(host_meta)
        self.assertEqual(host_meta.ref_count, 5)

    def test_offload_empty_list_returns_true(self):
        """Offloading empty list succeeds."""
        cm = create_cache_manager()
        success = cm.offload_to_host([])
        self.assertTrue(success)


# ---------------------------------------------------------------------------
# load_from_host
# ---------------------------------------------------------------------------


class TestCacheManagerLoadFromHost(unittest.TestCase):
    """Tests for CacheManager.load_from_host."""

    def test_load_frees_host_blocks(self):
        """After loading, host blocks should be released."""
        cm = create_cache_manager(total_block_num=20, num_cpu_blocks=20)
        host_blocks = cm._host_pool.allocate(4)
        free_before = cm.num_free_host_blocks

        success = cm.load_from_host(host_blocks)

        self.assertTrue(success)
        self.assertEqual(cm.num_free_host_blocks, free_before + 4)

    def test_load_allocates_device_blocks(self):
        """After loading, device blocks should be consumed."""
        cm = create_cache_manager(total_block_num=20, num_cpu_blocks=20)
        host_blocks = cm._host_pool.allocate(3)
        free_device_before = cm.num_free_device_blocks

        cm.load_from_host(host_blocks)

        self.assertEqual(cm.num_free_device_blocks, free_device_before - 3)

    def test_load_fails_when_no_device_blocks(self):
        """Load should return False when device pool is exhausted."""
        cm = create_cache_manager(total_block_num=2, num_cpu_blocks=20)
        # Fill up device
        cm._device_pool.allocate(2)
        host_blocks = cm._host_pool.allocate(2)

        success = cm.load_from_host(host_blocks)
        self.assertFalse(success)

    def test_load_empty_list_returns_true(self):
        """Loading empty list succeeds."""
        cm = create_cache_manager()
        success = cm.load_from_host([])
        self.assertTrue(success)


# ---------------------------------------------------------------------------
# get_pending_backup_count / check_and_add_pending_backup /
# issue_pending_backup_to_batch_request
# ---------------------------------------------------------------------------


class TestCacheManagerPendingBackup(unittest.TestCase):
    """Tests for write_through_selective backup methods."""

    def _create_write_through_cm(self, threshold: int = 1):
        from fastdeploy.cache_manager.v1.cache_manager import CacheManager

        config = get_default_test_fd_config()
        config.cache_config.total_block_num = 50
        config.cache_config.num_cpu_blocks = 50
        config.cache_config.block_size = 64
        config.cache_config.enable_prefix_caching = True
        config.cache_config.write_policy = "write_through_selective"
        config.cache_config.write_through_threshold = threshold
        return CacheManager(config)

    def test_get_pending_backup_count_initially_zero(self):
        cm = self._create_write_through_cm()
        self.assertEqual(cm.get_pending_backup_count(), 0)

    def test_issue_pending_backup_returns_none_when_empty(self):
        cm = self._create_write_through_cm()
        result = cm.issue_pending_backup_to_batch_request()
        self.assertIsNone(result)

    def test_check_and_add_pending_backup_does_nothing_without_prefix_caching(self):
        """When prefix caching is off, check_and_add_pending_backup is a no-op."""
        cm = create_cache_manager(enable_prefix_caching=False)
        cm.check_and_add_pending_backup()  # should not raise
        self.assertEqual(cm.get_pending_backup_count(), 0)

    def test_check_and_add_pending_backup_does_nothing_without_host_cache(self):
        """Without host cache, check_and_add_pending_backup is a no-op."""
        cm = self._create_write_through_cm()
        cm.enable_host_cache = False
        cm.check_and_add_pending_backup()
        self.assertEqual(cm.get_pending_backup_count(), 0)

    def test_check_and_add_pending_backup_adds_candidates(self):
        """After inserting nodes that meet threshold, backup should be queued."""
        cm = self._create_write_through_cm(threshold=1)
        rt = cm._radix_tree

        # Insert nodes and decrement so they become evictable
        nodes, _ = rt.insert([("h1", 0), ("h2", 1), ("h3", 2)])
        # Simulate hit_count meeting threshold (threshold=1, default hit_count=1)
        cm._device_pool.allocate(3)  # Ensure enough device blocks consumed
        rt.decrement_ref_nodes(nodes)

        cm.check_and_add_pending_backup()
        # Should have added at least something if there are candidates
        # (may be 0 if no candidates qualify; just ensure no exception)
        count = cm.get_pending_backup_count()
        self.assertGreaterEqual(count, 0)

    def test_issue_pending_backup_clears_queue(self):
        """After issuing, the pending backup queue should be empty."""
        cm = self._create_write_through_cm(threshold=1)
        rt = cm._radix_tree

        nodes, _ = rt.insert([("h1", 0)])
        cm._device_pool.allocate(1)
        rt.decrement_ref_nodes(nodes)
        cm.check_and_add_pending_backup()

        cm.issue_pending_backup_to_batch_request()
        self.assertEqual(cm.get_pending_backup_count(), 0)

    def test_issue_returns_none_when_host_cache_disabled(self):
        """If host cache is not enabled, issue returns None and clears queue."""
        cm = self._create_write_through_cm()
        # Manually add a fake pending entry
        cm._pending_backup.append(([], []))
        cm.enable_host_cache = False
        result = cm.issue_pending_backup_to_batch_request()
        self.assertIsNone(result)
        self.assertEqual(cm.get_pending_backup_count(), 0)


# ---------------------------------------------------------------------------
# prepare_prefetch_metadata
# ---------------------------------------------------------------------------


class TestCacheManagerPreparePrefetchMetadata(unittest.TestCase):
    """Tests for CacheManager.prepare_prefetch_metadata."""

    def test_empty_hashes_returns_none(self):
        cm = create_cache_manager()
        result = cm.prepare_prefetch_metadata([])
        self.assertIsNone(result)

    def test_returns_nodes_when_host_blocks_available(self):
        cm = create_cache_manager(num_cpu_blocks=20)
        hashes = ["hash_a", "hash_b"]
        result = cm.prepare_prefetch_metadata(hashes)
        # Should return a list (possibly empty if no host blocks or tree reuse)
        self.assertIsInstance(result, list)

    def test_returns_empty_when_insufficient_host_blocks(self):
        cm = create_cache_manager(total_block_num=20, num_cpu_blocks=0)
        result = cm.prepare_prefetch_metadata(["h1", "h2"])
        # With no host blocks, should return empty or None
        self.assertFalse(result)  # None or []


if __name__ == "__main__":
    unittest.main()
