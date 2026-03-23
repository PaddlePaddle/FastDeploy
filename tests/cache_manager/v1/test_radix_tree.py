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

"""
Unit tests for RadixTree in cache_manager/v1.

Tests cover:
- Basic operations: insert, find_prefix, increment_ref_nodes, decrement_ref_nodes
- Eviction: evict_host_nodes, evict_device_to_host
- Edge cases and error handling

Run with:
    source .venv/py310/bin/activate
    pytest tests/cache_manager/v1/test_radix_tree.py -v
"""

import time

from fastdeploy.cache_manager.v1.metadata import CacheStatus
from fastdeploy.cache_manager.v1.radix_tree import RadixTree


class TestRadixTreeInit:
    """Tests for RadixTree initialization."""

    def test_init_default(self):
        """Test default initialization."""
        tree = RadixTree()
        assert tree.node_count() == 1  # Only root
        assert tree._enable_host_cache is False

    def test_init_with_host_cache(self):
        """Test initialization with host cache enabled."""
        tree = RadixTree(enable_host_cache=True)
        assert tree._enable_host_cache is True

    def test_get_stats(self):
        """Test get_stats returns correct structure."""
        tree = RadixTree()
        stats = tree.get_stats()
        assert stats.node_count == 1
        assert stats.evictable_count == 0
        # Test to_dict
        stats_dict = stats.to_dict()
        assert "node_count" in stats_dict
        assert "evictable_count" in stats_dict


class TestRadixTreeInsert:
    """Tests for insert operation."""

    def test_insert_single_block(self):
        """Test inserting a single block."""
        tree = RadixTree()
        result, _ = tree.insert([("hash1", 1)])
        assert len(result) == 1  # Returns list of nodes
        assert tree.node_count() == 2  # root + 1 node

    def test_insert_multiple_blocks(self):
        """Test inserting multiple blocks in sequence."""
        tree = RadixTree()
        result, _ = tree.insert([("hash1", 1), ("hash2", 2), ("hash3", 3)])
        assert len(result) == 3
        assert tree.node_count() == 4  # root + 3 nodes

    def test_insert_empty_list(self):
        """Test inserting empty list returns empty list."""
        tree = RadixTree()
        result, _ = tree.insert([])
        assert result == []
        assert tree.node_count() == 1

    def test_insert_shared_prefix(self):
        """Test inserting sequences with shared prefix."""
        tree = RadixTree()
        # Insert first sequence
        tree.insert([("hash1", 1), ("hash2", 2)])
        # Insert second sequence sharing first block
        tree.insert([("hash1", 1), ("hash3", 3)])

        # Should reuse the first node, only add one new node
        assert tree.node_count() == 4  # root + 3 unique nodes (hash1, hash2, hash3)

    def test_insert_same_sequence_twice(self):
        """Test inserting the same sequence twice increases ref_count."""
        tree = RadixTree()
        tree.insert([("hash1", 1), ("hash2", 2)])
        tree.insert([("hash1", 1), ("hash2", 2)])

        # Should reuse nodes, not create new ones
        assert tree.node_count() == 3  # root + 2 nodes


class TestRadixTreeFindPrefix:
    """Tests for find_prefix operation."""

    def test_find_prefix_full_match(self):
        """Test finding a full prefix match."""
        tree = RadixTree()
        tree.insert([("hash1", 1), ("hash2", 2), ("hash3", 3)])

        nodes = tree.find_prefix(["hash1", "hash2", "hash3"])
        assert len(nodes) == 3
        block_ids = [node.block_id for node in nodes]
        assert block_ids == [1, 2, 3]

    def test_find_prefix_partial_match(self):
        """Test finding a partial prefix match."""
        tree = RadixTree()
        tree.insert([("hash1", 1), ("hash2", 2), ("hash3", 3)])

        nodes = tree.find_prefix(["hash1", "hash2", "hash4"])
        assert len(nodes) == 2
        block_ids = [node.block_id for node in nodes]
        assert block_ids == [1, 2]

    def test_find_prefix_no_match(self):
        """Test finding no prefix match."""
        tree = RadixTree()
        tree.insert([("hash1", 1), ("hash2", 2)])

        nodes = tree.find_prefix(["hash3", "hash4"])
        assert len(nodes) == 0

    def test_find_prefix_empty_query(self):
        """Test finding prefix with empty query."""
        tree = RadixTree()
        tree.insert([("hash1", 1)])

        nodes = tree.find_prefix([])
        assert len(nodes) == 0


class TestRadixTreeRefCount:
    """Tests for reference count operations."""

    def test_increment_ref_nodes(self):
        """Test incrementing reference count for nodes."""
        tree = RadixTree()
        nodes, _ = tree.insert([("hash1", 1), ("hash2", 2)])

        # Release nodes first
        tree.decrement_ref_nodes(nodes)
        assert len(tree._evictable_set) == 2

        # Increment again - should remove from evictable
        tree.increment_ref_nodes(nodes)
        assert len(tree._evictable_set) == 0

    def test_decrement_ref_nodes(self):
        """Test decrementing reference count for nodes."""
        tree = RadixTree()
        nodes, _ = tree.insert([("hash1", 1), ("hash2", 2)])

        assert len(tree._evictable_set) == 0

        # Decrement ref count
        tree.decrement_ref_nodes(nodes)
        assert len(tree._evictable_set) == 2

    def test_decrement_ref_nodes_shared_prefix(self):
        """Test decrementing with shared prefix."""
        tree = RadixTree()
        nodes1, _ = tree.insert([("hash1", 1), ("hash2", 2)])
        nodes2, _ = tree.insert([("hash1", 1), ("hash3", 3)])

        # Release first sequence
        tree.decrement_ref_nodes(nodes1)
        # hash2 should be evictable, hash1 still has ref=1
        assert len(tree._evictable_set) == 1

        # Release second sequence
        tree.decrement_ref_nodes(nodes2)
        # Now hash1 and hash3 should be evictable (hash2 already was)
        assert len(tree._evictable_set) == 3


class TestEvictDeviceToHost:
    """Tests for evict_device_to_host method."""

    def test_basic_evict_to_host(self):
        """Test basic device-to-host eviction."""
        tree = RadixTree(enable_host_cache=True)
        nodes, _ = tree.insert([("h1", 10), ("h2", 20), ("h3", 30)])
        tree.decrement_ref_nodes(nodes)

        result = tree.evict_device_to_host(3, [100, 101, 102])
        assert sorted(result) == [10, 20, 30]

        stats = tree.get_stats()
        assert stats.evictable_device_count == 0
        assert stats.evictable_host_count == 3

        # Verify nodes now have HOST status and new block_ids
        for node in nodes:
            assert node.cache_status == CacheStatus.HOST
            assert node.block_id in [100, 101, 102]

    def test_evict_partial(self):
        """Test evicting only part of the evictable nodes."""
        tree = RadixTree(enable_host_cache=True)
        nodes, _ = tree.insert([("h1", 1), ("h2", 2), ("h3", 3)])
        tree.decrement_ref_nodes(nodes)

        # Evict only 1 out of 3
        result = tree.evict_device_to_host(1, [100])
        assert result == [1]

        stats = tree.get_stats()
        assert stats.evictable_device_count == 2
        assert stats.evictable_host_count == 1

    def test_evict_with_shared_prefix_non_evictable(self):
        """Test eviction skips non-evictable nodes (ref_count > 0)."""
        tree = RadixTree(enable_host_cache=True)

        # Insert two sequences sharing prefix: h1->h2
        nodes_a, _ = tree.insert([("h1", 1), ("h2", 2), ("h3", 3)])
        tree.insert([("h1", 1), ("h2", 2), ("h4", 4)])

        # Release only sequence A: h3 evictable, h1 and h2 still ref=2
        tree.decrement_ref_nodes(nodes_a)

        stats = tree.get_stats()
        assert stats.evictable_device_count == 1  # only h3

        # Evict h3 to host
        result = tree.evict_device_to_host(1, [100])
        assert result == [3]

        # h3 should now be on host
        for node in nodes_a:
            if node.hash_value == "h3":
                assert node.cache_status == CacheStatus.HOST
                assert node.block_id == 100

    def test_evict_skips_host_nodes_in_heap(self):
        """Test that HOST nodes already in heap are skipped."""
        tree = RadixTree(enable_host_cache=True)

        # Insert and release sequence A
        nodes_a, _ = tree.insert([("h1", 1), ("h2", 2)])
        tree.decrement_ref_nodes(nodes_a)

        # Evict A to host
        tree.evict_device_to_host(2, [100, 101])

        # Insert and release sequence B
        nodes_b, _ = tree.insert([("h3", 3), ("h4", 4)])
        tree.decrement_ref_nodes(nodes_b)

        # Now heap has: host(h1), host(h2), device(h3), device(h4)
        # Try to evict 2 device blocks - should skip host nodes
        result = tree.evict_device_to_host(2, [200, 201])
        assert sorted(result) == [3, 4]

        stats = tree.get_stats()
        assert stats.evictable_device_count == 0
        assert stats.evictable_host_count == 4

    def test_evict_to_host_then_reuse_in_find_prefix(self):
        """Test that evicted HOST nodes can still be found by find_prefix."""
        tree = RadixTree(enable_host_cache=True)

        nodes, _ = tree.insert([("h1", 1), ("h2", 2)])
        tree.decrement_ref_nodes(nodes)

        # Evict to host
        tree.evict_device_to_host(2, [100, 101])

        # find_prefix should still match (HOST nodes are not skipped)
        matched = tree.find_prefix(["h1", "h2"])
        assert len(matched) == 2
        block_ids = [n.block_id for n in matched]
        assert block_ids == [100, 101]

    def test_evict_to_host_then_swap_back_to_device(self):
        """Test full cycle: insert -> evict to host -> swap back to device."""
        tree = RadixTree(enable_host_cache=True)

        nodes, _ = tree.insert([("h1", 1), ("h2", 2)])
        tree.decrement_ref_nodes(nodes)

        # Evict to host
        tree.evict_device_to_host(2, [100, 101])
        for node in nodes:
            assert node.cache_status == CacheStatus.HOST

        # Swap back to device
        original_host_ids = tree.swap_to_device(nodes, [1, 2])
        assert sorted(original_host_ids) == [100, 101]
        for node in nodes:
            assert node.cache_status == CacheStatus.SWAP_TO_DEVICE

        # Complete swap
        tree.complete_swap_to_device(nodes)
        for node in nodes:
            assert node.cache_status == CacheStatus.DEVICE

    def test_evict_precheck_insufficient_evictable(self):
        """Test pre-check returns None when not enough evictable DEVICE nodes."""
        tree = RadixTree(enable_host_cache=True)

        # Insert but do NOT decrement (ref_count=1, not evictable)
        tree.insert([("h1", 1)])

        stats = tree.get_stats()
        assert stats.evictable_device_count == 0

        result = tree.evict_device_to_host(1, [100])
        assert result is None

    def test_evict_to_host_preserves_tree_structure(self):
        """Test that eviction preserves tree parent-child relationships."""
        tree = RadixTree(enable_host_cache=True)

        nodes, _ = tree.insert([("h1", 1), ("h2", 2), ("h3", 3)])
        tree.decrement_ref_nodes(nodes)

        # Evict all to host
        tree.evict_device_to_host(3, [100, 101, 102])

        # Verify tree structure is intact
        assert tree.node_count() == 4  # root + 3 nodes

        root = tree._root
        assert "h1" in root.children
        assert "h2" in root.children["h1"].children
        assert "h3" in root.children["h1"].children["h2"].children

    def test_evict_to_host_multiple_times(self):
        """Test evicting in multiple rounds."""
        tree = RadixTree(enable_host_cache=True)

        nodes, _ = tree.insert([("h1", 1), ("h2", 2), ("h3", 3), ("h4", 4)])
        tree.decrement_ref_nodes(nodes)

        # Round 1: evict 2 blocks
        result1 = tree.evict_device_to_host(2, [100, 101])
        assert sorted(result1) == [1, 2]

        stats = tree.get_stats()
        assert stats.evictable_device_count == 2
        assert stats.evictable_host_count == 2

        # Round 2: evict remaining 2 blocks
        result2 = tree.evict_device_to_host(2, [102, 103])
        assert sorted(result2) == [3, 4]

        stats = tree.get_stats()
        assert stats.evictable_device_count == 0
        assert stats.evictable_host_count == 4


class TestRadixTreeEviction:
    """Tests for eviction operations."""

    def test_evict_host_nodes(self):
        """Test evicting HOST nodes."""
        tree = RadixTree(enable_host_cache=True)
        nodes, _ = tree.insert([("hash1", 1), ("hash2", 2)])
        tree.decrement_ref_nodes(nodes)

        # First, evict device to host
        device_ids = tree.evict_device_to_host(2, [101, 102])
        assert device_ids == [1, 2]

        # Now nodes are on host, evict them
        host_ids = tree.evict_host_nodes(2)
        assert sorted(host_ids) == [101, 102]
        assert tree.node_count() == 1  # Only root

    def test_evict_device_to_host(self):
        """Test evicting DEVICE nodes to host."""
        tree = RadixTree(enable_host_cache=True)
        nodes, _ = tree.insert([("hash1", 1), ("hash2", 2)])
        tree.decrement_ref_nodes(nodes)

        device_ids = tree.evict_device_to_host(2, [101, 102])
        assert sorted(device_ids) == [1, 2]

        # Check nodes are now on host
        stats = tree.get_stats()
        assert stats.evictable_host_count == 2
        assert stats.evictable_device_count == 0

    def test_evict_device_to_host_not_enough_blocks(self):
        """Test eviction when not enough evictable blocks."""
        tree = RadixTree(enable_host_cache=True)
        nodes, _ = tree.insert([("hash1", 1)])
        tree.decrement_ref_nodes(nodes)

        # Try to evict more than available
        result = tree.evict_device_to_host(5, [101, 102, 103, 104, 105])
        assert result is None

    def test_evict_device_to_host_mismatched_host_ids(self):
        """Test eviction with insufficient host_block_ids."""
        tree = RadixTree(enable_host_cache=True)
        nodes, _ = tree.insert([("hash1", 1), ("hash2", 2)])
        tree.decrement_ref_nodes(nodes)

        # Not enough host block ids
        result = tree.evict_device_to_host(2, [101])  # Only 1 host id
        assert result is None

    def test_evict_host_nodes_empty(self):
        """Test evicting when no host nodes available."""
        tree = RadixTree()

        result = tree.evict_host_nodes(1)
        assert result is None

    def test_evict_zero_blocks(self):
        """Test evicting zero blocks returns empty list."""
        tree = RadixTree()

        result = tree.evict_host_nodes(0)
        assert result == []

        result = tree.evict_device_to_host(0, [])
        assert result == []


class TestRadixTreeReset:
    """Tests for reset operation."""

    def test_reset_clears_all(self):
        """Test reset clears all data."""
        tree = RadixTree()
        nodes, _ = tree.insert([("hash1", 1), ("hash2", 2)])
        tree.decrement_ref_nodes(nodes)

        tree.reset()

        assert tree.node_count() == 1
        assert len(tree._evictable_set) == 0
        assert len(tree._evictable_device_heap) == 0
        assert len(tree._evictable_host_heap) == 0
        assert len(tree._node_id_to_node) == 0


class TestRadixTreeFullWorkflow:
    """Tests for complete workflow scenarios."""

    def test_workflow_shared_prefix_eviction(self):
        """Test complete workflow with shared prefix and eviction."""
        tree = RadixTree(enable_host_cache=True)

        # Insert two sequences sharing a prefix
        nodes_a, _ = tree.insert([("h1", 1), ("h2", 2), ("h3", 3)])  # Sequence A
        _ = tree.insert([("h1", 1), ("h2", 2), ("h4", 4)])  # Sequence B

        # Release sequence A
        tree.decrement_ref_nodes(nodes_a)

        # h3 should be evictable, but h1 and h2 still have ref_count=1
        assert len(tree._evictable_set) == 1

        # Find prefix for new sequence should still match h1, h2
        matched_nodes = tree.find_prefix(["h1", "h2", "h5"])
        assert len(matched_nodes) == 2
        block_ids = [node.block_id for node in matched_nodes]
        assert block_ids == [1, 2]

    def test_workflow_evict_device_to_host_then_remove(self):
        """Test workflow: evict to host, then remove from host."""
        tree = RadixTree(enable_host_cache=True)

        # Insert and release
        nodes, _ = tree.insert([("h1", 1), ("h2", 2)])
        tree.decrement_ref_nodes(nodes)

        # Evict device to host
        device_ids = tree.evict_device_to_host(2, [101, 102])
        assert sorted(device_ids) == [1, 2]

        # Nodes should be on host now and evictable again
        stats = tree.get_stats()
        assert stats.evictable_host_count == 2

        # Now remove from host
        host_ids = tree.evict_host_nodes(2)
        assert sorted(host_ids) == [101, 102]
        assert tree.node_count() == 1


class TestRadixTreeEdgeCases:
    """Tests for edge cases and error handling."""

    def test_evict_not_enough_blocks(self):
        """Test eviction when not enough evictable blocks."""
        tree = RadixTree(enable_host_cache=True)
        nodes, _ = tree.insert([("h1", 1)])
        tree.decrement_ref_nodes(nodes)

        # Try to evict more than available
        result = tree.evict_device_to_host(5, [101, 102, 103, 104, 105])
        assert result is None

        # Node should still be evictable
        assert len(tree._evictable_set) == 1

    def test_node_id_uniqueness(self):
        """Test that each node has a unique node_id."""
        tree = RadixTree()
        tree.insert([("h1", 1), ("h2", 2), ("h3", 3)])

        node_ids = set()
        for node_id, node in tree._node_id_to_node.items():
            assert node_id == node.node_id
            node_ids.add(node_id)

        assert len(node_ids) == 3  # All unique

    def test_eviction_order_lru(self):
        """Test that eviction follows LRU order."""
        tree = RadixTree(enable_host_cache=True)

        # Insert multiple blocks
        nodes, _ = tree.insert([("h1", 1), ("h2", 2), ("h3", 3)])
        tree.decrement_ref_nodes(nodes)

        # Wait a bit and access h2
        time.sleep(0.01)
        _ = tree.find_prefix(["h1", "h2"])
        # h2 is now more recently accessed

        # Evict - should start with least recently used
        device_ids = tree.evict_device_to_host(3, [101, 102, 103])
        assert len(device_ids) == 3
        # h1 should be evicted first (least recently accessed after find_prefix)
        assert device_ids[0] == 1
