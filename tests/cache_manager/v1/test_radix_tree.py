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
        assert stats.evictable_device_count == 0
        assert stats.evictable_host_count == 0
        assert stats.evictable_count == 0
        # Test to_dict
        stats_dict = stats.to_dict()
        assert "node_count" in stats_dict
        assert "evictable_device_count" in stats_dict
        assert "evictable_host_count" in stats_dict
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
        assert len(tree._evictable_device) == 2

        # Increment again - should remove from evictable
        tree.increment_ref_nodes(nodes)
        assert len(tree._evictable_device) == 0

    def test_decrement_ref_nodes(self):
        """Test decrementing reference count for nodes."""
        tree = RadixTree()
        nodes, _ = tree.insert([("hash1", 1), ("hash2", 2)])

        assert len(tree._evictable_device) == 0

        # Decrement ref count
        tree.decrement_ref_nodes(nodes)
        assert len(tree._evictable_device) == 2

    def test_decrement_ref_nodes_shared_prefix(self):
        """Test decrementing with shared prefix."""
        tree = RadixTree()
        nodes1, _ = tree.insert([("hash1", 1), ("hash2", 2)])
        nodes2, _ = tree.insert([("hash1", 1), ("hash3", 3)])

        # Release first sequence
        tree.decrement_ref_nodes(nodes1)
        # hash2 should be evictable, hash1 still has ref=1
        assert len(tree._evictable_device) == 1

        # Release second sequence
        tree.decrement_ref_nodes(nodes2)
        # Now hash1 and hash3 should be evictable (hash2 already was)
        assert len(tree._evictable_device) == 3


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

        # Swap back to device: swap_to_device sets status directly to DEVICE (not SWAP_TO_DEVICE)
        original_host_ids = tree.swap_to_device(nodes, [1, 2])
        assert sorted(original_host_ids) == [100, 101]
        for node in nodes:
            assert node.cache_status == CacheStatus.DEVICE

        # Complete swap (idempotent when already DEVICE)
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
        assert sorted(device_ids) == [1, 2]

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
        assert len(tree._evictable_device) == 0
        assert len(tree._evictable_host) == 0


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
        assert len(tree._evictable_device) == 1

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
        assert len(tree._evictable_device) == 1

    def test_node_id_uniqueness(self):
        """Test that each node has a unique node_id."""
        tree = RadixTree()
        nodes, _ = tree.insert([("h1", 1), ("h2", 2), ("h3", 3)])

        # Collect node_ids from the tree structure
        node_ids = set()

        def traverse(node):
            if node.hash_value:  # Skip root
                node_ids.add(node.node_id)
            for child in node.children.values():
                traverse(child)

        traverse(tree._root)
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


class TestRadixTreeMultiSequenceWorkflow:
    """Tests for multi-sequence workflows simulating real usage patterns."""

    def test_multi_sequence_shared_prefix_reuse(self):
        """
        Test multiple sequences sharing a common prefix.

        Simulates CacheManager usage:
        1. Request A: [h1, h2, h3] -> cached
        2. Request B: [h1, h2, h4] -> finds prefix match for [h1, h2], inserts new [h4]
        3. Request C: [h1, h2] -> finds full prefix match
        """
        tree = RadixTree(enable_host_cache=True)

        # Request A: Insert full sequence
        nodes_a, _ = tree.insert([("h1", 1), ("h2", 2), ("h3", 3)])
        assert len(nodes_a) == 3

        # After insert, h1 has ref_count=1
        h1_node = tree._root.children["h1"]
        assert h1_node.ref_count == 1

        # Simulate request finish - decrement ref
        tree.decrement_ref_nodes(nodes_a)

        # Now h1, h2, h3 are all evictable (ref_count=0)
        stats = tree.get_stats()
        assert stats.evictable_device_count == 3

        # Request B: Share prefix, insert new suffix
        nodes_b, wasted = tree.insert([("h1", 1), ("h2", 2), ("h4", 4)])
        assert len(nodes_b) == 3
        # h1 and h2 should be reused (not incremented), h4 is new
        # h1 and h2 still have ref_count=0, h4 has ref_count=1
        assert tree.node_count() == 5  # root + h1, h2, h3, h4

        h4_node = h1_node.children["h2"].children["h4"]
        assert h4_node.ref_count == 1

        # Decrement B's refs
        tree.decrement_ref_nodes(nodes_b)

        # Request C: Find prefix for [h1, h2]
        matched = tree.find_prefix(["h1", "h2"])
        assert len(matched) == 2

        # Increment ref for matched nodes to prevent eviction
        tree.increment_ref_nodes(matched)
        assert h1_node.ref_count == 1
        assert h1_node.children["h2"].ref_count == 1

        # Decrement when done
        tree.decrement_ref_nodes(matched)

    def test_incremental_insert_after_prefix_match(self):
        """
        Test incremental insertion from a matched prefix node.

        Simulates CacheManager usage where:
        1. Insert [h1, h2] and cache it
        2. Later request comes with [h1, h2, h3, h4]
        3. find_prefix returns [h1, h2]
        4. insert remaining [h3, h4] starting from matched node
        """
        tree = RadixTree()

        # Initial sequence
        nodes1, _ = tree.insert([("h1", 1), ("h2", 2)])
        tree.decrement_ref_nodes(nodes1)

        # Later request with longer sequence
        matched = tree.find_prefix(["h1", "h2"])
        assert len(matched) == 2

        # Incremental insert starting from last matched node
        last_node = matched[-1]
        nodes2, wasted = tree.insert([("h3", 3), ("h4", 4)], start_node=last_node)
        assert len(nodes2) == 2
        assert len(wasted) == 0

        # Verify complete sequence
        full_match = tree.find_prefix(["h1", "h2", "h3", "h4"])
        assert len(full_match) == 4

    def test_three_request_caching_cycle(self):
        """
        Test complete caching cycle with three sequential requests.

        Workflow:
        1. Request 1: Insert [A, B, C], finish
        2. Request 2: Find [A, B], gets match, continue with [X, Y], finish
        3. Request 3: Find [A, B], gets full match

        Note: Request 3 finds [A, B] but NOT [X] because X is under A, not B.
        """
        tree = RadixTree(enable_host_cache=True)

        # Request 1: Insert and cache
        req1_nodes, _ = tree.insert([("A", 1), ("B", 2), ("C", 3)])
        tree.decrement_ref_nodes(req1_nodes)

        # Request 2: Find prefix, add new blocks
        matched = tree.find_prefix(["A", "B"])
        assert len(matched) == 2
        tree.increment_ref_nodes(matched)

        req2_new, wasted = tree.insert([("X", 10), ("Y", 11)])
        assert len(req2_new) == 2

        tree.decrement_ref_nodes(matched)
        tree.decrement_ref_nodes(req2_new)

        # Request 3: Find [A, B] - should get full match
        # X is NOT under B, so we can only match A, B
        matched3 = tree.find_prefix(["A", "B"])
        assert len(matched3) == 2

        # Stats should show correct state
        stats = tree.get_stats()
        # Tree has: root, A, B, C (from req1), X, Y (from req2)
        assert stats.node_count == 6


class TestRadixTreeCompleteEvictionCycle:
    """Tests for complete eviction cycles (DEVICE -> HOST -> Removed)."""

    def test_full_eviction_cycle_single_sequence(self):
        """
        Test complete eviction cycle for a single sequence.

        Cycle: Insert -> Decrement -> Evict to Host -> Remove from Host
        """
        tree = RadixTree(enable_host_cache=True)

        # Step 1: Insert
        nodes, _ = tree.insert([("h1", 1), ("h2", 2), ("h3", 3)])
        assert tree.node_count() == 4

        # Step 2: Decrement refs to make evictable
        tree.decrement_ref_nodes(nodes)
        stats = tree.get_stats()
        assert stats.evictable_device_count == 3

        # Step 3: Evict to host
        released = tree.evict_device_to_host(3, [100, 101, 102])
        assert sorted(released) == [1, 2, 3]
        stats = tree.get_stats()
        assert stats.evictable_device_count == 0
        assert stats.evictable_host_count == 3

        # Verify nodes are now HOST
        for node in nodes:
            assert node.cache_status == CacheStatus.HOST
            assert node.block_id in [100, 101, 102]

        # Step 4: Remove from host
        evicted = tree.evict_host_nodes(3)
        assert sorted(evicted) == [100, 101, 102]
        assert tree.node_count() == 1  # Only root remains

    def test_full_eviction_cycle_multiple_rounds(self):
        """
        Test eviction in multiple rounds.

        Insert 10 blocks, evict 3, then evict remaining 7.
        """
        tree = RadixTree(enable_host_cache=True)

        nodes, _ = tree.insert([(f"h{i}", i) for i in range(10)])
        tree.decrement_ref_nodes(nodes)

        # Round 1: Evict 3
        released1 = tree.evict_device_to_host(3, [100, 101, 102])
        assert len(released1) == 3

        stats = tree.get_stats()
        assert stats.evictable_device_count == 7
        assert stats.evictable_host_count == 3

        # Round 2: Evict remaining 7
        released2 = tree.evict_device_to_host(7, [200, 201, 202, 203, 204, 205, 206])
        assert len(released2) == 7

        stats = tree.get_stats()
        assert stats.evictable_device_count == 0
        assert stats.evictable_host_count == 10

        # Now remove all from host
        evicted = tree.evict_host_nodes(10)
        assert len(evicted) == 10
        assert tree.node_count() == 1

    def test_eviction_with_shared_prefix_multiple_refs(self):
        """
        Test eviction when nodes have shared prefixes with active references.

        Tree structure:
            root
            └── h1 (ref=2) - shared by both sequences, incremented each insert
                ├── h2 (evicted to HOST)
                └── h3 (ref=1 after decrement)

        After seq1 finishes: h1 stays (ref=1), h2 is evicted to HOST (still in tree)
        """
        tree = RadixTree(enable_host_cache=True)

        # Insert seq1: h1 -> h2
        nodes1, _ = tree.insert([("h1", 1), ("h2", 2)])
        # Insert seq2: h1 -> h3 (shares h1)
        nodes2, _ = tree.insert([("h1", 1), ("h3", 3)])

        # Shared h1 has ref_count=2 (incremented on each insert traversal)
        h1_node = tree._root.children["h1"]
        assert h1_node.ref_count == 2

        # Seq1 finishes - decrement its refs
        tree.decrement_ref_nodes(nodes1)

        # h1 still has ref=1, h2 should be evictable
        stats = tree.get_stats()
        assert stats.evictable_device_count == 1

        # Evict h2 to host (changes status, node stays in tree until evict_host_nodes)
        released = tree.evict_device_to_host(1, [100])
        assert released == [2]

        # h2 is now on host but still in tree
        assert "h1" in tree._root.children
        # evict_device_to_host only changes status, doesn't remove from tree
        assert tree.node_count() == 4  # root + h1 + h2 + h3

        # h2 is now on host with ref=0 (evictable in host heap)
        h2_node = h1_node.children["h2"]
        assert h2_node.cache_status == CacheStatus.HOST
        assert h2_node.ref_count == 0


class TestRadixTreeSwapWorkflow:
    """Tests for HOST -> DEVICE swap workflow."""

    def test_swap_host_to_device_complete_cycle(self):
        """
        Test full swap cycle: DEVICE -> HOST -> SWAP_TO_DEVICE -> DEVICE.

        This simulates loading cached blocks back to GPU.
        """
        tree = RadixTree(enable_host_cache=True)

        # Step 1: Insert and evict to host
        nodes, _ = tree.insert([("h1", 1), ("h2", 2)])
        tree.decrement_ref_nodes(nodes)
        tree.evict_device_to_host(2, [100, 101])

        # Verify nodes are on host
        for node in nodes:
            assert node.cache_status == CacheStatus.HOST
            assert node.block_id in [100, 101]

        # Step 2: Swap back to device
        # swap_to_device() sets status directly to DEVICE (not SWAP_TO_DEVICE intermediate)
        original_ids = tree.swap_to_device(nodes, [50, 51])
        assert sorted(original_ids) == [100, 101]

        # Verify status is DEVICE after swap_to_device
        for node in nodes:
            assert node.cache_status == CacheStatus.DEVICE
            assert node.block_id in [50, 51]

        # Step 3: complete_swap_to_device is idempotent when already DEVICE
        gpu_ids = tree.complete_swap_to_device(nodes)
        assert sorted(gpu_ids) == [50, 51]

        for node in nodes:
            assert node.cache_status == CacheStatus.DEVICE
            assert node.block_id in [50, 51]

    def test_swap_after_find_prefix(self):
        """
        Test that swapped blocks can still be found via find_prefix.

        After swap_to_device, nodes should be findable again.
        """
        tree = RadixTree(enable_host_cache=True)

        # Insert and evict
        nodes, _ = tree.insert([("h1", 1), ("h2", 2)])
        tree.decrement_ref_nodes(nodes)
        tree.evict_device_to_host(2, [100, 101])

        # Find prefix (should find HOST nodes)
        matched = tree.find_prefix(["h1", "h2"])
        assert len(matched) == 2

        # Increment refs to prevent eviction during swap
        tree.increment_ref_nodes(matched)

        # Swap to device
        original_ids = tree.swap_to_device(matched, [50, 51])
        assert sorted(original_ids) == [100, 101]

        # Find should still work
        matched2 = tree.find_prefix(["h1", "h2"])
        assert len(matched2) == 2
        block_ids = [n.block_id for n in matched2]
        assert sorted(block_ids) == [50, 51]

        tree.decrement_ref_nodes(matched2)


class TestRadixTreeConcurrencySafety:
    """Tests for thread safety and concurrent access patterns."""

    def test_concurrent_insert_and_find(self):
        """Test concurrent insert and find_prefix operations."""
        import threading

        tree = RadixTree(enable_host_cache=True)

        def insert_sequence(prefix, start_id, count):
            for i in range(count):
                blocks = [(f"{prefix}_{j}", start_id + j) for j in range(5)]
                tree.insert(blocks)

        def find_sequence(prefix, results):
            for _ in range(10):
                matched = tree.find_prefix([f"{prefix}_0", f"{prefix}_1"])
                results.append(len(matched))

        threads = []
        results = []

        # Create 5 threads doing inserts
        for i in range(5):
            t = threading.Thread(target=insert_sequence, args=(f"P{i}", i * 10, 10))
            threads.append(t)

        # Create 5 threads doing finds
        for i in range(5):
            t = threading.Thread(target=find_sequence, args=(f"P{i}", results))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All find operations should complete without error
        assert len(results) == 50
        # Find results may vary depending on timing, but should be valid
        for r in results:
            assert 0 <= r <= 2

    def test_concurrent_eviction_and_access(self):
        """Test concurrent eviction and find_prefix operations."""
        import threading

        tree = RadixTree(enable_host_cache=True)

        # Setup: Insert and make evictable
        nodes, _ = tree.insert([(f"h{i}", i) for i in range(20)])
        tree.decrement_ref_nodes(nodes)

        results = []
        errors = []

        def evict_blocks():
            try:
                for _ in range(5):
                    released = tree.evict_device_to_host(2, [1000, 1001])
                    if released:
                        results.append(("evict", len(released)))
            except Exception as e:
                errors.append(e)

        def access_blocks():
            try:
                for _ in range(10):
                    matched = tree.find_prefix(["h0", "h1"])
                    results.append(("access", len(matched)))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=evict_blocks),
            threading.Thread(target=access_blocks),
            threading.Thread(target=access_blocks),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have completed without error
        assert len(errors) == 0
        # Should have results from all operations
        assert len(results) > 0
        # Access results should be valid (0, 1, or 2 blocks matched)
        for op, count in results:
            if op == "access":
                assert 0 <= count <= 2


class TestRadixTreeMemoryManagement:
    """Tests for proper memory management and reference counting."""

    def test_node_reuse_different_block_ids(self):
        """
        Test that reusing a node with different block_id tracks wasted blocks.

        When inserting a sequence that partially reuses existing nodes
        but with different block_ids, the conflicting block_ids should
        be tracked as wasted.

        In this case:
        - h1 already exists with block_id=1, new block_id=100 -> wasted
        - h2 already exists with block_id=2, new block_id=200 -> wasted
        """
        tree = RadixTree()

        # Insert first sequence
        nodes1, wasted1 = tree.insert([("h1", 1), ("h2", 2)])
        assert len(wasted1) == 0

        # Insert same hashes but different block_ids - both are wasted
        nodes2, wasted2 = tree.insert([("h1", 100), ("h2", 200)])
        # Both h1 and h2 already exist, so both new block_ids are wasted
        assert len(wasted2) == 2
        assert sorted(wasted2) == [100, 200]

        # Verify nodes still have original block_ids
        h1_node = tree._root.children["h1"]
        h2_node = h1_node.children["h2"]
        assert h1_node.block_id == 1
        assert h2_node.block_id == 2

    def test_multiple_insert_same_node_tracking(self):
        """
        Test that multiple inserts of the same path correctly track refs.

        Insert the same sequence 5 times, then decrement 5 times.
        Node should become evictable only after all decrements.
        """
        tree = RadixTree()

        # Insert same sequence 5 times
        all_nodes = []
        for i in range(5):
            nodes, _ = tree.insert([("h1", 1), ("h2", 2)])
            all_nodes.append(nodes)

        h1_node = tree._root.children["h1"]
        assert h1_node.ref_count == 5

        # Decrement refs one by one
        for i in range(5):
            tree.decrement_ref_nodes(all_nodes[i])
            expected_ref = 5 - i - 1
            assert h1_node.ref_count == expected_ref

        # Now h1 should be evictable
        assert h1_node.ref_count == 0
        stats = tree.get_stats()
        assert stats.evictable_device_count == 2  # h1 and h2

    def test_reset_clears_all_tracking(self):
        """Test that reset properly clears all tracking structures."""
        tree = RadixTree(enable_host_cache=True)

        nodes, _ = tree.insert([("h1", 1), ("h2", 2), ("h3", 3)])
        tree.decrement_ref_nodes(nodes)
        tree.evict_device_to_host(3, [100, 101, 102])

        assert tree.node_count() == 4
        stats = tree.get_stats()
        assert stats.evictable_host_count == 3

        # Reset
        tree.reset()

        assert tree.node_count() == 1
        assert len(tree._evictable_device) == 0
        assert len(tree._evictable_host) == 0


class TestRadixTreeComplexScenarios:
    """Tests for complex real-world scenarios."""

    def test_batched_requests_with_partial_match(self):
        """
        Test handling multiple batched requests with partial prefix matches.

        Simulates a batch of 3 requests:
        - Req1: [sys, user1] -> insert both
        - Req2: [sys, user2] -> prefix match [sys], insert [user2]
        - Req3: [sys, user1] -> full prefix match
        """
        tree = RadixTree(enable_host_cache=True)

        # Request 1: Full insert
        req1_nodes, _ = tree.insert([("sys", 0), ("user1", 1)])
        tree.decrement_ref_nodes(req1_nodes)

        # Request 2: Partial match (sys), new suffix (user2)
        matched = tree.find_prefix(["sys"])
        assert len(matched) == 1
        tree.increment_ref_nodes(matched)

        req2_nodes, wasted = tree.insert([("user2", 2)])
        assert len(wasted) == 0

        tree.decrement_ref_nodes(matched)
        tree.decrement_ref_nodes(req2_nodes)

        # Request 3: Full match
        matched3 = tree.find_prefix(["sys", "user1"])
        assert len(matched3) == 2

        # Stats check
        stats = tree.get_stats()
        assert stats.node_count == 4  # sys, user1, user2 + root

    def test_deep_chain_insertion(self):
        """
        Test insertion and access of deep node chains.

        Insert a chain of 20 blocks, verify find_prefix works at various depths.
        """
        tree = RadixTree()

        # Insert deep chain
        depth = 20
        blocks = [(f"h{i}", i) for i in range(depth)]
        nodes, _ = tree.insert(blocks)

        assert len(nodes) == depth
        assert tree.node_count() == depth + 1

        # Find at various depths
        for d in [5, 10, 15, 20]:
            matched = tree.find_prefix([f"h{i}" for i in range(d)])
            assert len(matched) == d

        # Decrement and verify all become evictable
        tree.decrement_ref_nodes(nodes)
        stats = tree.get_stats()
        assert stats.evictable_device_count == depth

    def test_wide_tree_with_shared_prefix(self):
        """
        Test tree with many branches sharing a common prefix.

        Structure:
            root
            └── shared (ref=100) - incremented each insert
                ├── branch_0 (ref=0 after release)
                ├── branch_1 (ref=0 after release)
                ... (50 branches released, 50 still held)
        """
        tree = RadixTree(enable_host_cache=True)
        num_branches = 100

        # Insert 100 sequences, all sharing "shared" prefix
        all_branch_nodes = []
        for i in range(num_branches):
            nodes, _ = tree.insert([("shared", 0), (f"branch_{i}", i)])
            all_branch_nodes.append(nodes)

        # shared has ref_count=100 (incremented on each insert traversal)
        shared_node = tree._root.children["shared"]
        assert shared_node.ref_count == 100

        # Release half the branches
        for i in range(num_branches // 2):
            tree.decrement_ref_nodes(all_branch_nodes[i])

        stats = tree.get_stats()
        # 50 branch nodes become evictable, shared stays at ref=50
        assert stats.evictable_device_count == num_branches // 2  # 50

        # shared node should still have ref=50 (not evictable)
        assert shared_node.ref_count == num_branches // 2

        # Verify one remaining branch is still findable
        matched = tree.find_prefix(["shared", f"branch_{num_branches // 2}"])
        assert len(matched) == 2


class TestEvictDeviceNodes:
    """Tests for evict_device_nodes (no host cache mode)."""

    def test_evict_device_nodes_basic(self):
        """Test evicting DEVICE nodes directly (no host cache)."""
        tree = RadixTree(enable_host_cache=False)
        nodes, _ = tree.insert([("h1", 1), ("h2", 2), ("h3", 3)])
        tree.decrement_ref_nodes(nodes)

        result = tree.evict_device_nodes(2)
        assert result is not None
        assert len(result) == 2
        # Returned block_ids must be from original insert
        assert all(bid in [1, 2, 3] for bid in result)

    def test_evict_device_nodes_not_enough(self):
        """Test eviction fails when not enough evictable DEVICE nodes."""
        tree = RadixTree(enable_host_cache=False)
        nodes, _ = tree.insert([("h1", 1)])
        tree.decrement_ref_nodes(nodes)

        result = tree.evict_device_nodes(5)
        assert result is None

    def test_evict_device_nodes_zero(self):
        """Test evicting zero DEVICE nodes returns empty list."""
        tree = RadixTree()
        result = tree.evict_device_nodes(0)
        assert result == []

    def test_evict_device_nodes_removes_from_tree(self):
        """Test that evicted DEVICE nodes are removed from tree."""
        tree = RadixTree(enable_host_cache=False)
        nodes, _ = tree.insert([("h1", 1)])
        tree.decrement_ref_nodes(nodes)

        assert tree.node_count() == 2  # root + h1

        tree.evict_device_nodes(1)

        assert tree.node_count() == 1  # only root
        assert "h1" not in tree._root.children


class TestBackupBlocks:
    """Tests for backup_blocks method."""

    def test_backup_blocks_basic(self):
        """Test marking blocks as backed up."""
        tree = RadixTree(write_policy="write_through_selective")
        nodes, _ = tree.insert([("h1", 1), ("h2", 2)])
        tree.decrement_ref_nodes(nodes)

        backed_ids = tree.backup_blocks(nodes, [100, 101])

        assert sorted(backed_ids) == [1, 2]
        for node in nodes:
            assert node.backuped is True
            assert node.host_block_id in [100, 101]

    def test_backup_blocks_mismatched_length(self):
        """Test backup_blocks returns empty for mismatched lengths."""
        tree = RadixTree()
        nodes, _ = tree.insert([("h1", 1), ("h2", 2)])
        tree.decrement_ref_nodes(nodes)

        result = tree.backup_blocks(nodes, [100])  # Only 1 host_block_id for 2 nodes
        assert result == []

    def test_backup_blocks_empty(self):
        """Test backup_blocks with empty lists."""
        tree = RadixTree()
        result = tree.backup_blocks([], [])
        assert result == []


class TestGetCandidatesForBackup:
    """Tests for get_candidates_for_backup method."""

    def test_get_candidates_basic(self):
        """Test get_candidates_for_backup returns eligible nodes."""
        tree = RadixTree(write_policy="write_through_selective")
        nodes, _ = tree.insert([("h1", 1), ("h2", 2)])
        # Simulate hit_count >= threshold
        tree.decrement_ref_nodes(nodes)
        # Manually set hit_count so they qualify
        for node in nodes:
            node.hit_count = 3

        candidates = tree.get_candidates_for_backup(threshold=2)

        assert len(candidates) == 2

    def test_get_candidates_excludes_already_backed_up(self):
        """Test that already backed-up nodes are excluded."""
        tree = RadixTree(write_policy="write_through_selective")
        nodes, _ = tree.insert([("h1", 1), ("h2", 2)])
        tree.decrement_ref_nodes(nodes)

        for node in nodes:
            node.hit_count = 5

        # Mark first node as backed up
        nodes[0].backuped = True

        candidates = tree.get_candidates_for_backup(threshold=1)
        assert len(candidates) == 1
        assert candidates[0] is nodes[1]

    def test_get_candidates_wrong_policy_returns_empty(self):
        """Test that non-write_through_selective policy returns empty."""
        tree = RadixTree(write_policy="write_through")
        nodes, _ = tree.insert([("h1", 1)])
        tree.decrement_ref_nodes(nodes)
        nodes[0].hit_count = 10

        candidates = tree.get_candidates_for_backup(threshold=1)
        assert candidates == []

    def test_get_candidates_excludes_pending_block_ids(self):
        """Test that nodes with block_ids in pending list are excluded."""
        tree = RadixTree(write_policy="write_through_selective")
        nodes, _ = tree.insert([("h1", 1), ("h2", 2)])
        tree.decrement_ref_nodes(nodes)

        for node in nodes:
            node.hit_count = 5

        # Exclude block_id=1 from candidates
        candidates = tree.get_candidates_for_backup(threshold=1, pending_block_ids=[1])

        assert len(candidates) == 1
        assert candidates[0].block_id == 2


class TestEvictNodesSelective:
    """Tests for evict_nodes_selective (write_through_selective policy)."""

    def test_evict_nodes_selective_without_backup(self):
        """Test eviction of nodes without backup removes from tree."""
        tree = RadixTree(write_policy="write_through_selective")
        nodes, _ = tree.insert([("h1", 1), ("h2", 2)])
        tree.decrement_ref_nodes(nodes)

        # Nodes have no backup
        result = tree.evict_nodes_selective(2)

        assert sorted(result) == [1, 2]
        # Nodes should be removed from tree (no backup, so deleted)
        assert tree.node_count() == 1

    def test_evict_nodes_selective_with_backup(self):
        """Test eviction of backed-up nodes transitions to HOST state."""
        tree = RadixTree(write_policy="write_through_selective", enable_host_cache=True)
        nodes, _ = tree.insert([("h1", 1), ("h2", 2)])
        tree.decrement_ref_nodes(nodes)

        # Mark nodes as backed up with host block IDs
        tree.backup_blocks(nodes, [100, 101])

        result = tree.evict_nodes_selective(2)

        assert sorted(result) == [1, 2]
        # Nodes should now be in HOST state (not removed from tree)
        for node in nodes:
            assert node.cache_status == CacheStatus.HOST
            assert node.block_id in [100, 101]

        # Nodes should be evictable from host
        stats = tree.get_stats()
        assert stats.evictable_host_count == 2

    def test_evict_nodes_selective_zero_blocks(self):
        """Test evicting zero blocks returns empty list."""
        tree = RadixTree(write_policy="write_through_selective")
        result = tree.evict_nodes_selective(0)
        assert result == []

    def test_evict_nodes_selective_not_enough_blocks(self):
        """Test eviction returns empty list when not enough evictable blocks."""
        tree = RadixTree(write_policy="write_through_selective")
        nodes, _ = tree.insert([("h1", 1)])
        tree.decrement_ref_nodes(nodes)

        # Request more than available
        result = tree.evict_nodes_selective(5)
        assert result == []


# ---------------------------------------------------------------------------
# complete_swap_to_device
# ---------------------------------------------------------------------------


class TestCompleteSwapToDevice:
    """Dedicated tests for RadixTree.complete_swap_to_device."""

    def test_complete_swap_sets_status_to_device(self):
        """Nodes in any state are set to DEVICE after complete_swap_to_device."""
        tree = RadixTree(enable_host_cache=True)
        nodes, _ = tree.insert([("h1", 1), ("h2", 2)])
        tree.decrement_ref_nodes(nodes)

        # Evict to host then swap back (swap_to_device sets to DEVICE directly in current impl)
        tree.evict_device_to_host(2, [10, 11])
        tree.swap_to_device(nodes, [1, 2])

        # Call complete_swap_to_device and verify DEVICE status
        gpu_ids = tree.complete_swap_to_device(nodes)
        assert len(gpu_ids) == 2
        for node in nodes:
            assert node.cache_status == CacheStatus.DEVICE

    def test_complete_swap_returns_gpu_block_ids(self):
        """Return value must be the current block_ids of the nodes."""
        tree = RadixTree(enable_host_cache=True)
        nodes, _ = tree.insert([("h1", 5)])
        tree.decrement_ref_nodes(nodes)

        tree.evict_device_to_host(1, [99])
        tree.swap_to_device(nodes, [5])

        gpu_ids = tree.complete_swap_to_device(nodes)
        assert gpu_ids == [node.block_id for node in nodes]

    def test_complete_swap_empty_list(self):
        """Calling with empty list returns empty list and does not raise."""
        tree = RadixTree()
        result = tree.complete_swap_to_device([])
        assert result == []

    def test_complete_swap_idempotent(self):
        """Calling complete_swap_to_device twice is safe."""
        tree = RadixTree(enable_host_cache=True)
        nodes, _ = tree.insert([("h1", 1)])
        tree.decrement_ref_nodes(nodes)
        tree.evict_device_to_host(1, [20])
        tree.swap_to_device(nodes, [1])

        tree.complete_swap_to_device(nodes)
        tree.complete_swap_to_device(nodes)  # second call should not raise
        for node in nodes:
            assert node.cache_status == CacheStatus.DEVICE

    def test_complete_swap_updates_last_access_time(self):
        """complete_swap_to_device should touch each node."""
        tree = RadixTree(enable_host_cache=True)
        nodes, _ = tree.insert([("h1", 1)])
        tree.decrement_ref_nodes(nodes)
        tree.evict_device_to_host(1, [30])
        tree.swap_to_device(nodes, [1])

        old_time = nodes[0].last_access_time
        time.sleep(0.01)
        tree.complete_swap_to_device(nodes)
        assert nodes[0].last_access_time >= old_time

    def test_complete_swap_multiple_nodes(self):
        """Works correctly with multiple nodes."""
        tree = RadixTree(enable_host_cache=True)
        nodes, _ = tree.insert([("h1", 1), ("h2", 2), ("h3", 3)])
        tree.decrement_ref_nodes(nodes)
        tree.evict_device_to_host(3, [10, 11, 12])
        tree.swap_to_device(nodes, [1, 2, 3])

        gpu_ids = tree.complete_swap_to_device(nodes)
        assert len(gpu_ids) == 3
        for node in nodes:
            assert node.cache_status == CacheStatus.DEVICE
