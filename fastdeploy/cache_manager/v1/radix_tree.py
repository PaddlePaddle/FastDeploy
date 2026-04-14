"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import heapq
import threading
from typing import Dict, List, Optional, Tuple

from fastdeploy.utils import get_logger

from .metadata import BlockNode, CacheStatus, RadixTreeStats

logger = get_logger("radix_tree", "cache_manager.log")


class RadixTree:
    """
    Radix tree for efficient prefix matching in KV cache.

    Used to find matching prefixes across different sequences,
    enabling KV cache reuse for shared prefixes.

    Uses separate min-heaps for DEVICE and HOST evictable nodes with true deletion,
    ensuring heap contents are always consistent with the evictable set.

    API Usage Guidelines
    ====================

    1. Reference Count Management (CRITICAL)
    -----------------------------------------
    The reference count (ref_count) determines whether a node can be evicted.
    A node is evictable ONLY when ref_count == 0.

    IMPORTANT: You MUST pair increment_ref_nodes() and decrement_ref_nodes() calls:
    - After insert(): nodes have ref_count >= 1, NOT evictable
    - After decrement_ref_nodes(): ref_count decreases, may become evictable
    - After increment_ref_nodes(): ref_count increases, removed from evictable set

    WARNING: Unbalanced ref_count management can cause:
    - Memory leaks: nodes never become evictable (ref_count > 0 forever)
    - Premature eviction: nodes evicted while still in use (ref_count == 0)

    Example:
        nodes, wasted_ids = tree.insert(blocks)  # ref_count = 1, wasted_ids may be non-empty if nodes were reused
        if wasted_ids:
            # Release wasted block_ids that were not used due to node reuse
            release_blocks(wasted_ids)
        # ... use the nodes ...
        tree.decrement_ref_nodes(nodes)       # ref_count = 0, now evictable
        # Do NOT use nodes after decrement - they may be evicted!

    2. Eviction Operation Order
    ---------------------------
    The correct eviction order is:

        DEVICE -> HOST -> Storage

    Step 1: evict_device_to_host() - Move DEVICE nodes to HOST
        - Input: num_blocks, host_block_ids (pre-allocated)
        - Output: released device block_ids
        - Nodes transition: DEVICE -> HOST (still in tree)

    Step 2: evict_host_nodes() - Remove HOST nodes permanently
        - Input: num_blocks
        - Output: evicted host block_ids
        - Nodes removed from tree completely

    WARNING: Do NOT call evict_host_nodes() before evict_device_to_host() for
    the same nodes - this will fail since nodes are still in DEVICE state.

    3. Atomicity Guarantee
    ----------------------
    All eviction methods provide atomic operation:
    - Pre-check: verify enough evictable nodes exist
    - If pre-check fails, return None immediately (no partial eviction)
    - If success, all requested blocks are processed

    Check return value:
    - None: Not enough evictable blocks, operation failed
    - Empty list: num_blocks == 0, nothing to do
    - List of block_ids: Success

    4. Thread Safety
    ----------------
    All public methods are thread-safe using RLock.
    However, be careful with the following pattern:

    WARNING: Do NOT hold references to nodes across method calls:
        # DANGEROUS - node may be evicted by another thread
        nodes = tree.find_prefix(hashes)
        # ... some operation without lock ...
        tree.increment_ref_nodes(nodes)  # nodes may already be evicted!

    Instead, use the returned nodes immediately:
        nodes = tree.find_prefix(hashes)
        tree.increment_ref_nodes(nodes)  # Safe: immediate operation

    5. Node Lifecycle
    -----------------
    Node states and valid transitions:

        [New] --insert()--> DEVICE (ref_count >= 1)
        DEVICE --decrement_ref()--> DEVICE (ref_count == 0, evictable)
        DEVICE --evict_device_to_host()--> HOST (ref_count == 0)
        HOST --evict_host_nodes()--> [Deleted from tree]

        HOST --swap_to_device()--> SWAP_TO_DEVICE
        SWAP_TO_DEVICE --complete_swap_to_device()--> DEVICE

    WARNING: Once a node's ref_count becomes 0, it can be evicted at any time.
    Do NOT access or modify a node after decrementing its ref_count unless
    you increment it first.

    6. Common Pitfalls
    ------------------
    a) Forgetting to decrement ref_count after use:
       -> Memory leak, blocks never released

    b) Decrementing ref_count multiple times:
       -> ref_count becomes negative, undefined behavior

    c) Using nodes after decrement_ref_nodes():
       -> Nodes may be evicted, accessing invalid memory

    d) Evicting nodes with ref_count > 0:
       -> Not possible, eviction methods skip non-zero ref_count nodes

    e) Calling find_prefix() on DELETING/SWAP_TO_HOST nodes:
       -> These states are skipped, prefix match stops at these nodes
    """

    def __init__(
        self,
        enable_host_cache: bool = False,
        write_policy: str = "write_through",
    ):
        """
        Initialize the radix tree.

        Args:
            enable_host_cache: If True, evict() moves nodes to HOST state
                              instead of removing them from tree.
            write_policy: Write policy for backup to lower tier.
                          - "write_through": Every matched node triggers backup check
                          - "write_through_selective": Only nodes with hit_count >= threshold trigger backup
                          - "write_back": Backup only when evicted (not implemented yet)
        """
        self._root = BlockNode()
        self._lock = threading.RLock()
        self._node_count = 1  # Root node
        self._enable_host_cache = enable_host_cache
        self._write_policy = write_policy

        # Use dict for O(1) add/remove instead of heap's O(n) removal
        # Format: {node_id: (last_access_time, node)}
        self._evictable_device: Dict[str, Tuple[float, BlockNode]] = {}
        self._evictable_host: Dict[str, Tuple[float, BlockNode]] = {}

    def insert(
        self,
        blocks: List[Tuple[str, int]],
        cache_status: CacheStatus = CacheStatus.DEVICE,
        start_node: Optional[BlockNode] = None,
    ) -> Tuple[List[BlockNode], List[int]]:
        """
        Insert a sequence of blocks into the tree.

        Args:
            blocks: List of (block_hash, block_id) tuples.
                    Each tuple represents a complete block.
            cache_status: Initial cache status for new nodes.
                         Defaults to DEVICE.
            start_node: Node to start insertion from. If None, starts from root.
                       Used for incremental insertion after prefix match.

        Returns:
            Tuple of (result_nodes, wasted_block_ids):
            - result_nodes: List of inserted or updated BlockNode objects.
            - wasted_block_ids: List of block_ids that were not used due to
              node reuse (should be released by caller).
        """
        result_nodes = []
        wasted_block_ids = []

        if not blocks:
            return result_nodes, wasted_block_ids

        with self._lock:
            node = self._root if start_node is None else start_node
            for i, (block_hash, block_id) in enumerate(blocks):
                if block_hash not in node.children:
                    # Create new BlockNode with block_id, parent, and hash_value
                    new_node = BlockNode(
                        block_id=block_id,
                        parent=node,
                        hash_value=block_hash,
                        cache_status=cache_status,
                    )
                    node.children[block_hash] = new_node
                    self._node_count += 1
                else:
                    # Node already exists for this hash - the new block_id is wasted
                    existing_node = node.children[block_hash]
                    if existing_node.block_id != block_id:
                        # Track the wasted block_id for caller to release
                        wasted_block_ids.append(block_id)

                node = node.children[block_hash]
                # Increment ref and update evictable status
                node.increment_ref()
                # If node in evictable, remove it from evictable dict
                if node.cache_status == CacheStatus.DEVICE and node.node_id in self._evictable_device:
                    del self._evictable_device[node.node_id]
                elif node.cache_status == CacheStatus.HOST and node.node_id in self._evictable_host:
                    del self._evictable_host[node.node_id]
                result_nodes.append(node)

        return result_nodes, wasted_block_ids

    def find_prefix(
        self,
        block_hashes: List[str],
    ) -> List[BlockNode]:
        """
        Find the longest matching prefix.

        Args:
            block_hashes: List of block hash values to match.

        Returns:
            List of matched BlockNode objects in order.
            Empty list if no match found.
        """
        matched_nodes = []

        with self._lock:
            node = self._root
            for i, block_hash in enumerate(block_hashes):
                if block_hash not in node.children:
                    break

                node = node.children[block_hash]
                if node.cache_status in (CacheStatus.DELETING, CacheStatus.SWAP_TO_HOST):
                    break

                node.touch()
                matched_nodes.append(node)

        return matched_nodes

    def increment_ref_nodes(self, nodes: List[BlockNode]) -> None:
        """
        Increment reference count for a list of nodes.

        Removes nodes from evictable set (no longer available for eviction).
        Also updates last_access_time for each node.

        Args:
            nodes: List of BlockNode objects to increment ref_count.
        """
        if not nodes:
            return
        with self._lock:
            for node in nodes:
                node.increment_ref()
                node.hit_count += 1
                node.touch()
                self._remove_from_evictable(node)

    def decrement_ref_nodes(self, nodes: List[BlockNode]) -> None:
        """
        Decrement reference count for a list of nodes.

        When ref_count becomes 0, the node is added to evictable heap
        and becomes available for eviction. Also updates last_access_time.

        Args:
            nodes: List of BlockNode objects to decrement ref_count.
        """
        if not nodes:
            return
        with self._lock:
            for node in nodes:
                old_ref = node.ref_count
                node.decrement_ref()
                node.touch()
                # If ref_count goes from 1 to 0, add to evictable
                if old_ref == 1 and node.ref_count == 0:
                    self._add_to_evictable(node)

    def reset(self) -> None:
        """
        Reset the tree to initial state.

        Clears all nodes except root, evictable tracking, and node mappings.
        """
        with self._lock:
            self._root = BlockNode(block_id=0)
            self._node_count = 1
            self._evictable_device.clear()
            self._evictable_host.clear()

    def get_stats(self) -> RadixTreeStats:
        """
        Get tree statistics snapshot.

        Returns a snapshot of all tree statistics. Using a snapshot ensures
        consistent values across all fields in a single call.

        Returns:
            RadixTreeStats containing all tree statistics.
        """
        return RadixTreeStats(
            node_count=self._node_count,
            evictable_device_count=len(self._evictable_device),
            evictable_host_count=len(self._evictable_host),
        )

    def node_count(self) -> int:
        """Get total number of nodes in the tree."""
        return self._node_count

    def evict_host_nodes(
        self,
        num_blocks: int,
    ) -> Optional[List[int]]:
        """
        Evict HOST nodes from the tree.

        Removes HOST nodes permanently and returns their block_ids.

        Args:
            num_blocks: Number of HOST blocks to evict

        Returns:
            List of evicted host block_ids, or None if not enough
            evictable HOST blocks.
        """
        if num_blocks == 0:
            return []

        with self._lock:
            if len(self._evictable_host) < num_blocks:
                return None

            nodes = self._get_lru_nodes(self._evictable_host, num_blocks)
            evicted_block_ids = []

            for node in nodes:
                self._remove_node_from_tree(node)
                evicted_block_ids.append(node.block_id)

            logger.debug(
                f"evict_host_nodes: evicted={evicted_block_ids}, " f"remaining_host={len(self._evictable_host)}"
            )

        return evicted_block_ids

    def _get_lru_nodes(
        self,
        evictable_dict: Dict[str, Tuple[float, BlockNode]],
        num_blocks: int,
    ) -> List[BlockNode]:
        """
        Get the coldest (LRU) nodes from an evictable dict.

        Args:
            evictable_dict: The evictable dict to get nodes from (_evictable_device or _evictable_host).
            num_blocks: Number of nodes to get.

        Returns:
            List of BlockNode objects in LRU order (coldest first).
        """
        if num_blocks <= 0 or not evictable_dict:
            return []

        smallest = heapq.nsmallest(
            min(num_blocks, len(evictable_dict)), evictable_dict.items(), key=lambda item: item[1][0]
        )

        nodes = [node for _, (_, node) in smallest]
        for node_id, _ in smallest:
            del evictable_dict[node_id]
        return nodes

    def evict_device_nodes(
        self,
        num_blocks: int,
    ) -> Optional[List[int]]:
        """
        Evict DEVICE nodes from the tree directly.

        Removes DEVICE nodes permanently without moving to HOST.
        This is used when host cache is disabled.

        Args:
            num_blocks: Number of DEVICE blocks to evict.

        Returns:
            List of evicted device block_ids, or None if not enough
            evictable DEVICE blocks.
        """
        if num_blocks == 0:
            return []

        with self._lock:
            if len(self._evictable_device) < num_blocks:
                return None

            nodes = self._get_lru_nodes(self._evictable_device, num_blocks)
            evicted_block_ids = []

            for node in nodes:
                self._remove_node_from_tree(node)
                evicted_block_ids.append(node.block_id)

            logger.debug(
                f"evict_device_nodes: evicted={evicted_block_ids}, " f"remaining_device={len(self._evictable_device)}"
            )

        return evicted_block_ids

    def evict_device_to_host(
        self,
        num_blocks: int,
        host_block_ids: List[int],
    ) -> Optional[List[int]]:
        """
        Evict DEVICE nodes to host memory.

        Changes node status from DEVICE to HOST and updates block_id
        to the provided host_block_ids.

        Args:
            num_blocks: Number of DEVICE blocks to evict
            host_block_ids: Pre-allocated host block IDs to use

        Returns:
            List of released device block_ids, or None if not enough
            evictable DEVICE blocks.
        """
        if num_blocks == 0:
            return []

        if len(host_block_ids) < num_blocks:
            return None

        released_block_ids = []

        with self._lock:
            if len(self._evictable_device) < num_blocks:
                return None

            nodes = self._get_lru_nodes(self._evictable_device, num_blocks)
            released_block_ids = []

            for i, node in enumerate(nodes):
                # Save the original device block_id
                original_block_id = node.block_id
                new_host_block_id = host_block_ids[i]

                # Update status and block_id
                node.cache_status = CacheStatus.HOST
                node.block_id = new_host_block_id
                node.touch()

                # Add to host evictable dict
                self._evictable_host[node.node_id] = (node.last_access_time, node)

                released_block_ids.append(original_block_id)

            logger.debug(
                f"evict_device_to_host: released_device={released_block_ids} -> host={host_block_ids[:len(released_block_ids)]}, "
                f"evictable_device={len(self._evictable_device)}, evictable_host={len(self._evictable_host)}"
            )

        return released_block_ids

    def _add_to_evictable(self, node: BlockNode) -> None:
        """
        Add a node to the appropriate evictable dict based on cache status.
        """
        if node.cache_status == CacheStatus.DEVICE:
            if node.node_id not in self._evictable_device:
                self._evictable_device[node.node_id] = (node.last_access_time, node)
        elif node.cache_status == CacheStatus.HOST:
            if node.node_id not in self._evictable_host:
                self._evictable_host[node.node_id] = (node.last_access_time, node)

    def _remove_from_evictable(self, node: BlockNode) -> None:
        """
        Remove a node from evictable tracking (O(1) deletion from dict).
        """
        if node.cache_status == CacheStatus.DEVICE and node.node_id in self._evictable_device:
            del self._evictable_device[node.node_id]
        elif node.cache_status == CacheStatus.HOST and node.node_id in self._evictable_host:
            del self._evictable_host[node.node_id]

    def _remove_node_from_tree(self, node: BlockNode) -> None:
        """
        Remove a single node from the tree permanently.

        Args:
            node: Node to remove
        """
        if node.parent is None:
            return  # Cannot remove root

        # Remove from parent's children
        if node.hash_value and node.hash_value in node.parent.children:
            del node.parent.children[node.hash_value]
            self._node_count -= 1

    def swap_to_device(
        self,
        nodes: List[BlockNode],
        gpu_block_ids: List[int],
    ) -> List[int]:
        """
        Swap CPU blocks to device.

        Changes node status to SWAP_TO_DEVICE and updates block_id to GPU block ID.
        This is used when loading host blocks back to device memory.

        Args:
            nodes: List of BlockNode objects on host to swap to device.
                   Caller guarantees all nodes are on HOST.
            gpu_block_ids: Corresponding GPU block IDs

        Returns:
            List of original host block_ids
        """
        if len(nodes) != len(gpu_block_ids):
            return []

        original_block_ids = []

        with self._lock:
            for node, gpu_block_id in zip(nodes, gpu_block_ids):
                # Save the original host block_id
                original_block_ids.append(node.block_id)

                # Remove from evictable before changing status
                self._remove_from_evictable(node)

                # Update status to SWAP_TO_DEVICE and block_id to GPU block ID
                node.cache_status = CacheStatus.DEVICE  # Temporary status for test
                node.block_id = gpu_block_id
                node.touch()

        return original_block_ids

    def complete_swap_to_device(
        self,
        nodes: List[BlockNode],
    ) -> List[int]:
        """
        Complete the swap to device operation.

        Changes node status from SWAP_TO_DEVICE to DEVICE.
        This should be called after the actual data transfer is complete.

        Args:
            nodes: List of BlockNode objects that were swapped to device

        Returns:
            List of GPU block_ids
        """
        gpu_block_ids = []

        with self._lock:
            for node in nodes:
                # Update status to DEVICE
                node.cache_status = CacheStatus.DEVICE
                node.touch()

                gpu_block_ids.append(node.block_id)

        return gpu_block_ids

    def backup_blocks(
        self,
        nodes: List[BlockNode],
        host_block_ids: List[int],
    ) -> List[int]:
        """
        Mark blocks as backed up and record their host block IDs.

        This method marks the given nodes as backuped and stores the
        host block IDs. It does NOT perform the actual data transfer -
        that should be done by the caller via cache_evict_metadata.

        Args:
            nodes: List of BlockNode objects to backup
            host_block_ids: Corresponding host block IDs for the backup

        Returns:
            List of device block IDs that were marked as backuped
        """
        if len(nodes) != len(host_block_ids):
            return []

        backed_up_ids = []

        with self._lock:
            for node, host_block_id in zip(nodes, host_block_ids):
                node.backuped = True
                node.host_block_id = host_block_id
                backed_up_ids.append(node.block_id)

        return backed_up_ids

    def get_candidates_for_backup(self, threshold: int, pending_block_ids: list[int] = []) -> List[BlockNode]:
        """
        Get nodes that are candidates for backup based on write_through_selective policy.

        Returns evictable device nodes that:
        1. Have hit_count >= threshold
        2. Are not already backed up

        Args:
            threshold: Minimum hit_count required for backup candidacy.
            pending_block_ids: List of block IDs already in the pending backup queue,
                               used to avoid duplicate scheduling.

        Returns:
            List of BlockNode objects that are candidates for backup,
            sorted by LRU (coldest first).
        """
        if self._write_policy != "write_through_selective":
            return []

        candidates = []
        with self._lock:
            for node_id, (_, node) in self._evictable_device.items():
                if not node.backuped and node.hit_count >= threshold and node.block_id not in pending_block_ids:
                    candidates.append(node)

            # Sort by LRU (oldest last_access_time first)
            candidates.sort(key=lambda n: n.last_access_time)

        return candidates

    def evict_nodes_selective(
        self,
        num_blocks: int,
    ) -> List[int]:
        """
        Evict device nodes with write_through_selective optimization.

        First selects the coldest (LRU) nodes, then categorizes them:
        - without_backup: Release directly (cold data, no transfer needed)
        - with_backup: Update metadata to HOST (data already in host)

        Args:
            num_blocks: Number of blocks to evict

        Returns:
            List of released device block IDs
        """
        if num_blocks <= 0:
            return []

        with self._lock:
            if len(self._evictable_device) < num_blocks:
                return []

            # Get LRU nodes first (this pops them from _evictable_device)
            nodes = self._get_lru_nodes(self._evictable_device, num_blocks)

            released_device_ids = []
            for node in nodes:
                if node.backuped:
                    released_device_ids.append(node.block_id)

                    node.cache_status = CacheStatus.HOST
                    node.block_id = node.host_block_id
                    node.touch()
                    # Move to host evictable
                    self._evictable_host[node.node_id] = (node.last_access_time, node)
                else:
                    self._remove_node_from_tree(node)
                    released_device_ids.append(node.block_id)

            return released_device_ids
