"""
RadixTree implementation for prefix matching in KV cache.
"""

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

    def __init__(self, enable_host_cache: bool = False):
        """
        Initialize the radix tree.

        Args:
            enable_host_cache: If True, evict() moves nodes to HOST state
                              instead of removing them from tree.
        """
        self._root = BlockNode()
        self._lock = threading.RLock()
        self._node_count = 1  # Root node
        self._enable_host_cache = enable_host_cache

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
                    logger.debug(
                        f"[DEBUG] find_prefix path[{i}]: hash={block_hash[:8]}... "
                        f"MISMATCH (not in children), total_matched={len(matched_nodes)}"
                    )
                    break

                node = node.children[block_hash]
                if node.cache_status in (CacheStatus.DELETING, CacheStatus.SWAP_TO_HOST):
                    logger.debug(
                        f"[DEBUG] find_prefix path[{i}]: hash={block_hash[:8]}... "
                        f"status={node.cache_status.name}, block_id={node.block_id}, "
                        f"ref={node.ref_count}, SKIP (deleting/swapping)"
                    )
                    break

                logger.debug(
                    f"[DEBUG] find_prefix path[{i}]: hash={block_hash[:8]}... "
                    f"status={node.cache_status.name}, block_id={node.block_id}, "
                    f"ref={node.ref_count}"
                )
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

        evicted_block_ids = []

        with self._lock:
            if len(self._evictable_host) < num_blocks:
                return None

            for _ in range(num_blocks):
                # Find LRU node (smallest last_access_time)
                lru_node_id = min(self._evictable_host.keys(),
                                  key=lambda nid: self._evictable_host[nid][0])
                _, node = self._evictable_host.pop(lru_node_id)

                logger.debug(
                    f"[DEBUG] evict_host_nodes: -HOST block_id={node.block_id}, "
                    f"device={len(self._evictable_device)}, "
                    f"host={len(self._evictable_host)}"
                )

                self._remove_node_from_tree(node)
                evicted_block_ids.append(node.block_id)

        return evicted_block_ids

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

        evicted_block_ids = []

        with self._lock:
            if len(self._evictable_device) < num_blocks:
                return None

            for _ in range(num_blocks):
                # Find LRU node (smallest last_access_time)
                lru_node_id = min(self._evictable_device.keys(),
                                  key=lambda nid: self._evictable_device[nid][0])
                _, node = self._evictable_device.pop(lru_node_id)

                logger.debug(
                    f"[DEBUG] evict_device_nodes: -DEVICE block_id={node.block_id}, "
                    f"device={len(self._evictable_device)}, "
                    f"host={len(self._evictable_host)}"
                )

                self._remove_node_from_tree(node)
                evicted_block_ids.append(node.block_id)

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
            logger.debug("[DEBUG] evict_device_to_host: num_blocks=0, nothing to do")
            return []

        if len(host_block_ids) < num_blocks:
            logger.debug(
                f"[DEBUG] evict_device_to_host: not enough host_block_ids, "
                f"need={num_blocks}, got={len(host_block_ids)}"
            )
            return None

        released_block_ids = []

        with self._lock:
            if len(self._evictable_device) < num_blocks:
                logger.debug(
                    f"[DEBUG] evict_device_to_host: pre-check failed, "
                    f"need={num_blocks}, device={len(self._evictable_device)}"
                )
                return None

            logger.debug(
                f"[DEBUG] evict_device_to_host: start, "
                f"num_blocks={num_blocks}, host_block_ids={host_block_ids}, "
                f"device={len(self._evictable_device)}, "
                f"host={len(self._evictable_host)}"
            )

            for i in range(num_blocks):
                # Find LRU node (smallest last_access_time)
                lru_node_id = min(self._evictable_device.keys(),
                                  key=lambda nid: self._evictable_device[nid][0])
                _, node = self._evictable_device.pop(lru_node_id)

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
                    f"[DEBUG] evict_device_to_host: DEVICE block_id={original_block_id} -> HOST block_id={new_host_block_id}, "
                    f"device={len(self._evictable_device)}, "
                    f"host={len(self._evictable_host)}"
                )

            logger.debug(
                f"[DEBUG] evict_device_to_host: done, "
                f"released_device_block_ids={released_block_ids}, "
                f"device={len(self._evictable_device)}, "
                f"host={len(self._evictable_host)}"
            )

        return released_block_ids

    def _add_to_evictable(self, node: BlockNode) -> None:
        """
        Add a node to the appropriate evictable dict based on cache status.
        """
        if node.cache_status == CacheStatus.DEVICE:
            if node.node_id not in self._evictable_device:
                self._evictable_device[node.node_id] = (node.last_access_time, node)
                logger.debug(
                    f"[DEBUG] _add_to_evictable: +{node.cache_status.name} block_id={node.block_id}, "
                    f"device={len(self._evictable_device)}, "
                    f"host={len(self._evictable_host)}"
                )
        elif node.cache_status == CacheStatus.HOST:
            if node.node_id not in self._evictable_host:
                self._evictable_host[node.node_id] = (node.last_access_time, node)
                logger.debug(
                    f"[DEBUG] _add_to_evictable: +{node.cache_status.name} block_id={node.block_id}, "
                    f"device={len(self._evictable_device)}, "
                    f"host={len(self._evictable_host)}"
                )

    def _remove_from_evictable(self, node: BlockNode) -> None:
        """
        Remove a node from evictable tracking (O(1) deletion from dict).
        """
        if node.cache_status == CacheStatus.DEVICE and node.node_id in self._evictable_device:
            del self._evictable_device[node.node_id]
            logger.debug(
                f"[DEBUG] _remove_from_evictable: -{node.cache_status.name} block_id={node.block_id}, "
                f"device={len(self._evictable_device)}, "
                f"host={len(self._evictable_host)}"
            )
        elif node.cache_status == CacheStatus.HOST and node.node_id in self._evictable_host:
            del self._evictable_host[node.node_id]
            logger.debug(
                f"[DEBUG] _remove_from_evictable: -{node.cache_status.name} block_id={node.block_id}, "
                f"device={len(self._evictable_device)}, "
                f"host={len(self._evictable_host)}"
            )

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
