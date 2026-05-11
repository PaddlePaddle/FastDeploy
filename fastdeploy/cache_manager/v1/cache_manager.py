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

from __future__ import annotations

import threading
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from fastdeploy.utils import get_logger

if TYPE_CHECKING:
    from fastdeploy.engine.request import Request
    from fastdeploy.config import FDConfig
    from fastdeploy.cache_manager.v1.storage import StorageScheduler

from .base import KVCacheBase
from .block_pool import DeviceBlockPool, HostBlockPool
from .cache_utils import storage_key_for_block
from .metadata import (
    BlockNode,
    CacheLevel,
    CacheStatus,
    CacheSwapMetadata,
    MatchResult,
    PendingPrefetch,
    StorageMetadata,
)
from .radix_tree import RadixTree
from .storage import create_storage_scheduler

logger = get_logger("prefix_cache_manager", "cache_manager.log")


class CacheManager(KVCacheBase):
    """
    Cache Manager for Scheduler process.

    Inherits from KVCacheBase and uniquely owns DeviceBlockPool and HostBlockPool.
    Responsible for block allocation/release, cache matching, and eviction decisions.

    Three-level cache hierarchy:
        Level 1: Device (GPU) - Fastest access, directly used for inference
        Level 2: Host (CPU) - Medium speed, needs to be loaded to Device
        Level 3: Storage - Slowest, needs to be fetched to Host first

    Attributes:
        device_pool: DeviceBlockPool instance.
        host_pool: HostBlockPool instance.
        radix_tree: RadixTree instance for prefix matching.
    """

    def __init__(
        self,
        config: "FDConfig",
    ):
        """
        Initialize the Cache Manager.

        Args:
            config: FDConfig instance containing all fastdeploy configuration
        """
        super().__init__(config)

        # Extract configuration from FDConfig
        self.num_gpu_blocks = self.cache_config.total_block_num
        self.num_cpu_blocks = self.cache_config.num_cpu_blocks
        self.block_size = self.cache_config.block_size
        self.enable_host_cache = self.num_cpu_blocks > 0
        self.enable_prefix_caching = self.cache_config.enable_prefix_caching

        # Write policy for backup (write_through, write_through_selective, write_back)
        # Normalize write_policy: "write_through" is a special case of "write_through_selective" with threshold=1
        self._write_policy = self.cache_config.write_policy
        self._write_through_threshold = self.cache_config.write_through_threshold
        if self._write_policy == "write_through":
            self._write_through_threshold = 1
            self._write_policy = "write_through_selective"

        # Thread safety
        self._lock = threading.RLock()

        # Initialize block pools
        self._device_pool = DeviceBlockPool(
            num_blocks=self.num_gpu_blocks,
            block_size=self.block_size,
        )
        self._host_pool = HostBlockPool(
            num_blocks=self.num_cpu_blocks,
            block_size=self.block_size,
        )

        # Initialize radix tree for prefix matching
        self._radix_tree = None
        if self.enable_prefix_caching:
            self._radix_tree = RadixTree(
                enable_host_cache=self.enable_host_cache,
                write_policy=self._write_policy,
            )

        # Pending backup list: nodes waiting to be backed up, to be issued via request's cache_evict_metadata
        self._pending_backup: List[Tuple[List[BlockNode], List[int]]] = []
        self._pending_block_ids: List[int] = []

        # Mapping from host_block_id -> BlockNode for LOADING_FROM_STORAGE blocks,
        # used to quickly update status to HOST once prefetch completes.
        self._prefetch_node_map: Dict[int, BlockNode] = {}

        # Pending prefetch queue: tasks waiting to be dispatched by scheduler
        self._pending_prefetch_list: List[PendingPrefetch] = []
        self._pending_prefetch_lock = threading.Lock()

        # Storage scheduler (create using factory method if backend is configured)
        self._storage_scheduler = create_storage_scheduler(self.cache_config)

        # Eviction tracking
        self._eviction_in_progress = False

        self._initialized = True

        logger.info(
            f"CacheManager initialized, num_gpu_blocks: {self.num_gpu_blocks}, "
            f"num_cpu_blocks: {self.num_cpu_blocks}, block_size: {self.block_size}, "
            f"enable_prefix_caching: {self.enable_prefix_caching}, "
            f"enable_host_cache: {self.enable_host_cache}, "
            f"write_policy: {self._write_policy}, "
            f"write_through_threshold: {self._write_through_threshold}"
        )

    # ============ Properties ============

    @property
    def device_pool(self) -> DeviceBlockPool:
        """Get the device block pool."""
        return self._device_pool

    @property
    def host_pool(self) -> HostBlockPool:
        """Get the host block pool."""
        return self._host_pool

    @property
    def radix_tree(self) -> RadixTree:
        """Get the radix tree."""
        return self._radix_tree

    @property
    def num_free_device_blocks(self) -> int:
        """Get number of free device blocks."""
        return self._device_pool.available_blocks()

    @property
    def num_free_host_blocks(self) -> int:
        """Get number of free host blocks."""
        return self._host_pool.available_blocks()

    @property
    def storage_scheduler(self) -> Optional["StorageScheduler"]:
        """Get the storage scheduler."""
        return self._storage_scheduler

    # ============ Block Allocation/Release Methods ============

    def can_allocate_device_blocks(self, num: int) -> bool:
        """
        Check if current resources can allocate the specified number of device blocks.

        Args:
            num: Number of blocks to check

        Returns:
            True if allocation is possible, False otherwise
        """
        if self._device_pool.available_blocks() >= num:
            return True

        elif self.enable_prefix_caching:
            stats = self._radix_tree.get_stats()
            if self._device_pool.available_blocks() + stats.evictable_device_count >= num:
                return True

        return False

    def can_allocate_host_blocks(self, num: int) -> bool:
        """
        Check if current resources can allocate the specified number of host blocks.

        Args:
            num: Number of blocks to check

        Returns:
            True if allocation is possible, False otherwise
        """
        if self._host_pool.available_blocks() >= num:
            return True

        elif self.enable_prefix_caching:
            stats = self._radix_tree.get_stats()
            if self._host_pool.available_blocks() + stats.evictable_host_count >= num:
                return True

        return False

    def allocate_device_blocks(
        self,
        request: Request,
        num_blocks: int,
    ) -> Optional[List[int]]:
        """
        Allocate device blocks for a request.

        This method handles:
        1. Evicting device blocks if needed
        2. Swapping host blocks to device if matched
        3. Inserting new blocks into RadixTree

        Args:
            request: Request object containing match result and prompt hashes
            num_blocks: Number of new device blocks to allocate

        Returns:
            List of allocated device block indices, or empty list if allocation failed
        """
        try:
            with self._lock:
                match_result = request.match_result

                need_block_num = num_blocks

                if not self.can_allocate_device_blocks(need_block_num):
                    return []

                if need_block_num > self._device_pool.available_blocks():
                    evicted_result = self._evict_blocks(need_block_num - self._device_pool.available_blocks())
                    if evicted_result is None:
                        logger.error(f"evict_device_blocks failed, request_id: {request.request_id}")
                        return []

                    if self.enable_host_cache and self._write_policy == "write_back":
                        evicted_blocks, host_block_ids = evicted_result
                        if len(evicted_blocks) != len(host_block_ids):
                            logger.error(
                                f"evict_blocks to host failed, request_id: {request.request_id}, "
                                f"evicted_blocks: {evicted_blocks}, host_block_ids: {host_block_ids}"
                            )
                            return []
                        request.cache_evict_metadata.append(
                            CacheSwapMetadata(
                                src_block_ids=evicted_blocks,
                                dst_block_ids=host_block_ids,
                                src_type=CacheLevel.DEVICE,
                                dst_type=CacheLevel.HOST,
                            )
                        )

                allocated = self._device_pool.allocate(need_block_num)
                if allocated is None:
                    logger.error(
                        f"allocate device blocks failed, request_id: {request.request_id}, need: {need_block_num}"
                    )
                    return []

                if self.enable_host_cache and match_result.matched_host_nums > 0:
                    device_blocks = allocated[: match_result.matched_host_nums]

                    free_host_block_ids = self._radix_tree.swap_to_device(match_result.host_nodes, device_blocks)
                    logger.debug(
                        f"[allocate_device_blocks] request_id={request.request_id} "
                        f"swap host->device: host_block_ids={free_host_block_ids} -> device_block_ids={device_blocks}"
                    )

                    request.cache_swap_metadata.append(
                        CacheSwapMetadata(
                            src_block_ids=free_host_block_ids,
                            dst_block_ids=device_blocks,
                            src_type=CacheLevel.HOST,
                            dst_type=CacheLevel.DEVICE,
                        )
                    )

                    if self._write_policy == "write_through_selective":
                        self._radix_tree.backup_blocks(match_result.host_nodes, free_host_block_ids)
                    else:
                        self.free_host_blocks(free_host_block_ids)

                    match_result.device_nodes.extend(match_result.host_nodes)
                    match_result.host_nodes = []

                if self.enable_prefix_caching:
                    block_hashes = request.prompt_hashes[match_result.matched_device_nums :]
                    all_device_blocks = request.block_tables + allocated
                    uncached_device_blocks = all_device_blocks[match_result.matched_device_nums :]
                    num_block_lens = min(len(uncached_device_blocks), len(block_hashes))

                    if num_block_lens > 0:
                        blocks = list(zip(block_hashes[:num_block_lens], uncached_device_blocks[:num_block_lens]))
                        start_node = match_result.device_nodes[-1] if match_result.device_nodes else None

                        device_nodes, wasted_block_ids = self._radix_tree.insert(blocks=blocks, start_node=start_node)
                        match_result.device_nodes.extend(device_nodes)

                        inserted_block_ids = [n.block_id for n in device_nodes]
                        logger.debug(
                            f"[allocate_device_blocks] request_id={request.request_id} "
                            f"newly allocated={allocated} "
                            f"inserted_into_path_block_ids={inserted_block_ids} "
                            f"wasted_block_ids(not_in_path)={wasted_block_ids}"
                        )

                        # Release any blocks that were wasted due to node reuse
                        # and update allocated with actual block_ids
                        if wasted_block_ids:
                            match_result.uncached_block_ids.extend(wasted_block_ids)

                return allocated
        except Exception as e:
            logger.error(f"allocate_device_blocks error: {e}, {str(traceback.format_exc())}")
            return []

    def allocate_host_blocks(self, num: int) -> List[int]:
        """
        Allocate host blocks from the pool.

        Args:
            num: Number of blocks to allocate

        Returns:
            List of allocated block indices (may be fewer than requested or empty on error)
        """
        try:
            if self._host_pool.available_blocks() < num:
                evict_blocks = self._radix_tree.evict_host_nodes(num - self._host_pool.available_blocks())
                if evict_blocks is not None:
                    self._host_pool.release(evict_blocks)
            return self._host_pool.allocate(num) or []
        except Exception as e:
            logger.error(f"allocate_host_blocks error: {e}, {str(traceback.format_exc())}")
            return []

    def free_device_blocks(self, block_ids: List[int]) -> None:
        """
        Free device blocks back to the pool.

        Args:
            block_ids: List of block indices to free
        """
        if not block_ids:
            return

        with self._lock:
            self._device_pool.release(block_ids)

    def free_host_blocks(self, block_ids: List[int]) -> None:
        """
        Free host blocks back to the pool.

        Args:
            block_ids: List of block indices to free
        """
        if not block_ids:
            return
        self._host_pool.release(block_ids)

    def free_all_device_blocks(self) -> int:
        """
        Free all device blocks.

        Returns:
            Number of blocks freed
        """
        with self._lock:
            freed = self._device_pool.used_blocks()
            self._device_pool.reset()
            return freed

    def free_all_host_blocks(self) -> int:
        """
        Free all host blocks.

        Returns:
            Number of blocks freed
        """
        with self._lock:
            freed = self._host_pool.used_blocks()
            self._host_pool.reset()
            return freed

    def resize_device_pool(self, new_num_blocks: int) -> bool:
        """
        Resize the device block pool.

        Supports both expansion and shrinking. Shrinking will fail if
        there are more used blocks than the new size.

        Args:
            new_num_blocks: New total number of blocks for device pool

        Returns:
            True if resize was successful, False otherwise
        """
        logger.info(f"resize_device_pool: {self._device_pool.available_blocks()} -> {new_num_blocks}")
        with self._lock:
            if self._device_pool.resize(new_num_blocks):
                self.num_gpu_blocks = new_num_blocks
                return True
            return False

    # ============ Legacy Compatibility Methods ============
    # These methods provide backward compatibility with PrefixCacheManager interface
    # for resource_manager.py

    @property
    def gpu_free_block_list(self) -> List[int]:
        """
        Get list of free GPU block indices (legacy alias).

        Returns list of available device block IDs for compatibility
        with PrefixCacheManager.gpu_free_block_list.
        """
        # Return list representation of available blocks
        return list(self._device_pool._free_blocks)

    @property
    def available_gpu_resource(self) -> float:
        """
        Get available GPU resource ratio (legacy alias).

        Returns the ratio of free blocks to total blocks.
        """
        if self.num_gpu_blocks == 0:
            return 0.0
        return self._device_pool.available_blocks() / self.num_gpu_blocks

    def allocate_gpu_blocks(self, request: Request, num_blocks: int) -> Optional[List[int]]:
        """
        Allocate GPU blocks (legacy alias for allocate_device_blocks).

        Args:
            request: Request object containing match result
            num_blocks: Number of blocks to allocate

        Returns:
            List of allocated block indices, or None if allocation failed
        """
        return self.allocate_device_blocks(request, num_blocks)

    def can_allocate_gpu_blocks(self, num_blocks: int) -> bool:
        """
        Check if GPU blocks can be allocated (legacy alias).

        Args:
            num_blocks: Number of blocks to check

        Returns:
            True if allocation is possible, False otherwise
        """
        return self.can_allocate_device_blocks(num_blocks)

    def update_cache_config(self, new_cfg) -> None:
        """
        Update cache configuration.

        Args:
            new_cfg: New cache configuration object
        """
        self.cache_config = new_cfg
        new_num_blocks = getattr(new_cfg, "total_block_num", None)
        if new_num_blocks is not None:
            self.resize_device_pool(new_num_blocks)

    # ============ Three-Level Cache Matching ============

    def match_prefix(
        self,
        request: Request,
        skip_storage: bool = False,
    ) -> None:
        """
        Execute three-level cache matching (Device -> Host -> Storage).

        This is the main entry point for prefix matching during scheduling.
        Only effective when prefix caching is enabled. The result is stored
        in request._match_result.

        Args:
            request: Request object containing prompt hashes
            skip_storage: If True, skip storage-level matching

        Returns:
            None. Match result is stored in request._match_result.
        """
        if not self.enable_prefix_caching or self._radix_tree is None:
            request._match_result = MatchResult()
            return

        with self._lock:
            try:
                result = MatchResult()
                block_hashes = request.prompt_hashes

                # Step 1: Match Device and Host cache via RadixTree
                matched_nodes = self._radix_tree.find_prefix(block_hashes)

                #   Split matched_nodes into device blocks and host blocks
                if self.enable_host_cache:
                    for node in matched_nodes:
                        if node.is_on_device():
                            result.device_nodes.append(node)
                        elif node.is_on_host():
                            result.host_nodes.append(node)
                else:
                    result.device_nodes = matched_nodes

                #   Calculate remaining hashes to match
                matched_count = result.matched_device_nums + result.matched_host_nums
                remaining_hashes = block_hashes[matched_count:]

                # Step 2: Match Storage (if enabled and not skipped)
                if not skip_storage and self._storage_scheduler and remaining_hashes:
                    storage_matches = self._match_storage(remaining_hashes)
                    start_node = matched_nodes[-1] if matched_nodes else None
                    result.storage_nodes = self.prepare_prefetch_metadata(storage_matches, start_node=start_node)

                # Step 3: Increment ref count for matched blocks(only scheduling phase)
                if skip_storage:
                    self._radix_tree.increment_ref_nodes(matched_nodes)

                logger.info(
                    f"match_prefix for request_id: {request.request_id} total_hashes: {len(block_hashes)}, "
                    f"total_matched: {result.total_matched_blocks} (device_blocks={result.matched_device_nums}, "
                    f"host_blocks={result.matched_host_nums}, storage_hashes={result.matched_storage_nums})"
                )

                matched_device_ids = [n.block_id for n in result.device_nodes]
                matched_host_ids = [n.block_id for n in result.host_nodes]
                logger.debug(
                    f"[match_prefix] request_id={request.request_id} "
                    f"matched_device_block_ids={matched_device_ids} "
                    f"matched_host_block_ids={matched_host_ids}"
                )
                request._match_result = result
            except Exception as e:
                logger.error(f"match_prefix error: {e}, {str(traceback.format_exc())}")

    def _match_storage(self, hash_values: List[str]) -> List[str]:
        """
        Match hash values against storage.

        Checks each hash for existence in storage and returns the longest
        consecutive prefix of hashes that are all present (prefix semantics
        are required because a cache miss in the middle breaks prefetch continuity).

        Probes both rank=0 "key" and "value" kinds: a block is considered present
        only when both exist.  This avoids false positives from partial writes where
        only one kind was stored, and prevents LRU asymmetry (probing only "key"
        would keep it hot while "value" gets evicted by Mooncake).

        Storage key format (see cache_utils.storage_key_for_block):
            "{hash_value}_0_key"  /  "{hash_value}_0_value"

        Args:
            hash_values: List of block hash values to check, in prefix order.

        Returns:
            The leading sub-list of hash_values whose blocks all exist in storage.
            For example, if hash_values = [h0, h1, h2, h3] and h2 is missing,
            returns [h0, h1].
        """
        if not self._storage_scheduler:
            return []

        try:
            if not self._storage_scheduler.is_connected():
                logger.warning("_match_storage: storage scheduler disconnected, skipping storage match")
                return []

            # Probe both key and value kinds for rank=0.
            # Interleaved: [h0_key, h0_value, h1_key, h1_value, ...]
            probe_keys = []
            for h in hash_values:
                probe_keys.append(storage_key_for_block(h, 0, "key"))
                probe_keys.append(storage_key_for_block(h, 0, "value"))

            exist_flags = self._storage_scheduler.batch_exists(probe_keys)

            # A block is present only when both key and value exist.
            matched = []
            for i, h in enumerate(hash_values):
                key_ok = exist_flags[i * 2]
                val_ok = exist_flags[i * 2 + 1]
                if not (key_ok and val_ok):
                    break
                matched.append(h)

            logger.debug(
                f"[CacheManager] _match_storage: probing {len(hash_values)} blocks "
                f"({len(probe_keys)} keys), matched={len(matched)}"
            )
            return matched
        except Exception:
            logger.warning("_match_storage failed", exc_info=True)
            return []

    # ============ Eviction Methods ============

    def _evict_blocks(self, num_blocks: int) -> Optional[List[int]]:
        """
        Evict device blocks to free device memory.

        In write_through_selective policy:
        - Blocks with backup (backuped=True): Update metadata only, no actual data transfer needed
        - Blocks without backup but hit_count >= threshold: Trigger emergency backup, then evict
        - Blocks without backup and hit_count < threshold: Release directly

        Eviction flow (for other policies):
        1. Try to allocate host block ids for device->host eviction
        2. If not enough host blocks, evict host nodes first to free host blocks
        3. Evict device blocks to host using RadixTree.evict_device_to_host()
        4. Free the evicted device blocks back to the pool

        Args:
            num_blocks: Number of device blocks to evict

        Returns:
            List of evicted device block ids, or None if eviction failed
        """
        if not self.enable_prefix_caching or self._radix_tree is None:
            logger.warning("_evict_blocks: prefix caching not enabled")
            return None

        if num_blocks <= 0:
            return [], []

        try:
            with self._lock:
                host_block_ids = []

                # Step 1: Check if we have enough evictable device blocks
                stats = self._radix_tree.get_stats()
                if stats.evictable_device_count < num_blocks:
                    logger.warning(
                        f"_evict_blocks: not enough evictable device blocks, "
                        f"needed: {num_blocks}, available: {stats.evictable_device_count}"
                    )
                    return None

                # Step 2: Handle eviction based on write policy
                if self.enable_host_cache:
                    if self._write_policy == "write_through_selective":
                        # write_through_selective policy: optimize eviction based on backup status
                        released_device_ids = self._radix_tree.evict_nodes_selective(num_blocks=num_blocks)
                    elif self._write_policy == "write_back":
                        # write_back policy:: allocate host blocks and evict to host
                        host_block_ids = self.allocate_host_blocks(num_blocks)
                        if host_block_ids is None or len(host_block_ids) < num_blocks:
                            logger.warning("_evict_blocks: failed to allocate host blocks")
                            return None

                        released_device_ids = self._radix_tree.evict_device_to_host(
                            num_blocks=num_blocks,
                            host_block_ids=host_block_ids,
                        )
                else:
                    # No host cache, evict device nodes directly
                    released_device_ids = self._radix_tree.evict_device_nodes(num_blocks)

                if released_device_ids is None:
                    return None

                # Step 3: Free the evicted device blocks
                self._device_pool.release(released_device_ids)

                logger.debug(
                    f"[_evict_blocks] evicted_device_block_ids={released_device_ids} "
                    f"host_block_ids={host_block_ids} "
                    f"write_policy={self._write_policy} "
                    f"free_device_after={self._device_pool.available_blocks()}"
                )

                return released_device_ids, host_block_ids
        except Exception as e:
            logger.error(f"_evict_blocks error: {e}, {str(traceback.format_exc())}")
            return None

    # ============ Request Lifecycle Methods ============

    def request_finish(
        self,
        request: Request,
    ) -> None:
        """
        Update cache state when a request finishes.

        This method:
        1. Inserts new blocks into the RadixTree (for caching)
        2. Decrements reference counts for matched blocks
        3. Releases blocks that cannot be cached:
           - Blocks without hash (partial blocks)
           - Blocks wasted due to node reuse

        Note: Blocks successfully inserted into RadixTree are managed by
        the tree and will be freed when evicted.

        Only effective when prefix caching is enabled.

        Args:
            request: Request object containing match result and block tables
        """
        with self._lock:
            try:
                if self.enable_prefix_caching and self._radix_tree is not None:
                    match_result = request.match_result

                    block_hashes = request.prompt_hashes[match_result.matched_device_nums :]
                    device_blocks = request.block_tables[match_result.matched_device_nums :]
                    num_block_lens = min(len(device_blocks), len(block_hashes))

                    if num_block_lens > 0:
                        blocks = list(zip(block_hashes[:num_block_lens], device_blocks[:num_block_lens]))
                        start_node = match_result.device_nodes[-1] if match_result.device_nodes else None

                        device_nodes, wasted_block_ids = self._radix_tree.insert(blocks=blocks, start_node=start_node)
                        match_result.device_nodes.extend(device_nodes)

                        # Release blocks that were wasted due to node reuse
                        if wasted_block_ids:
                            match_result.uncached_block_ids.extend(wasted_block_ids)

                    # Release uncached blocks
                    uncached_blocks = match_result.uncached_block_ids
                    uncached_blocks.extend(request.block_tables[match_result.matched_device_nums :])

                    # Decrement ref count - blocks become evictable if ref_count reaches 0
                    self._radix_tree.decrement_ref_nodes(match_result.device_nodes)
                    self._device_pool.release(uncached_blocks)

                    cached_block_ids = [n.block_id for n in match_result.device_nodes]
                    logger.debug(
                        f"[request_finish] request_id={request.request_id} "
                        f"cached_block_ids(in_radix_tree)={cached_block_ids} "
                        f"released_uncached_block_ids={uncached_blocks}"
                    )
                    logger.info(
                        f"request {request.request_id} finished, cached blocks: {match_result.matched_device_nums}, "
                        f"uncached blocks freed: {len(uncached_blocks)}, "
                        f"total_free: {self._device_pool.available_blocks()}"
                    )
                else:
                    self._device_pool.release(request.block_tables)

                    logger.debug(
                        f"[request_finish] request_id={request.request_id} "
                        f"prefix_caching=disabled released_block_ids={request.block_tables}"
                    )
                    logger.info(
                        f"request {request.request_id} finished, release blocks: {len(request.block_tables)}, "
                        f"total_free: {self._device_pool.available_blocks()}"
                    )
            except Exception as e:
                logger.error(f"request_finish error: {e}, {str(traceback.format_exc())}")

    # ============ Write-through Selective Backup Methods ============

    def get_pending_backup_count(self) -> int:
        """
        Get the number of pending backup tasks.

        Returns:
            Number of pending backup tasks in the queue.
        """
        return len(self._pending_backup)

    def issue_pending_backup_to_batch_request(
        self,
    ) -> Optional[CacheSwapMetadata]:
        """
        Issue pending backup tasks and return a CacheSwapMetadata for BatchRequest.

        This method is called during scheduling to prepare pending backup tasks
        to be attached to a BatchRequest. The BatchRequest will pass this metadata
        to the worker, which will execute the backup (Device->Host transfer).

        Returns:
            CacheSwapMetadata containing backup tasks, or None if no pending backup.
        """
        if not self._pending_backup:
            return None

        if not self.enable_host_cache or not self._radix_tree:
            # No host cache, clear pending backup
            self._pending_backup.clear()
            return None

        try:
            with self._lock:
                if not self._pending_backup:
                    return None

                all_device_block_ids = []
                all_host_block_ids = []
                all_hash_values = []
                freed_host_ids = []

                for nodes, host_block_ids in self._pending_backup:
                    # Filter out nodes that are no longer valid (already evicted, etc.)
                    valid_nodes = []
                    valid_host_ids = []

                    for node, host_block_id in zip(nodes, host_block_ids):
                        # Check if node is still in evictable_device and not already backed up
                        if (
                            node.node_id in self._radix_tree._evictable_device
                            and not node.backuped
                            and node.cache_status == CacheStatus.DEVICE
                        ):
                            valid_nodes.append(node)
                            valid_host_ids.append(host_block_id)
                        else:
                            # Node no longer valid, release the allocated host block
                            freed_host_ids.append(host_block_id)

                    if valid_nodes:
                        # Mark nodes as backed up
                        self._radix_tree.backup_blocks(valid_nodes, valid_host_ids)

                        # Collect device block IDs and hash values
                        all_device_block_ids.extend([node.block_id for node in valid_nodes])
                        all_host_block_ids.extend(valid_host_ids)
                        all_hash_values.extend([node.hash_value for node in valid_nodes])

                # Release invalid host block allocations
                if freed_host_ids:
                    self._host_pool.release(freed_host_ids)

                # Clear pending backup
                self._pending_backup.clear()
                self._pending_block_ids.clear()

                # Create and return CacheSwapMetadata
                if all_device_block_ids:
                    evict_metadata = CacheSwapMetadata(
                        src_block_ids=all_device_block_ids,
                        dst_block_ids=all_host_block_ids,
                        src_type=CacheLevel.DEVICE,
                        dst_type=CacheLevel.HOST,
                        hash_values=all_hash_values,
                    )
                    return evict_metadata

                return None

        except Exception as e:
            logger.error(f"issue_pending_backup_to_batch_request error: {e}, {str(traceback.format_exc())}")
            # Clear pending backup on error to avoid infinite accumulation
            self._pending_backup.clear()
            self._pending_block_ids.clear()
            return None

    def check_and_add_pending_backup(
        self,
    ) -> None:
        """
        Check for nodes that meet backup criteria and add them to pending backup queue.

        This method is called after request_finish to check if any nodes
        in the radix tree meet the write_through_selective backup criteria.

        For write_through_selective policy:
        - Nodes with hit_count >= threshold that are not yet backed up
        - are added to the pending backup queue

        The pending backup will be issued to the next scheduled request.
        """
        if not self.enable_host_cache or not self._radix_tree:
            return

        if self._write_policy != "write_through_selective":
            return

        try:
            with self._lock:
                # Get candidates from radix tree
                candidates = self._radix_tree.get_candidates_for_backup(
                    self._write_through_threshold,
                    self._pending_block_ids,
                )

                if not candidates:
                    return

                # Allocate host blocks for backup
                host_block_ids = self.allocate_host_blocks(len(candidates))
                if host_block_ids is None or len(host_block_ids) < len(candidates):
                    logger.warning(
                        f"check_and_add_pending_backup: failed to allocate host blocks, "
                        f"needed={len(candidates)}, got={len(host_block_ids) if host_block_ids else 0}"
                    )
                    if host_block_ids:
                        self._host_pool.release(host_block_ids)
                    return

                # Add to pending backup queue
                self._pending_backup.append((candidates, host_block_ids))
                self._pending_block_ids.extend([node.block_id for node in candidates])

        except Exception as e:
            logger.error(f"check_and_add_pending_backup error: {e}, {str(traceback.format_exc())}")

    def load_from_host(self, block_indices: List[int]) -> bool:
        """
        Load blocks from host to device memory.

        This is a coordination method. Actual data transfer happens in Worker.

        Args:
            block_indices: List of host block indices to load

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                # Allocate device blocks
                dev_indices = self._device_pool.allocate(len(block_indices))
                if dev_indices is None or len(dev_indices) != len(block_indices):
                    if dev_indices:
                        self._device_pool.release(dev_indices)
                    return False

                # Perform the load (actual data transfer would happen in Worker)

                # Release host blocks
                self._host_pool.release(block_indices)

                return True
        except Exception as e:
            logger.error(f"load_from_host error: {e}, {str(traceback.format_exc())}")
            return False

    # ============ Prefetch Methods ============

    def prefetch_storage(self, request: "Request") -> bool:
        """
        Execute storage matching and enqueue prefetch info for later dispatch.

        Called from the preprocess thread. Does match_prefix(skip_storage=False)
        to probe storage, allocate host blocks, and enqueue PendingPrefetch
        into the pending list. The scheduler will drain and dispatch later.

        Args:
            request: The request to prefetch cache for.

        Returns:
            True if storage blocks were matched and enqueued, False otherwise.
        """
        if not self.enable_prefix_caching:
            return False

        self.match_prefix(request, skip_storage=False)
        match_result = request.match_result
        request.match_result = None

        if match_result is None or match_result.matched_storage_nums == 0:
            return False

        storage_nodes = match_result.storage_nodes
        host_block_ids = [node.block_id for node in storage_nodes]
        hash_values = [node.hash_value for node in storage_nodes]

        metadata = StorageMetadata(
            hash_values=hash_values,
            block_ids=host_block_ids,
            direction="load",
        )

        pending = PendingPrefetch(
            request_id=request.request_id,
            metadata=metadata,
            host_block_ids=host_block_ids,
        )
        with self._pending_prefetch_lock:
            self._pending_prefetch_list.append(pending)

        logger.info(
            f"[Debug][StoragePrefetch] request_id={request.request_id} "
            f"storage_matched={match_result.matched_storage_nums} blocks, enqueued for dispatch"
        )
        return True

    def drain_pending_prefetches(self) -> List[PendingPrefetch]:
        """Atomically drain all pending prefetch tasks for scheduler dispatch."""
        with self._pending_prefetch_lock:
            items = self._pending_prefetch_list
            self._pending_prefetch_list = []
            return items

    def prepare_prefetch_metadata(
        self,
        storage_hashes: List[str],
        start_node: Optional["BlockNode"] = None,
    ) -> Optional[List["BlockNode"]]:
        """
        Prepare metadata for storage prefetch operation.

        Called when storage cache is matched, allocates host blocks
        for the prefetch target.

        Args:
            storage_hashes: List of storage hash values to prefetch
            start_node: Node to start insertion from in the radix tree.
                        Must be the last matched node from find_prefix so that
                        the new LOADING_FROM_STORAGE nodes are attached as proper
                        extensions of the existing prefix chain.

        Returns:
            List of BlockNode objects if successful, None or empty list otherwise.
            Each node's block_id contains the actual block assigned
            (may differ from originally allocated if node was reused).
        """
        if not storage_hashes:
            return []

        try:
            with self._lock:
                # Check if we have enough host blocks
                if not self.can_allocate_host_blocks(len(storage_hashes)):
                    return []

                # Allocate host blocks for prefetch (evicts evictable host blocks if needed)
                host_block_ids = self.allocate_host_blocks(len(storage_hashes))
                if not host_block_ids:
                    return []

                blocks = list(zip(storage_hashes, host_block_ids))
                prefetch_nodes, wasted_block_ids = self._radix_tree.insert(
                    blocks=blocks, cache_status=CacheStatus.LOADING_FROM_STORAGE, start_node=start_node
                )
                # Release any blocks that were wasted due to node reuse
                if wasted_block_ids:
                    self._host_pool.release(wasted_block_ids)

                # Register only truly new LOADING_FROM_STORAGE nodes.
                # insert() reuses existing nodes without updating their status, so nodes
                # that were already HOST/DEVICE must be excluded — they don't need a
                # storage transfer and would trigger a spurious "unexpected status" warning
                # in update_storage_blocks_to_host.
                actual_prefetch_nodes = []
                for node in prefetch_nodes:
                    if node.cache_status == CacheStatus.LOADING_FROM_STORAGE:
                        self._prefetch_node_map[node.block_id] = node
                        actual_prefetch_nodes.append(node)

                return actual_prefetch_nodes
        except Exception as e:
            logger.error(f"prepare_prefetch_metadata error: {e}, {str(traceback.format_exc())}")
            return []

    def update_storage_blocks_to_host(self, host_block_ids: List[int]) -> None:
        """
        Mark storage-prefetched blocks as HOST after data transfer completes.

        Called by Scheduler when all TP workers report prefetch done for a batch
        of blocks. Transitions block status LOADING_FROM_STORAGE → HOST so that
        these blocks become eligible for swap-in scheduling.

        Args:
            host_block_ids: List of host block IDs that finished loading.
        """
        if not host_block_ids:
            return
        try:
            with self._lock:
                updated = 0
                for block_id in host_block_ids:
                    node = self._prefetch_node_map.pop(block_id, None)
                    if node is None:
                        logger.warning(
                            f"[StoragePrefetch] update_storage_blocks_to_host: "
                            f"block_id={block_id} not found in prefetch_node_map"
                        )
                        continue
                    if node.cache_status == CacheStatus.LOADING_FROM_STORAGE:
                        node.cache_status = CacheStatus.HOST
                        updated += 1
                    else:
                        logger.warning(
                            f"[StoragePrefetch] update_storage_blocks_to_host: "
                            f"block_id={block_id} unexpected status={node.cache_status}"
                        )
                logger.info(
                    f"[StoragePrefetch] update_storage_blocks_to_host: "
                    f"requested={len(host_block_ids)}, updated={updated}"
                )
        except Exception as e:
            logger.error(f"update_storage_blocks_to_host error: {e}, {str(traceback.format_exc())}")

    def abort_prefetch_blocks(self, host_block_ids: List[int]) -> None:
        """
        Abort in-flight prefetch blocks on failure.

        Removes nodes from the prefetch_node_map, deletes them from the RadixTree,
        and releases their host pool blocks. Called when the storage→CPU transfer
        fails so that LOADING_FROM_STORAGE blocks do not leak.

        Args:
            host_block_ids: List of host block IDs whose prefetch should be aborted.
        """
        if not host_block_ids:
            return
        try:
            with self._lock:
                for block_id in host_block_ids:
                    node = self._prefetch_node_map.pop(block_id, None)
                    if node is None:
                        continue
                    self._radix_tree._remove_node_from_tree(node)
                self._host_pool.release(host_block_ids)
            logger.warning(f"[StoragePrefetch] abort_prefetch_blocks: released {len(host_block_ids)} host blocks")
        except Exception as e:
            logger.error(f"abort_prefetch_blocks error: {e}, {str(traceback.format_exc())}")

    # ============ Reset Methods ============

    def reset_cache(self) -> bool:
        """
        Reset cache state.

        Implements abstract method from KVCacheBase.
        Clears block pools and radix tree.

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                self._device_pool.reset()
                self._host_pool.reset()
                if self._radix_tree is not None:
                    self._radix_tree.reset()
                self._eviction_in_progress = False
            logger.info("reset_cache: all cache state cleared")
            return True
        except Exception as e:
            logger.error(f"reset_cache failed: {e}, {str(traceback.format_exc())}")
            return False

    # ============ Statistics Methods ============

    def get_stats(self) -> Dict[str, Any]:
        """Get cache manager statistics."""
        return {
            "initialized": self._initialized,
            "num_gpu_blocks": self.num_gpu_blocks,
            "num_cpu_blocks": self.num_cpu_blocks,
            "block_size": self.block_size,
            "device_pool": self._device_pool.get_stats(),
            "host_pool": self._host_pool.get_stats(),
            "radix_tree": self._radix_tree.get_stats() if self._radix_tree else None,
            "num_free_device_blocks": self.num_free_device_blocks,
            "num_free_host_blocks": self.num_free_host_blocks,
            "storage_enabled": self._storage_scheduler is not None,
        }

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory usage information
        """
        device_stats = self._device_pool.get_stats()
        host_stats = self._host_pool.get_stats()

        return {
            "device": {
                "total_blocks": device_stats["num_blocks"],
                "used_blocks": device_stats["used"],
                "free_blocks": device_stats["available"],
                "usage_percent": (
                    device_stats["used"] / device_stats["num_blocks"] * 100 if device_stats["num_blocks"] > 0 else 0
                ),
            },
            "host": {
                "total_blocks": host_stats["num_blocks"],
                "used_blocks": host_stats["used"],
                "free_blocks": host_stats["available"],
                "usage_percent": (
                    host_stats["used"] / host_stats["num_blocks"] * 100 if host_stats["num_blocks"] > 0 else 0
                ),
            },
        }
