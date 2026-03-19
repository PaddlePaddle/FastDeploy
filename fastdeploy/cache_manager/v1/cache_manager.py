"""
CacheManager - Scheduler-side cache management.

Responsible for:
- Managing DeviceBlockPool and HostBlockPool
- Block allocation and release
- RadixTree for prefix matching
- Storage operations coordination
- Three-level cache matching (Device → Host → Storage)
"""

import threading
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastdeploy.engine.request import Request
from fastdeploy.utils import get_logger

if TYPE_CHECKING:
    from fastdeploy.config import FDConfig
    from fastdeploy.cache_manager.v1.storage import StorageScheduler

from .base import KVCacheBase
from .block_pool import DeviceBlockPool, HostBlockPool
from .metadata import BlockNode, CacheStatus, CacheSwapMetadata, MatchResult
from .radix_tree import RadixTree
from .storage import create_storage_scheduler

logger = get_logger("prefix_cache_manager", "cache_manager.log")


def _debug_log_radix_tree_state(request_id: str, operation: str, radix_tree, device_pool=None, host_pool=None):
    """DEBUG: 打印 radix tree 和 pool 的状态"""
    if radix_tree is None:
        return
    stats = radix_tree.get_stats()
    device_available = device_pool.available_blocks() if device_pool else 0
    host_available = host_pool.available_blocks() if host_pool else 0
    logger.debug(
        f"[DEBUG] {operation} request_id={request_id} "
        f"radix_tree: node_count={stats.node_count}, "
        f"evictable_device={stats.evictable_device_count}, "
        f"evictable_host={stats.evictable_host_count} | "
        f"pools: device_available={device_available}, host_available={host_available}"
    )


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
        self.cache_config = config.cache_config
        self.num_gpu_blocks = self.cache_config.total_block_num
        self.num_cpu_blocks = self.cache_config.num_cpu_blocks
        self.block_size = self.cache_config.block_size
        self.enable_host_cache = self.num_cpu_blocks > 0
        self.enable_prefix_caching = self.cache_config.enable_prefix_caching

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
            self._radix_tree = RadixTree(enable_host_cache=self.enable_host_cache)

        # Storage scheduler (create using factory method if backend is configured)
        self._storage_scheduler = create_storage_scheduler(self.cache_config)

        # Eviction tracking
        self._eviction_in_progress = False

        self._initialized = True

        logger.info(
            f"CacheManager initialized, num_gpu_blocks: {self.num_gpu_blocks}, "
            f"num_cpu_blocks: {self.num_cpu_blocks}, block_size: {self.block_size}, "
            f"enable_prefix_caching: {self.enable_prefix_caching}, "
            f"enable_host_cache: {self.enable_host_cache}"
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

                need_block_num = match_result.matched_host_nums + num_blocks

                if not self.can_allocate_device_blocks(need_block_num):
                    return []

                if need_block_num > self._device_pool.available_blocks():
                    evicted_blocks, host_block_ids = self._evict_blocks(
                        need_block_num - self._device_pool.available_blocks()
                    )
                    if evicted_blocks is None:
                        logger.error(f"evict_device_blocks failed, request_id: {request.request_id}")
                        return []

                    if self.enable_host_cache:
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
                                src_type="device",
                                dst_type="host",
                            )
                        )

                allocated = self._device_pool.allocate(need_block_num)
                if allocated is None:
                    logger.error(
                        f"allocate device blocks failed, request_id: {request.request_id}, need: {need_block_num}"
                    )
                    return []

                # DEBUG LOG: 分配的 blocks
                logger.debug(
                    f"[DEBUG] allocate_device_blocks request_id={request.request_id} "
                    f"allocated_blocks={allocated}, need_block_num={need_block_num}, "
                    f"new_blocks_num={num_blocks}, matched_host_nums={match_result.matched_host_nums}"
                )

                if self.enable_host_cache and match_result.matched_host_nums > 0:
                    device_blocks = allocated[: match_result.matched_host_nums]

                    # DEBUG LOG: swap host to device
                    logger.debug(
                        f"[DEBUG] swap_host_to_device request_id={request.request_id} "
                        f"host_nodes={[n.block_id for n in match_result.host_nodes]}, "
                        f"target_device_blocks={device_blocks}"
                    )

                    free_host_block_ids = self._radix_tree.swap_to_device(match_result.host_nodes, device_blocks)

                    request.cache_swap_metadata.append(
                        CacheSwapMetadata(
                            src_block_ids=free_host_block_ids,
                            dst_block_ids=device_blocks,
                            src_type="host",
                            dst_type="device",
                        )
                    )

                    # DEBUG LOG: swap 完成后释放的 host blocks
                    logger.debug(
                        f"[DEBUG] swap_host_to_device done request_id={request.request_id} "
                        f"freed_host_blocks={free_host_block_ids}"
                    )

                    self.free_host_blocks(free_host_block_ids)

                    match_result.device_nodes.extend(match_result.host_nodes)
                    match_result.host_nodes = []

                    # DEBUG LOG: radix tree 状态
                    _debug_log_radix_tree_state(
                        request.request_id,
                        "allocate_device_after_swap",
                        self._radix_tree,
                        self._device_pool,
                        self._host_pool,
                    )

                if self.enable_prefix_caching:
                    block_hashes = request.prompt_hashes[match_result.matched_device_nums :]
                    all_device_blocks = request.block_tables + allocated
                    uncached_device_blocks = all_device_blocks[match_result.matched_device_nums :]
                    num_block_lens = min(len(uncached_device_blocks), len(block_hashes))

                    # DEBUG LOG: insert 参数
                    logger.debug(
                        f"[DEBUG] allocate_device_blocks insert_params request_id={request.request_id} "
                        f"num_blocks={num_blocks}, num_block_lens={num_block_lens}, "
                        f"block_hashes_len={len(block_hashes)}, "
                        f"uncached_device_blocks={uncached_device_blocks}"
                    )

                    if num_block_lens > 0:
                        blocks = list(zip(block_hashes[:num_block_lens], uncached_device_blocks[:num_block_lens]))
                        start_node = match_result.device_nodes[-1] if match_result.device_nodes else None

                        # DEBUG LOG: insert 前状态
                        logger.debug(
                            f"[DEBUG] allocate_device_blocks before_insert request_id={request.request_id} "
                            f"blocks_len={len(blocks)}, blocks={blocks}, "
                            f"start_node_block_id={start_node.block_id if start_node else None}"
                        )

                        device_nodes, wasted_block_ids = self._radix_tree.insert(blocks=blocks, start_node=start_node)
                        match_result.device_nodes.extend(device_nodes)

                        for node in device_nodes:
                            logger.debug(
                                f"[DEBUG] allocate_device_blocks, ref_count: {node.ref_count}, "
                                f"evictable: {node.node_id in self._radix_tree._evictable_set}, block_id: {node.block_id}"
                            )

                        # DEBUG LOG: insert 结果
                        logger.debug(
                            f"[DEBUG] allocate_device_blocks after_insert request_id={request.request_id} "
                            f"wasted_block_ids={wasted_block_ids}"
                        )

                        # Release any blocks that were wasted due to node reuse
                        # and update allocated with actual block_ids
                        if wasted_block_ids:
                            match_result.uncached_block_ids.extend(wasted_block_ids)

                        # DEBUG LOG: 最终 uncached_device_blocks
                        logger.debug(
                            f"[DEBUG] allocate_device_blocks final_blocks request_id={request.request_id} "
                            f"allocated={allocated}"
                        )

                        # DEBUG LOG: radix tree 状态
                        _debug_log_radix_tree_state(
                            request.request_id,
                            "allocate_device_after_insert",
                            self._radix_tree,
                            self._device_pool,
                            self._host_pool,
                        )

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
                    logger.debug(
                        f"evict_host_nodes: {evict_blocks}, free host blocks: {self._host_pool.available_blocks()}"
                    )

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
            # DEBUG LOG: 释放 device blocks
            logger.debug(f"[DEBUG] free_device_blocks block_ids={block_ids}")
            self._device_pool.release(block_ids)

    def free_host_blocks(self, block_ids: List[int]) -> None:
        """
        Free host blocks back to the pool.

        Args:
            block_ids: List of block indices to free
        """
        if not block_ids:
            return
        # DEBUG LOG: 释放 host blocks
        logger.debug(f"[DEBUG] free_host_blocks block_ids={block_ids}")
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

    def write_cache_to_storage(self, req: Any) -> None:
        """
        Write request cache to storage if storage is enabled.

        Args:
            req: The request object containing cache data to write
        """
        if self._storage_scheduler is None:
            return
        # TODO: Implement storage write logic when storage is enabled
        pass

    @property
    def gpu_free_block_list(self) -> List[int]:
        """
        Get list of free GPU block indices (legacy alias).

        Returns list of available device block IDs for compatibility
        with PrefixCacheManager.gpu_free_block_list.
        """
        # Return list representation of available blocks
        return list(range(self._device_pool.available_blocks()))

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
                    result.storage_nodes = self.prepare_prefetch_metadata(storage_matches)

                # Step 3: Increment ref count for matched blocks(only first match node)
                if not (self._storage_scheduler and skip_storage):
                    self._radix_tree.increment_ref_nodes(matched_nodes)

                # DEBUG LOG: 匹配结果详情
                for node in matched_nodes:
                    logger.debug(f"[DEBUG] matched node: block_id={node.block_id}, ref_count={node.ref_count}")

                # DEBUG LOG: radix tree 状态
                _debug_log_radix_tree_state(
                    request.request_id,
                    "match_prefix_after_match",
                    self._radix_tree,
                    self._device_pool,
                    self._host_pool,
                )

                logger.info(
                    f"match_prefix for request_id: {request.request_id} total_hashes: {len(block_hashes)}, "
                    f"total_matched: {result.total_matched_blocks} (device_blocks={result.matched_device_nums}, "
                    f"host_blocks={result.matched_host_nums}, storage_hashes={result.matched_storage_nums})"
                )
                request._match_result = result
            except Exception as e:
                logger.error(f"match_prefix error: {e}, {str(traceback.format_exc())}")

    def _match_storage(self, hash_values: List[str]) -> List[str]:
        """
        Match hash values against storage.

        Args:
            hash_values: List of hash values to check

        Returns:
            List of hashes that exist in storage
        """
        if not self._storage_scheduler:
            return []

        try:
            if not self._storage_scheduler.is_connected():
                self._storage_scheduler.connect()

            existence_map = self._storage_scheduler.query(hash_values)
            return [h for h, exists in existence_map.items() if exists]
        except Exception:
            return []

    # ============ Eviction Methods ============

    def _evict_blocks(self, num_blocks: int) -> Optional[List[int]]:
        """
        Evict device blocks to free device memory.

        Eviction flow:
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
            return []

        try:
            with self._lock:
                # DEBUG LOG: radix tree 状态 - 驱逐前
                _debug_log_radix_tree_state(
                    "", "evict_blocks_before", self._radix_tree, self._device_pool, self._host_pool
                )

                # Step 1: Check if we have enough evictable device blocks
                stats = self._radix_tree.get_stats()
                if stats.evictable_device_count < num_blocks:
                    logger.warning(
                        f"_evict_blocks: not enough evictable device blocks, "
                        f"needed: {num_blocks}, available: {stats.evictable_device_count}"
                    )
                    return None

                # Step 2: Try to allocate host blocks for eviction target
                host_block_ids = []
                if self.enable_host_cache:
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

                # Step 3: Free the evicted device blocks
                self._device_pool.release(released_device_ids)

                # DEBUG LOG: radix tree 状态 - 驱逐后
                _debug_log_radix_tree_state(
                    "", f"evict_blocks_after(num={num_blocks})", self._radix_tree, self._device_pool, self._host_pool
                )
                logger.debug(f"[DEBUG] _evict_blocks done released_device_ids={released_device_ids}")

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
                # DEBUG LOG: 请求结束时的 block_tables
                logger.debug(
                    f"[DEBUG] request_finish start request_id={request.request_id} "
                    f"block_tables={request.block_tables}"
                )

                if self.enable_prefix_caching and self._radix_tree is not None:
                    match_result = request.match_result

                    block_hashes = request.prompt_hashes[match_result.matched_device_nums :]
                    device_blocks = request.block_tables[match_result.matched_device_nums :]
                    num_block_lens = min(len(device_blocks), len(block_hashes))

                    # DEBUG LOG: insert 参数
                    logger.debug(
                        f"[DEBUG] request_finish insert_params request_id={request.request_id} "
                        f"device_blocks_len={len(device_blocks)}, num_block_lens={num_block_lens}, "
                        f"block_hashes_len={len(block_hashes)}, device_blocks={device_blocks}"
                    )

                    if num_block_lens > 0:
                        blocks = list(zip(block_hashes[:num_block_lens], device_blocks[:num_block_lens]))
                        start_node = match_result.device_nodes[-1] if match_result.device_nodes else None

                        # DEBUG LOG: insert 前状态
                        logger.debug(
                            f"[DEBUG] request_finish before_insert request_id={request.request_id} "
                            f"blocks_len={len(blocks)}, blocks={blocks}, "
                            f"start_node_block_id={start_node.block_id if start_node else None}"
                        )

                        device_nodes, wasted_block_ids = self._radix_tree.insert(blocks=blocks, start_node=start_node)
                        match_result.device_nodes.extend(device_nodes)

                        # DEBUG LOG: insert 结果
                        logger.debug(
                            f"[DEBUG] request_finish after_insert request_id={request.request_id} "
                            f"device_nodes_len={len(device_nodes)}, "
                            f"device_nodes_block_ids={[n.block_id for n in device_nodes]}, "
                            f"wasted_block_ids={wasted_block_ids}"
                        )

                        # Release blocks that were wasted due to node reuse
                        if wasted_block_ids:
                            # DEBUG LOG: 浪费的 blocks
                            logger.debug(
                                f"[DEBUG] request_finish wasted_blocks request_id={request.request_id} "
                                f"wasted_block_ids={wasted_block_ids}"
                            )
                            match_result.uncached_block_ids.extend(wasted_block_ids)

                        # DEBUG LOG: radix tree 状态 - insert 后
                        _debug_log_radix_tree_state(
                            request.request_id,
                            "request_finish_after_insert",
                            self._radix_tree,
                            self._device_pool,
                            self._host_pool,
                        )

                    # DEBUG LOG: 释放 uncached blocks
                    uncached_blocks = match_result.uncached_block_ids
                    uncached_blocks.extend(request.block_tables[match_result.matched_device_nums :])

                    logger.debug(
                        f"[DEBUG] request_finish release_uncached_blocks request_id={request.request_id} "
                        f"uncached_blocks={uncached_blocks}"
                    )

                    # Decrement ref count - blocks become evictable if ref_count reaches 0
                    self._radix_tree.decrement_ref_nodes(match_result.device_nodes)
                    self._device_pool.release(uncached_blocks)

                    # DEBUG LOG: radix tree 状态 - 最终
                    _debug_log_radix_tree_state(
                        request.request_id,
                        "request_finish_final",
                        self._radix_tree,
                        self._device_pool,
                        self._host_pool,
                    )

                    logger.info(
                        f"request {request.request_id} finished, cached blocks: {match_result.matched_device_nums}, "
                        f"uncached blocks freed: {len(uncached_blocks)}, "
                        f"total_free: {self._device_pool.available_blocks()}"
                    )
                else:
                    self._device_pool.release(request.block_tables)

                    logger.info(
                        f"request {request.request_id} finished, release blocks: {len(request.block_tables)}, "
                        f"total_free: {self._device_pool.available_blocks()}"
                    )
            except Exception as e:
                logger.error(f"request_finish error: {e}, {str(traceback.format_exc())}")

    # ============ Host/Device Transfer Coordination ============

    def offload_to_host(self, block_indices: List[int]) -> bool:
        """
        Offload blocks from device to host memory.

        This is a coordination method. Actual data transfer happens in Worker.

        Args:
            block_indices: List of block indices to offload

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                # Allocate host blocks
                host_indices = self._host_pool.allocate(len(block_indices))
                if host_indices is None or len(host_indices) != len(block_indices):
                    # Not enough host memory, release what we allocated
                    if host_indices:
                        self._host_pool.release(host_indices)
                    return False

                # Perform the offload (actual data transfer would happen in Worker)
                for i, dev_idx in enumerate(block_indices):
                    host_idx = host_indices[i]
                    metadata = self._device_pool.get_metadata(dev_idx)
                    if metadata:
                        self._host_pool.set_metadata(host_idx, metadata)

                # Release device blocks
                self._device_pool.release(block_indices)

                return True
        except Exception as e:
            logger.error(f"offload_to_host error: {e}, {str(traceback.format_exc())}")
            return False

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

    def prepare_prefetch_metadata(
        self,
        storage_hashes: List[str],
    ) -> Optional[List["BlockNode"]]:
        """
        Prepare metadata for storage prefetch operation.

        Called when storage cache is matched, allocates host blocks
        for the prefetch target.

        Args:
            storage_hashes: List of storage hash values to prefetch

        Returns:
            List of BlockNode objects if successful, None or empty list otherwise.
            Each node's block_id contains the actual block assigned
            (may differ from originally allocated if node was reused).
        """
        if not storage_hashes:
            return None

        try:
            with self._lock:
                # Check if we have enough host blocks
                if not self.can_allocate_host_blocks(len(storage_hashes)):
                    return []

                # Allocate host blocks for prefetch
                host_block_ids = self._host_pool.allocate(len(storage_hashes))
                if host_block_ids is None or len(host_block_ids) == 0:
                    return []

                blocks = list(zip(storage_hashes, host_block_ids))
                prefetch_nodes, wasted_block_ids = self._radix_tree.insert(
                    blocks=blocks, cache_status=CacheStatus.LOADING_FROM_STORAGE
                )
                # Release any blocks that were wasted due to node reuse
                if wasted_block_ids:
                    self._host_pool.release(wasted_block_ids)

                return prefetch_nodes
        except Exception as e:
            logger.error(f"prepare_prefetch_metadata error: {e}, {str(traceback.format_exc())}")
            return []

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
