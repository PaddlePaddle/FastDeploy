"""
CacheController - Worker-side cache control.

Responsible for:
- Managing cache transfer operations
- Layer-by-layer transfer synchronization
- Cross-node transfer via TransferConnector

Note: CacheController does NOT manage BlockPool. BlockPool is managed
by CacheManager in the Scheduler process. CacheController only handles
data transfer operations based on block IDs provided by Scheduler.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import paddle
from paddleformers.utils.log import logger

if TYPE_CHECKING:
    from fastdeploy.config import FDConfig

# Import ops for CPU cache allocation
from fastdeploy.cache_manager.ops import cuda_host_alloc

from .base import KVCacheBase
from .cache_utils import LayerDoneCounter
from .metadata import (
    AsyncTaskHandler,
    CacheSwapMetadata,
    PDTransferMetadata,
    StorageMetadata,
    TransferResult,
    TransferStatus,
    TransferTask,
)
from .transfer_manager import CacheTransferManager


class CacheController(KVCacheBase):
    """
    Cache Controller for Worker process.

    Inherits KVCacheBase, handles transfer tasks by block index only, does NOT manage BlockPool.
    BlockPool is managed by CacheManager. CacheController only executes transfers
    based on block IDs provided by Scheduler.

    All transfer methods are async - they submit tasks and return immediately,
    returning an AsyncTaskHandler for the caller to track completion.

    Three-level cache hierarchy:
        Level 1: Device (GPU) - Fastest access, directly used for inference
        Level 2: Host (CPU) - Medium speed, needs to be loaded to Device
        Level 3: Storage - Slowest, needs to be fetched to Host first

    Attributes:
        transfer_manager: CacheTransferManager instance.
        layer_counter: LayerDoneCounter instance.
        num_layers: Total number of model layers.
    """

    def __init__(self, config: "FDConfig", local_rank: int, device_id: int):
        """
        Initialize the Cache Controller.

        Args:
            config: FDConfig instance containing all fastdeploy configuration
        """
        super().__init__(config)

        # Extract configuration from FDConfig
        self.model_config = config.model_config
        self.cache_config = config.cache_config
        self.quant_config = config.quant_config
        self.parallel_config = config.parallel_config

        self._num_layers = self.model_config.num_hidden_layers
        self._local_rank = local_rank
        self._device_id = device_id

        # cache_kvs_map: stores created kv cache tensors by name
        self.cache_kvs_map: Dict[str, Any] = {}
        # host_cache_kvs_map: stores Host (pinned memory) kv cache tensors by name for swap space
        self.host_cache_kvs_map: Dict[str, Any] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Thread pool executor for async operations
        # Used to wrap synchronous transfer operations into async tasks
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cache_transfer")

        # Initialize transfer manager
        self._transfer_manager = CacheTransferManager(config, local_rank, device_id)

        # Initialize layer done counter
        self._layer_counter = LayerDoneCounter(self._num_layers)

        # Active transfer tasks
        self._active_tasks: Dict[str, TransferTask] = {}

        # Active async handlers
        self._async_handlers: Dict[str, AsyncTaskHandler] = {}

        self._initialized = True

    # ============ Properties ============

    @property
    def transfer_manager(self) -> CacheTransferManager:
        """Get the transfer manager."""
        return self._transfer_manager

    @property
    def layer_counter(self) -> LayerDoneCounter:
        """Get the layer done counter."""
        return self._layer_counter

    # ============ Helper Methods ============

    def _get_kv_cache_quant_type(self) -> Optional[str]:
        """Get KV cache quantization type."""
        if (
            self.quant_config
            and hasattr(self.quant_config, "kv_cache_quant_type")
            and self.quant_config.kv_cache_quant_type is not None
        ):
            return self.quant_config.kv_cache_quant_type
        return None

    def _is_fp8_quantization(self, quant_type: Optional[str] = None) -> bool:
        """Check if using fp8 quantization."""
        if quant_type is None:
            quant_type = self._get_kv_cache_quant_type()
        return quant_type == "block_wise_fp8"

    def _get_cache_names(self, layer_idx: int) -> Dict[str, str]:
        """
        Generate cache names for a layer.

        Args:
            layer_idx: Layer index.

        Returns:
            Dictionary with cache names: {
                "key": "key_caches_{layer}_rank{rank}.device{device}",
                "value": "value_caches_{layer}_rank{rank}.device{device}",
                "key_scale": "key_cache_scales_{layer}_rank{rank}.device{device}",
                "value_scale": "value_cache_scales_{layer}_rank{rank}.device{device}",
            }
        """
        local_rank = self._local_rank % self.parallel_config.tensor_parallel_size

        return {
            "key": f"key_caches_{layer_idx}_rank{local_rank}.device{self._device_id}",
            "value": f"value_caches_{layer_idx}_rank{local_rank}.device{self._device_id}",
            "key_scale": f"key_cache_scales_{layer_idx}_rank{local_rank}.device{self._device_id}",
            "value_scale": f"value_cache_scales_{layer_idx}_rank{local_rank}.device{self._device_id}",
        }

    # ============ KV Cache Management ============

    def get_kv_caches(self) -> Optional[Dict[str, Any]]:
        """
        Get the current KV Cache tensor dictionary.

        Returns:
            KV Cache tensor dictionary, None if not initialized.
        """
        with self._lock:
            return self.cache_kvs_map

    def initialize_kv_cache(
        self,
        attn_backend: Any,
        num_gpu_blocks: int,
    ) -> List[Any]:
        """
        Initialize KV Cache tensors.

        Create KV Cache tensors on GPU for storing attention Key and Value.

        Args:
            attn_backend: Attention backend instance for getting kv cache shape.
            num_gpu_blocks: Maximum number of blocks on GPU.

        Returns:
            cache_kvs_list: KV Cache tensor list in [key_cache_layer0, value_cache_layer0, ...] order.
        """
        # Get kv cache quantization type
        kv_cache_quant_type = self._get_kv_cache_quant_type()

        # Get kv cache shape
        key_cache_shape, value_cache_shape = attn_backend.get_kv_cache_shape(
            max_num_blocks=num_gpu_blocks, kv_cache_quant_type=kv_cache_quant_type
        )

        # Get scale shape for block_wise_fp8 quantization
        kv_cache_scale_shape = None
        if self._is_fp8_quantization(kv_cache_quant_type):
            kv_cache_scale_shape = [key_cache_shape[0], key_cache_shape[1], key_cache_shape[2]]

        logger.info(f"Initializing kv cache for all layers. num_layers={self._num_layers}")
        cache_kvs_list = []

        for i in range(self._num_layers):
            # Generate cache names
            cache_names = self._get_cache_names(i)

            logger.info(f"..creating kv cache for layer {i}: key:{key_cache_shape}, value:{value_cache_shape}")

            # Create key cache and value cache
            key_cache = paddle.full(shape=key_cache_shape, fill_value=0, dtype=self.model_config.dtype)
            self.cache_kvs_map[cache_names["key"]] = key_cache

            val_cache = paddle.full(shape=value_cache_shape, fill_value=0, dtype=self.model_config.dtype)
            self.cache_kvs_map[cache_names["value"]] = val_cache
            cache_kvs_list.extend([key_cache, val_cache])

            # Create scale caches for block_wise_fp8 quantization
            if self._is_fp8_quantization(kv_cache_quant_type) and kv_cache_scale_shape:
                key_cache_scales = paddle.full(
                    shape=kv_cache_scale_shape, fill_value=0, dtype=paddle.get_default_dtype()
                )
                val_cache_scales = paddle.full(
                    shape=kv_cache_scale_shape, fill_value=0, dtype=paddle.get_default_dtype()
                )
                self.cache_kvs_map[cache_names["key_scale"]] = key_cache_scales
                self.cache_kvs_map[cache_names["value_scale"]] = val_cache_scales
                cache_kvs_list.extend([key_cache_scales, val_cache_scales])

        paddle.device.cuda.empty_cache()
        logger.info("kv cache is initialized!")

        # Share cache_kvs_map with transfer manager for data transfer operations
        self._transfer_manager.set_cache_kvs_map(self.cache_kvs_map)

        # Initialize host cache
        self.initialize_host_cache(attn_backend)

        return cache_kvs_list

    def initialize_host_cache(
        self,
        attn_backend: Any,
    ) -> Dict[str, Any]:
        """
        Initialize Host (Pinned Memory) KV Cache.

        Use cuda_host_alloc to allocate pinned memory for fast Host-Device data transfer.
        Called during initialization to create Host-side swap space.

        Args:
            attn_backend: Attention backend instance for getting kv cache shape.

        Returns:
            host_cache_kvs_map: Host KV Cache pointer dictionary, indexed by name.
        """
        num_host_blocks = self.cache_config.num_cpu_blocks
        if num_host_blocks == 0:
            logger.info("[CacheController] No swap space (Host cache) specified, skipping initialization.")
            return

        if len(self.host_cache_kvs_map) > 0:
            return

        # Get kv cache quantization type
        kv_cache_quant_type = self._get_kv_cache_quant_type()

        # Get kv cache shape (pass num_host_blocks as max_num_blocks for host cache)
        key_cache_shape, value_cache_shape = attn_backend.get_kv_cache_shape(
            max_num_blocks=num_host_blocks, kv_cache_quant_type=kv_cache_quant_type
        )

        # Calculate cache sizes (elements per block per layer)
        key_cache_size = key_cache_shape[1] * key_cache_shape[2] * key_cache_shape[3]
        if value_cache_shape:
            value_cache_size = value_cache_shape[1] * value_cache_shape[2] * value_cache_shape[3]
        else:
            value_cache_size = 0

        # Get cache dtype and bytes per element
        cache_dtype = self.cache_config.cache_dtype
        cache_item_bytes = self.cache_config.get_cache_bytes(cache_dtype)

        # Calculate total bytes to allocate
        key_need_to_allocate_bytes = num_host_blocks * cache_item_bytes * key_cache_size
        value_need_to_allocate_bytes = num_host_blocks * cache_item_bytes * value_cache_size

        # Calculate scale sizes for block_wise_fp8 quantization
        scales_key_need_to_allocate_bytes = 0
        scales_value_need_to_allocate_bytes = 0
        cache_scale_shape = None
        if self._is_fp8_quantization(kv_cache_quant_type):
            cache_scales_size = key_cache_shape[1] * key_cache_shape[2]
            # Scale tensor uses default dtype (float32)
            scale_bytes = 4  # float32
            scales_key_need_to_allocate_bytes = num_host_blocks * scale_bytes * cache_scales_size
            scales_value_need_to_allocate_bytes = num_host_blocks * scale_bytes * cache_scales_size
            cache_scale_shape = [num_host_blocks, key_cache_shape[1], key_cache_shape[2]]

        total_size_gb = (key_need_to_allocate_bytes + value_need_to_allocate_bytes) / (1024**3)
        logger.info(
            f"[CacheController] Host swap space size: {total_size_gb:.2f}GB, " f"num_host_blocks: {num_host_blocks}"
        )

        logger.info(f"[CacheController] Initializing swap space (Host cache) for {self._num_layers} layers.")

        # Allocate Host cache for each layer
        for i in range(self._num_layers):
            # Generate cache names
            cache_names = self._get_cache_names(i)

            logger.info(
                f"[CacheController] Creating Host cache for layer {i}: "
                f"key={(key_need_to_allocate_bytes / 1024 ** 3):.2f}GB, "
                f"value={(value_need_to_allocate_bytes / 1024 ** 3):.2f}GB"
            )

            # Allocate key cache using cuda_host_alloc (pinned memory)
            self.host_cache_kvs_map[cache_names["key"]] = cuda_host_alloc(key_need_to_allocate_bytes)

            # Allocate scale cache for block_wise_fp8 quantization
            if self._is_fp8_quantization(kv_cache_quant_type):
                self.host_cache_kvs_map[cache_names["key_scale"]] = cuda_host_alloc(scales_key_need_to_allocate_bytes)

            # Allocate value cache if needed
            if value_need_to_allocate_bytes > 0:
                self.host_cache_kvs_map[cache_names["value"]] = cuda_host_alloc(value_need_to_allocate_bytes)
                if self._is_fp8_quantization(kv_cache_quant_type):
                    self.host_cache_kvs_map[cache_names["value_scale"]] = cuda_host_alloc(
                        scales_value_need_to_allocate_bytes
                    )

        logger.info(f"[CacheController] Swap space (Host cache) is ready for {self._num_layers} layers!")

        # Store shapes for later use
        self._host_key_cache_shape = [num_host_blocks] + list(key_cache_shape[1:])
        self._host_value_cache_shape = [num_host_blocks] + list(value_cache_shape[1:]) if value_cache_shape else None
        self._host_cache_scale_shape = cache_scale_shape
        self._num_host_blocks = num_host_blocks

        # Share host_cache_kvs_map with transfer manager
        self._transfer_manager.set_host_cache_kvs_map(self.host_cache_kvs_map)

    def get_host_cache_kvs_map(self) -> Dict[str, Any]:
        """
        Get the Host KV Cache pointer dictionary.

        Returns:
            Host KV Cache pointer dictionary, empty dict if not initialized.
        """
        return self.host_cache_kvs_map

    # ============ Worker Methods ============

    def _submit_swap_task(
        self,
        meta: CacheSwapMetadata,
        src_location: str,
        dst_location: str,
        transfer_fn_all: callable,
        transfer_fn_layer: callable,
    ) -> None:
        """
        Submit a single swap transfer task (internal method).

        Creates an independent async transfer task for each CacheSwapMetadata.
        The handler is saved in meta.async_handler for upstream tracking.

        Transfer mode is determined by global config self._transfer_manager.swap_all_layers.

        Args:
            meta: CacheSwapMetadata containing src_block_ids and dst_block_ids.
            src_location: Source location ("host" or "device").
            dst_location: Destination location ("device" or "host").
            transfer_fn_all: All-layer transfer function, signature (src_ids, dst_ids) -> bool.
            transfer_fn_layer: Layer-by-layer transfer function, signature (layer_indices, on_layer_complete, src_ids, dst_ids) -> bool.
        """
        handler = AsyncTaskHandler()
        meta.async_handler = handler
        task_id = handler.task_id

        src_block_ids = meta.src_block_ids
        dst_block_ids = meta.dst_block_ids

        if not src_block_ids or not dst_block_ids:
            logger.info(
                f"[SwapTask] task_id={task_id} skip: empty block_ids " f"src={src_block_ids}, dst={dst_block_ids}"
            )
            meta.success = False
            meta.error_message = "Empty block IDs in CacheSwapMetadata"
            handler.set_error(meta.error_message)
            return

        use_all_layers = self._transfer_manager.swap_all_layers
        layers_to_transfer = list(range(self._num_layers))
        mode = "all_layers" if use_all_layers else "layer_by_layer"

        logger.info(
            f"[SwapTask] submit task_id={task_id} {src_location}->{dst_location} "
            f"src_block_ids={src_block_ids} dst_block_ids={dst_block_ids} "
            f"num_blocks={len(src_block_ids)} mode={mode}"
        )

        task = TransferTask(
            task_id=task_id,
            src_location=src_location,
            dst_location=dst_location,
            block_indices=list(zip(src_block_ids, dst_block_ids)),
            layer_indices=layers_to_transfer,
            status=TransferStatus.PENDING,
        )

        with self._lock:
            self._active_tasks[task_id] = task
            self._async_handlers[task_id] = handler
            self._layer_counter.start_transfer(task_id)
            task.status = TransferStatus.IN_PROGRESS

        def _on_layer_complete(layer_idx: int) -> None:
            self._layer_counter.mark_layer_done(task_id, layer_idx)

        def _do_transfer():
            try:
                start_time = time.time()
                if use_all_layers:
                    success = transfer_fn_all(src_block_ids, dst_block_ids)
                    elapsed = time.time() - start_time
                    if success:
                        for layer_idx in layers_to_transfer:
                            _on_layer_complete(layer_idx)
                    result = TransferResult(
                        src_block_ids=src_block_ids,
                        dst_block_ids=dst_block_ids,
                        src_type=src_location,
                        dst_type=dst_location,
                        success=success,
                        error_message=None if success else f"All-layer {src_location}→{dst_location} transfer failed",
                    )
                    logger.info(
                        f"[SwapTask] task_id={task_id} all_layers transfer "
                        f"{'success' if success else 'FAILED'} "
                        f"elapsed={elapsed:.3f}s "
                        f"src={src_block_ids} dst={dst_block_ids}"
                    )
                else:
                    success = transfer_fn_layer(
                        layers_to_transfer,
                        _on_layer_complete,
                        src_block_ids,
                        dst_block_ids,
                    )
                    elapsed = time.time() - start_time
                    result = TransferResult(
                        src_block_ids=src_block_ids,
                        dst_block_ids=dst_block_ids,
                        src_type=src_location,
                        dst_type=dst_location,
                        success=success,
                        error_message=(
                            None if success else f"Layer-by-layer {src_location}→{dst_location} transfer failed"
                        ),
                    )
                    logger.info(
                        f"[SwapTask] task_id={task_id} layer_by_layer transfer "
                        f"{'success' if success else 'FAILED'} "
                        f"elapsed={elapsed:.3f}s "
                        f"src={src_block_ids} dst={dst_block_ids}"
                    )

                with self._lock:
                    task = self._active_tasks.get(task_id)
                    if task:
                        task.status = TransferStatus.COMPLETED if result.success else TransferStatus.FAILED
                        task.completed_time = time.time()
                        if not result.success:
                            task.error_message = result.error_message

                # Update metadata with result
                meta.success = result.success
                meta.error_message = result.error_message
                handler.set_result(result)

                total_elapsed = time.time() - start_time
                logger.info(
                    f"[SwapTask] task_id={task_id} {src_location}->{dst_location} "
                    f"{'SUCCESS' if result.success else 'FAILED'} "
                    f"num_blocks={len(src_block_ids)} total_elapsed={total_elapsed:.3f}s"
                )

            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(
                    f"[SwapTask] task_id={task_id} {src_location}->{dst_location} "
                    f"EXCEPTION: {e}\n{traceback.format_exc()}"
                )
                with self._lock:
                    task = self._active_tasks.get(task_id)
                    if task:
                        task.status = TransferStatus.FAILED
                        task.error_message = str(e)
                meta.success = False
                meta.error_message = str(e)
                handler.set_error(str(e))
            finally:
                self._layer_counter.clear_transfer(task_id)

        self._executor.submit(_do_transfer)

    def load_host_to_device(
        self,
        swap_metadata: list[CacheSwapMetadata],
    ) -> None:
        """
        Load host cache to device (async).

        Creates an independent async transfer task for each CacheSwapMetadata, executed in parallel.
        Each task's AsyncTaskHandler is saved in the corresponding CacheSwapMetadata.async_handler,
        allowing the caller to track each task's execution status.

        Uses layer-by-layer transfer strategy to overlap with forward computation.
        Each layer's completion is marked via LayerDoneCounter.

        Args:
            swap_metadata: CacheSwapMetadata list, each element containing:
                - src_block_ids: Source host block IDs
                - dst_block_ids: Destination device block IDs
        """
        for meta in swap_metadata:
            self._submit_swap_task(
                meta=meta,
                src_location="host",
                dst_location="device",
                transfer_fn_all=lambda src_ids, dst_ids: self._transfer_manager.load_to_device_all_layers(
                    src_ids, dst_ids
                ),
                transfer_fn_layer=lambda layer_indices, on_layer_complete, src_ids, dst_ids: self._transfer_manager.load_layers_to_device(
                    layer_indices=layer_indices,
                    host_block_ids=src_ids,
                    device_block_ids=dst_ids,
                    on_layer_complete=on_layer_complete,
                ),
            )
        logger.info(
            f"[LoadHostToDevice] submitted {len(swap_metadata)} swap task(s), "
            f"total_blocks={sum(len(m.src_block_ids) for m in swap_metadata)}"
        )

    def evict_device_to_host(
        self,
        swap_metadata: list[CacheSwapMetadata],
    ) -> None:
        """
        Evict device cache to host (async).

        Creates an independent async transfer task for each CacheSwapMetadata, executed in parallel.
        Each task's AsyncTaskHandler is saved in the corresponding CacheSwapMetadata.async_handler,
        allowing the caller to track each task's execution status.

        Args:
            swap_metadata: CacheSwapMetadata list, each element containing:
                - src_block_ids: Source device block IDs
                - dst_block_ids: Destination host block IDs
        """
        for meta in swap_metadata:
            self._submit_swap_task(
                meta=meta,
                src_location="device",
                dst_location="host",
                transfer_fn_all=lambda src_ids, dst_ids: self._transfer_manager.evict_to_host_all_layers(
                    src_ids, dst_ids
                ),
                transfer_fn_layer=lambda layer_indices, on_layer_complete, src_ids, dst_ids: self._transfer_manager.evict_layers_to_host(
                    layer_indices=layer_indices,
                    device_block_ids=src_ids,
                    host_block_ids=dst_ids,
                    on_layer_complete=on_layer_complete,
                ),
            )
        logger.info(
            f"[EvictDeviceToHost] submitted {len(swap_metadata)} swap task(s), "
            f"total_blocks={sum(len(m.src_block_ids) for m in swap_metadata)}"
        )

    def prefetch_from_storage(
        self,
        metadata: StorageMetadata,
    ) -> AsyncTaskHandler:
        """
        Prefetch storage cache to host (async).

        When Scheduler matches cache in storage, Worker uses this method
        to pull data from storage to host.

        Args:
            metadata: Storage transfer metadata, containing:
                - hash_values: Hash values to fetch
                - block_ids: Destination host block IDs (pre-allocated by Scheduler)
                - Other storage-specific parameters

        Returns:
            AsyncTaskHandler for tracking the async transfer task.
        """

        handler = AsyncTaskHandler()

        # TODO: Implement storage prefetch logic
        handler.set_error("Storage prefetch not implemented yet")

        return handler

    def backup_device_to_storage(
        self,
        device_block_ids: List[int],
        metadata: StorageMetadata,
    ) -> AsyncTaskHandler:
        """
        Backup device cache to storage (async).

        Backup KV cache from device memory to external storage
        for reuse by subsequent requests.

        Args:
            device_block_ids: Device block IDs to backup.
            metadata: Storage transfer metadata.

        Returns:
            AsyncTaskHandler for tracking the async transfer task.
        """

        handler = AsyncTaskHandler()

        # TODO: Implement storage backup logic
        handler.set_error("Storage backup not implemented yet")

        return handler

    def backup_host_to_storage(
        self,
        host_block_ids: List[int],
        metadata: StorageMetadata,
    ) -> AsyncTaskHandler:
        """
        Backup host cache to storage (async).

        Backup KV cache from host memory to external storage.

        Args:
            host_block_ids: Host block IDs to backup.
            metadata: Storage transfer metadata.

        Returns:
            AsyncTaskHandler for tracking the async transfer task.
        """

        handler = AsyncTaskHandler()

        # TODO: Implement storage backup logic
        handler.set_error("Storage backup not implemented yet")

        return handler

    def send_to_node(
        self,
        metadata: PDTransferMetadata,
    ) -> AsyncTaskHandler:
        """
        Send cache to another node (PD separation, async).

        In PD separation architecture, P node uses this method
        to send KV cache to D node.

        Args:
            metadata: PD transfer metadata, containing:
                - target_node_id: Target node identifier
                - block_ids: Block IDs to transfer
                - Other transfer-specific parameters

        Returns:
            AsyncTaskHandler for tracking the async transfer task.
        """

        handler = AsyncTaskHandler()

        # TODO: Implement PD separation transfer logic
        handler.set_error("PD transfer not implemented yet")

        return handler

    def wait_for_transfer_from_node(
        self,
        metadata: PDTransferMetadata,
    ) -> AsyncTaskHandler:
        """
        Wait for cache transfer from another node (PD separation, async).

        In PD separation architecture, D node uses this method
        to wait for P node to send KV cache.

        Args:
            metadata: PD transfer metadata, containing:
                - source_node_id: Source node identifier
                - block_ids: Block IDs to receive
                - Other transfer-specific parameters

        Returns:
            AsyncTaskHandler for tracking the async transfer task.
        """

        handler = AsyncTaskHandler()

        # TODO: Implement PD separation transfer wait logic
        handler.set_error("PD transfer not implemented yet")

        return handler

    # ============ Transfer Status Methods ============

    def get_transfer_status(self, transfer_id: str) -> Optional[TransferStatus]:
        """
        Get the status of a transfer task.

        Args:
            transfer_id: Unique identifier for the transfer

        Returns:
            Current transfer status or None if not found
        """
        with self._lock:
            if transfer_id not in self._active_tasks:
                return None
            return self._active_tasks[transfer_id].status

    def cancel_transfer(self, transfer_id: str) -> bool:
        """
        Cancel an active transfer.

        Args:
            transfer_id: Unique identifier for the transfer

        Returns:
            True if cancellation was successful
        """
        with self._lock:
            if transfer_id not in self._active_tasks:
                return False

            task = self._active_tasks[transfer_id]
            if task.status in [TransferStatus.COMPLETED, TransferStatus.FAILED]:
                return False

            task.status = TransferStatus.CANCELLED
            self._layer_counter.clear_transfer(transfer_id)

            # Cancel async handler
            if transfer_id in self._async_handlers:
                self._async_handlers[transfer_id].cancel()

            return self._transfer_manager.cancel_task(transfer_id)

    def get_async_handler(self, transfer_id: str) -> Optional[AsyncTaskHandler]:
        """
        Get the async handler for a transfer.

        Args:
            transfer_id: Unique identifier for the transfer

        Returns:
            AsyncTaskHandler or None if not found
        """
        return self._async_handlers.get(transfer_id)

    # ============ Layer Done Methods ============

    def mark_layer_done(self, transfer_id: str, layer_idx: int) -> bool:
        """
        Mark a layer as completed for a transfer.

        Args:
            transfer_id: Unique identifier for the transfer
            layer_idx: Index of the completed layer

        Returns:
            True if this was the last layer
        """
        return self._layer_counter.mark_layer_done(transfer_id, layer_idx)

    def is_layer_done(self, transfer_id: str, layer_idx: int) -> bool:
        """
        Check if a layer is completed.

        Args:
            transfer_id: Unique identifier for the transfer
            layer_idx: Index of the layer

        Returns:
            True if the layer is completed
        """
        return self._layer_counter.is_layer_done(transfer_id, layer_idx)

    def is_transfer_complete(self, transfer_id: str) -> bool:
        """
        Check if all layers are completed for a transfer.

        Args:
            transfer_id: Unique identifier for the transfer

        Returns:
            True if all layers are completed
        """
        return self._layer_counter.is_transfer_complete(transfer_id)

    def wait_for_layer(
        self,
        transfer_id: str,
        layer_idx: int,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Wait for a specific layer to complete.

        This is used by the forward computation thread to wait for
        layer transfer completion before using the cache.

        Args:
            transfer_id: Unique identifier for the transfer
            layer_idx: Index of the layer to wait for
            timeout: Maximum wait time in seconds

        Returns:
            True if layer completed, False if timeout or transfer not found
        """
        # Polling wait (could be optimized with events)
        start_time = time.time()
        while True:
            if self._layer_counter.is_layer_done(transfer_id, layer_idx):
                return True

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False

            time.sleep(0.001)  # Small sleep to avoid busy waiting

    def register_layer_callback(
        self,
        transfer_id: str,
        callback: Callable[[int], None],
    ) -> None:
        """
        Register a callback for layer completion.

        Args:
            transfer_id: Unique identifier for the transfer
            callback: Function to call when each layer completes
        """
        self._layer_counter.register_callback(transfer_id, callback)

    # ============ Progress Methods ============

    def get_progress(self, transfer_id: str) -> Dict[str, Any]:
        """
        Get transfer progress.

        Args:
            transfer_id: Unique identifier for the transfer

        Returns:
            Dictionary with progress information
        """
        with self._lock:
            if transfer_id not in self._active_tasks:
                return {"error": "Transfer not found"}

            task = self._active_tasks[transfer_id]
            completed = self._layer_counter.get_completed_count(transfer_id)
            total = len(task.layer_indices)

            return {
                "transfer_id": transfer_id,
                "status": task.status.value,
                "completed_layers": completed,
                "total_layers": total,
                "progress": completed / total if total > 0 else 0,
                "elapsed_time": self._layer_counter.get_elapsed_time(transfer_id),
            }

    # ============ Public Interface Implementation ============

    def reset_cache(self) -> bool:
        """
        Reset all cache state.

        Clears active tasks and resets layer counter.
        """
        try:
            with self._lock:
                # Cancel all active tasks
                for task_id, task in self._active_tasks.items():
                    if task.status in [TransferStatus.PENDING, TransferStatus.IN_PROGRESS]:
                        task.status = TransferStatus.CANCELLED

                self._layer_counter.reset()
                self._active_tasks.clear()
                self._async_handlers.clear()

            return True
        except Exception:
            return False

    def reset_controller_cache(self, reset_external: bool = False) -> bool:
        """
        Reset controller cache state.

        Args:
            reset_external: If True, also reset external storage cache

        Returns:
            True if successful, False otherwise
        """
        success = self.reset_cache()

        # Reset external storage if requested
        if reset_external and self._transfer_manager.storage_connector:
            try:
                # TODO: Call storage connector clear method
                pass
            except Exception:
                pass

        return success

    # ============ Statistics Methods ============

    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        with self._lock:
            status_counts = {}
            for status in TransferStatus:
                status_counts[status.value] = sum(1 for task in self._active_tasks.values() if task.status == status)

            return {
                "initialized": self._initialized,
                "num_layers": self._num_layers,
                "active_transfers": len(self._active_tasks),
                "status_counts": status_counts,
                "layer_counter": self._layer_counter.get_stats(),
                "transfer_manager": self._transfer_manager.get_stats(),
            }

    def start(self) -> None:
        """Start the transfer manager."""
        self._transfer_manager.start()

    def stop(self) -> None:
        """Stop the transfer manager and shutdown thread pool."""
        self._transfer_manager.stop()
        # Shutdown thread pool executor
        self._executor.shutdown(wait=False)
