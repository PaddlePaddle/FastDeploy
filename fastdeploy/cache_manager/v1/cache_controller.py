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

import ctypes
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import paddle
from paddleformers.utils.log import logger

if TYPE_CHECKING:
    from fastdeploy.config import FDConfig

# Import ops for CPU cache allocation
from fastdeploy.cache_manager.ops import cuda_host_alloc, cuda_host_free

from .base import KVCacheBase
from .cache_utils import LayerDoneCounter
from .metadata import (
    AsyncTaskHandler,
    CacheLevel,
    CacheSwapMetadata,
    PDTransferMetadata,
    StorageMetadata,
    TransferResult,
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
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cache_transfer")

        # Initialize transfer manager
        self._transfer_manager = CacheTransferManager(config, local_rank, device_id)

        # Note: LayerDoneCounter is no longer a singleton
        # Each submit_swap_tasks call creates a new LayerDoneCounter instance
        self._layer_done_counter = None

        # Pending evict LayerDoneCounters for write_back mode ordering
        self._pending_evict_counters: List["LayerDoneCounter"] = []

        self._initialized = True

        # NUMA binding flag
        self._numa_bound = False

    @property
    def write_policy(self) -> Optional[str]:
        """Get the write policy for cache operations."""
        if self.cache_config and hasattr(self.cache_config, "write_policy"):
            return self.cache_config.write_policy
        return None

    def _should_wait_for_swap_out(self) -> bool:
        """
        Determine if swap-out operations should wait synchronously.

        Returns:
            True if write_policy is 'write_back', otherwise False.
        """
        return self.write_policy == "write_back"

    def submit_swap_tasks(
        self,
        evict_metadata: Optional["CacheSwapMetadata"],
        swap_in_metadata: Optional["CacheSwapMetadata"],
    ) -> Optional["LayerDoneCounter"]:
        """
        Submit evict and swap-in tasks with proper synchronization.

        Logic:
        1. Before submitting evict, wait for existing pending evict counters to complete
        2. write_back: Wait for evict to complete before submitting swap-in
        3. Other policies: Submit both evict and swap-in immediately

        Args:
            evict_metadata: CacheSwapMetadata for device-to-host eviction (can be None)
            swap_in_metadata: CacheSwapMetadata for host-to-device swap-in (can be None)

        Returns:
            LayerDoneCounter for swap-in task, or None if no swap-in metadata provided.
        """
        # Step 1: Wait for existing pending evict counters before submitting new evict
        self._wait_for_pending_evict_counters()

        # Step 2: Submit evict task if provided
        # Note: evict returns LayerDoneCounter but we don't wait on it layer-by-layer
        # (except in write_back mode where we wait synchronously via wait_all)
        if evict_metadata is not None:
            evict_counter = self.evict_device_to_host(evict_metadata)
            self._pending_evict_counters.append(evict_counter)

            # Step 3: For write_back, wait for evict to complete before submitting swap-in
            if self._should_wait_for_swap_out():
                self._wait_for_pending_evict_counters()

        # Step 4: Submit swap-in task if provided
        # Returns LayerDoneCounter for tracking layer completion
        if swap_in_metadata is not None:
            self._layer_done_counter = self.load_host_to_device(swap_in_metadata)
            return self._layer_done_counter

        return None

    def _wait_for_pending_evict_counters(self) -> None:
        """
        Wait for all pending evict counters to complete.

        This is called before submitting new evict tasks to ensure proper ordering.
        Uses LayerDoneCounter.wait_all() for efficient waiting.
        """
        if not self._pending_evict_counters:
            return

        evict_wait_start = time.time()
        evict_length = len(self._pending_evict_counters)

        for counter in self._pending_evict_counters:
            counter.wait_all()

        self._pending_evict_counters.clear()
        evict_wait_ms = (time.time() - evict_wait_start) * 1000
        if evict_wait_ms > 0.1:
            logger.info(f"cache evict wait time: {evict_wait_ms:.2f}ms, {evict_length} pending evictions")

    # ============ Properties ============

    @property
    def transfer_manager(self) -> CacheTransferManager:
        """Get the transfer manager."""
        return self._transfer_manager

    @property
    def swap_layer_done_counter(self) -> Optional["LayerDoneCounter"]:
        """Get the layer done counter for layer swap."""
        return self._layer_done_counter

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

        # Quantized KV cache (int8/fp8/etc.) uses uint8 storage (1 byte per element).
        # Non-quantized cache uses the model's compute dtype (e.g., bfloat16).
        cache_dtype = "uint8" if kv_cache_quant_type is not None else self.model_config.dtype

        for i in range(self._num_layers):
            # Generate cache names
            cache_names = self._get_cache_names(i)

            logger.info(f"..creating kv cache for layer {i}: key:{key_cache_shape}, value:{value_cache_shape}")

            # Create key cache
            key_cache = paddle.full(shape=key_cache_shape, fill_value=0, dtype=cache_dtype)
            self.cache_kvs_map[cache_names["key"]] = key_cache

            if value_cache_shape:
                val_cache = paddle.full(shape=value_cache_shape, fill_value=0, dtype=cache_dtype)
                self.cache_kvs_map[cache_names["value"]] = val_cache
                cache_kvs_list.extend([key_cache, val_cache])
            else:
                cache_kvs_list.extend([key_cache])

            # Create scale caches for block_wise_fp8 quantization
            if self._is_fp8_quantization(kv_cache_quant_type) and kv_cache_scale_shape:
                key_cache_scales = paddle.full(
                    shape=kv_cache_scale_shape, fill_value=0, dtype=paddle.get_default_dtype()
                )
                self.cache_kvs_map[cache_names["key_scale"]] = key_cache_scales
                if value_cache_shape:
                    val_cache_scales = paddle.full(
                        shape=kv_cache_scale_shape, fill_value=0, dtype=paddle.get_default_dtype()
                    )
                    self.cache_kvs_map[cache_names["value_scale"]] = val_cache_scales
                    cache_kvs_list.extend([key_cache_scales, val_cache_scales])
                else:
                    cache_kvs_list.extend([key_cache_scales])

        paddle.device.cuda.empty_cache()
        logger.info("kv cache is initialized!")

        # Share cache_kvs_map with transfer manager for data transfer operations
        self._transfer_manager.set_cache_kvs_map(self.cache_kvs_map)

        # Initialize host cache
        self.initialize_host_cache(attn_backend)

        return cache_kvs_list

    def initialize_mtp_kv_cache(
        self,
        attn_backend: Any,
        num_gpu_blocks: int,
        num_mtp_layers: int,
        layer_offset: int,
    ) -> List[Any]:
        """
        Initialize MTP (speculative decode) KV Cache tensors.

        MTP cache layers use indices [layer_offset, layer_offset + num_mtp_layers),
        so they share the same cache_kvs_map namespace as the main model cache but
        with non-overlapping layer indices.  All subsequent transfer operations
        via CacheController automatically cover MTP layers as well because they
        live in the same cache_kvs_map.

        Args:
            attn_backend: MTP attention backend instance (proposer.attn_backends[0]).
            num_gpu_blocks: Number of GPU blocks for MTP (already expanded by ratio).
            num_mtp_layers: Number of MTP model layers (proposer.model_config.num_hidden_layers).
            layer_offset: Starting layer index, equals main model num_hidden_layers.

        Returns:
            cache_kvs_list: KV Cache tensor list in [key_layer0, val_layer0, ...] order.
        """
        kv_cache_quant_type = self._get_kv_cache_quant_type()

        key_cache_shape, value_cache_shape = attn_backend.get_kv_cache_shape(
            max_num_blocks=num_gpu_blocks, kv_cache_quant_type=kv_cache_quant_type
        )

        kv_cache_scale_shape = None
        if self._is_fp8_quantization(kv_cache_quant_type):
            kv_cache_scale_shape = [key_cache_shape[0], key_cache_shape[1], key_cache_shape[2]]

        logger.info(
            f"[CacheController] Initializing MTP kv cache for {num_mtp_layers} layers "
            f"(layer_offset={layer_offset}, num_gpu_blocks={num_gpu_blocks})."
        )
        cache_kvs_list = []

        # Quantized KV cache uses uint8 storage; non-quantized uses model compute dtype.
        cache_dtype = "uint8" if kv_cache_quant_type is not None else self.model_config.dtype

        for i in range(layer_offset, layer_offset + num_mtp_layers):
            cache_names = self._get_cache_names(i)

            key_cache = paddle.full(shape=key_cache_shape, fill_value=0, dtype=cache_dtype)
            self.cache_kvs_map[cache_names["key"]] = key_cache

            if value_cache_shape:
                val_cache = paddle.full(shape=value_cache_shape, fill_value=0, dtype=cache_dtype)
                self.cache_kvs_map[cache_names["value"]] = val_cache
                cache_kvs_list.extend([key_cache, val_cache])
            else:
                cache_kvs_list.extend([key_cache])

            if self._is_fp8_quantization(kv_cache_quant_type) and kv_cache_scale_shape:
                key_cache_scales = paddle.full(
                    shape=kv_cache_scale_shape, fill_value=0, dtype=paddle.get_default_dtype()
                )
                self.cache_kvs_map[cache_names["key_scale"]] = key_cache_scales
                if value_cache_shape:
                    val_cache_scales = paddle.full(
                        shape=kv_cache_scale_shape, fill_value=0, dtype=paddle.get_default_dtype()
                    )
                    self.cache_kvs_map[cache_names["value_scale"]] = val_cache_scales
                    cache_kvs_list.extend([key_cache_scales, val_cache_scales])
                else:
                    cache_kvs_list.extend([key_cache_scales])

        paddle.device.cuda.empty_cache()
        logger.info("[CacheController] MTP kv cache initialized!")

        # Refresh transfer manager so it sees the full map (main + MTP layers)
        self._transfer_manager.set_cache_kvs_map(self.cache_kvs_map)

        return cache_kvs_list

    def _get_numa_node_for_gpu(self, device_id: int) -> int:
        """
        Get the NUMA node closest to the specified GPU device.

        Tries multiple methods in order:
        1. nvidia-smi topo -C -i <gpu_id> (fastest and most reliable)
        2. /sys/class/nvidia-gpu/ (direct sysfs)
        3. /sys/bus/pci/devices/ (fallback)

        Args:
            device_id: CUDA device ID.

        Returns:
            NUMA node index, or -1 if cannot be determined.
        """
        try:
            # Method 1: Use nvidia-smi topo -C -i (fastest, SGLang-style)
            # This directly outputs the NUMA ID for the specific GPU
            try:
                import subprocess

                result = subprocess.run(
                    ["nvidia-smi", "topo", "-C", "-i", str(device_id)], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    output_line = result.stdout.strip()
                    prefix = "NUMA IDs of closest CPU:"
                    if output_line.startswith(prefix):
                        numa_str = output_line[len(prefix) :].strip()
                        # Handle comma-separated or range values (e.g., "0" or "0,1" or "0-1")
                        if numa_str:
                            # Take the first NUMA node if multiple are listed
                            first_numa = numa_str.split(",")[0].split("-")[0].strip()
                            if first_numa.isdigit():
                                return int(first_numa)
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                logger.debug(f"[CacheController] nvidia-smi topo -C method failed: {e}")

            # Method 2: Try to read from /sys filesystem
            sys_path = f"/sys/class/nvidia-gpu/nvidia{device_id}/device/numa_node"
            if os.path.exists(sys_path):
                with open(sys_path, "r") as f:
                    return int(f.read().strip())

            # Method 3: Fallback - check all NVIDIA PCI devices
            import glob

            numa_paths = glob.glob("/sys/bus/pci/devices/*/numa_node")
            for path in numa_paths:
                vendor_path = path.replace("numa_node", "vendor")
                if os.path.exists(vendor_path):
                    with open(vendor_path, "r") as f:
                        vendor = f.read().strip()
                        if vendor == "0x10de":  # NVIDIA vendor ID
                            with open(path, "r") as f:
                                return int(f.read().strip())

            return -1
        except Exception as e:
            logger.debug(f"[CacheController] Failed to get NUMA node for GPU {device_id}: {e}")
            return -1

    def _bind_to_closest_numa_node(self) -> bool:
        """
        Bind current thread and memory allocation to the NUMA node closest to the GPU.

        This should be called before allocating host memory to ensure the memory
        is allocated on the NUMA node local to the GPU, reducing cross-NUMA access
        latency during H2D transfers.

        Returns:
            True if binding was successful, False otherwise.
        """
        if self._numa_bound:
            return True

        try:
            # Load libnuma
            try:
                libnuma = ctypes.CDLL("libnuma.so.1")
            except OSError:
                try:
                    libnuma = ctypes.CDLL("libnuma.so")
                except OSError:
                    logger.warning("[CacheController] libnuma not found, NUMA binding skipped")
                    return False

            # Check if NUMA is available
            if libnuma.numa_available() < 0:
                logger.warning("[CacheController] NUMA is not available on this system")
                return False

            # Get NUMA node for current GPU
            numa_node = self._get_numa_node_for_gpu(self._device_id)

            if numa_node < 0:
                logger.warning(f"[CacheController] Could not determine NUMA node for GPU {self._device_id}")
                return False

            # Bind current thread to specific NUMA node
            # numa_run_on_node binds the current thread to run on the specified node
            result = libnuma.numa_run_on_node(numa_node)
            if result < 0:
                logger.warning(f"[CacheController] numa_run_on_node({numa_node}) failed")
                return False

            # Set memory allocation preference to the specified NUMA node
            # This affects subsequent memory allocations (including cudaHostAlloc)
            libnuma.numa_set_preferred(numa_node)

            self._numa_bound = True
            logger.info(
                f"[CacheController] NUMA binding successful: " f"GPU {self._device_id} bound to NUMA node {numa_node}"
            )
            return True

        except Exception as e:
            logger.warning(f"[CacheController] NUMA binding failed: {e}")
            return False

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

        # Step 0: Bind to closest NUMA node before allocating host memory
        # This ensures subsequent cuda_host_alloc allocations are on the local NUMA node
        if not self._numa_bound:
            self._bind_to_closest_numa_node()

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

        num_layers = self._num_layers + self.config.speculative_config.num_extra_cache_layer

        per_layer_size_gb = (key_need_to_allocate_bytes + value_need_to_allocate_bytes) / (1024**3)
        actual_alloc_gb = per_layer_size_gb * num_layers
        logger.info(
            f"[CacheController] Host swap space allocated: {actual_alloc_gb:.2f}GB "
            f"({per_layer_size_gb:.2f}GB per layer x {num_layers} layers), "
            f"num_host_blocks: {num_host_blocks}"
        )

        logger.info(f"[CacheController] Initializing swap space (Host cache) for {num_layers} layers.")

        # Allocate Host cache for each layer
        for i in range(num_layers):
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

        logger.info(f"[CacheController] Swap space (Host cache) is ready for {num_layers} layers!")

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
        src_location: CacheLevel,
        dst_location: CacheLevel,
        transfer_fn_all: callable,
        transfer_fn_layer: callable,
        force_all_layers: bool = False,
    ) -> LayerDoneCounter:
        """
        Submit a single swap transfer task (internal method).

        Creates a LayerDoneCounter for tracking layer completion.
        The counter is returned to the caller for later waiting.

        H2D (load) always uses layer-by-layer mode for compute-transfer overlap.
        D2H (evict) always uses all-layers mode via _output_stream (fire-and-forget).

        Args:
            meta: CacheSwapMetadata containing src_block_ids and dst_block_ids.
            src_location: Source cache level (CacheLevel.HOST or CacheLevel.DEVICE).
            dst_location: Destination cache level (CacheLevel.DEVICE or CacheLevel.HOST).
            transfer_fn_all: All-layer transfer function, signature (src_ids, dst_ids) -> bool.
            transfer_fn_layer: Layer-by-layer transfer function, signature (layer_indices, on_layer_complete, src_ids, dst_ids) -> bool.
            force_all_layers: If True, always use all-layers mode (used for D2H evict).

        Returns:
            LayerDoneCounter instance for tracking layer completion.
        """
        # Create LayerDoneCounter for this transfer (independent sync primitive)
        layer_counter = LayerDoneCounter(self._num_layers)

        src_block_ids = meta.src_block_ids
        dst_block_ids = meta.dst_block_ids

        if not src_block_ids or not dst_block_ids:
            logger.info(f"[SwapTask] skip: empty block_ids src={src_block_ids}, dst={dst_block_ids}")
            meta.success = False
            meta.error_message = "Empty block IDs in CacheSwapMetadata"
            return layer_counter

        layers_to_transfer = list(range(self._num_layers))

        def _on_layer_complete(layer_idx: int) -> None:
            """Callback called after each layer's H2D kernel is submitted to input_stream.

            Records a CUDA event on input_stream so that wait_for_layer() can
            synchronize on the actual transfer stream (cross-stream dependency).
            """
            # Record event on _input_stream so wait_for_layer() waits for the real H2D transfer.
            # Must use input_stream (not Paddle default stream) to capture the correct dependency.
            stream_event = self._transfer_manager.record_input_stream_event()
            if stream_event is not None:
                layer_counter.set_layer_event(layer_idx, stream_event)

            # Mark layer done (adds to _completed_layers, unblocks polling fallback)
            layer_counter.mark_layer_done(layer_idx)

        def _do_transfer():
            try:
                start_time = time.time()
                if force_all_layers:
                    success = transfer_fn_all(src_block_ids, dst_block_ids)
                    elapsed = time.time() - start_time
                    if success:
                        # For H2D transfers: record event on _input_stream so that
                        # wait_all() synchronizes on the actual transfer stream, not
                        # Paddle's default stream. set_layer_event must be called
                        # before mark_all_done() so wait_all()'s loop finds the event.
                        if dst_location == CacheLevel.DEVICE:
                            stream_event = self._transfer_manager.record_input_stream_event()
                            if stream_event is not None:
                                layer_counter.set_layer_event(self._num_layers - 1, stream_event)

                        # Mark all layers done at once
                        layer_counter.mark_all_done()

                    result = TransferResult(
                        src_block_ids=src_block_ids,
                        dst_block_ids=dst_block_ids,
                        src_type=src_location,
                        dst_type=dst_location,
                        success=success,
                        error_message=(
                            None if success else f"All-layer {src_location.value}→{dst_location.value} transfer failed"
                        ),
                    )
                    logger.debug(
                        f"[SwapTask] all_layers {src_location.value}->{dst_location.value} "
                        f"{'success' if success else 'FAILED'} "
                        f"src={src_block_ids} dst={dst_block_ids} elapsed={elapsed*1000:.3f}ms"
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
                            None
                            if success
                            else f"Layer-by-layer {src_location.value}→{dst_location.value} transfer failed"
                        ),
                    )
                    logger.debug(
                        f"[SwapTask] layer_by_layer {src_location.value}->{dst_location.value} "
                        f"{'success' if success else 'FAILED'} "
                        f"src={src_block_ids} dst={dst_block_ids} elapsed={elapsed*1000:.3f}ms"
                    )

                # Update metadata with result
                meta.success = result.success
                meta.error_message = result.error_message

            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(
                    f"[SwapTask] {src_location.value}->{dst_location.value} "
                    f"EXCEPTION: {e}\n{traceback.format_exc()}"
                )
                meta.success = False
                meta.error_message = str(e)
            finally:
                # Cleanup CUDA events when transfer is complete
                layer_counter.cleanup()

        self._executor.submit(_do_transfer)
        return layer_counter

    def load_host_to_device(
        self,
        swap_metadata: CacheSwapMetadata,
    ) -> LayerDoneCounter:
        """
        Load host cache to device (async).

        Creates an async transfer task and returns LayerDoneCounter
        for tracking layer completion.

        Args:
            swap_metadata: CacheSwapMetadata containing:
                - src_block_ids: Source host block IDs
                - dst_block_ids: Destination device block IDs

        Returns:
            LayerDoneCounter for tracking layer completion.
        """
        layer_counter = self._submit_swap_task(
            meta=swap_metadata,
            src_location=CacheLevel.HOST,
            dst_location=CacheLevel.DEVICE,
            transfer_fn_all=None,
            transfer_fn_layer=lambda layer_indices, on_layer_complete, src_ids, dst_ids: self._transfer_manager.load_layers_to_device_async(
                layer_indices=layer_indices,
                host_block_ids=src_ids,
                device_block_ids=dst_ids,
                on_layer_complete=on_layer_complete,
            ),
        )
        return layer_counter

    def evict_device_to_host(
        self,
        swap_metadata: CacheSwapMetadata,
    ) -> LayerDoneCounter:
        """
        Evict device cache to host (async).

        Creates an async transfer task and returns LayerDoneCounter
        for tracking layer completion.

        Args:
            swap_metadata: CacheSwapMetadata containing:
                - src_block_ids: Source device block IDs
                - dst_block_ids: Destination host block IDs

        Returns:
            LayerDoneCounter for tracking layer completion.
        """
        layer_counter = self._submit_swap_task(
            meta=swap_metadata,
            src_location=CacheLevel.DEVICE,
            dst_location=CacheLevel.HOST,
            transfer_fn_all=lambda src_ids, dst_ids: self._transfer_manager.evict_to_host_async(src_ids, dst_ids),
            transfer_fn_layer=None,
            force_all_layers=True,  # Eviction always uses output_stream for all-layers async transfer
        )
        return layer_counter

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

    # ============ Public Interface Implementation ============

    def reset_cache(self) -> bool:
        """
        Reset cache state (clear content only, do NOT free storage).

        This method only clears the transfer state:
        - Clears pending evict counters

        It does NOT free any storage (GPU memory, CPU pinned memory, or storage).
        Use free_cache() to release storage resources.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with self._lock:
                # Clear pending evict counters
                self._pending_evict_counters.clear()
            return True
        except Exception:
            return False

    def free_cache(self, clear_storage: bool = False) -> bool:
        """
        Free all cache storage (GPU memory + CPU pinned memory + storage).

        This releases all underlying storage resources, not just clears content.
        Use this when shutting down or wanting to fully release cache resources.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # First reset transfer state
            self.reset_cache()

            # Free GPU cache
            self.free_gpu_cache()

            # Free CPU cache (pinned memory)
            self._free_host_cache()

            # Clear storage
            if clear_storage:
                self._clear_storage()

            return True
        except Exception:
            return False

    def free_gpu_cache(self) -> None:
        """Free GPU cache tensors stored in cache_kvs_map."""
        if not hasattr(self, "cache_kvs_map") or not self.cache_kvs_map:
            return

        logger.info(f"[CacheController] Freeing GPU cache memory, {len(self.cache_kvs_map)} tensors.")
        self.cache_kvs_map.clear()
        paddle.device.cuda.empty_cache()
        logger.info("[CacheController] GPU cache memory released.")

    def _clear_storage(self) -> None:
        """Clear storage connector cache."""
        storage_connector = getattr(self._transfer_manager, "_storage_connector", None)
        if not storage_connector:
            return

        try:
            if hasattr(storage_connector, "clear") and callable(storage_connector.clear):
                count = storage_connector.clear()
                logger.info(f"[CacheController] Cleared {count} entries from storage.")
            elif hasattr(storage_connector, "disconnect") and callable(storage_connector.disconnect):
                storage_connector.disconnect()
                logger.info("[CacheController] Storage connector disconnected.")
        except Exception as e:
            logger.warning(f"[CacheController] Failed to clear storage: {e}")

    # ============ Statistics Methods ============

    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        with self._lock:
            return {
                "initialized": self._initialized,
                "num_layers": self._num_layers,
                "pending_evict_counters": len(self._pending_evict_counters),
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

    def __del__(self) -> None:
        """Destructor to release pinned host memory."""
        try:
            self._free_host_cache()
        except Exception:
            pass

    def _free_host_cache(self) -> None:
        """Free pinned host memory allocated for swap space."""
        if not hasattr(self, "host_cache_kvs_map"):
            return

        if not self.host_cache_kvs_map:
            return

        logger.info(f"[CacheController] Freeing host cache memory, {len(self.host_cache_kvs_map)} tensors.")
        for name, ptr in list(self.host_cache_kvs_map.items()):
            if ptr != 0:
                try:
                    cuda_host_free(ptr)
                except Exception as e:
                    logger.warning(f"[CacheController] Failed to free host cache {name}: {e}")
        self.host_cache_kvs_map.clear()
        logger.info("[CacheController] Host cache memory released.")
