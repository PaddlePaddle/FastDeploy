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

import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import paddle
from paddleformers.utils.log import logger

# Import cupy for independent CUDA stream management
try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False
    logger.warning("cupy not available, falling back to synchronous transfers")

# Import ops for cache swap
from fastdeploy.cache_manager.ops import (
    swap_cache_per_layer,  # sync fallback (used when cupy not available)
)
from fastdeploy.cache_manager.ops import (
    swap_cache_per_layer_async,  # async per-layer op (no cudaStreamSynchronize)
)
from fastdeploy.cache_manager.ops import swap_cache_all_layers
from fastdeploy.cache_manager.v1.cache_utils import storage_key_for_block
from fastdeploy.cache_manager.v1.storage import create_storage_connector
from fastdeploy.cache_manager.v1.transfer import create_transfer_connector

if TYPE_CHECKING:
    from fastdeploy.config import FDConfig


class CacheTransferManager:
    """
    KV Cache Transfer Manager.

    H2D (load): layer-by-layer on _input_stream, overlaps with forward compute.
    D2H (evict): all-layers on _output_stream, fire-and-forget.

    Data organization:
    1. Name-indexed storage (_cache_kvs_map, _host_cache_kvs_map): for building layer indices
    2. Layer-indexed storage (_device_key_caches, etc.): passed to swap operators

    Attributes:
        config: FDConfig instance.
    """

    def __init__(
        self,
        config: "FDConfig",
        local_rank: int = 0,
        device_id: int = 0,
    ):
        """
        Initialize the transfer manager.

        Args:
            config: FDConfig instance.
            local_rank: Local rank for tensor parallel.
            device_id: Device ID.
        """
        self.config = config
        self.cache_config = config.cache_config
        self.quant_config = config.quant_config

        self._local_rank = local_rank
        self._device_id = device_id
        self._num_layers = config.model_config.num_hidden_layers
        self._cache_dtype = config.cache_config.cache_dtype
        self._num_host_blocks = self.cache_config.num_cpu_blocks or 0

        self._lock = threading.RLock()

        # ============ Async Transfer Streams (cupy-based) ============
        # Two independent CUDA streams for fully async transfer
        # _input_stream: H2D transfer (load to device, layer-by-layer)
        # _output_stream: D2H transfer (evict to host, all-layers)
        # They run in parallel without waiting for each other
        # Using cupy to avoid affecting Paddle's internal stream state
        if _HAS_CUPY and paddle.is_compiled_with_cuda():
            self._cupy_device_id = cp.cuda.runtime.getDevice()
            logger.info(
                f"[TransferManager] Creating streams: local_rank={self._local_rank}, device_id={self._device_id}, "
                f"cupy_device_id={self._cupy_device_id}"
            )
            with cp.cuda.Device(self._cupy_device_id):
                self._input_stream = cp.cuda.Stream(non_blocking=False)
                self._output_stream = cp.cuda.Stream(non_blocking=False)
            logger.info(
                f"[TransferManager] Using cupy streams: input={id(self._input_stream)}, output={id(self._output_stream)}"
            )
        else:
            self._input_stream = None
            self._output_stream = None
            logger.warning("[TransferManager] cupy not available, async transfers disabled")

        # ============ KV Cache Data Storage ============
        # Name-indexed storage (used to build layer-indexed structures below)
        self._cache_kvs_map: Dict[str, Any] = {}
        self._host_cache_kvs_map: Dict[str, Any] = {}

        # Layer-indexed lists (for all-layer transfers, compatible with swap_cache_all_layers operator)
        # Device cache tensors per layer (GPU)
        self._device_key_caches: List[Any] = []  # key cache per layer
        self._device_value_caches: List[Any] = []  # value cache per layer
        self._device_key_scales: List[Any] = []  # key scales (fp8)
        self._device_value_scales: List[Any] = []  # value scales (fp8)

        # Host cache pointers per layer (CPU pinned memory)
        self._host_key_ptrs: List[int] = []  # key host pointers
        self._host_value_ptrs: List[int] = []  # value host pointers
        self._host_key_scales_ptrs: List[int] = []  # key scale pointers (fp8)
        self._host_value_scales_ptrs: List[int] = []  # value scale pointers (fp8)

        # ============ Connectors (for future use) ============
        # connect() is deferred to set_host_block_shape() so that cpu_cache_size
        # can be computed from the actual block shape before connecting.
        self._storage_connector = create_storage_connector(
            self.cache_config,
            tp_rank=self._local_rank,
        )
        self._transfer_connector = create_transfer_connector(self.cache_config)

        # ============ Host block stride (bytes per block per layer) ============
        # Set by set_host_block_shape() after host cache is allocated.
        self._host_key_block_stride_bytes: int = 0
        self._host_value_block_stride_bytes: int = 0
        self._host_scale_block_stride_bytes: int = 0

    # ============ Cache Map Setters ============

    @property
    def cache_kvs_map(self) -> Dict[str, Any]:
        return self._cache_kvs_map

    def set_cache_kvs_map(self, cache_kvs_map: Dict[str, Any]) -> None:
        """
        Share the KV cache tensor map from CacheController.

        Args:
            cache_kvs_map: Dictionary mapping cache names to tensors.
                Format: {
                    "key_caches_{layer_id}_rank{rank}.device{device}": paddle.Tensor,
                    "value_caches_{layer_id}_rank{rank}.device{device}": paddle.Tensor,
                    "key_cache_scales_{layer_id}_rank{rank}.device{device}": paddle.Tensor,  # fp8
                    "value_cache_scales_{layer_id}_rank{rank}.device{device}": paddle.Tensor, # fp8
                    ...
                }
        """
        with self._lock:
            self._cache_kvs_map = cache_kvs_map
            self._build_device_layer_indices()

    def _build_device_layer_indices(self) -> None:
        """Build layer-indexed Device cache lists from _cache_kvs_map."""
        if not self._cache_kvs_map:
            return

        self._device_key_caches = []
        self._device_value_caches = []
        self._device_key_scales = []
        self._device_value_scales = []

        for layer_idx in range(self._num_layers):
            key_name = f"key_caches_{layer_idx}_rank{self._local_rank}.device{self._device_id}"
            val_name = f"value_caches_{layer_idx}_rank{self._local_rank}.device{self._device_id}"
            key_scale_name = f"key_cache_scales_{layer_idx}_rank{self._local_rank}.device{self._device_id}"
            val_scale_name = f"value_cache_scales_{layer_idx}_rank{self._local_rank}.device{self._device_id}"

            self._device_key_caches.append(self._cache_kvs_map.get(key_name))
            self._device_value_caches.append(self._cache_kvs_map.get(val_name))

            if self._is_fp8_quantization():
                self._device_key_scales.append(self._cache_kvs_map.get(key_scale_name))
                self._device_value_scales.append(self._cache_kvs_map.get(val_scale_name))

    @property
    def host_cache_kvs_map(self) -> Dict[str, Any]:
        return self._host_cache_kvs_map

    def set_host_cache_kvs_map(self, host_cache_kvs_map: Dict[str, Any]) -> None:
        """
        Share the Host KV cache tensor map from CacheController.

        Args:
            host_cache_kvs_map: Dictionary mapping cache names to Host pointers (int).
                Format: {
                    "key_caches_{layer_id}_rank{rank}.device{device}": pointer (int),
                    ...
                }
        """
        with self._lock:
            self._host_cache_kvs_map = host_cache_kvs_map
            self._build_host_layer_indices()
            self._register_host_buffers()

    def _register_host_buffers(self) -> None:
        """Register all per-layer host buffers with the storage connector for zero-copy RDMA."""
        if self._storage_connector is None:
            return
        if self._num_host_blocks <= 0 or not self._host_key_ptrs:
            return
        if self._host_key_block_stride_bytes <= 0:
            return

        layer_total_bytes = self._num_host_blocks * self._host_key_block_stride_bytes
        for layer_idx in range(len(self._host_key_ptrs)):
            key_ptr = self._host_key_ptrs[layer_idx]
            if key_ptr:
                try:
                    self._storage_connector.register_buffer(key_ptr, layer_total_bytes)
                except Exception as e:
                    logger.warning(f"[TransferManager] register_buffer key layer {layer_idx} failed: {e}")
            if self._is_fp8_quantization():
                val_ptr = self._host_value_ptrs[layer_idx] if layer_idx < len(self._host_value_ptrs) else 0
            else:
                val_ptr = self._host_value_ptrs[layer_idx] if layer_idx < len(self._host_value_ptrs) else 0
            if val_ptr:
                val_total = self._num_host_blocks * self._host_value_block_stride_bytes
                try:
                    self._storage_connector.register_buffer(val_ptr, val_total)
                except Exception as e:
                    logger.warning(f"[TransferManager] register_buffer value layer {layer_idx} failed: {e}")

    def _build_host_layer_indices(self) -> None:
        """Build layer-indexed Host pointer lists from _host_cache_kvs_map."""
        if self._num_host_blocks <= 0:
            return
        if not self._host_cache_kvs_map:
            return
        if self._num_layers == 0:
            return

        self._host_key_ptrs = []
        self._host_value_ptrs = []
        self._host_key_scales_ptrs = []
        self._host_value_scales_ptrs = []

        for layer_idx in range(self._num_layers):
            key_name = f"key_caches_{layer_idx}_rank{self._local_rank}.device{self._device_id}"
            val_name = f"value_caches_{layer_idx}_rank{self._local_rank}.device{self._device_id}"
            key_scale_name = f"key_cache_scales_{layer_idx}_rank{self._local_rank}.device{self._device_id}"
            val_scale_name = f"value_cache_scales_{layer_idx}_rank{self._local_rank}.device{self._device_id}"

            self._host_key_ptrs.append(self._host_cache_kvs_map.get(key_name, 0))
            self._host_value_ptrs.append(self._host_cache_kvs_map.get(val_name, 0))

            if self._is_fp8_quantization():
                self._host_key_scales_ptrs.append(self._host_cache_kvs_map.get(key_scale_name, 0))
                self._host_value_scales_ptrs.append(self._host_cache_kvs_map.get(val_scale_name, 0))

    # ============ Host Block Shape ============

    def set_host_block_shape(
        self,
        key_shape: List[int],
        value_shape: Optional[List[int]],
        scale_shape: Optional[List[int]],
        cache_item_bytes: int,
        scale_item_bytes: int = 4,
    ) -> None:
        """
        Set per-layer host block shape for stride calculation.

        Must be called after host cache is allocated (initialize_swap_space)
        so that prefetch_from_storage / backup_to_storage can compute the
        correct byte offset for each block_id.

        Args:
            key_shape:   [num_host_blocks, dim1, dim2, dim3] per-layer key cache shape.
            value_shape: [num_host_blocks, dim1, dim2, dim3] per-layer value cache shape, or None.
            scale_shape: [num_host_blocks, dim1, dim2] per-layer scale shape (fp8), or None.
            cache_item_bytes: Bytes per cache element (e.g. 2 for float16).
            scale_item_bytes: Bytes per scale element (default 4 for float32).
        """
        with self._lock:
            # stride = elements per block per layer * bytes per element
            # key_shape = [num_blocks, d1, d2, d3]  → per-block stride = d1*d2*d3 * bytes
            self._host_key_block_stride_bytes = (
                int(key_shape[1]) * int(key_shape[2]) * int(key_shape[3]) * cache_item_bytes
            )
            if value_shape:
                self._host_value_block_stride_bytes = (
                    int(value_shape[1]) * int(value_shape[2]) * int(value_shape[3]) * cache_item_bytes
                )
            else:
                self._host_value_block_stride_bytes = self._host_key_block_stride_bytes
            if scale_shape:
                self._host_scale_block_stride_bytes = int(scale_shape[1]) * int(scale_shape[2]) * scale_item_bytes
            else:
                self._host_scale_block_stride_bytes = 0

            # Connect storage connector now that block strides are known.
            # cpu_cache_size = total pinned CPU memory across all layers
            # (key + value, plus fp8 scales when present).
            if self._storage_connector is not None and not self._storage_connector.is_connected():
                cpu_cache_size = (
                    self._num_host_blocks
                    * self._num_layers
                    * (self._host_key_block_stride_bytes + self._host_value_block_stride_bytes)
                )
                if self._is_fp8_quantization() and self._host_scale_block_stride_bytes > 0:
                    cpu_cache_size += (
                        self._num_host_blocks
                        * self._num_layers
                        * self._host_scale_block_stride_bytes
                        * 2  # key scale + value scale
                    )

                self._storage_connector._cpu_cache_size = cpu_cache_size
                logger.info(
                    f"[TransferManager] Connecting storage connector: "
                    f"tp_rank={self._local_rank}, cpu_cache_size={cpu_cache_size / 1024**3:.3f} GB"
                )
                self._storage_connector.connect()
                # connect() completes RDMA initialization; now all three conditions for
                # _register_host_buffers are satisfied (_host_key_ptrs set, strides > 0,
                # connector connected), so register host pinned memory as RDMA MR.
                self._register_host_buffers()

    # ============ Metadata Properties ============

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

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def cache_dtype(self) -> str:
        return self._cache_dtype

    @property
    def has_cache_scale(self) -> bool:
        """Check if cache has scale tensors (fp8)."""
        return self._is_fp8_quantization()

    @property
    def num_host_blocks(self) -> int:
        return self._num_host_blocks

    # ============ Layer Indexed Access ============

    def get_device_key_cache(self, layer_idx: int) -> Optional[Any]:
        """Get Device key cache tensor for a specific layer."""
        if 0 <= layer_idx < len(self._device_key_caches):
            return self._device_key_caches[layer_idx]
        return None

    def get_device_value_cache(self, layer_idx: int) -> Optional[Any]:
        """Get Device value cache tensor for a specific layer."""
        if 0 <= layer_idx < len(self._device_value_caches):
            return self._device_value_caches[layer_idx]
        return None

    def get_host_key_ptr(self, layer_idx: int) -> int:
        """Get Host key cache pointer for a specific layer."""
        if self._num_host_blocks <= 0:
            return 0
        if 0 <= layer_idx < len(self._host_key_ptrs):
            return self._host_key_ptrs[layer_idx]
        return 0

    def get_host_value_ptr(self, layer_idx: int) -> int:
        """Get Host value cache pointer for a specific layer."""
        if self._num_host_blocks <= 0:
            return 0
        if 0 <= layer_idx < len(self._host_value_ptrs):
            return self._host_value_ptrs[layer_idx]
        return 0

    # ============ Internal Sync Fallbacks (used when cupy not available) ============

    def _swap_all_layers(
        self,
        device_block_ids: List[int],
        host_block_ids: List[int],
        mode: int,
    ) -> bool:
        """
        Synchronous all-layer transfer fallback (used when cupy streams unavailable).

        Args:
            device_block_ids: Device block IDs to swap.
            host_block_ids: Host block IDs to swap.
            mode: 0=Device→Host (evict), 1=Host→Device (load).
        """
        if self._num_host_blocks <= 0:
            return False

        try:
            swap_cache_all_layers(
                self._device_key_caches,
                self._host_key_ptrs,
                self._num_host_blocks,
                device_block_ids,
                host_block_ids,
                self._device_id,
                mode,
            )
            swap_cache_all_layers(
                self._device_value_caches,
                self._host_value_ptrs,
                self._num_host_blocks,
                device_block_ids,
                host_block_ids,
                self._device_id,
                mode,
            )
            if self._is_fp8_quantization() and self._device_key_scales and self._host_key_scales_ptrs:
                swap_cache_all_layers(
                    self._device_key_scales,
                    self._host_key_scales_ptrs,
                    self._num_host_blocks,
                    device_block_ids,
                    host_block_ids,
                    self._device_id,
                    mode,
                )
                swap_cache_all_layers(
                    self._device_value_scales,
                    self._host_value_scales_ptrs,
                    self._num_host_blocks,
                    device_block_ids,
                    host_block_ids,
                    self._device_id,
                    mode,
                )
            return True
        except Exception:
            import traceback

            traceback.print_exc()
            return False

    def _swap_single_layer(
        self,
        layer_idx: int,
        device_block_ids: List[int],
        host_block_ids: List[int],
        mode: int,
    ) -> bool:
        """
        Synchronous single-layer transfer fallback (used when cupy streams unavailable).

        Args:
            layer_idx: Layer index to transfer.
            device_block_ids: Device block IDs to swap.
            host_block_ids: Host block IDs to swap.
            mode: 0=Device→Host (evict), 1=Host→Device (load).
        """
        if self._num_host_blocks <= 0:
            return False
        if not device_block_ids or not host_block_ids:
            return False
        if len(device_block_ids) != len(host_block_ids):
            return False

        try:
            key_cache = self.get_device_key_cache(layer_idx)
            value_cache = self.get_device_value_cache(layer_idx)
            if key_cache is None or value_cache is None:
                return False

            key_ptr = self.get_host_key_ptr(layer_idx)
            value_ptr = self.get_host_value_ptr(layer_idx)
            if key_ptr == 0 or value_ptr == 0:
                return False

            swap_cache_per_layer(
                key_cache,
                key_ptr,
                self._num_host_blocks,
                device_block_ids,
                host_block_ids,
                self._device_id,
                mode,
            )
            swap_cache_per_layer(
                value_cache,
                value_ptr,
                self._num_host_blocks,
                device_block_ids,
                host_block_ids,
                self._device_id,
                mode,
            )
            return True
        except Exception:
            import traceback

            traceback.print_exc()
            return False

    # ============ Async Transfer Methods ============

    def _swap_all_layers_async(
        self,
        device_block_ids: List[int],
        host_block_ids: List[int],
        mode: int,
    ) -> bool:
        """
        Async all-layer transfer on dedicated stream.

        D2H uses _output_stream (fire-and-forget).
        H2D uses _input_stream (but H2D always goes through _swap_single_layer_async).
        Falls back to _swap_all_layers if cupy not available.

        Args:
            device_block_ids: Device block IDs to swap.
            host_block_ids: Host block IDs to swap.
            mode: 0=Device→Host (evict), 1=Host→Device (load).
        """
        if self._num_host_blocks <= 0:
            return False

        if self._input_stream is None or self._output_stream is None:
            return self._swap_all_layers(device_block_ids, host_block_ids, mode)

        stream = self._output_stream if mode == 0 else self._input_stream
        try:
            logger.debug(
                f"[TransferManager] _swap_all_layers_async: local_rank={self._local_rank}, device_id={self._device_id}, "
                f"cupy_device_id={self._cupy_device_id}, stream_device={stream.device_id}, mode={mode}"
            )
            with cp.cuda.Device(self._cupy_device_id):
                with stream:
                    swap_cache_all_layers(
                        self._device_key_caches,
                        self._host_key_ptrs,
                        self._num_host_blocks,
                        device_block_ids,
                        host_block_ids,
                        self._device_id,
                        mode,
                    )
                    swap_cache_all_layers(
                        self._device_value_caches,
                        self._host_value_ptrs,
                        self._num_host_blocks,
                        device_block_ids,
                        host_block_ids,
                        self._device_id,
                        mode,
                    )
                    if self._is_fp8_quantization() and self._device_key_scales and self._host_key_scales_ptrs:
                        swap_cache_all_layers(
                            self._device_key_scales,
                            self._host_key_scales_ptrs,
                            self._num_host_blocks,
                            device_block_ids,
                            host_block_ids,
                            self._device_id,
                            mode,
                        )
                        swap_cache_all_layers(
                            self._device_value_scales,
                            self._host_value_scales_ptrs,
                            self._num_host_blocks,
                            device_block_ids,
                            host_block_ids,
                            self._device_id,
                            mode,
                        )
            return True
        except Exception:
            import traceback

            traceback.print_exc()
            return False

    def _swap_single_layer_async(
        self,
        layer_idx: int,
        device_block_ids: List[int],
        host_block_ids: List[int],
        mode: int,
    ) -> bool:
        """
        Async single-layer transfer on _input_stream (H2D) or _output_stream (D2H).

        Falls back to _swap_single_layer if cupy not available.

        Args:
            layer_idx: Layer index to transfer.
            device_block_ids: Device block IDs to swap.
            host_block_ids: Host block IDs to swap.
            mode: 0=Device→Host (evict), 1=Host→Device (load).
        """
        if self._num_host_blocks <= 0:
            return False

        if self._input_stream is None or self._output_stream is None:
            return self._swap_single_layer(layer_idx, device_block_ids, host_block_ids, mode)

        stream = self._output_stream if mode == 0 else self._input_stream
        key_cache = self.get_device_key_cache(layer_idx)
        value_cache = self.get_device_value_cache(layer_idx)
        if key_cache is None or value_cache is None:
            return False

        key_ptr = self.get_host_key_ptr(layer_idx)
        value_ptr = self.get_host_value_ptr(layer_idx)
        if key_ptr == 0 or value_ptr == 0:
            return False

        try:
            with cp.cuda.Device(self._cupy_device_id):
                with stream:
                    swap_cache_per_layer_async(
                        key_cache,
                        key_ptr,
                        self._num_host_blocks,
                        device_block_ids,
                        host_block_ids,
                        self._device_id,
                        mode,
                    )
                    swap_cache_per_layer_async(
                        value_cache,
                        value_ptr,
                        self._num_host_blocks,
                        device_block_ids,
                        host_block_ids,
                        self._device_id,
                        mode,
                    )
            return True
        except Exception:
            import traceback

            traceback.print_exc()
            return False

    # ============ Public Async API ============

    def evict_to_host_async(
        self,
        device_block_ids: List[int],
        host_block_ids: List[int],
    ) -> bool:
        """
        Async evict all layers of KV Cache from Device to Host (D2H).

        Runs on _output_stream, fire-and-forget.

        Args:
            device_block_ids: Device block IDs to evict.
            host_block_ids: Host block IDs to receive.
        """
        return self._swap_all_layers_async(device_block_ids, host_block_ids, mode=0)

    def load_layers_to_device_async(
        self,
        layer_indices: List[int],
        host_block_ids: List[int],
        device_block_ids: List[int],
        on_layer_complete: Optional[callable] = None,
    ) -> bool:
        """
        Async load KV Cache from Host to Device layer-by-layer (H2D).

        Each layer runs on _input_stream. Overlaps with forward compute:
        the callback is invoked after each layer's kernel is submitted so
        the forward thread can start using that layer's data once the event fires.

        Args:
            layer_indices: Layer indices to load.
            host_block_ids: Host block IDs to load from.
            device_block_ids: Device block IDs to receive.
            on_layer_complete: Optional callback(layer_idx) after each layer is submitted.
        """
        if self._num_host_blocks <= 0:
            return False

        all_success = True
        for layer_idx in layer_indices:
            success = self._swap_single_layer_async(layer_idx, device_block_ids, host_block_ids, mode=1)
            if not success:
                all_success = False
            if on_layer_complete is not None:
                try:
                    on_layer_complete(layer_idx)
                except Exception:
                    pass
        return all_success

    # ============ Stream Utilities ============

    def sync_input_stream(self):
        """Wait for all pending _input_stream (H2D) transfers to complete."""
        if self._input_stream is not None:
            self._input_stream.synchronize()

    def sync_output_stream(self):
        """Wait for all pending _output_stream (D2H) transfers to complete."""
        if self._output_stream is not None:
            self._output_stream.synchronize()

    def record_input_stream_event(self) -> Any:
        """
        Record a CUDA event on _input_stream and return it.

        Used by _on_layer_complete callback in CacheController so that
        LayerDoneCounter.wait_for_layer() can synchronize on the actual
        H2D transfer stream rather than Paddle's default stream.

        Returns:
            cupy.cuda.Event if cupy streams are available, else None.
        """
        if not _HAS_CUPY or self._input_stream is None:
            return None
        try:
            with cp.cuda.Device(self._cupy_device_id):
                event = cp.cuda.Event()
                with self._input_stream:
                    event.record()
            return event
        except Exception as e:
            logger.warning(f"[TransferManager] Failed to record input_stream event: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get transfer manager statistics."""
        return {
            "num_layers": self._num_layers,
            "local_rank": self._local_rank,
            "device_id": self._device_id,
            "cache_dtype": self._cache_dtype,
            "num_host_blocks": self._num_host_blocks,
            "has_device_cache": len(self._device_key_caches) > 0,
            "has_host_cache": len(self._host_key_ptrs) > 0,
            "is_fp8": self._is_fp8_quantization(),
        }

    # ============ Storage Transfer API ============
    #
    # Key format (per-layer):
    #   K cache: "{hash_value}_{local_rank}_key_{layer_idx}"
    #   V cache: "{hash_value}_{local_rank}_value_{layer_idx}"
    #   K scale: "{hash_value}_{local_rank}_key_scale_{layer_idx}"  (fp8 only)
    #   V scale: "{hash_value}_{local_rank}_value_scale_{layer_idx}" (fp8 only)
    #
    # Each layer's data is stored independently with its own key, allowing
    # direct zero-copy RDMA between per-layer host memory and remote storage
    # via connector.batch_get / batch_set (backed by batch_get_into / batch_put_from).

    def _compute_layer_block_ptr(self, kind: str, layer_idx: int, cpu_block_id: int) -> int:
        """Compute raw pointer for a specific layer+block of a given kind."""
        if kind == "key":
            return self._host_key_ptrs[layer_idx] + cpu_block_id * self._host_key_block_stride_bytes
        elif kind == "value":
            return self._host_value_ptrs[layer_idx] + cpu_block_id * self._host_value_block_stride_bytes
        elif kind == "key_scale":
            return self._host_key_scales_ptrs[layer_idx] + cpu_block_id * self._host_scale_block_stride_bytes
        elif kind == "value_scale":
            return self._host_value_scales_ptrs[layer_idx] + cpu_block_id * self._host_scale_block_stride_bytes
        return 0

    def _get_layer_block_size(self, kind: str) -> int:
        """Return byte size for a single layer's block of a given kind."""
        if kind == "key":
            return self._host_key_block_stride_bytes
        elif kind == "value":
            return self._host_value_block_stride_bytes
        elif kind in ("key_scale", "value_scale"):
            return self._host_scale_block_stride_bytes
        return 0

    def _build_per_layer_io_args(
        self,
        hash_list: List[str],
        cpu_block_list: List[int],
    ) -> tuple:
        """Build flat per-layer keys, pointers, and sizes for direct connector calls.

        Returns:
            (flat_keys, flat_ptrs, flat_sizes, flat_block_indices, flat_kinds)
            All lists have length = len(hash_list) * num_layers * num_kinds.
            flat_block_indices maps flat_idx -> position in hash_list/cpu_block_list.
            flat_kinds maps flat_idx -> kind string (for rollback).
        """
        is_fp8 = self._is_fp8_quantization()
        kinds = ["key", "value"]
        if is_fp8 and self._host_scale_block_stride_bytes > 0:
            kinds.extend(["key_scale", "value_scale"])

        flat_keys: List[str] = []
        flat_ptrs: List[int] = []
        flat_sizes: List[int] = []
        flat_block_indices: List[int] = []
        flat_kinds: List[str] = []

        for bi, (hash_val, cpu_block_id) in enumerate(zip(hash_list, cpu_block_list)):
            for layer_idx in range(self._num_layers):
                for kind in kinds:
                    flat_keys.append(storage_key_for_block(hash_val, self._local_rank, kind, layer_idx))
                    flat_ptrs.append(self._compute_layer_block_ptr(kind, layer_idx, cpu_block_id))
                    flat_sizes.append(self._get_layer_block_size(kind))
                    flat_block_indices.append(bi)
                    flat_kinds.append(kind)

        return flat_keys, flat_ptrs, flat_sizes, flat_block_indices, flat_kinds

    def prefetch_from_storage(
        self,
        hash_list: List[str],
        cpu_block_list: List[int],
    ) -> List[bool]:
        """
        Batch-prefetch KV cache blocks from remote storage into CPU host memory.

        Uses per-layer storage keys.  Each layer's data is read directly from
        storage into per-layer host buffers via zero-copy RDMA (batch_get_into).

        Storage key per block, per layer:
            ``"{hash}_{rank}_key_{layer_idx}"`` / ``"{hash}_{rank}_value_{layer_idx}"``

        Args:
            hash_list:      List of block hash values (one per block).
            cpu_block_list: List of target CPU block IDs (same length as hash_list).

        Returns:
            List[bool]: True for each block that was fully retrieved successfully.
        """
        if self._storage_connector is None or not self._storage_connector.is_connected():
            logger.warning("[TransferManager] prefetch_from_storage: connector not ready")
            return [False] * len(hash_list)

        if len(hash_list) != len(cpu_block_list):
            raise ValueError("hash_list and cpu_block_list must have the same length")

        if not hash_list:
            return []

        if not self._host_key_ptrs or self._host_key_block_stride_bytes <= 0:
            logger.warning(
                "[TransferManager] prefetch_from_storage: host cache not ready "
                "(call set_host_block_shape after initialize_swap_space)"
            )
            return [False] * len(hash_list)

        flat_keys, flat_ptrs, flat_sizes, flat_block_indices, flat_kinds = self._build_per_layer_io_args(
            hash_list, cpu_block_list
        )
        results = self._storage_connector.batch_get(flat_keys, flat_ptrs, flat_sizes)

        num_blocks = len(hash_list)
        block_success = [True] * num_blocks
        for flat_idx, ok in enumerate(results):
            if not ok:
                block_success[flat_block_indices[flat_idx]] = False

        # Diagnostic: probe which per-layer keys are missing for failed blocks
        failed_indices = [i for i, ok in enumerate(block_success) if not ok]
        if failed_indices and self._storage_connector is not None:
            probe_keys = []
            probe_labels = []
            for i in failed_indices:
                for layer_idx in range(self._num_layers):
                    for kind in ["key", "value"]:
                        probe_keys.append(storage_key_for_block(hash_list[i], self._local_rank, kind, layer_idx))
                        probe_labels.append((i, cpu_block_list[i], hash_list[i], layer_idx, kind))

            try:
                exist_flags = self._storage_connector.batch_exists(probe_keys)
                block_diag: Dict[int, Dict] = {}
                for (bi, cpu_bid, h, layer_idx, kind), ok in zip(probe_labels, exist_flags):
                    if bi not in block_diag:
                        block_diag[bi] = {"cpu_bid": cpu_bid, "hash": h, "missing": [], "existing": []}
                    label = f"{kind}_l{layer_idx}"
                    if ok:
                        block_diag[bi]["existing"].append(label)
                    else:
                        block_diag[bi]["missing"].append(label)

                partial_missing = {bi: v for bi, v in block_diag.items() if v["missing"]}
                pure_transfer_err = {bi: v for bi, v in block_diag.items() if not v["missing"]}

                if partial_missing:
                    detail = [
                        f"cpu_block={v['cpu_bid']} hash={v['hash'][:16]}.. "
                        f"missing={v['missing'][:4]}{'...' if len(v['missing']) > 4 else ''}"
                        for v in partial_missing.values()
                    ]
                    logger.warning(
                        f"[TransferManager] prefetch_from_storage: {len(partial_missing)} block(s) have missing per-layer keys — "
                        + "; ".join(detail)
                    )
                if pure_transfer_err:
                    detail = [f"cpu_block={v['cpu_bid']} hash={v['hash'][:16]}.." for v in pure_transfer_err.values()]
                    logger.warning(
                        f"[TransferManager] prefetch_from_storage: {len(pure_transfer_err)} block(s) keys exist but transfer failed — "
                        + ", ".join(detail)
                    )
            except Exception as e:
                logger.warning(f"[TransferManager] prefetch_from_storage: failed to probe missing keys: {e}")

        return block_success

    def backup_to_storage(
        self,
        cpu_block_list: List[int],
        hash_list: List[str],
    ) -> List[bool]:
        """
        Batch-backup KV cache blocks from CPU host memory to remote storage.

        Uses per-layer storage keys.  Each layer's data is written directly
        from per-layer host buffers to storage via zero-copy RDMA (batch_put_from).

        Storage key per block, per layer:
            ``"{hash}_{rank}_key_{layer_idx}"`` / ``"{hash}_{rank}_value_{layer_idx}"``

        Args:
            cpu_block_list: List of source CPU block IDs.
            hash_list:      List of block hash values (same length as cpu_block_list).

        Returns:
            List[bool]: True for each block that was fully stored successfully.
        """
        if self._storage_connector is None or not self._storage_connector.is_connected():
            logger.warning("[TransferManager] backup_to_storage: connector not ready")
            return [False] * len(cpu_block_list)

        if len(cpu_block_list) != len(hash_list):
            raise ValueError("cpu_block_list and hash_list must have the same length")

        if not cpu_block_list:
            return []

        if not self._host_key_ptrs or self._host_key_block_stride_bytes <= 0:
            logger.warning(
                "[TransferManager] backup_to_storage: host cache not ready "
                "(call set_host_block_shape after initialize_swap_space)"
            )
            return [False] * len(cpu_block_list)

        flat_keys, flat_ptrs, flat_sizes, flat_block_indices, flat_kinds = self._build_per_layer_io_args(
            hash_list, cpu_block_list
        )
        results = self._storage_connector.batch_set(flat_keys, flat_ptrs, flat_sizes)

        num_blocks = len(cpu_block_list)
        block_success = [True] * num_blocks
        block_written_keys: Dict[int, List[str]] = {}

        for flat_idx, ok in enumerate(results):
            bi = flat_block_indices[flat_idx]
            if ok:
                block_written_keys.setdefault(bi, []).append(flat_keys[flat_idx])
            else:
                block_success[bi] = False

        # Rollback: delete successfully-written per-layer keys of partially-failed blocks.
        # If a block failed but some of its layer-kind keys were written, delete those
        # keys so the block appears fully absent in storage.
        keys_to_rollback = [key for bi, keys in block_written_keys.items() if not block_success[bi] for key in keys]
        if keys_to_rollback:
            logger.warning(
                f"[TransferManager] backup_to_storage: partial write on {len(keys_to_rollback)} key(s), rolling back"
            )
            self._storage_connector.batch_delete(keys_to_rollback)

        failed = [(cpu_block_list[i], hash_list[i]) for i, ok in enumerate(block_success) if not ok]
        if failed:
            logger.warning(
                f"[TransferManager] backup_to_storage: {len(failed)}/{len(cpu_block_list)} block(s) failed — "
                + ", ".join(f"cpu_block={cb} hash={h[:16]}.." for cb, h in failed)
            )

        return block_success
