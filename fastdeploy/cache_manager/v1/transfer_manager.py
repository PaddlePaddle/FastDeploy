"""
CacheTransferManager - Manages cache transfer operations.

Responsible for:
- Coordinating Host↔Device transfers (synchronous only)

Note: All methods in CacheTransferManager are synchronous.
Async operations are handled by CacheController, not here.
"""

import threading
from typing import Any, Dict, List, Optional

# Import ops for cache swap
from fastdeploy.cache_manager.ops import swap_cache_all_layers
from fastdeploy.cache_manager.v1.storage import create_storage_connector
from fastdeploy.cache_manager.v1.transfer import create_transfer_connector
from fastdeploy.config import FDConfig


class CacheTransferManager:
    """
    KV Cache Transfer Manager.

    Coordinates Host↔Device transfers (synchronous operations only).
    Created in Worker process, held by CacheController.

    Data organization:
    1. Name-indexed storage (_cache_kvs_map, _host_cache_kvs_map): for single-layer access
    2. Layer-indexed storage (_device_key_caches, etc.): for all-layer transfers,
       compatible with swap_cache_all_layers operator

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

        self.swap_all_layers = True

        self._lock = threading.RLock()

        # ============ KV Cache Data Storage ============
        # Name-indexed storage (for single-layer access)
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
        self._storage_connector = create_storage_connector(self.cache_config)
        self._transfer_connector = create_transfer_connector(self.cache_config)

    # ============ KV Cache Map Sharing ============

    @property
    def cache_kvs_map(self) -> Dict[str, Any]:
        """
        Get the shared KV cache tensor map.

        Returns:
            Dict[str, Any]: The KV cache tensor dictionary.
        """
        return self._cache_kvs_map

    def set_cache_kvs_map(self, cache_kvs_map: Dict[str, Any]) -> None:
        """
        Share the KV cache tensor map from CacheController.

        This method allows CacheController to share its created KV cache tensors
        with CacheTransferManager, enabling direct access to KV cache data
        during transfer operations (Host↔Device, Storage, etc.).

        Also parses cache_kvs_map and builds layer-indexed data structures
        for compatibility with swap_cache_all_layers operator.

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
        """
        Parse layer-indexed Device cache lists from _cache_kvs_map.

        Builds the following lists:
        - _device_key_caches: key cache per layer
        - _device_value_caches: value cache per layer
        - _device_key_scales: key scales per layer (fp8)
        - _device_value_scales: value scales per layer (fp8)
        """
        if not self._cache_kvs_map:
            return

        # Build layer-indexed lists
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
        """
        Get the shared Host KV cache tensor map.

        Returns:
            Dict[str, Any]: The Host KV cache tensor dictionary.
        """
        return self._host_cache_kvs_map

    def set_host_cache_kvs_map(self, host_cache_kvs_map: Dict[str, Any]) -> None:
        """
        Share the Host KV cache tensor map from CacheController.

        This method allows CacheController to share its created Host KV cache tensors
        with CacheTransferManager, enabling direct access to Host cache data
        during host-device transfer operations.

        Also parses host_cache_kvs_map and builds layer-indexed Host pointer lists
        for compatibility with swap_cache_all_layers operator.

        Args:
            host_cache_kvs_map: Dictionary mapping cache names to Host tensors.
                Format: {
                    "key_caches_{layer_id}_rank{rank}.device{device}": pointer (int),
                    "value_caches_{layer_id}_rank{rank}.device{device}": pointer (int),
                    "key_cache_scales_{layer_id}_rank{rank}.device{device}": pointer (int),  # fp8
                    "value_cache_scales_{layer_id}_rank{rank}.device{device}": pointer (int), # fp8
                    ...
                }
        """
        with self._lock:
            self._host_cache_kvs_map = host_cache_kvs_map
            self._build_host_layer_indices()

    def _build_host_layer_indices(self) -> None:
        """
        Parse layer-indexed Host pointer lists from _host_cache_kvs_map.

        Builds the following lists:
        - _host_key_ptrs: key cache host pointers per layer
        - _host_value_ptrs: value cache host pointers per layer
        - _host_key_scales_ptrs: key scale host pointers per layer (fp8)
        - _host_value_scales_ptrs: value scale host pointers per layer (fp8)
        """
        # Early return if no host cache configured
        if self._num_host_blocks <= 0:
            return

        if not self._host_cache_kvs_map:
            return

        if self._num_layers == 0:
            return

        # Build layer-indexed Host pointer lists
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

    def get_host_cache_tensor(self, cache_name: str) -> Optional[Any]:
        """
        Get a specific Host cache tensor by name.

        Args:
            cache_name: Name of the cache tensor (e.g., "key_caches_0_rank0.device0").

        Returns:
            The Host cache tensor if found, None otherwise.
        """
        # Early return if no host cache configured
        if self._num_host_blocks <= 0:
            return None
        return self._host_cache_kvs_map.get(cache_name)

    def get_host_layer_caches(self, layer_idx: int) -> Dict[str, Any]:
        """
        Get all Host cache tensors for a specific layer.

        Args:
            layer_idx: Layer index.

        Returns:
            Dictionary containing key and value Host caches for the layer.
        """
        # Early return if no host cache configured
        if self._num_host_blocks <= 0:
            return {}

        layer_caches = {}
        for name, tensor in self._host_cache_kvs_map.items():
            if f"_{layer_idx}_" in name:
                layer_caches[name] = tensor
        return layer_caches

    def get_cache_tensor(self, cache_name: str) -> Optional[Any]:
        """
        Get a specific cache tensor by name.

        Args:
            cache_name: Name of the cache tensor (e.g., "key_caches_0_rank0.device0").

        Returns:
            The cache tensor if found, None otherwise.
        """
        return self._cache_kvs_map.get(cache_name)

    def get_layer_caches(self, layer_idx: int) -> Dict[str, Any]:
        """
        Get all cache tensors for a specific layer.

        Args:
            layer_idx: Layer index.

        Returns:
            Dictionary containing key and value caches for the layer.
        """
        layer_caches = {}
        for name, tensor in self._cache_kvs_map.items():
            if f"_{layer_idx}_" in name:
                layer_caches[name] = tensor
        return layer_caches

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
        """Get the number of layers."""
        return self._num_layers

    @property
    def local_rank(self) -> int:
        """Get the local rank."""
        return self._local_rank

    @property
    def device_id(self) -> int:
        """Get the device ID."""
        return self._device_id

    @property
    def cache_dtype(self) -> str:
        """Get the cache dtype."""
        return self._cache_dtype

    @property
    def has_cache_scale(self) -> bool:
        """Check if cache has scale tensors (fp8)."""
        return self._is_fp8_quantization()

    @property
    def num_host_blocks(self) -> int:
        """Get the number of Host blocks."""
        return self._num_host_blocks

    # ============ Device/Host Layer Indexed Access ============

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
        # Early return if no host cache configured
        if self._num_host_blocks <= 0:
            return 0
        if 0 <= layer_idx < len(self._host_key_ptrs):
            return self._host_key_ptrs[layer_idx]
        return 0

    def get_host_value_ptr(self, layer_idx: int) -> int:
        """Get Host value cache pointer for a specific layer."""
        # Early return if no host cache configured
        if self._num_host_blocks <= 0:
            return 0
        if 0 <= layer_idx < len(self._host_value_ptrs):
            return self._host_value_ptrs[layer_idx]
        return 0

    # ============ All-Layer Synchronous Swap Methods ============

    def _swap_all_layers(
        self,
        device_block_ids: List[int],
        host_block_ids: List[int],
        mode: int,
    ) -> bool:
        """
        Synchronous all-layer transfer (directly calls swap_cache_all_layers operator).

        Transfers KV cache data for all layers at once, supporting consecutive
        block merge transfer optimization.

        Args:
            device_block_ids: Device block IDs to swap.
            host_block_ids: Host block IDs to swap (corresponding to device_block_ids).
            mode: Transfer mode, 0=Device→Host (evict), 1=Host→Device (load).

        Returns:
            True if transfer succeeded, False if failed.
        """
        # Early return if no host cache configured
        if self._num_host_blocks <= 0:
            return False

        try:
            # Swap key caches
            swap_cache_all_layers(
                self._device_key_caches,
                self._host_key_ptrs,
                self._num_host_blocks,
                device_block_ids,
                host_block_ids,
                self._device_id,
                mode,
            )

            # Swap value caches
            swap_cache_all_layers(
                self._device_value_caches,
                self._host_value_ptrs,
                self._num_host_blocks,
                device_block_ids,
                host_block_ids,
                self._device_id,
                mode,
            )

            # Swap scales for fp8
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

    def evict_to_host_all_layers(
        self,
        device_block_ids: List[int],
        host_block_ids: List[int],
    ) -> bool:
        """
        Evict all layers of KV Cache from Device to Host (synchronous).

        Args:
            device_block_ids: Device block IDs to evict.
            host_block_ids: Host block IDs to receive (corresponding to device_block_ids).

        Returns:
            True if transfer succeeded, False if failed.
        """
        # Early return if no host cache configured
        if self._num_host_blocks <= 0:
            return False

        if self.swap_all_layers:
            return self._swap_all_layers(device_block_ids, host_block_ids, mode=0)
        else:
            # TODO: Support per-layer transfer
            return False

    def load_to_device_all_layers(
        self,
        host_block_ids: List[int],
        device_block_ids: List[int],
    ) -> bool:
        """
        Load all layers of KV Cache from Host to Device (synchronous).

        Args:
            host_block_ids: Host block IDs to load from.
            device_block_ids: Device block IDs to receive (corresponding to host_block_ids).

        Returns:
            True if transfer succeeded, False if failed.
        """
        # Early return if no host cache configured
        if self._num_host_blocks <= 0:
            return False

        if self.swap_all_layers:
            return self._swap_all_layers(device_block_ids, host_block_ids, mode=1)
        else:
            # TODO: Support per-layer transfer
            return False

    def _validate_swap_params(
        self,
        device_block_ids: List[int],
        host_block_ids: List[int],
    ) -> bool:
        """
        Validate swap parameters.

        Args:
            device_block_ids: Device block IDs.
            host_block_ids: Host block IDs.

        Returns:
            True if parameters are valid, False if invalid.
        """
        # Early return if no host cache configured
        if self._num_host_blocks <= 0:
            return False

        if not device_block_ids or not host_block_ids:
            return False

        if len(device_block_ids) != len(host_block_ids):
            return False

        if not self._device_key_caches or not self._device_value_caches:
            return False

        if not self._host_key_ptrs or not self._host_value_ptrs:
            return False

        return True

    # ============ Per-Layer Synchronous Swap Methods ============

    def _swap_single_layer(
        self,
        layer_idx: int,
        device_block_ids: List[int],
        host_block_ids: List[int],
        mode: int,
    ) -> bool:
        """
        Synchronous single-layer transfer.

        Transfers KV cache data for a single layer using swap_cache_all_layers
        operator with single-element lists.

        Args:
            layer_idx: Layer index to transfer.
            device_block_ids: Device block IDs to swap.
            host_block_ids: Host block IDs to swap (corresponding to device_block_ids).
            mode: Transfer mode, 0=Device→Host (evict), 1=Host→Device (load).

        Returns:
            True if transfer succeeded, False if failed.
        """
        # Early return if no host cache configured
        if self._num_host_blocks <= 0:
            return False

        if not device_block_ids or not host_block_ids:
            return False

        if len(device_block_ids) != len(host_block_ids):
            return False

        try:
            # Get device cache tensors for this layer
            key_cache = self.get_device_key_cache(layer_idx)
            value_cache = self.get_device_value_cache(layer_idx)

            if key_cache is None or value_cache is None:
                return False

            # Get host pointers for this layer
            key_ptr = self.get_host_key_ptr(layer_idx)
            value_ptr = self.get_host_value_ptr(layer_idx)

            if key_ptr == 0 or value_ptr == 0:
                return False

            # Swap key cache for this layer (using single-element lists)
            swap_cache_all_layers(
                [key_cache],
                [key_ptr],
                self._num_host_blocks,
                device_block_ids,
                host_block_ids,
                self._device_id,
                mode,
            )

            # Swap value cache for this layer
            swap_cache_all_layers(
                [value_cache],
                [value_ptr],
                self._num_host_blocks,
                device_block_ids,
                host_block_ids,
                self._device_id,
                mode,
            )

            # Swap scales for fp8 if needed
            if self._is_fp8_quantization():
                key_scale = self._device_key_scales[layer_idx] if layer_idx < len(self._device_key_scales) else None
                value_scale = (
                    self._device_value_scales[layer_idx] if layer_idx < len(self._device_value_scales) else None
                )
                key_scale_ptr = (
                    self._host_key_scales_ptrs[layer_idx] if layer_idx < len(self._host_key_scales_ptrs) else 0
                )
                value_scale_ptr = (
                    self._host_value_scales_ptrs[layer_idx] if layer_idx < len(self._host_value_scales_ptrs) else 0
                )

                if key_scale is not None and key_scale_ptr > 0:
                    swap_cache_all_layers(
                        [key_scale],
                        [key_scale_ptr],
                        self._num_host_blocks,
                        device_block_ids,
                        host_block_ids,
                        self._device_id,
                        mode,
                    )
                if value_scale is not None and value_scale_ptr > 0:
                    swap_cache_all_layers(
                        [value_scale],
                        [value_scale_ptr],
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

    def evict_layer_to_host(
        self,
        layer_idx: int,
        device_block_ids: List[int],
        host_block_ids: List[int],
    ) -> bool:
        """
        Evict a single layer of KV Cache from Device to Host (synchronous).

        Args:
            layer_idx: Layer index to evict.
            device_block_ids: Device block IDs to evict.
            host_block_ids: Host block IDs to receive (corresponding to device_block_ids).

        Returns:
            True if transfer succeeded, False if failed.
        """
        # Early return if no host cache configured
        if self._num_host_blocks <= 0:
            return False

        return self._swap_single_layer(layer_idx, device_block_ids, host_block_ids, mode=0)

    def load_layer_to_device(
        self,
        layer_idx: int,
        host_block_ids: List[int],
        device_block_ids: List[int],
    ) -> bool:
        """
        Load a single layer of KV Cache from Host to Device (synchronous).

        Args:
            layer_idx: Layer index to load.
            host_block_ids: Host block IDs to load from.
            device_block_ids: Device block IDs to receive (corresponding to host_block_ids).

        Returns:
            True if transfer succeeded, False if failed.
        """
        # Early return if no host cache configured
        if self._num_host_blocks <= 0:
            return False

        return self._swap_single_layer(layer_idx, device_block_ids, host_block_ids, mode=1)

    def evict_layers_to_host(
        self,
        layer_indices: List[int],
        device_block_ids: List[int],
        host_block_ids: List[int],
        on_layer_complete: Optional[callable] = None,
    ) -> bool:
        """
        Evict multiple layers of KV Cache from Device to Host (synchronous, layer-by-layer).

        This method transfers layers one by one, calling the callback after each layer
        completes. This allows overlapping transfer with forward computation.

        Args:
            layer_indices: Layer indices to evict.
            device_block_ids: Device block IDs to evict.
            host_block_ids: Host block IDs to receive.
            on_layer_complete: Optional callback(layer_idx) called after each layer completes.

        Returns:
            True if all transfers succeeded, False if any failed.
        """
        # Early return if no host cache configured
        if self._num_host_blocks <= 0:
            return False

        all_success = True
        for layer_idx in layer_indices:
            success = self.evict_layer_to_host(layer_idx, device_block_ids, host_block_ids)
            if not success:
                all_success = False
            if on_layer_complete is not None:
                try:
                    on_layer_complete(layer_idx)
                except Exception:
                    pass
        return all_success

    def load_layers_to_device(
        self,
        layer_indices: List[int],
        host_block_ids: List[int],
        device_block_ids: List[int],
        on_layer_complete: Optional[callable] = None,
    ) -> bool:
        """
        Load multiple layers of KV Cache from Host to Device (synchronous, layer-by-layer).

        This method transfers layers one by one, calling the callback after each layer
        completes. This allows overlapping transfer with forward computation.

        Args:
            layer_indices: Layer indices to load.
            host_block_ids: Host block IDs to load from.
            device_block_ids: Device block IDs to receive.
            on_layer_complete: Optional callback(layer_idx) called after each layer completes.

        Returns:
            True if all transfers succeeded, False if any failed.
        """
        # Early return if no host cache configured
        if self._num_host_blocks <= 0:
            return False

        all_success = True
        for layer_idx in layer_indices:
            success = self.load_layer_to_device(layer_idx, host_block_ids, device_block_ids)
            if not success:
                all_success = False
            if on_layer_complete is not None:
                try:
                    on_layer_complete(layer_idx)
                except Exception:
                    pass
        return all_success

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
