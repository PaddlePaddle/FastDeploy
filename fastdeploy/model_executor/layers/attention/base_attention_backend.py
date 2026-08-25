"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# Adapt from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/base_attn_backend.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import paddle
from paddleformers.utils.log import logger

try:
    from fastdeploy.cache_manager.ops import cuda_host_alloc, cuda_host_free
except Exception:  # pragma: no cover - host alloc unavailable on some platforms
    cuda_host_alloc = None
    cuda_host_free = None

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta


@dataclass
class AttentionMetadata(ABC):
    pass


class AttentionBackend(ABC):
    """The base class of attention backends"""

    @abstractmethod
    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize the forward metadata."""
        raise NotImplementedError

    def _get_identity_rotary_embs(self, original_rotary_embs: paddle.Tensor) -> paddle.Tensor:
        """
        Create identity rotary embeddings (cos=1, sin=0) that make RoPE a no-op.

        This is used when RoPE has already been applied externally (e.g., by PaddleFormers).
        The identity transformation ensures: x * cos(0) + y * sin(0) = x, preserving the input.

        Text models pack rotary embs as [2, batch, seq, 1, head_dim] (axis 0 = [cos, sin]),
        while multimodal models use [batch, 2, 1, max_len, 1, head_dim] (axis 1 = [cos, sin]),
        so the cos/sin axis is located by shape.

        NOTE: Shape can change between prefill/decode, so we check if cached shape matches.
        """
        # Check if we need to recreate (shape mismatch or not cached)
        need_recreate = (
            not hasattr(self, "_identity_rotary_embs")
            or self._identity_rotary_embs is None
            or self._identity_rotary_embs.shape != original_rotary_embs.shape
        )

        if need_recreate:
            # Create identity RoPE: cos=1, sin=0
            identity = paddle.zeros_like(original_rotary_embs)
            if identity.shape[0] != 2 and len(identity.shape) > 1 and identity.shape[1] == 2:
                # Multimodal layout: [batch, 2, 1, max_len, 1, head_dim]
                identity[:, 0] = 1.0  # cos = 1
                identity[:, 1] = 0.0  # sin = 0
            else:
                # Text layout: [2, batch, seq, 1, head_dim]
                identity[0] = 1.0  # cos = 1
                identity[1] = 0.0  # sin = 0
            self._identity_rotary_embs = identity

        return self._identity_rotary_embs

    def create_kv_cache(
        self,
        num_layers: int,
        num_blocks: int,
        cache_dtype: Any,
        kv_cache_quant_type: Optional[str] = None,
        layer_offset: int = 0,
    ) -> Dict[Tuple[str, int], paddle.Tensor]:
        """
        Allocate KV cache tensors for a range of layers.

        Default implementation is GQA/MHA: per layer allocates `key` + `value`
        (+ fp8 scales when `kv_cache_quant_type == "block_wise_fp8"`). MLA/DSA
        backends override with their variant-specific layout.

        Args:
            num_layers: Number of layers to allocate cache for.
            num_blocks: Block count used to size each layer's cache tensors.
            cache_dtype: Paddle dtype (or string) for key/value tensors.
                Ignored by backends that hard-code their dtype (e.g. DSA uint8).
            kv_cache_quant_type: KV cache quantization type, or None.
            layer_offset: Global index of the first layer; resulting dict keys
                use absolute layer indices in
                ``[layer_offset, layer_offset + num_layers)``.

        Returns:
            Dict keyed by ``(role, layer_idx)`` where role is one of
            ``key / value / key_scale / value_scale / indexer`` and
            ``layer_idx`` is the absolute layer index. Callers translate role
            names into storage names and register the tensors.
        """
        key_shape, value_shape = self.get_kv_cache_shape(
            max_num_blocks=num_blocks, kv_cache_quant_type=kv_cache_quant_type
        )
        logger.info(
            f"[create_kv_cache][{type(self).__name__}] num_layers={num_layers} "
            f"layer_offset={layer_offset} key_shape={key_shape} "
            f"value_shape={value_shape} dtype={cache_dtype} "
            f"kv_cache_quant_type={kv_cache_quant_type}"
        )
        has_value = bool(value_shape)
        is_fp8 = kv_cache_quant_type == "block_wise_fp8"
        scale_shape = [key_shape[0], key_shape[1], key_shape[2]] if is_fp8 else None
        scale_dtype = paddle.get_default_dtype() if is_fp8 else None

        caches: Dict[Tuple[str, int], paddle.Tensor] = {}
        for layer_idx in range(layer_offset, layer_offset + num_layers):
            caches[("key", layer_idx)] = paddle.full(shape=key_shape, fill_value=0, dtype=cache_dtype)
            if has_value:
                caches[("value", layer_idx)] = paddle.full(shape=value_shape, fill_value=0, dtype=cache_dtype)
            if is_fp8:
                caches[("key_scale", layer_idx)] = paddle.full(shape=scale_shape, fill_value=0, dtype=scale_dtype)
                if has_value:
                    caches[("value_scale", layer_idx)] = paddle.full(
                        shape=scale_shape, fill_value=0, dtype=scale_dtype
                    )
        return caches

    def create_host_kv_cache(
        self,
        num_layers: int,
        num_blocks: int,
        cache_item_bytes: int,
        kv_cache_quant_type: Optional[str] = None,
        layer_offset: int = 0,
    ) -> Dict[Tuple[str, int], Any]:
        """
        Allocate pinned-memory host KV cache for a range of layers.

        Default GQA/MHA implementation allocates, per layer: `key` host buffer
        (+ `value` if this backend has a separate value cache) + fp8 scale
        buffers when applicable. Host buffers are raw pinned-memory pointers
        from ``cuda_host_alloc``; dtype-specific sizing is folded into
        ``cache_item_bytes`` by the caller.

        Args:
            num_layers: Number of layers to allocate host cache for.
            num_blocks: Host block count used for sizing.
            cache_item_bytes: Bytes per element for key/value buffers.
            kv_cache_quant_type: KV cache quantization type, or None.
            layer_offset: Global index of the first layer.

        Returns:
            Dict keyed by ``(role, layer_idx)``. Empty dict if host alloc is
            unavailable on the current platform.
        """
        if cuda_host_alloc is None:
            raise RuntimeError(
                f"[create_host_kv_cache][{type(self).__name__}] cuda_host_alloc " "is not available on this platform"
            )

        key_shape, value_shape = self.get_kv_cache_shape(
            max_num_blocks=num_blocks, kv_cache_quant_type=kv_cache_quant_type
        )
        has_value = bool(value_shape)
        is_fp8 = kv_cache_quant_type == "block_wise_fp8"

        # Elements per block per layer.
        key_elems = key_shape[1] * key_shape[2] * key_shape[3]
        value_elems = value_shape[1] * value_shape[2] * value_shape[3] if has_value else 0

        key_bytes = num_blocks * cache_item_bytes * key_elems
        value_bytes = num_blocks * cache_item_bytes * value_elems

        # fp8 scales use float32 (4 bytes), shape [num_blocks, k1, k2].
        scale_elems = key_shape[1] * key_shape[2] if is_fp8 else 0
        scale_bytes = num_blocks * 4 * scale_elems if is_fp8 else 0

        logger.info(
            f"[create_host_kv_cache][{type(self).__name__}] num_layers={num_layers} "
            f"layer_offset={layer_offset} num_blocks={num_blocks} "
            f"key_bytes_per_layer={key_bytes} value_bytes_per_layer={value_bytes} "
            f"scale_bytes_per_layer={scale_bytes} kv_cache_quant_type={kv_cache_quant_type}"
        )

        out: Dict[Tuple[str, int], Any] = {}
        for layer_idx in range(layer_offset, layer_offset + num_layers):
            out[("key", layer_idx)] = cuda_host_alloc(key_bytes)
            if is_fp8:
                out[("key_scale", layer_idx)] = cuda_host_alloc(scale_bytes)
            if has_value:
                out[("value", layer_idx)] = cuda_host_alloc(value_bytes)
                if is_fp8:
                    out[("value_scale", layer_idx)] = cuda_host_alloc(scale_bytes)
        return out

    def free_host_kv_cache(self, host_caches: Dict[Any, Any]) -> None:
        """
        Release pinned-memory host KV cache buffers.

        Accepts the dict returned by :meth:`create_host_kv_cache` or any
        mapping whose values are pinned-memory pointers (ints) produced by
        ``cuda_host_alloc``. Zero/None pointers are skipped. Individual
        frees that raise are logged and swallowed so one stale pointer
        does not block the rest. The input mapping is cleared on return.

        Args:
            host_caches: Mapping whose values are pinned-memory pointers.
        """
        if not host_caches:
            return

        if cuda_host_free is None:
            logger.warning(
                f"[free_host_kv_cache][{type(self).__name__}] cuda_host_free "
                "is not available on this platform; leaking pinned memory."
            )
            host_caches.clear()
            return

        logger.info(f"[free_host_kv_cache][{type(self).__name__}] freeing " f"{len(host_caches)} host cache buffers.")
        for name, ptr in list(host_caches.items()):
            if not ptr:
                continue
            try:
                cuda_host_free(ptr)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    f"[free_host_kv_cache][{type(self).__name__}] failed to " f"free host cache {name}: {e}"
                )
        host_caches.clear()

    def forward(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: paddle.nn.Layer,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """
        Run a forward.
        args:
            q: The query tensor.
            k: The key tensor.
            v: The value tensor.
            layer: The layer that will be used for the forward.
            compressed_kv: optional compressed key-value cache (for MLA)
            k_pe: optional key positional encoding (for MLA)
            forward_meta: The forward metadata.
        """
        if forward_meta.forward_mode.is_mixed():
            return self.forward_mixed(
                q,
                k,
                v,
                qkv,
                compressed_kv,
                k_pe,
                layer,
                forward_meta,
            )
        elif forward_meta.forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                qkv,
                compressed_kv,
                k_pe,
                layer,
                forward_meta,
            )
        elif forward_meta.forward_mode.is_native():
            return self.forward_native_backend(
                q,
                k,
                v,
                qkv,
                layer,
                forward_meta,
            )
        else:
            return self.forward_extend(
                q,
                k,
                v,
                qkv,
                compressed_kv,
                k_pe,
                layer,
                forward_meta,
            )

    def forward_mixed(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: paddle.nn.Layer,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """Run a forward for mix."""
        raise NotImplementedError

    def forward_decode(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: paddle.nn.Layer,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """Run a forward for decode."""
        raise NotImplementedError

    def forward_extend(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: paddle.nn.Layer,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """Run a forward for extend."""
        raise NotImplementedError

    def forward_native_backend(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        layer: paddle.nn.Layer,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """Run a forward for native."""
        raise NotImplementedError
