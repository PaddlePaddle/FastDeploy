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
from typing import TYPE_CHECKING, Any, Dict, Optional

import paddle
from paddleformers.utils.log import logger

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta


@dataclass
class AttentionMetadata(ABC):
    pass


class AttentionBackend(ABC):
    """The base class of attention backends"""

    # Whether KV cache tensors produced by `create_kv_cache` must be pinned via
    # `set_data_ipc` so the paddle allocator cannot relocate them during
    # CUDAGraph capture/replay. Variant backends (MLA/DSA) override to True.
    pin_kv_cache_for_cudagraph: bool = False

    @abstractmethod
    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize the forward metadata."""
        raise NotImplementedError

    def create_kv_cache(
        self,
        max_num_blocks: int,
        cache_dtype: Any,
        kv_cache_quant_type: Optional[str] = None,
    ) -> Dict[str, paddle.Tensor]:
        """
        Allocate KV cache tensors for a single layer.

        Default implementation is GQA/MHA: `key` + `value` (+ fp8 scales when
        `kv_cache_quant_type == "block_wise_fp8"`). MLA/DSA backends override.

        Args:
            max_num_blocks: Block count used to size the cache tensors.
            cache_dtype: Paddle dtype (or string) for key/value tensors.
                Ignored by backends that hard-code their dtype (e.g. DSA uint8).
            kv_cache_quant_type: KV cache quantization type, or None.

        Returns:
            Dict mapping role names (`key`, `value`, `key_scale`,
            `value_scale`, `indexer`, ...) to allocated tensors. Callers
            translate roles into storage names and register them.
        """
        key_shape, value_shape = self.get_kv_cache_shape(
            max_num_blocks=max_num_blocks, kv_cache_quant_type=kv_cache_quant_type
        )
        logger.info(
            f"[create_kv_cache][{type(self).__name__}] key_shape={key_shape} "
            f"value_shape={value_shape} dtype={cache_dtype} "
            f"kv_cache_quant_type={kv_cache_quant_type}"
        )
        caches: Dict[str, paddle.Tensor] = {
            "key": paddle.full(shape=key_shape, fill_value=0, dtype=cache_dtype),
        }
        if value_shape:
            caches["value"] = paddle.full(shape=value_shape, fill_value=0, dtype=cache_dtype)
        if kv_cache_quant_type == "block_wise_fp8":
            scale_shape = [key_shape[0], key_shape[1], key_shape[2]]
            caches["key_scale"] = paddle.full(shape=scale_shape, fill_value=0, dtype=paddle.get_default_dtype())
            if value_shape:
                caches["value_scale"] = paddle.full(shape=scale_shape, fill_value=0, dtype=paddle.get_default_dtype())
        return caches

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
