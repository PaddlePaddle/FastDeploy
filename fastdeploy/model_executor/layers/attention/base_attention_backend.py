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
from typing import TYPE_CHECKING

import paddle

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
