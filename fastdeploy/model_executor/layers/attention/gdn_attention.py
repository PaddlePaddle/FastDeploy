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

"""GDN linear attention trampoline layer.

Analogous to ``Attention`` for standard softmax attention — instantiated in the
model layer's ``__init__`` and called in ``forward``, delegating to the backend
on ``forward_meta``.

Usage::

    class Qwen3_5GatedDeltaNet(nn.Layer):
        def __init__(self, fd_config, layer_id, ...):
            ...
            self.gdn_attn = GDNAttention(fd_config, layer_id)

        def forward(self, forward_meta, hidden_states):
            ...
            core_attn_out = self.gdn_attn(mixed_qkv, a, b, self, forward_meta)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import paddle
from paddle import nn

from fastdeploy.config import FDConfig

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta


class GDNAttention(nn.Layer):
    """GDN (Gated Delta Network) linear attention trampoline.

    Mirrors the role of :class:`Attention` for softmax attention:
    the model layer holds ``self.gdn_attn = GDNAttention(fd_config, layer_id)``
    and calls ``self.gdn_attn(mixed_qkv, a, b, self, forward_meta)`` in its
    forward.

    Internally delegates to ``forward_meta.gdn_attn_backend.forward()``.
    """

    def __init__(self, fd_config: FDConfig, layer_id: int) -> None:
        super().__init__()
        self.fd_config = fd_config
        self.layer_id = layer_id

    def forward(
        self,
        mixed_qkv: paddle.Tensor,
        a: paddle.Tensor,
        b: paddle.Tensor,
        layer: nn.Layer,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """Forward pass — delegates to the GDN attention backend.

        Args:
            mixed_qkv: Projected QKV tensor ``[num_tokens, conv_dim]``.
            a: Gating input a ``[num_tokens, num_v_heads]``.
            b: Gating input b ``[num_tokens, num_v_heads]``.
            layer: The parent ``Qwen3_5GatedDeltaNet`` instance (provides
                ``conv_weight``, ``A_log_local``, ``dt_bias_local``, etc.).
            forward_meta: Per-step forward metadata.

        Returns:
            Attention output ``[num_tokens, num_v_heads_local, head_v_dim]``.
        """
        return forward_meta.gdn_attn_backend.forward(mixed_qkv, a, b, layer, forward_meta)
