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

from typing import List

import paddle
from paddleformers.transformers.activations import ACT2FN

from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import fused_neox_rope_embedding, gelu_tanh
elif current_platform.is_iluvatar():
    from fastdeploy.model_executor.ops.iluvatar import fused_neox_rope_embedding


def rotate_half(x):
    Dh = x.shape[-1]
    if Dh == -1:
        Dh = paddle.shape(x)[-1]
    x1 = x[..., : Dh // 2]
    x2 = x[..., Dh // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(x, cos, sin):
    """Apply rotary embeddings to float32 query/key tensors.

    Callers should cast lower precision inputs to float32 before calling this
    helper, and cast the result back afterwards if needed.
    """
    if x.dtype != paddle.float32:
        raise TypeError(f"apply_rotary_pos_emb_vision expects float32 input, got {x.dtype}")
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def native_neox_rope_embedding(qkv, cos, sin, num_heads):
    if qkv.dim() == 3:
        B, seq_length, D = qkv.shape
        if seq_length == -1:
            _, seq_length, _ = paddle.shape(qkv)
        token_count = B * seq_length
    else:
        token_count, D = qkv.shape
        if token_count == -1:
            token_count, _ = paddle.shape(qkv)
    qkv = qkv.reshape([token_count, 3, num_heads, -1])
    q_dtype = qkv.dtype
    if q_dtype != paddle.float32:
        qk = qkv[:, :2].astype("float32")
        q, k = qk[:, 0], qk[:, 1]
    else:
        q, k = qkv[:, 0], qkv[:, 1]
    v = qkv[:, 2]
    q = apply_rotary_pos_emb_vision(q, cos, sin)
    k = apply_rotary_pos_emb_vision(k, cos, sin)
    if q.dtype != q_dtype:
        q = q.astype(q_dtype)
        k = k.astype(q_dtype)
    return q, k, v


jit_unified_marker = paddle.jit.marker.unified if hasattr(paddle.jit.marker, "unified") else lambda fn: fn


@jit_unified_marker
def neox_rope_embedding(
    qkv: paddle.Tensor, cos_emb: paddle.Tensor, sin_emb: paddle.Tensor, num_heads: int, head_dim: int
) -> List[paddle.Tensor]:
    if (current_platform.is_cuda() or current_platform.is_iluvatar()) and paddle.in_dynamic_mode():
        return fused_neox_rope_embedding(qkv, cos_emb, sin_emb, num_heads, head_dim)
    else:
        return native_neox_rope_embedding(qkv, cos_emb, sin_emb, num_heads)


@jit_unified_marker
def get_activation_fn(hidden_act: str):
    if hidden_act == "gelu_pytorch_tanh":
        if current_platform.is_cuda() and paddle.in_dynamic_mode():
            return gelu_tanh
        else:
            return ACT2FN["gelu_new"]
    else:
        return ACT2FN[hidden_act]
