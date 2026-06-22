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

import os

import paddle
from paddleformers.transformers.activations import ACT2FN

from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import fused_neox_rope_embedding, gelu_tanh
elif current_platform.is_maca():
    # Metax C500 (SM 80) has gelu_tanh fused op available, but not fused_neox_rope_embedding.
    # Enable gelu_tanh to avoid Python fallback in SigLIP MLP (27 layers × 1 GELU each).
    from fastdeploy.model_executor.ops.gpu import gelu_tanh

    # Fused RoPE kernel (rotate-half convention) compiled from apply_rope_qkv.cu.
    # Replaces the Python fallback in SigLIP vision encoder (27 layers × 1 RoPE each,
    # measured at 27.5% of vision encoder time — see optimization_log.md §0.1.1).
    _ROPE_SO = os.path.join(os.path.dirname(__file__), "apply_rope_qkv_pd_.so")
    _MACA_FUSED_ROPE_OK = False
    if os.path.exists(_ROPE_SO):
        try:
            paddle.utils.cpp_extension.load_op_meta_info_and_register_op(_ROPE_SO)
            _MACA_FUSED_ROPE_OK = True
        except Exception:
            pass


def rotate_half(x):
    Dh = x.shape[-1]
    if Dh == -1:
        Dh = paddle.shape(x)[-1]
    x1 = x[..., : Dh // 2]
    x2 = x[..., Dh // 2 :]
    return paddle.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(x, cos, sin):
    orig_dtype = x.dtype
    x = x.astype("float32")
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed.astype(orig_dtype)


def native_neox_rope_embedding(qkv, cos, sin, num_heads):
    B, seq_length, D = qkv.shape
    if seq_length == -1:
        _, seq_length, _ = paddle.shape(qkv)
    qkv = qkv.reshape(
        [
            seq_length,
            3,
            num_heads,
            -1,
        ]
    ).transpose(perm=[1, 0, 2, 3])
    q, k, v = qkv.unbind(axis=0)
    q = apply_rotary_pos_emb_vision(q, cos, sin)
    k = apply_rotary_pos_emb_vision(k, cos, sin)
    return q, k, v


def maca_fused_neox_rope_embedding(qkv, cos, sin, num_heads, head_dim):
    """Metax fused RoPE kernel — rotate-half convention.
    Input: qkv [B, seq, 3*num_heads*head_dim], cos/sin [seq, 1, head_dim]
    Output: q, k, v each [seq, num_heads, head_dim]
    """
    B, seq_length, D = qkv.shape
    if seq_length == -1:
        _, seq_length, _ = paddle.shape(qkv)
    # Flatten batch+seq → tokens (kernel expects 2D qkv)
    qkv_flat = qkv.reshape([-1, D])
    q_out, k_out, v_out = paddle._C_ops._run_custom_op(
        "apply_rope_qkv",
        qkv_flat,
        cos,
        sin,
        num_heads,
        num_heads,
        head_dim,
    )
    return q_out, k_out, v_out




jit_unified_marker = paddle.jit.marker.unified if hasattr(paddle.jit.marker, "unified") else lambda fn: fn


@jit_unified_marker
def neox_rope_embedding(
    qkv: paddle.Tensor, cos_emb: paddle.Tensor, sin_emb: paddle.Tensor, num_heads: int, head_dim: int
) -> List[paddle.Tensor]:
    if current_platform.is_cuda() and paddle.in_dynamic_mode():
        return fused_neox_rope_embedding(qkv, cos_emb, sin_emb, num_heads, head_dim)
    elif current_platform.is_maca() and _MACA_FUSED_ROPE_OK and paddle.in_dynamic_mode():
        return maca_fused_neox_rope_embedding(qkv, cos_emb, sin_emb, num_heads, head_dim)
    else:
        return native_neox_rope_embedding(qkv, cos_emb, sin_emb, num_heads)


@jit_unified_marker
def get_activation_fn(hidden_act: str):
    if hidden_act == "gelu_pytorch_tanh":
        if (current_platform.is_cuda() or current_platform.is_maca()) and paddle.in_dynamic_mode():
            return gelu_tanh
        else:
            return ACT2FN["gelu_new"]
    else:
        return ACT2FN[hidden_act]


# ── eager-mode dispatch cache ─────────────────────────────────────
# Avoid repeated platform / dynamic-mode checks on every encoder-layer
# call.  Resolved once on first invocation, then the fast path is a
# single boolean branch (no function calls, no attribute lookups).
_USE_MACA_FUSED_ROPE = (
    current_platform.is_maca() and _MACA_FUSED_ROPE_OK and paddle.in_dynamic_mode()
)
_USE_CUDA_FUSED_ROPE = (
    current_platform.is_cuda() and paddle.in_dynamic_mode()
)


def neox_rope_embedding_eager(
    qkv, cos_emb, sin_emb, num_heads, head_dim
):
    """Hot-path RoPE dispatch — no platform checks, no decorator overhead."""
    if _USE_MACA_FUSED_ROPE:
        return maca_fused_neox_rope_embedding(qkv, cos_emb, sin_emb, num_heads, head_dim)
    elif _USE_CUDA_FUSED_ROPE:
        return fused_neox_rope_embedding(qkv, cos_emb, sin_emb, num_heads, head_dim)
    else:
        return native_neox_rope_embedding(qkv, cos_emb, sin_emb, num_heads)


# Pre-resolved activation for SigLIP MLP (gelu_pytorch_tanh).
_siglip_activation = get_activation_fn("gelu_pytorch_tanh")
