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

from typing import Optional

import paddle
import triton
from einops import rearrange

from .minimax_mamba_kernels import (
    _fwd_diag_kernel,
    _fwd_kv_parallel,
    _fwd_kv_reduce,
    _fwd_none_diag_kernel,
    _linear_attn_decode_kernel,
)


def lightning_attention(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    slope_rate: paddle.Tensor,
    kv_history: Optional[paddle.Tensor] = None,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    Implements the Lightning Attention mechanism for the prefill stage using Triton kernels.

    Args:
        q (paddle.Tensor): Query tensor of shape (B, H, N, D).
        k (paddle.Tensor): Key tensor of shape (B, H, N, D).
        v (paddle.Tensor): Value tensor of shape (B, H, N, E).
        slope_rate (paddle.Tensor): Slope tensor for attention decay.
        kv_history (Optional[paddle.Tensor]): KV history from previous steps.

    Returns:
        tuple[paddle.Tensor, paddle.Tensor]: A tuple containing the output tensor and the updated KV history.
    """
    # Ensure tensors are contiguous for Triton kernels.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if slope_rate.dim() > 1:
        slope_rate = slope_rate.squeeze()
    slope_rate = slope_rate.contiguous()

    B, H, N, D = q.shape
    E = v.shape[-1]
    compute_dtype = q.dtype
    o = paddle.empty_like(v)

    if kv_history is None:
        kv_history_in = paddle.zeros((B, H, D, E), dtype=paddle.float32).to(q.place)
    else:
        kv_history_in = kv_history.clone().contiguous()

    # --- Kernel Grid and Block Size Configuration ---
    BLOCK = 256
    NUM_BLOCK = triton.cdiv(N, BLOCK)
    CBLOCK_diag = 32
    NUM_CBLOCK_diag = BLOCK // CBLOCK_diag

    # --- Launch Triton Kernels ---
    grid_diag = (B * H * NUM_BLOCK, NUM_CBLOCK_diag)
    _fwd_diag_kernel[grid_diag](
        q, k, v, o, slope_rate, b=B, h=H, n=N, d=D, e=E, BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK, CBLOCK=CBLOCK_diag
    )

    array = (paddle.arange(0, BLOCK, dtype="float32") + 1).to(q.place)
    k_decay = paddle.exp(-slope_rate.reshape((H, 1)) * (BLOCK - array.reshape((1, -1))))
    k_decay = k_decay.astype(compute_dtype)

    NUM_FBLOCK = 1
    D_FBLOCK = D // NUM_FBLOCK
    E_FBLOCK = E // NUM_FBLOCK
    CBLOCK_kv = 64
    NUM_CBLOCK_kv = BLOCK // CBLOCK_kv
    kv_intermediate = paddle.empty((B, H, NUM_BLOCK, D, E), dtype=paddle.float32).to(q.place)

    grid_kv_parallel = (B * H, NUM_BLOCK)
    _fwd_kv_parallel[grid_kv_parallel](
        k,
        v,
        k_decay,
        kv_intermediate,
        b=B,
        h=H,
        n=N,
        d=D,
        e=E,
        BLOCK=BLOCK,
        NUM_BLOCK=NUM_BLOCK,
        D_FBLOCK=D_FBLOCK,
        E_FBLOCK=E_FBLOCK,
        NUM_FBLOCK=NUM_FBLOCK,
        CBLOCK=CBLOCK_kv,
        NUM_CBLOCK=NUM_CBLOCK_kv,
    )

    grid_kv_reduce = (B * H, NUM_FBLOCK)
    _fwd_kv_reduce[grid_kv_reduce](
        slope_rate,
        kv_intermediate,
        kv_history_in,
        b=B,
        h=H,
        n=N,
        d=D,
        e=E,
        BLOCK=BLOCK,
        NUM_BLOCK=NUM_BLOCK,
        D_FBLOCK=D_FBLOCK,
        E_FBLOCK=E_FBLOCK,
    )

    grid_none_diag = (B * H, NUM_BLOCK * NUM_CBLOCK_diag, NUM_FBLOCK)
    _fwd_none_diag_kernel[grid_none_diag](
        q,
        o,
        slope_rate,
        kv_intermediate,
        b=B,
        h=H,
        n=N,
        d=D,
        e=E,
        BLOCK=BLOCK,
        NUM_BLOCK=NUM_BLOCK,
        E_FBLOCK=E_FBLOCK,
        CBLOCK=CBLOCK_diag,
        NUM_CBLOCK=NUM_CBLOCK_diag,
    )

    return o, kv_history_in


def linear_decode_forward_triton(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    kv_caches: paddle.Tensor,
    slope_rate: paddle.Tensor,
    slot_idx: paddle.Tensor,
    BLOCK_SIZE: int = 32,
) -> paddle.Tensor:
    """
    Performs the forward pass for linear attention decoding using a Triton kernel.

    Args:
        q (paddle.Tensor): Query tensor of shape (B, H, 1, D).
        k (paddle.Tensor): Key tensor of shape (B, H, 1, D).
        v (paddle.Tensor): Value tensor of shape (B, H, 1, D).
        kv_caches (paddle.Tensor): KV cache tensor.
        slope_rate (paddle.Tensor): Slope tensor for attention decay.
        slot_idx (paddle.Tensor): Slot indices for accessing the KV cache.
        BLOCK_SIZE (int): The block size for the Triton kernel.

    Returns:
        paddle.Tensor: The output tensor of the attention operation.
    """
    B, H, _, D = q.shape
    assert tuple(k.shape) == (B, H, 1, D), f"Shape of k is {k.shape}, expected {(B, H, 1, D)}"
    assert tuple(v.shape) == (B, H, 1, D), f"Shape of v is {v.shape}, expected {(B, H, 1, D)}"

    output = paddle.empty_like(q)
    grid = (B, H, triton.cdiv(D, BLOCK_SIZE))

    qkv_b_stride, qkv_h_stride = q.strides[0], q.strides[1]
    cache_b_stride, cache_h_stride, cache_d0_stride, cache_d1_stride = (
        kv_caches.strides[0],
        kv_caches.strides[1],
        kv_caches.strides[2],
        kv_caches.strides[3],
    )

    _linear_attn_decode_kernel[grid](
        q,
        k,
        v,
        kv_caches,
        slope_rate,
        slot_idx,
        output,
        D=D,
        qkv_b_stride=qkv_b_stride,
        qkv_h_stride=qkv_h_stride,
        cache_b_stride=cache_b_stride,
        cache_h_stride=cache_h_stride,
        cache_d0_stride=cache_d0_stride,
        cache_d1_stride=cache_d1_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    output = rearrange(output, "b h n d -> b n (h d)")
    return output.squeeze(1).contiguous()