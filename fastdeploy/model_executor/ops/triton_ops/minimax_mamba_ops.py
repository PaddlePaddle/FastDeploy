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
"""
from typing import Optional
import paddle
import triton

from .minimax_mamba_kernels import (
    _fwd_diag_kernel,
    _fwd_kv_parallel,
    _fwd_kv_reduce,
    _fwd_none_diag_kernel,
    _linear_attn_decode_kernel,
)


class _Attention(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, q, k, v, s, kv_history_in):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()
        
        kv_history_compute = paddle.clone(kv_history_in)

        # Get input dimensions
        b, h, n, d = q.shape
        e = v.shape[-1]

        # Initialize the output tensor
        o = paddle.empty(shape=[b, h, n, e], dtype=q.dtype)

        BLOCK = 256
        NUM_BLOCK = triton.cdiv(n, BLOCK)
        CBLOCK_DIAG = 32
        NUM_CBLOCK_DIAG = BLOCK // CBLOCK_DIAG
        assert BLOCK % CBLOCK_DIAG == 0
        array = paddle.arange(0, BLOCK) + 1
        array_float = array.astype("float32")
        k_decay = paddle.exp(-s * (BLOCK - array_float.reshape([1, -1])))

        # Step 1
        grid_diag = (b * h * NUM_BLOCK, NUM_CBLOCK_DIAG)
        _fwd_diag_kernel[grid_diag](
            q, k, v, o, s,
            b=b, h=h, n=n, d=d, e=e,
            BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK, CBLOCK=CBLOCK_DIAG,
        )

        # Step 2
        NUM_FBLOCK = 1
        D_FBLOCK = d // NUM_FBLOCK
        E_FBLOCK = e // NUM_FBLOCK
        CBLOCK_KV_AND_NON_DIAG = 64
        NUM_CBLOCK_KV_AND_NON_DIAG = BLOCK // CBLOCK_KV_AND_NON_DIAG
        kv = paddle.empty(shape=[b, h, NUM_BLOCK, d, e], dtype="float32")
        grid_kv_parallel = (b * h, NUM_BLOCK)
        _fwd_kv_parallel[grid_kv_parallel](
            k, v, k_decay, kv,
            b=b, h=h, n=n, d=d, e=e,
            BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK, E_FBLOCK=E_FBLOCK, NUM_FBLOCK=NUM_FBLOCK,
            CBLOCK=CBLOCK_KV_AND_NON_DIAG, NUM_CBLOCK=NUM_CBLOCK_KV_AND_NON_DIAG,
        )

        # Step 3
        grid_kv_reduce = (b * h, NUM_FBLOCK)
        _fwd_kv_reduce[grid_kv_reduce](
            s, kv, kv_history_compute,  
            b=b, h=h, n=n, d=d, e=e,
            BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK,
            D_FBLOCK=D_FBLOCK, E_FBLOCK=E_FBLOCK,
        )

        # Step 4
        grid_none_diag = (b * h, NUM_BLOCK * NUM_CBLOCK_KV_AND_NON_DIAG)
        _fwd_none_diag_kernel[grid_none_diag](
            q, o, s, kv,
            b=b, h=h, n=n, d=d, e=e,
            BLOCK=BLOCK, NUM_BLOCK=NUM_BLOCK, E_FBLOCK=E_FBLOCK,
            CBLOCK=CBLOCK_KV_AND_NON_DIAG, NUM_CBLOCK=NUM_CBLOCK_KV_AND_NON_DIAG,
        )

        return o, kv_history_compute

    @staticmethod
    def backward(ctx, grad_output, grad_kv_history):
        raise NotImplementedError("Backward pass for lightning_attention is not implemented")

def lightning_attention(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    slope_rate: paddle.Tensor,
    kv_history: Optional[paddle.Tensor] = None,
    is_profiling: bool = False, 
    block_size: int = 256, 
) -> tuple[paddle.Tensor, paddle.Tensor]:

    d = q.shape[-1]
    e = v.shape[-1]

    if slope_rate.dim() == 1:
        slope_rate = slope_rate.reshape([1, -1, 1, 1])

    m = 128 if d >= 128 else 64
    if d % m != 0:
        raise ValueError(f"Head dimension d ({d}) must be divisible by chunk size m ({m})")

    arr = [m * i for i in range(d // m + 1)]
    if arr[-1] != d:
        arr.append(d)

    num_chunks = len(arr) - 1
    output = 0

    if kv_history is None:
        kv_history_for_loop = paddle.zeros(
            shape=[q.shape[0], q.shape[1], d, e], dtype="float32"
        )
    else:
        kv_history_for_loop = paddle.clone(kv_history).contiguous()

    final_kv_state = None
    for i in range(num_chunks):
        s = arr[i]
        e_chunk = arr[i + 1]

        q_chunk = q[..., s:e_chunk]
        k_chunk = k[..., s:e_chunk]

        o_chunk, updated_full_kv_history = _Attention.apply(q_chunk, k_chunk, v, slope_rate, kv_history_for_loop)

        output = output + o_chunk

        # update kv_history
        kv_history_for_loop = updated_full_kv_history
        final_kv_state = updated_full_kv_history

    return output.astype(k.dtype), final_kv_state


def linear_decode_forward_triton(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    kv_caches: paddle.Tensor,
    slope_rate: paddle.Tensor,
    slot_idx: paddle.Tensor,
    BLOCK_SIZE: int = 32,
) -> paddle.Tensor:
    B, H, _, D = q.shape
    assert tuple(k.shape) == (B, H, 1, D), f"Shape of k is {k.shape}, expected {(B, H, 1, D)}"
    assert tuple(v.shape) == (B, H, 1, D), f"Shape of v is {v.shape}, expected {(B, H, 1, D)}"
    from einops import rearrange
    output = paddle.empty_like(q)
    grid = (B, H, triton.cdiv(D, BLOCK_SIZE))

    qkv_b_stride, qkv_h_stride = q.strides[0], q.strides[1]
    cache_b_stride, cache_h_stride, cache_d0_stride, cache_d1_stride = (kv_caches.strides[0], kv_caches.strides[1], kv_caches.strides[2], kv_caches.strides[3])

    _linear_attn_decode_kernel[grid](q, k, v, kv_caches, slope_rate, slot_idx, output, D=D, qkv_b_stride=qkv_b_stride, qkv_h_stride=qkv_h_stride, cache_b_stride=cache_b_stride, cache_h_stride=cache_h_stride, cache_d0_stride=cache_d0_stride, cache_d1_stride=cache_d1_stride, BLOCK_SIZE=BLOCK_SIZE)
    output = rearrange(output, "b h n d -> b n (h d)")
    return output.squeeze(1).contiguous()