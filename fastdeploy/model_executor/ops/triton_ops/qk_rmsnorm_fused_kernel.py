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

import triton
import triton.language as tl

from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)
from fastdeploy.utils import ceil_div


@enable_compat_on_triton_kernel
@triton.jit
def qk_rmsnorm_fused_kernel(
    x_ptr,
    q_weight_ptr,
    k_weight_ptr,
    M,
    q_size,
    kv_size,
    eps,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    pid = tl.program_id(0)

    heads_per_token = tl.cdiv(num_q_heads, BLOCK_HEADS)
    token_id = pid // heads_per_token
    head_block = pid % heads_per_token

    if token_id >= M:
        return

    offs_h = tl.arange(0, BLOCK_HEADS)
    offs_d = tl.arange(0, head_dim)

    head_ids = head_block * BLOCK_HEADS + offs_h

    q_mask = head_ids < num_q_heads
    kv_mask = head_ids < num_kv_heads

    row_base = token_id * (q_size + 2 * kv_size)

    # -------------------
    # Q RMSNorm
    # -------------------
    q_ptrs = x_ptr + row_base + head_ids[:, None] * head_dim + offs_d[None, :]

    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    q_var = tl.sum(q * q, axis=1) / head_dim
    q_hat = q * tl.rsqrt(q_var[:, None] + eps)

    q_w = tl.load(q_weight_ptr + offs_d).to(tl.float32)
    q_out = q_hat * q_w[None, :]

    tl.store(
        q_ptrs,
        q_out,
        mask=q_mask[:, None],
    )

    # -------------------
    # K RMSNorm
    # -------------------
    k_ptrs = x_ptr + row_base + q_size + head_ids[:, None] * head_dim + offs_d[None, :]

    k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
    k_var = tl.sum(k * k, axis=1) / head_dim
    k_hat = k * tl.rsqrt(k_var[:, None] + eps)

    k_w = tl.load(k_weight_ptr + offs_d).to(tl.float32)
    k_out = k_hat * k_w[None, :]

    tl.store(
        k_ptrs,
        k_out,
        mask=kv_mask[:, None],
    )


def qk_rmsnorm_fused(
    qkv_out,
    q_norm_weight,
    k_norm_weight,
    eps,
    q_size,
    kv_size,
    head_dim,
):
    assert qkv_out.ndim == 2
    M, _ = qkv_out.shape

    num_q_heads = q_size // head_dim
    num_kv_heads = kv_size // head_dim

    BLOCK_HEADS = 4 if num_q_heads <= 32 else 8

    grid = (M * ceil_div(num_q_heads, BLOCK_HEADS),)

    qk_rmsnorm_fused_kernel[grid](
        x_ptr=qkv_out,
        q_weight_ptr=q_norm_weight,
        k_weight_ptr=k_norm_weight,
        M=M,
        q_size=q_size,
        kv_size=kv_size,
        eps=eps,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_HEADS=BLOCK_HEADS,
        num_warps=2,
    )
    return qkv_out
