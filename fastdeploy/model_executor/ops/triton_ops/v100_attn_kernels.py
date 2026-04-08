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

Triton kernels for V100 (SM70) attention backend (Triton fallback path).

Used when the CUDA C++ custom op (v100_decode_attention) is unavailable.
Three kernels:
1. v100_write_kv_cache_kernel     - write K/V to block-based cache
2. v100_decode_fused_kernel       - fused flash-decoding (single/multi split)
3. v100_decode_attn_stage2        - merge partial outputs across splits
"""

import triton
import triton.language as tl

from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)
from fastdeploy.utils import ceil_div

# ---------------------------------------------------------------------------
# Kernel 1: Write KV to block cache
# ---------------------------------------------------------------------------


@enable_compat_on_triton_kernel
@triton.jit
def v100_write_kv_cache_kernel(
    k_ptr,  # [num_tokens, kv_num_heads, head_dim]
    v_ptr,  # [num_tokens, kv_num_heads, head_dim]
    key_cache_ptr,  # [max_num_blocks, kv_num_heads, block_size, head_dim]
    value_cache_ptr,  # [max_num_blocks, kv_num_heads, block_size, head_dim]
    block_tables_ptr,  # [batch_size, max_blocks_per_seq]
    positions_ptr,  # [num_tokens] int64
    batch_id_per_token_ptr,  # [num_tokens] int32
    num_tokens,
    max_blocks_per_seq,
    block_size: tl.constexpr,
    kv_num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Each program handles one (token, kv_head) pair."""
    pid = tl.program_id(0)
    token_id = pid // kv_num_heads
    head_id = pid % kv_num_heads

    if token_id >= num_tokens:
        return

    pos = tl.load(positions_ptr + token_id).to(tl.int32)
    batch_id = tl.load(batch_id_per_token_ptr + token_id)

    block_idx = pos // block_size
    block_offset = pos % block_size

    # physical block from block_tables
    physical_block = tl.load(block_tables_ptr + batch_id * max_blocks_per_seq + block_idx)

    # Guard: skip if block freed (preempted), physical_block == -1
    valid_block = physical_block >= 0

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    # Source: k_ptr[token_id, head_id, :head_dim]
    k_src_base = token_id * kv_num_heads * head_dim + head_id * head_dim
    k_vals = tl.load(k_ptr + k_src_base + offs_d, mask=d_mask, other=0.0)

    v_src_base = token_id * kv_num_heads * head_dim + head_id * head_dim
    v_vals = tl.load(v_ptr + v_src_base + offs_d, mask=d_mask, other=0.0)

    # Dest: cache[physical_block, head_id, block_offset, :head_dim]
    # cache layout: [max_num_blocks, kv_num_heads, block_size, head_dim]
    cache_base = (
        physical_block * (kv_num_heads * block_size * head_dim)
        + head_id * (block_size * head_dim)
        + block_offset * head_dim
    )
    tl.store(key_cache_ptr + cache_base + offs_d, k_vals, mask=d_mask & valid_block)
    tl.store(value_cache_ptr + cache_base + offs_d, v_vals, mask=d_mask & valid_block)


def v100_write_kv_cache(
    k,  # paddle.Tensor [num_tokens, kv_num_heads, head_dim]
    v,  # paddle.Tensor [num_tokens, kv_num_heads, head_dim]
    key_cache,  # paddle.Tensor [max_num_blocks, kv_num_heads, block_size, head_dim]
    value_cache,  # paddle.Tensor [max_num_blocks, kv_num_heads, block_size, head_dim]
    block_tables,  # paddle.Tensor [batch_size, max_blocks_per_seq]
    positions,  # paddle.Tensor [num_tokens] int64
    batch_id_per_token,  # paddle.Tensor [num_tokens] int32
):
    """Write K/V to block-based cache using a Triton kernel."""
    num_tokens = k.shape[0]
    kv_num_heads = k.shape[1]
    head_dim = k.shape[2]
    block_size = key_cache.shape[2]
    max_blocks_per_seq = block_tables.shape[1]

    # BLOCK_D must be >= head_dim, power of 2
    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (num_tokens * kv_num_heads,)
    v100_write_kv_cache_kernel[grid](
        k_ptr=k,
        v_ptr=v,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_tables,
        positions_ptr=positions,
        batch_id_per_token_ptr=batch_id_per_token,
        num_tokens=num_tokens,
        max_blocks_per_seq=max_blocks_per_seq,
        block_size=block_size,
        kv_num_heads=kv_num_heads,
        head_dim=head_dim,
        BLOCK_D=BLOCK_D,
        num_warps=2,
    )


# ---------------------------------------------------------------------------
# Kernel 2: Fused decode attention (stage1 with optional stage2)
# ---------------------------------------------------------------------------


@enable_compat_on_triton_kernel
@triton.jit
def v100_decode_fused_kernel(
    q_ptr,  # [num_tokens, num_heads, head_dim]
    key_cache_ptr,  # [max_num_blocks, kv_num_heads, block_size, head_dim]
    value_cache_ptr,  # [max_num_blocks, kv_num_heads, block_size, head_dim]
    output_ptr,  # [num_tokens, num_heads, head_dim]
    block_tables_ptr,  # [batch_size, max_blocks_per_seq]
    seq_lens_ptr,  # [batch_size] int32 - total kv length (including new token)
    q_start_loc_ptr,  # [batch_size] int32
    partial_out_ptr,  # [batch_size, num_heads, num_kv_splits, head_dim] float32 (unused if SINGLE_SPLIT)
    partial_lse_ptr,  # [batch_size, num_heads, num_kv_splits] float32 (unused if SINGLE_SPLIT)
    sm_scale,
    max_blocks_per_seq,
    num_heads: tl.constexpr,
    kv_num_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    num_kv_splits: tl.constexpr,
    MAX_BLOCKS_PER_SPLIT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SINGLE_SPLIT: tl.constexpr,  # True: write directly to output (skip stage2)
):
    """
    Stage1 kernel that writes directly to output when SINGLE_SPLIT=True.
    Grid: (batch_size, num_heads, num_kv_splits)
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_split = tl.program_id(2)

    total_kv_len = tl.load(seq_lens_ptr + pid_batch)
    if total_kv_len <= 0:
        return

    kv_head_id = pid_head // group_size

    # Determine KV range for this split
    total_kv_blocks = tl.cdiv(total_kv_len, block_size)
    blocks_per_split = tl.cdiv(total_kv_blocks, num_kv_splits)
    split_start_block = pid_split * blocks_per_split
    split_end_block = tl.minimum((pid_split + 1) * blocks_per_split, total_kv_blocks)

    if split_start_block >= total_kv_blocks:
        return

    # Load Q
    q_start = tl.load(q_start_loc_ptr + pid_batch)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    q_base = q_start * num_heads * head_dim + pid_head * head_dim
    q_vec = tl.load(q_ptr + q_base + offs_d, mask=d_mask, other=0.0).to(tl.float32)

    # Online softmax state
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for bi in range(MAX_BLOCKS_PER_SPLIT):
        block_idx = split_start_block + bi
        if block_idx < split_end_block:
            physical_block = tl.load(block_tables_ptr + pid_batch * max_blocks_per_seq + block_idx)

            # Guard: skip freed block (preempted, physical_block == -1)
            if physical_block >= 0:
                block_start_pos = block_idx * block_size
                valid_tokens = tl.minimum(block_size, total_kv_len - block_start_pos)

                kv_range = tl.arange(0, block_size)
                kv_mask = kv_range < valid_tokens

                k_base = physical_block * (kv_num_heads * block_size * head_dim) + kv_head_id * (block_size * head_dim)
                k_ptrs = k_base + kv_range[:, None] * head_dim + offs_d[None, :]
                k_vals = tl.load(key_cache_ptr + k_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0).to(
                    tl.float32
                )

                qk = tl.sum(q_vec[None, :] * k_vals, axis=1) * sm_scale
                qk = tl.where(kv_mask, qk, float("-inf"))

                m_new = tl.maximum(m_i, tl.max(qk, axis=0))
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(qk - m_new)
                l_i = l_i * alpha + tl.sum(p, axis=0)

                v_base = physical_block * (kv_num_heads * block_size * head_dim) + kv_head_id * (block_size * head_dim)
                v_ptrs = v_base + kv_range[:, None] * head_dim + offs_d[None, :]
                v_vals = tl.load(value_cache_ptr + v_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0).to(
                    tl.float32
                )

                acc = acc * alpha + tl.sum(p[:, None] * v_vals, axis=0)
                m_i = m_new

    if SINGLE_SPLIT:
        # Write final output directly (no stage2 needed)
        out_base = q_start * num_heads * head_dim + pid_head * head_dim
        tl.store(output_ptr + out_base + offs_d, acc / l_i, mask=d_mask)
    else:
        # Write partial output + LSE for stage2 merging
        out_base = (
            pid_batch * (num_heads * num_kv_splits * head_dim)
            + pid_head * (num_kv_splits * head_dim)
            + pid_split * head_dim
        )
        tl.store(partial_out_ptr + out_base + offs_d, acc / l_i, mask=d_mask)

        lse = m_i + tl.log(l_i)
        lse_base = pid_batch * (num_heads * num_kv_splits) + pid_head * num_kv_splits + pid_split
        tl.store(partial_lse_ptr + lse_base, lse)


@enable_compat_on_triton_kernel
@triton.jit
def v100_decode_attn_stage2(
    partial_out_ptr,  # [batch_size, num_heads, num_kv_splits, head_dim] float32
    partial_lse_ptr,  # [batch_size, num_heads, num_kv_splits] float32
    output_ptr,  # [num_tokens, num_heads, head_dim]
    q_start_loc_ptr,  # [batch_size] int32
    seq_lens_ptr,  # [batch_size] int32
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_kv_splits: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Stage 2: Merge partial outputs from all splits for each (batch, head)."""
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)

    total_kv_len = tl.load(seq_lens_ptr + pid_batch)
    if total_kv_len <= 0:
        return

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    # Find max LSE across splits
    max_lse = float("-inf")
    for s in range(num_kv_splits):
        lse_idx = pid_batch * (num_heads * num_kv_splits) + pid_head * num_kv_splits + s
        lse_val = tl.load(partial_lse_ptr + lse_idx)
        max_lse = tl.maximum(max_lse, lse_val)

    # Merge: weighted sum with LSE-based rescaling
    sum_exp = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for s in range(num_kv_splits):
        lse_idx = pid_batch * (num_heads * num_kv_splits) + pid_head * num_kv_splits + s
        lse_val = tl.load(partial_lse_ptr + lse_idx)

        # Guard against empty splits: lse=-inf means no valid KV tokens were processed.
        is_valid = lse_val > float("-inf")
        w = tl.where(is_valid, tl.exp(lse_val - max_lse), 0.0)
        sum_exp += w

        out_base = (
            pid_batch * (num_heads * num_kv_splits * head_dim) + pid_head * (num_kv_splits * head_dim) + s * head_dim
        )
        partial = tl.load(partial_out_ptr + out_base + offs_d, mask=d_mask & is_valid, other=0.0)
        acc += w * partial

    # Normalize
    acc = acc / sum_exp

    # Write final output
    q_start = tl.load(q_start_loc_ptr + pid_batch)
    out_base = q_start * num_heads * head_dim + pid_head * head_dim
    tl.store(output_ptr + out_base + offs_d, acc, mask=d_mask)


def v100_decode_fused(
    q,  # paddle.Tensor [num_tokens, num_heads, head_dim]
    k_new,  # paddle.Tensor [num_tokens, kv_num_heads, head_dim] - new K after RoPE
    v_new,  # paddle.Tensor [num_tokens, kv_num_heads, head_dim] - new V
    key_cache,  # paddle.Tensor [max_num_blocks, kv_num_heads, block_size, head_dim]
    value_cache,  # paddle.Tensor [max_num_blocks, kv_num_heads, block_size, head_dim]
    output,  # paddle.Tensor [num_tokens, num_heads, head_dim]
    block_tables,  # paddle.Tensor [batch_size, max_blocks_per_seq]
    seq_lens,  # paddle.Tensor [batch_size] int32 - total kv lengths
    positions,  # paddle.Tensor [num_tokens] int64
    batch_id_per_token,  # paddle.Tensor [num_tokens] int32
    q_start_locs,  # paddle.Tensor [batch_size] int32
    num_heads,
    kv_num_heads,
    head_dim,
    sm_scale,
    max_kv_len,
    partial_out=None,  # Optional pre-allocated buffer
    partial_lse=None,  # Optional pre-allocated buffer
    skip_kv_write=False,  # Skip KV write if already written by v100_rope_write_cache
):
    """KV write + decode attention. Write KV first, then fused stage1+stage2.

    When num_kv_splits=1: 2 kernels (write_kv + fused_stage1 that writes output directly).
    When num_kv_splits>1: 3 kernels (write_kv + fused_stage1 + stage2).
    """
    import paddle

    batch_size = seq_lens.shape[0]
    block_size = key_cache.shape[2]
    max_blocks_per_seq = block_tables.shape[1]
    group_size = num_heads // kv_num_heads

    BLOCK_D = triton.next_power_of_2(head_dim)

    max_kv_blocks = ceil_div(max_kv_len, block_size) if max_kv_len > 0 else 1
    num_kv_splits = min(max(1, ceil_div(max_kv_blocks, 8)), 32)
    MAX_BLOCKS_PER_SPLIT = ceil_div(max_kv_blocks, num_kv_splits) + 1
    single_split = num_kv_splits == 1

    # Step 1: Write KV to cache (skip if already written by v100_rope_write_cache)
    if not skip_kv_write:
        v100_write_kv_cache(
            k_new,
            v_new,
            key_cache,
            value_cache,
            block_tables,
            positions,
            batch_id_per_token,
        )

    # Step 2: Fused attention (writes output directly when single_split)
    if not single_split:
        if partial_out is None or partial_lse is None:
            partial_out = paddle.zeros([batch_size, num_heads, num_kv_splits, head_dim], dtype="float32")
            partial_lse = paddle.full([batch_size, num_heads, num_kv_splits], float("-inf"), dtype="float32")

    grid = (batch_size, num_heads, num_kv_splits)
    v100_decode_fused_kernel[grid](
        q_ptr=q,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        output_ptr=output,
        block_tables_ptr=block_tables,
        seq_lens_ptr=seq_lens,
        q_start_loc_ptr=q_start_locs,
        partial_out_ptr=partial_out if not single_split else output,  # dummy, unused
        partial_lse_ptr=partial_lse if not single_split else seq_lens,  # dummy, unused
        sm_scale=sm_scale,
        max_blocks_per_seq=max_blocks_per_seq,
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        group_size=group_size,
        head_dim=head_dim,
        block_size=block_size,
        num_kv_splits=num_kv_splits,
        MAX_BLOCKS_PER_SPLIT=MAX_BLOCKS_PER_SPLIT,
        BLOCK_D=BLOCK_D,
        SINGLE_SPLIT=single_split,
        num_warps=4,
    )

    if not single_split:
        # Stage 2: merge partials
        grid_s2 = (batch_size, num_heads)
        v100_decode_attn_stage2[grid_s2](
            partial_out_ptr=partial_out,
            partial_lse_ptr=partial_lse,
            output_ptr=output,
            q_start_loc_ptr=q_start_locs,
            seq_lens_ptr=seq_lens,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_splits=num_kv_splits,
            BLOCK_D=BLOCK_D,
            num_warps=2,
        )
