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

Triton kernels for V100 (SM70) attention backend.

Five kernels that replace Python for-loops with GPU-side computation:
1. v100_compute_positions_kernel  - compute per-token positions
2. v100_fused_rope_kernel         - fused RoPE application on Q and K
3. v100_write_kv_cache_kernel     - write K/V to block-based cache
4. v100_decode_attn_stage1/stage2 - 2-stage flash-decoding for decode
5. v100_extend_attention_kernel   - tiled flash attention for prefill
"""

import triton
import triton.language as tl

from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)
from fastdeploy.utils import ceil_div

# ---------------------------------------------------------------------------
# Kernel 1: Compute per-token positions
# ---------------------------------------------------------------------------


@enable_compat_on_triton_kernel
@triton.jit
def v100_compute_positions_kernel(
    positions_ptr,  # output: [num_tokens] int64
    batch_id_per_token_ptr,  # [num_tokens] int32
    cu_seqlens_q_ptr,  # [batch_size + 1] int32
    seq_lens_encoder_ptr,  # [batch_size] int32
    seq_lens_decoder_ptr,  # [batch_size] int32
    seq_lens_this_time_ptr,  # [batch_size] int32
    num_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_tokens

    batch_id = tl.load(batch_id_per_token_ptr + offs, mask=mask, other=0)

    # within-sequence offset = token_idx - cu_seqlens_q[batch_id]
    cu_start = tl.load(cu_seqlens_q_ptr + batch_id, mask=mask, other=0)
    within_seq_offset = offs - cu_start

    encoder_len = tl.load(seq_lens_encoder_ptr + batch_id, mask=mask, other=0)
    decoder_len = tl.load(seq_lens_decoder_ptr + batch_id, mask=mask, other=0)
    this_time_len = tl.load(seq_lens_this_time_ptr + batch_id, mask=mask, other=0)

    # is_prefill: this_time == encoder_len AND decoder_len == 0
    is_prefill = (this_time_len == encoder_len) & (decoder_len == 0)

    # prefill: pos = within_seq_offset
    # decode:  pos = encoder_len + decoder_len + within_seq_offset
    pos = tl.where(is_prefill, within_seq_offset, encoder_len + decoder_len + within_seq_offset)

    tl.store(positions_ptr + offs, pos.to(tl.int64), mask=mask)


def v100_compute_positions(
    batch_id_per_token,  # paddle.Tensor [num_tokens] int32
    cu_seqlens_q,  # paddle.Tensor [batch_size+1] int32
    seq_lens_encoder,  # paddle.Tensor [batch_size] int32
    seq_lens_decoder,  # paddle.Tensor [batch_size] int32
    seq_lens_this_time,  # paddle.Tensor [batch_size] int32
):
    """Compute per-token positions on GPU, replacing Python for-loop."""
    import paddle

    num_tokens = batch_id_per_token.shape[0]
    positions = paddle.empty([num_tokens], dtype="int64")
    BLOCK_SIZE = 1024
    grid = (ceil_div(num_tokens, BLOCK_SIZE),)
    v100_compute_positions_kernel[grid](
        positions_ptr=positions,
        batch_id_per_token_ptr=batch_id_per_token,
        cu_seqlens_q_ptr=cu_seqlens_q,
        seq_lens_encoder_ptr=seq_lens_encoder,
        seq_lens_decoder_ptr=seq_lens_decoder,
        seq_lens_this_time_ptr=seq_lens_this_time,
        num_tokens=num_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return positions


# ---------------------------------------------------------------------------
# Kernel 2: Fused RoPE on Q and K
# ---------------------------------------------------------------------------


@enable_compat_on_triton_kernel
@triton.jit
def v100_fused_rope_kernel(
    q_ptr,  # [num_tokens, num_heads, head_dim]  in-place
    k_ptr,  # [num_tokens, kv_num_heads, head_dim] in-place
    cos_ptr,  # [max_seq_len, rotary_dim]
    sin_ptr,  # [max_seq_len, rotary_dim]
    positions_ptr,  # [num_tokens] int64
    num_tokens,
    num_heads: tl.constexpr,
    kv_num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,  # head_dim for interleaved, head_dim or head_dim//2 for neox
    max_seq_len,
    USE_NEOX_STYLE: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    """Each program handles one (token, head_block) pair."""
    pid = tl.program_id(0)
    total_head_blocks = tl.cdiv(num_heads, BLOCK_HEAD)
    token_id = pid // total_head_blocks
    head_block = pid % total_head_blocks

    if token_id >= num_tokens:
        return

    head_ids = head_block * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    q_mask = head_ids < num_heads
    kv_mask = head_ids < kv_num_heads

    pos = tl.load(positions_ptr + token_id).to(tl.int64)
    half_dim: tl.constexpr = head_dim // 2

    if USE_NEOX_STYLE:
        # Neox style: split into first half and second half
        # cos/sin could be rotary_dim wide; we use first half_dim elements
        cos_base = pos * rotary_dim
        sin_base = pos * rotary_dim

        offs_half = tl.arange(0, half_dim)
        # For neox: if rotary_dim == head_dim, slice to first half_dim
        # If rotary_dim == half_dim, use directly
        cos_vals = tl.load(cos_ptr + cos_base + offs_half, mask=offs_half < rotary_dim).to(tl.float32)
        sin_vals = tl.load(sin_ptr + sin_base + offs_half, mask=offs_half < rotary_dim).to(tl.float32)

        # ---- Apply to Q ----
        q_row_base = token_id * num_heads * head_dim
        q_ptrs_first = q_ptr + q_row_base + head_ids[:, None] * head_dim + offs_half[None, :]
        q_ptrs_second = q_ptr + q_row_base + head_ids[:, None] * head_dim + (half_dim + offs_half[None, :])

        q1 = tl.load(q_ptrs_first, mask=q_mask[:, None], other=0.0).to(tl.float32)
        q2 = tl.load(q_ptrs_second, mask=q_mask[:, None], other=0.0).to(tl.float32)

        q1_new = q1 * cos_vals[None, :] - q2 * sin_vals[None, :]
        q2_new = q2 * cos_vals[None, :] + q1 * sin_vals[None, :]

        tl.store(q_ptrs_first, q1_new, mask=q_mask[:, None])
        tl.store(q_ptrs_second, q2_new, mask=q_mask[:, None])

        # ---- Apply to K ----
        k_row_base = token_id * kv_num_heads * head_dim
        k_ptrs_first = k_ptr + k_row_base + head_ids[:, None] * head_dim + offs_half[None, :]
        k_ptrs_second = k_ptr + k_row_base + head_ids[:, None] * head_dim + (half_dim + offs_half[None, :])

        k1 = tl.load(k_ptrs_first, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        k2 = tl.load(k_ptrs_second, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        k1_new = k1 * cos_vals[None, :] - k2 * sin_vals[None, :]
        k2_new = k2 * cos_vals[None, :] + k1 * sin_vals[None, :]

        tl.store(k_ptrs_first, k1_new, mask=kv_mask[:, None])
        tl.store(k_ptrs_second, k2_new, mask=kv_mask[:, None])
    else:
        # Interleaved style: even/odd pairs
        # rotary_embs shape gives cos/sin of size half_dim (head_dim//2)
        cos_base = pos * rotary_dim
        sin_base = pos * rotary_dim

        offs_half = tl.arange(0, half_dim)
        cos_vals = tl.load(cos_ptr + cos_base + offs_half).to(tl.float32)
        sin_vals = tl.load(sin_ptr + sin_base + offs_half).to(tl.float32)

        # Even indices: 0, 2, 4, ...  Odd indices: 1, 3, 5, ...
        offs_even = offs_half * 2  # [0, 2, 4, ...]
        offs_odd = offs_half * 2 + 1  # [1, 3, 5, ...]

        # ---- Apply to Q ----
        q_row_base = token_id * num_heads * head_dim
        q_even_ptrs = q_ptr + q_row_base + head_ids[:, None] * head_dim + offs_even[None, :]
        q_odd_ptrs = q_ptr + q_row_base + head_ids[:, None] * head_dim + offs_odd[None, :]

        q_even = tl.load(q_even_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
        q_odd = tl.load(q_odd_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        q_even_new = q_even * cos_vals[None, :] - q_odd * sin_vals[None, :]
        q_odd_new = q_odd * cos_vals[None, :] + q_even * sin_vals[None, :]

        tl.store(q_even_ptrs, q_even_new, mask=q_mask[:, None])
        tl.store(q_odd_ptrs, q_odd_new, mask=q_mask[:, None])

        # ---- Apply to K ----
        k_row_base = token_id * kv_num_heads * head_dim
        k_even_ptrs = k_ptr + k_row_base + head_ids[:, None] * head_dim + offs_even[None, :]
        k_odd_ptrs = k_ptr + k_row_base + head_ids[:, None] * head_dim + offs_odd[None, :]

        k_even = tl.load(k_even_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        k_odd = tl.load(k_odd_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        k_even_new = k_even * cos_vals[None, :] - k_odd * sin_vals[None, :]
        k_odd_new = k_odd * cos_vals[None, :] + k_even * sin_vals[None, :]

        tl.store(k_even_ptrs, k_even_new, mask=kv_mask[:, None])
        tl.store(k_odd_ptrs, k_odd_new, mask=kv_mask[:, None])


def v100_fused_rope(
    q,  # paddle.Tensor [num_tokens, num_heads, head_dim]   in-place
    k,  # paddle.Tensor [num_tokens, kv_num_heads, head_dim] in-place
    rotary_embs,  # paddle.Tensor [2, 1, max_seq_len, 1, rotary_dim]
    positions,  # paddle.Tensor [num_tokens] int64
    use_neox_style,  # bool
):
    """Apply RoPE to Q and K in-place using a fused Triton kernel."""
    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    kv_num_heads = k.shape[1]
    head_dim = q.shape[2]
    rotary_dim = rotary_embs.shape[-1]
    max_seq_len = rotary_embs.shape[2]

    # rotary_embs: [2, 1, max_seq_len, 1, rotary_dim]
    # flatten cos/sin to [max_seq_len, rotary_dim] for kernel access
    cos = rotary_embs[0, 0, :, 0, :]  # [max_seq_len, rotary_dim]
    sin = rotary_embs[1, 0, :, 0, :]  # [max_seq_len, rotary_dim]

    # Ensure contiguous
    cos = cos.contiguous()
    sin = sin.contiguous()

    BLOCK_HEAD = 4 if num_heads <= 32 else 8
    grid = (num_tokens * ceil_div(num_heads, BLOCK_HEAD),)

    v100_fused_rope_kernel[grid](
        q_ptr=q,
        k_ptr=k,
        cos_ptr=cos,
        sin_ptr=sin,
        positions_ptr=positions,
        num_tokens=num_tokens,
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_seq_len=max_seq_len,
        USE_NEOX_STYLE=use_neox_style,
        BLOCK_HEAD=BLOCK_HEAD,
        num_warps=2,
    )


# ---------------------------------------------------------------------------
# Kernel 3: Write KV to block cache
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
    tl.store(key_cache_ptr + cache_base + offs_d, k_vals, mask=d_mask)
    tl.store(value_cache_ptr + cache_base + offs_d, v_vals, mask=d_mask)


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
# Kernel 4: Decode Attention (2-stage flash-decoding)
# ---------------------------------------------------------------------------


@enable_compat_on_triton_kernel
@triton.jit
def v100_decode_attn_stage1(
    q_ptr,  # [num_tokens, num_heads, head_dim]
    key_cache_ptr,  # [max_num_blocks, kv_num_heads, block_size, head_dim]
    value_cache_ptr,  # [max_num_blocks, kv_num_heads, block_size, head_dim]
    block_tables_ptr,  # [batch_size, max_blocks_per_seq]
    seq_lens_ptr,  # [batch_size] int32 - total kv length per sequence
    q_start_loc_ptr,  # [batch_size] int32 - start token index of each batch in q
    partial_out_ptr,  # [batch_size, num_heads, num_kv_splits, head_dim] float32
    partial_lse_ptr,  # [batch_size, num_heads, num_kv_splits] float32
    sm_scale,
    max_blocks_per_seq,
    num_heads: tl.constexpr,
    kv_num_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    num_kv_splits: tl.constexpr,
    MAX_BLOCKS_PER_SPLIT: tl.constexpr,  # constexpr upper bound for loop
    BLOCK_D: tl.constexpr,
):
    """
    Stage 1: Each program handles (batch, head, kv_split).
    Computes partial attention output + LSE for a split of KV blocks.
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

    # Load Q for this token (decode => q_len = 1)
    q_start = tl.load(q_start_loc_ptr + pid_batch)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    q_base = q_start * num_heads * head_dim + pid_head * head_dim
    q_vec = tl.load(q_ptr + q_base + offs_d, mask=d_mask, other=0.0).to(tl.float32)

    # Online softmax state
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Iterate over KV blocks in this split
    # Use constexpr MAX_BLOCKS_PER_SPLIT as loop bound, with runtime early exit
    for bi in range(MAX_BLOCKS_PER_SPLIT):
        block_idx = split_start_block + bi
        if block_idx >= split_end_block:
            break

        physical_block = tl.load(block_tables_ptr + pid_batch * max_blocks_per_seq + block_idx)

        # Number of valid tokens in this block
        block_start_pos = block_idx * block_size
        valid_tokens = tl.minimum(block_size, total_kv_len - block_start_pos)

        # Process all tokens in this block at once (block_size is constexpr)
        # block_size is typically 64 or 128, fits in one tile for SM70
        kv_range = tl.arange(0, block_size)
        kv_mask = kv_range < valid_tokens

        # Load K: cache[physical_block, kv_head_id, :, :]
        # cache layout: [max_num_blocks, kv_num_heads, block_size, head_dim]
        k_base = physical_block * (kv_num_heads * block_size * head_dim) + kv_head_id * (block_size * head_dim)
        k_ptrs = k_base + kv_range[:, None] * head_dim + offs_d[None, :]
        k_vals = tl.load(key_cache_ptr + k_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # QK^T: [block_size]
        qk = tl.sum(q_vec[None, :] * k_vals, axis=1) * sm_scale
        qk = tl.where(kv_mask, qk, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)

        l_i = l_i * alpha + tl.sum(p, axis=0)

        # Load V
        v_base = physical_block * (kv_num_heads * block_size * head_dim) + kv_head_id * (block_size * head_dim)
        v_ptrs = v_base + kv_range[:, None] * head_dim + offs_d[None, :]
        v_vals = tl.load(value_cache_ptr + v_ptrs, mask=kv_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # Update accumulator: acc = acc * alpha + p @ V
        acc = acc * alpha + tl.sum(p[:, None] * v_vals, axis=0)

        m_i = m_new

    # Store partial output and LSE
    # partial_out: [batch_size, num_heads, num_kv_splits, head_dim]
    out_base = (
        pid_batch * (num_heads * num_kv_splits * head_dim)
        + pid_head * (num_kv_splits * head_dim)
        + pid_split * head_dim
    )
    tl.store(partial_out_ptr + out_base + offs_d, acc, mask=d_mask)

    # LSE = m_i + log(l_i)
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

        # Skip if this split had no valid KV (lse = -inf)
        w = tl.exp(lse_val - max_lse)
        sum_exp += w

        out_base = (
            pid_batch * (num_heads * num_kv_splits * head_dim) + pid_head * (num_kv_splits * head_dim) + s * head_dim
        )
        partial = tl.load(partial_out_ptr + out_base + offs_d, mask=d_mask, other=0.0)
        acc += w * partial

    # Normalize
    acc = acc / sum_exp

    # Write final output
    q_start = tl.load(q_start_loc_ptr + pid_batch)
    out_base = q_start * num_heads * head_dim + pid_head * head_dim
    tl.store(output_ptr + out_base + offs_d, acc, mask=d_mask)


def v100_decode_attention(
    q,  # paddle.Tensor [num_tokens, num_heads, head_dim]
    key_cache,  # paddle.Tensor [max_num_blocks, kv_num_heads, block_size, head_dim]
    value_cache,  # paddle.Tensor [max_num_blocks, kv_num_heads, block_size, head_dim]
    output,  # paddle.Tensor [num_tokens, num_heads, head_dim]
    block_tables,  # paddle.Tensor [batch_size, max_blocks_per_seq]
    seq_lens,  # paddle.Tensor [batch_size] int32 - total kv lengths
    q_start_locs,  # paddle.Tensor [batch_size] int32 - start token index for each batch in q
    num_heads,
    kv_num_heads,
    head_dim,
    sm_scale,
):
    """2-stage flash-decoding for decode tokens."""
    import paddle

    batch_size = seq_lens.shape[0]
    block_size = key_cache.shape[2]
    max_blocks_per_seq = block_tables.shape[1]
    group_size = num_heads // kv_num_heads

    BLOCK_D = triton.next_power_of_2(head_dim)

    # Determine number of KV splits based on max seq len
    max_kv_len = int(seq_lens.max().item()) if batch_size > 0 else 0
    max_kv_blocks = ceil_div(max_kv_len, block_size) if max_kv_len > 0 else 1
    # Heuristic: aim for ~8 blocks per split
    num_kv_splits = min(max(1, ceil_div(max_kv_blocks, 8)), 32)
    # Constexpr upper bound for blocks per split
    MAX_BLOCKS_PER_SPLIT = ceil_div(max_kv_blocks, num_kv_splits) + 1

    # Allocate partial buffers
    partial_out = paddle.empty([batch_size, num_heads, num_kv_splits, head_dim], dtype="float32")
    partial_lse = paddle.full([batch_size, num_heads, num_kv_splits], float("-inf"), dtype="float32")

    # Stage 1
    grid_s1 = (batch_size, num_heads, num_kv_splits)
    v100_decode_attn_stage1[grid_s1](
        q_ptr=q,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_tables,
        seq_lens_ptr=seq_lens,
        q_start_loc_ptr=q_start_locs,
        partial_out_ptr=partial_out,
        partial_lse_ptr=partial_lse,
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
        num_warps=4,
    )

    # Stage 2
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


# ---------------------------------------------------------------------------
# Kernel 5: Extend (prefill) attention — tiled flash attention from block cache
# ---------------------------------------------------------------------------


@enable_compat_on_triton_kernel
@triton.jit
def v100_extend_attention_kernel(
    q_ptr,  # [num_tokens, num_heads, head_dim]
    key_cache_ptr,  # [max_num_blocks, kv_num_heads, block_size, head_dim]
    value_cache_ptr,  # [max_num_blocks, kv_num_heads, block_size, head_dim]
    output_ptr,  # [num_tokens, num_heads, head_dim]
    block_tables_ptr,  # [batch_size, max_blocks_per_seq]
    q_start_loc_ptr,  # [batch_size] int32 - start of this batch's q tokens in q_ptr
    q_seq_lens_ptr,  # [batch_size] int32 - number of q tokens for this batch
    kv_seq_lens_ptr,  # [batch_size] int32 - total kv length
    sm_scale,
    max_blocks_per_seq,
    num_heads: tl.constexpr,
    kv_num_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    is_causal: tl.constexpr,
    MAX_KV_BLOCKS: tl.constexpr,  # constexpr upper bound for kv block iterations
    BLOCK_M: tl.constexpr,  # 64
    BLOCK_N: tl.constexpr,  # 64
    BLOCK_D: tl.constexpr,
):
    """
    Tiled flash attention for prefill, reading K/V from block cache.
    Grid: (ceil_div(q_len, BLOCK_M), batch_size, num_heads)

    Uses element-wise multiply + reduce instead of tl.dot for SM70
    compatibility with fp32 and non-power-of-2 head_dim.
    """
    pid_m = tl.program_id(0)  # query tile index
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)

    q_len = tl.load(q_seq_lens_ptr + pid_batch)
    kv_len = tl.load(kv_seq_lens_ptr + pid_batch)

    if q_len <= 0 or kv_len <= 0:
        return

    q_tile_start = pid_m * BLOCK_M
    if q_tile_start >= q_len:
        return

    kv_head_id = pid_head // group_size

    q_start = tl.load(q_start_loc_ptr + pid_batch)

    offs_m = q_tile_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    m_mask = offs_m < q_len

    # Load Q tile: [BLOCK_M, BLOCK_D]
    q_base = (q_start + offs_m[:, None]) * num_heads * head_dim + pid_head * head_dim + offs_d[None, :]
    q_tile = tl.load(q_ptr + q_base, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    # Online softmax state per query in tile
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Position of each query token in the full sequence (for causal masking)
    # For prefill: q positions are (kv_len - q_len) + offset within q
    q_pos_base = kv_len - q_len

    # Iterate over all KV positions in BLOCK_N chunks
    # Use constexpr upper bound: MAX_KV_BLOCKS cache blocks, each with block_size / BLOCK_N tiles
    TILES_PER_BLOCK: tl.constexpr = (block_size + BLOCK_N - 1) // BLOCK_N
    total_kv_iters: tl.constexpr = MAX_KV_BLOCKS * TILES_PER_BLOCK

    for kv_iter in range(total_kv_iters):
        kv_start = kv_iter * BLOCK_N
        if kv_start >= kv_len:
            break

        kv_range = kv_start + tl.arange(0, BLOCK_N)
        kv_valid = kv_range < kv_len

        # Map kv positions to block cache
        kv_block_idx = kv_range // block_size
        kv_block_offset = kv_range % block_size

        # Load physical block numbers
        bt_ptrs = block_tables_ptr + pid_batch * max_blocks_per_seq + kv_block_idx
        physical_blocks = tl.load(bt_ptrs, mask=kv_valid, other=0)

        # Load K: [BLOCK_N, BLOCK_D]
        k_base = (
            physical_blocks[:, None] * (kv_num_heads * block_size * head_dim)
            + kv_head_id * (block_size * head_dim)
            + kv_block_offset[:, None] * head_dim
            + offs_d[None, :]
        )
        k_vals = tl.load(key_cache_ptr + k_base, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # QK^T: [BLOCK_M, BLOCK_N] — use element-wise broadcast multiply + reduce for SM70
        # q_tile: [BLOCK_M, BLOCK_D], k_vals: [BLOCK_N, BLOCK_D]
        # qk[m, n] = sum_d(q_tile[m, d] * k_vals[n, d]) * sm_scale
        qk = tl.dot(q_tile, tl.trans(k_vals)) * sm_scale

        # Apply causal mask
        if is_causal:
            q_positions = q_pos_base + offs_m
            causal_mask = q_positions[:, None] >= kv_range[None, :]
            qk = tl.where(causal_mask & kv_valid[None, :], qk, float("-inf"))
        else:
            qk = tl.where(kv_valid[None, :], qk, float("-inf"))

        # Also mask out invalid query positions
        qk = tl.where(m_mask[:, None], qk, float("-inf"))

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Load V: [BLOCK_N, BLOCK_D]
        v_base = (
            physical_blocks[:, None] * (kv_num_heads * block_size * head_dim)
            + kv_head_id * (block_size * head_dim)
            + kv_block_offset[:, None] * head_dim
            + offs_d[None, :]
        )
        v_vals = tl.load(value_cache_ptr + v_base, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # Update accumulator: acc = acc * alpha + P @ V
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v_vals)

        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Write output
    out_base = (q_start + offs_m[:, None]) * num_heads * head_dim + pid_head * head_dim + offs_d[None, :]
    tl.store(output_ptr + out_base, acc, mask=m_mask[:, None] & d_mask[None, :])


def v100_extend_attention(
    q,  # paddle.Tensor [num_tokens, num_heads, head_dim]
    key_cache,  # paddle.Tensor [max_num_blocks, kv_num_heads, block_size, head_dim]
    value_cache,  # paddle.Tensor [max_num_blocks, kv_num_heads, block_size, head_dim]
    output,  # paddle.Tensor [num_tokens, num_heads, head_dim]
    block_tables,  # paddle.Tensor [batch_size, max_blocks_per_seq]
    q_start_locs,  # paddle.Tensor [batch_size] int32
    q_seq_lens,  # paddle.Tensor [batch_size] int32
    kv_seq_lens,  # paddle.Tensor [batch_size] int32
    num_heads,
    kv_num_heads,
    head_dim,
    sm_scale,
    is_causal=True,
):
    """Tiled flash attention for prefill sequences, reading from block cache."""
    batch_size = q_seq_lens.shape[0]
    block_size = key_cache.shape[2]
    max_blocks_per_seq = block_tables.shape[1]
    group_size = num_heads // kv_num_heads

    BLOCK_M = 64  # SM70 friendly
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(head_dim)

    max_q_len = int(q_seq_lens.max().item()) if batch_size > 0 else 0
    max_kv_len = int(kv_seq_lens.max().item()) if batch_size > 0 else 0
    num_m_blocks = ceil_div(max_q_len, BLOCK_M)
    # Constexpr upper bound for kv iteration
    MAX_KV_BLOCKS = ceil_div(max_kv_len, block_size) + 1

    grid = (num_m_blocks, batch_size, num_heads)
    v100_extend_attention_kernel[grid](
        q_ptr=q,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        output_ptr=output,
        block_tables_ptr=block_tables,
        q_start_loc_ptr=q_start_locs,
        q_seq_lens_ptr=q_seq_lens,
        kv_seq_lens_ptr=kv_seq_lens,
        sm_scale=sm_scale,
        max_blocks_per_seq=max_blocks_per_seq,
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        group_size=group_size,
        head_dim=head_dim,
        block_size=block_size,
        is_causal=is_causal,
        MAX_KV_BLOCKS=MAX_KV_BLOCKS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Unified paged attention dispatcher
# ---------------------------------------------------------------------------


def v100_paged_attention(
    q,  # [num_tokens, num_heads, head_dim]
    key_cache,  # [max_num_blocks, kv_num_heads, block_size, head_dim]
    value_cache,  # [max_num_blocks, kv_num_heads, block_size, head_dim]
    output,  # [num_tokens, num_heads, head_dim]  pre-allocated
    block_tables,  # [batch_size, max_blocks_per_seq]
    seq_lens_this_time,  # [batch_size] int32 - q tokens per batch
    total_seq_lens,  # [batch_size] int32 - total kv per batch
    cu_seqlens_q,  # [batch_size + 1] int32
    batch_id_per_token,  # [num_tokens] int32
    num_heads,
    kv_num_heads,
    head_dim,
    is_causal=True,
):
    """
    Dispatch attention to decode kernel or extend kernel based on
    whether each sequence is decode (q_len=1) or prefill (q_len>1).

    For simplicity and to avoid complex splitting, we check if the entire
    batch is decode-only or contains prefill. If mixed, we handle separately.
    """
    import paddle

    sm_scale = head_dim**-0.5
    batch_size = seq_lens_this_time.shape[0]
    num_tokens = q.shape[0]

    if num_tokens == 0 or batch_size == 0:
        return

    # Check if all sequences are decode (q_len = 1)
    max_q_len = int(seq_lens_this_time.max().item())
    # For min check, filter out zero-length sequences
    active_mask = seq_lens_this_time > 0
    min_q_len = int(seq_lens_this_time[active_mask].min().item()) if active_mask.any() else 0

    if max_q_len == 1 and num_tokens == batch_size:
        # Pure decode batch
        q_start_locs = cu_seqlens_q[:batch_size]
        v100_decode_attention(
            q,
            key_cache,
            value_cache,
            output,
            block_tables,
            total_seq_lens,
            q_start_locs,
            num_heads,
            kv_num_heads,
            head_dim,
            sm_scale,
        )
    elif max_q_len > 1 and min_q_len > 1:
        # Pure prefill batch
        q_start_locs = cu_seqlens_q[:batch_size]
        v100_extend_attention(
            q,
            key_cache,
            value_cache,
            output,
            block_tables,
            q_start_locs,
            seq_lens_this_time,
            total_seq_lens,
            num_heads,
            kv_num_heads,
            head_dim,
            sm_scale,
            is_causal,
        )
    else:
        # Mixed batch: separate decode and prefill sequences
        seq_lens_this_time_cpu = seq_lens_this_time.numpy()
        total_seq_lens_cpu = total_seq_lens.numpy()

        decode_batch_ids = []
        prefill_batch_ids = []
        for i in range(batch_size):
            if seq_lens_this_time_cpu[i] <= 0:
                continue
            elif seq_lens_this_time_cpu[i] == 1:
                decode_batch_ids.append(i)
            else:
                prefill_batch_ids.append(i)

        # Handle decode sequences
        if decode_batch_ids:
            decode_total_lens = paddle.to_tensor([total_seq_lens_cpu[i] for i in decode_batch_ids], dtype="int32")
            decode_block_tables = block_tables[decode_batch_ids]
            # Gather decode Q tokens
            decode_q_indices = paddle.to_tensor([int(cu_seqlens_q[i].item()) for i in decode_batch_ids], dtype="int64")
            decode_q = q[decode_q_indices]  # [num_decode, num_heads, head_dim]

            decode_out = paddle.empty_like(decode_q)
            decode_q_starts = paddle.arange(0, len(decode_batch_ids), dtype="int32")

            v100_decode_attention(
                decode_q,
                key_cache,
                value_cache,
                decode_out,
                decode_block_tables,
                decode_total_lens,
                decode_q_starts,
                num_heads,
                kv_num_heads,
                head_dim,
                sm_scale,
            )
            # Scatter back
            for idx, batch_id in enumerate(decode_batch_ids):
                token_idx = int(cu_seqlens_q[batch_id].item())
                output[token_idx] = decode_out[idx]

        # Handle prefill sequences
        if prefill_batch_ids:
            for batch_id in prefill_batch_ids:
                q_start = int(cu_seqlens_q[batch_id].item())
                q_len = int(seq_lens_this_time_cpu[batch_id])
                kv_len = int(total_seq_lens_cpu[batch_id])

                q_seq = q[q_start : q_start + q_len]  # [q_len, num_heads, head_dim]
                out_seq = paddle.empty_like(q_seq)

                q_start_loc = paddle.to_tensor([0], dtype="int32")
                q_seq_len = paddle.to_tensor([q_len], dtype="int32")
                kv_seq_len = paddle.to_tensor([kv_len], dtype="int32")
                bt = block_tables[batch_id : batch_id + 1]

                v100_extend_attention(
                    q_seq,
                    key_cache,
                    value_cache,
                    out_seq,
                    bt,
                    q_start_loc,
                    q_seq_len,
                    kv_seq_len,
                    num_heads,
                    kv_num_heads,
                    head_dim,
                    sm_scale,
                    is_causal,
                )
                output[q_start : q_start + q_len] = out_seq
