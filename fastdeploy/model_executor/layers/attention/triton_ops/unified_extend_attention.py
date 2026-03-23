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

# Adapted from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/extend_attention.py
# Licensed under Apache License 2.0
#
# Modified by FastDeploy team for deterministic mode support with prefix caching.
# Key adaptation: FastDeploy uses paged KV cache [num_blocks, kv_heads, block_size, head_dim],
# while SGLang uses flat KV cache [total_tokens, kv_heads, head_dim].
"""

import paddle
import triton
import triton.language as tl

from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)

# ---------------------------------------------------------------------------
# Triton cumsum (CUDA Graph compatible, replaces paddle.cumsum / thrust)
# ---------------------------------------------------------------------------


@enable_compat_on_triton_kernel
@triton.jit  # pragma: no cover
def _cumsum_with_zero_prefix_kernel(input_ptr, output_ptr, n, BLOCK: tl.constexpr):
    """
    output[0] = 0, output[1:n+1] = cumsum(input[0:n]).
    Single program, handles n <= BLOCK.
    """
    tl.store(output_ptr, 0)
    idx = tl.arange(0, BLOCK)
    mask = idx < n
    val = tl.load(input_ptr + idx, mask=mask, other=0).to(tl.int32)
    cumval = tl.cumsum(val, axis=0)
    tl.store(output_ptr + 1 + idx, cumval, mask=mask)


def triton_cumsum_with_zero_prefix(x, n=None, out_buf=None):
    """
    Triton replacement for: paddle.concat([zeros([1], int32), cumsum(x).astype(int32)])
    Returns int32 tensor of shape [n+1].  CUDA Graph capture compatible.

    Args:
        out_buf: Optional pre-allocated buffer of size >= [n+1].
                 If provided, writes into out_buf[:n+1] instead of allocating.
    """
    if n is None:
        n = x.shape[0]
    out = out_buf[: n + 1] if out_buf is not None else paddle.empty([n + 1], dtype="int32")
    if n == 0:
        out[0:1] = paddle.zeros([1], dtype="int32")
        return out
    BLOCK = triton.next_power_of_2(n)
    _cumsum_with_zero_prefix_kernel[(1,)](x, out, n, BLOCK=BLOCK)
    return out


# ---------------------------------------------------------------------------
# Triton helper kernels (allocation-free elementwise ops for CUDA Graph)
# ---------------------------------------------------------------------------


@enable_compat_on_triton_kernel
@triton.jit  # pragma: no cover
def _indptr_to_lens_kernel(indptr_ptr, lens_ptr, n, BLOCK: tl.constexpr):
    """Compute lens[i] = indptr[i+1] - indptr[i] for i in [0, n)."""
    idx = tl.arange(0, BLOCK)
    mask = idx < n
    a = tl.load(indptr_ptr + idx + 1, mask=mask, other=0)
    b = tl.load(indptr_ptr + idx, mask=mask, other=0)
    tl.store(lens_ptr + idx, a - b, mask=mask)


@enable_compat_on_triton_kernel
@triton.jit  # pragma: no cover
def _elementwise_add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Compute out[i] = a[i] + b[i] for i in [0, n)."""
    idx = tl.arange(0, BLOCK)
    mask = idx < n
    a = tl.load(a_ptr + idx, mask=mask, other=0)
    b = tl.load(b_ptr + idx, mask=mask, other=0)
    tl.store(out_ptr + idx, a + b, mask=mask)


# ---------------------------------------------------------------------------
# Index building utilities
# ---------------------------------------------------------------------------


@enable_compat_on_triton_kernel
@triton.jit  # pragma: no cover
def _copy_unified_indices_kernel(
    prefix_kv_indptr,
    prefix_kv_indices,
    extend_start_loc,
    extend_seq_lens,
    extend_kv_indices,
    unified_kv_indptr,
    unified_kv_indices,
    bs,
):
    """
    Copy prefix and extend KV indices into a unified buffer.
    One program per sequence, internal loops for vectorized copy.
    """
    pid = tl.program_id(0)
    if pid >= bs:
        return

    prefix_start = tl.load(prefix_kv_indptr + pid)
    prefix_end = tl.load(prefix_kv_indptr + pid + 1)
    extend_start = tl.load(extend_start_loc + pid)
    extend_len = tl.load(extend_seq_lens + pid)

    prefix_len = prefix_end - prefix_start
    unified_start = tl.load(unified_kv_indptr + pid)

    BLOCK_SIZE: tl.constexpr = 128

    # Copy prefix indices
    for block_start in range(0, prefix_len, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < prefix_len
        vals = tl.load(prefix_kv_indices + prefix_start + offs, mask=mask, other=0)
        tl.store(unified_kv_indices + unified_start + offs, vals, mask=mask)

    # Copy extend indices
    for block_start in range(0, extend_len, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < extend_len
        vals = tl.load(extend_kv_indices + extend_start + offs, mask=mask, other=0)
        tl.store(unified_kv_indices + unified_start + prefix_len + offs, vals, mask=mask)


def build_unified_kv_indices(
    prefix_kv_indptr,
    prefix_kv_indices,
    extend_start_loc,
    extend_seq_lens,
    extend_kv_indices,
    bs,
    unified_kv_indptr_buf=None,
    unified_kv_indices_buf=None,
    prefix_lens_buf=None,
    unified_lens_buf=None,
):
    """
    Build unified KV indices from prefix and extend parts.
    Uses Triton cumsum (CUDA Graph compatible) for indptr, Triton kernel for index copy.

    Optional *_buf args: pre-allocated buffers for CUDA Graph compatibility.
    When None (default), tensors are allocated dynamically (backward-compatible).

    Returns:
        (unified_kv_indptr, unified_kv_indices, prefix_lens)
    """
    if prefix_lens_buf is not None and bs > 0:
        prefix_lens = prefix_lens_buf[:bs]
        BLOCK = triton.next_power_of_2(bs)
        _indptr_to_lens_kernel[(1,)](prefix_kv_indptr, prefix_lens, bs, BLOCK=BLOCK)
    else:
        prefix_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]

    if unified_lens_buf is not None and bs > 0:
        unified_lens = unified_lens_buf[:bs]
        BLOCK = triton.next_power_of_2(bs)
        _elementwise_add_kernel[(1,)](prefix_lens, extend_seq_lens, unified_lens, bs, BLOCK=BLOCK)
    else:
        unified_lens = prefix_lens + extend_seq_lens[:bs]

    unified_kv_indptr = triton_cumsum_with_zero_prefix(unified_lens, bs, out_buf=unified_kv_indptr_buf)

    total_len = prefix_kv_indices.shape[0] + extend_kv_indices.shape[0]
    if unified_kv_indices_buf is not None:
        unified_kv_indices = unified_kv_indices_buf[:total_len]
    else:
        unified_kv_indices = paddle.empty([total_len], dtype="int32")

    _copy_unified_indices_kernel[(bs,)](
        prefix_kv_indptr,
        prefix_kv_indices,
        extend_start_loc,
        extend_seq_lens,
        extend_kv_indices,
        unified_kv_indptr,
        unified_kv_indices,
        bs,
    )

    return unified_kv_indptr, unified_kv_indices, prefix_lens


@enable_compat_on_triton_kernel
@triton.jit  # pragma: no cover
def _build_kv_indices_kernel(
    block_tables_ptr,
    seq_lens_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    block_size: tl.constexpr,
    max_blocks_per_seq,
    BLOCK: tl.constexpr,
):
    """
    Build flat token-level KV indices from block_tables.
    One program per sequence.
    For token at position t in sequence s:
      physical_index = block_tables[s, t // block_size] * block_size + t % block_size
    """
    seq_id = tl.program_id(0)
    slen = tl.load(seq_lens_ptr + seq_id)
    dst_start = tl.load(kv_indptr_ptr + seq_id)

    row_base = seq_id * max_blocks_per_seq

    for off in range(0, slen, BLOCK):
        t = off + tl.arange(0, BLOCK)
        mask = t < slen
        block_idx = t // block_size
        in_block_off = t % block_size
        blk_id = tl.load(block_tables_ptr + row_base + block_idx, mask=mask, other=0)
        idx = blk_id * block_size + in_block_off
        tl.store(kv_indices_ptr + dst_start + t, idx, mask=mask)


@enable_compat_on_triton_kernel
@triton.jit  # pragma: no cover
def _scatter_extend_kv_indices_kernel(
    all_kv_indices_ptr,
    all_kv_indptr_ptr,
    prefix_lens_ptr,
    extend_start_loc_ptr,
    extend_seq_lens_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    """
    Scatter extend KV indices from all_kv_indices into a contiguous extend buffer.
    One program per sequence. Copies the extend portion (skipping prefix) of each sequence.
    """
    seq_id = tl.program_id(0)
    elen = tl.load(extend_seq_lens_ptr + seq_id)
    plen = tl.load(prefix_lens_ptr + seq_id)
    src_start = tl.load(all_kv_indptr_ptr + seq_id) + plen
    dst_start = tl.load(extend_start_loc_ptr + seq_id)

    for off in range(0, elen, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < elen
        val = tl.load(all_kv_indices_ptr + src_start + idx, mask=mask, other=0)
        tl.store(out_ptr + dst_start + idx, val, mask=mask)


def build_kv_indices_from_block_tables(
    block_tables, seq_lens, block_size, bs, total_kv_len=None, kv_indptr_buf=None, kv_indices_buf=None
):
    """
    Convert FastDeploy's block_tables to flat token-level KV indices.
    CUDA Graph capture compatible (uses Triton cumsum, no thrust).

    Optional *_buf args: pre-allocated buffers for CUDA Graph compatibility.
    """
    kv_indptr = triton_cumsum_with_zero_prefix(seq_lens[:bs], bs, out_buf=kv_indptr_buf)
    if total_kv_len is None:
        total_kv_len = int(paddle.sum(seq_lens[:bs]).item())
    if kv_indices_buf is not None:
        kv_indices = kv_indices_buf[: max(total_kv_len, 1)]
    else:
        kv_indices = paddle.empty([max(total_kv_len, 1)], dtype="int32")

    if bs > 0 and total_kv_len > 0:
        max_blocks_per_seq = block_tables.shape[1]
        _build_kv_indices_kernel[(bs,)](
            block_tables,
            seq_lens,
            kv_indptr,
            kv_indices,
            block_size,
            max_blocks_per_seq,
            BLOCK=128,
        )

    return kv_indptr, kv_indices


def build_kv_indices_from_block_tables_ref(block_tables, seq_lens, block_size, bs):
    """
    Reference (Python for-loop) implementation of build_kv_indices_from_block_tables.
    Uses .item() for GPU→CPU transfers — NOT compatible with CUDA Graph capture.
    Kept for correctness validation against the Triton version.
    """
    kv_indptr = paddle.concat(
        [
            paddle.zeros([1], dtype="int32"),
            paddle.cumsum(seq_lens[:bs]).astype("int32"),
        ]
    )
    total_kv_len = int(paddle.sum(seq_lens[:bs]).item())
    kv_indices = paddle.empty([max(total_kv_len, 1)], dtype="int32")

    for s in range(bs):
        slen = int(seq_lens[s].item())
        if slen == 0:
            continue
        start = int(kv_indptr[s].item())
        positions = paddle.arange(slen, dtype="int32")
        block_ids = block_tables[s, positions // block_size]
        offsets = positions % block_size
        kv_indices[start : start + slen] = block_ids * block_size + offsets

    return kv_indptr, kv_indices


# ---------------------------------------------------------------------------
# Triton pre_cache_len_concat (CUDA Graph compatible, GPU-only)
# ---------------------------------------------------------------------------


@enable_compat_on_triton_kernel
@triton.jit  # pragma: no cover
def _pre_cache_cu_seqlens_kernel(
    seq_lens_encoder_ptr,
    seq_lens_decoder_ptr,
    seq_lens_this_time_ptr,
    cu_seqlens_k_ptr,
    cache_len_ptr,
    loop_times_ptr,
    bsz,
    block_size: tl.constexpr,
    BLOCK_BSZ: tl.constexpr,
):
    """
    Compute cu_seqlens_k, cache_len, and loop_times per batch.
    Single program, vectorized over bsz.
    """
    tl.store(cu_seqlens_k_ptr, 0)
    bid = tl.arange(0, BLOCK_BSZ)
    mask = bid < bsz

    enc = tl.load(seq_lens_encoder_ptr + bid, mask=mask, other=0)
    dec = tl.load(seq_lens_decoder_ptr + bid, mask=mask, other=0)
    qlen = tl.load(seq_lens_this_time_ptr + bid, mask=mask, other=0)

    cache_len = tl.where(enc > 0, dec, 0)
    total_per_batch = cache_len + qlen

    cu_cumsum = tl.cumsum(total_per_batch, axis=0)
    tl.store(cu_seqlens_k_ptr + 1 + bid, cu_cumsum, mask=mask)
    tl.store(cache_len_ptr + bid, cache_len, mask=mask)

    lt = (cache_len + block_size - 1) // block_size
    tl.store(loop_times_ptr + bid, lt, mask=mask)


@enable_compat_on_triton_kernel
@triton.jit  # pragma: no cover
def _pre_cache_scatter_kernel(
    loop_times_ptr,
    gridx_offset_ptr,
    batch_ids_ptr,
    tile_ids_per_batch_ptr,
    BLOCK: tl.constexpr,
):
    """
    Scatter batch_ids and tile_ids for one batch.
    One program per batch (grid = bsz).
    """
    bid = tl.program_id(0)
    lt = tl.load(loop_times_ptr + bid)
    offset = tl.load(gridx_offset_ptr + bid)

    for off in range(0, lt, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < lt
        tl.store(batch_ids_ptr + offset + idx, bid, mask=m)
        tl.store(tile_ids_per_batch_ptr + offset + idx, idx, mask=m)


def pre_cache_len_concat_triton(
    seq_lens_encoder,
    seq_lens_decoder,
    seq_lens_this_time,
    bsz,
    block_size,
    max_tile_size_per_bs,
    cu_seqlens_k_buf=None,
    batch_ids_buf=None,
    tile_ids_buf=None,
    cache_len_buf=None,
    loop_times_buf=None,
    gridx_offset_buf=None,
):
    """
    GPU-only Triton replacement for pre_cache_len_concat C++ op.
    No D2H copy — CUDA Graph capture compatible.

    Two-phase approach:
      Phase 1: Vectorized kernel computes cu_seqlens_k, cache_len, loop_times
      Phase 2: Per-batch scatter kernel writes batch_ids and tile_ids

    Optional *_buf args: pre-allocated buffers for CUDA Graph compatibility.

    Returns:
        cu_seqlens_k: [bsz+1] int32, GPU
        batch_ids: [bsz * max_tile_size_per_bs] int32, GPU
        tile_ids_per_batch: [bsz * max_tile_size_per_bs] int32, GPU
    """
    cu_seqlens_k = (
        cu_seqlens_k_buf[: bsz + 1] if cu_seqlens_k_buf is not None else paddle.empty([bsz + 1], dtype="int32")
    )
    out_size = max(bsz * max_tile_size_per_bs, 1)
    batch_ids = batch_ids_buf[:out_size] if batch_ids_buf is not None else paddle.empty([out_size], dtype="int32")
    tile_ids = tile_ids_buf[:out_size] if tile_ids_buf is not None else paddle.empty([out_size], dtype="int32")

    if bsz > 0:
        BLOCK_BSZ = triton.next_power_of_2(bsz)
        _cache_len_buf = cache_len_buf[:bsz] if cache_len_buf is not None else paddle.empty([bsz], dtype="int32")
        _loop_times_buf = loop_times_buf[:bsz] if loop_times_buf is not None else paddle.empty([bsz], dtype="int32")

        # Phase 1: compute cu_seqlens_k, cache_len, loop_times
        _pre_cache_cu_seqlens_kernel[(1,)](
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            cu_seqlens_k,
            _cache_len_buf,
            _loop_times_buf,
            bsz=bsz,
            block_size=block_size,
            BLOCK_BSZ=BLOCK_BSZ,
        )

        # Phase 2: compute gridx_offset (exclusive prefix sum) and scatter
        gridx_offset = triton_cumsum_with_zero_prefix(_loop_times_buf, bsz, out_buf=gridx_offset_buf)
        _pre_cache_scatter_kernel[(bsz,)](
            _loop_times_buf,
            gridx_offset,
            batch_ids,
            tile_ids,
            BLOCK=128,
        )
    else:
        cu_seqlens_k[0:1] = paddle.zeros([1], dtype="int32")

    return cu_seqlens_k, batch_ids, tile_ids


def pre_cache_len_concat_ref(
    seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, bsz, block_size, max_tile_size_per_bs
):
    """
    Reference (Python for-loop) implementation of pre_cache_len_concat.
    Uses .item() — NOT CUDA Graph compatible. Kept for validation.
    """
    cu_seqlens_k = paddle.zeros([bsz + 1], dtype="int32")
    batch_ids = paddle.empty([max(bsz * max_tile_size_per_bs, 1)], dtype="int32")
    tile_ids = paddle.empty([max(bsz * max_tile_size_per_bs, 1)], dtype="int32")

    gridx = 0
    total_tokens = 0
    for bid in range(bsz):
        enc = int(seq_lens_encoder[bid].item())
        dec = int(seq_lens_decoder[bid].item())
        qlen = int(seq_lens_this_time[bid].item())
        cache_len = dec if enc > 0 else 0
        loop_times = (cache_len + block_size - 1) // block_size
        for tile_id in range(loop_times):
            batch_ids[gridx] = bid
            tile_ids[gridx] = tile_id
            gridx += 1
        total_tokens += cache_len + qlen
        cu_seqlens_k[bid + 1] = total_tokens

    return cu_seqlens_k, batch_ids, tile_ids


# ---------------------------------------------------------------------------
# Triton attention kernel (unified, deterministic)
# ---------------------------------------------------------------------------


@enable_compat_on_triton_kernel
@triton.jit  # pragma: no cover
def _fwd_kernel_unified(
    Q,
    O,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    prefix_lens,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_obs,
    stride_oh,
    # K_Buffer strides: [num_blocks, kv_heads, block_size, head_dim]
    stride_kb,  # dim0: block
    stride_kh,  # dim1: head
    stride_kt,  # dim2: token offset in block
    # V_Buffer strides: [num_blocks, kv_heads, block_size, head_dim]
    stride_vb,
    stride_vh,
    stride_vt,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Unified 1-stage extend attention kernel for deterministic inference.
    Both prefix and extend KV are accessed through unified kv_indices,
    ensuring identical accumulation order regardless of cache hit/miss.
    """
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    # Load sequence metadata
    cur_seq_q_start = tl.load(qo_indptr + cur_seq)
    cur_seq_q_len = tl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start
    cur_seq_kv_start = tl.load(kv_indptr + cur_seq)
    cur_seq_kv_len = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start
    cur_seq_prefix_len = tl.load(prefix_lens + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_q_len
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    # Load Q block: Q shape is [num_tokens, num_heads, head_dim]
    offs_q = (
        (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q + offs_q, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Initialize online softmax accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # Unified loop over all KV (prefix + extend)
    for start_n in range(0, cur_seq_kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_kv_len

        # Build mask: bounds + causal
        final_mask = mask_m[:, None] & mask_n[None, :]

        if IS_CAUSAL:
            # Prefix tokens: always visible (no causal mask)
            # Extend tokens: apply standard causal mask
            q_idx = cur_block_m * BLOCK_M + offs_m[:, None]
            k_idx_in_total = start_n + offs_n[None, :]
            k_is_extend = k_idx_in_total >= cur_seq_prefix_len
            k_idx_in_extend = k_idx_in_total - cur_seq_prefix_len
            causal_mask = tl.where(k_is_extend, q_idx >= k_idx_in_extend, True)
            final_mask &= causal_mask

        # Load KV indices (flat token indices: block_id * block_size + offset)
        offs_kv_loc = tl.load(
            kv_indices + cur_seq_kv_start + start_n + offs_n,
            mask=mask_n,
            other=0,
        )

        # Decompose flat index into (block_id, offset_in_block)
        kv_block_ids = offs_kv_loc // KV_BLOCK_SIZE
        kv_offsets = offs_kv_loc % KV_BLOCK_SIZE

        # Load K: cache shape [num_blocks, kv_heads, block_size, head_dim]
        # addr = block_id * stride_kb + head * stride_kh + offset * stride_kt + d
        offs_buf_k = (
            kv_block_ids[None, :] * stride_kb
            + cur_kv_head * stride_kh
            + kv_offsets[None, :] * stride_kt
            + offs_d[:, None]
        )
        k = tl.load(K_Buffer + offs_buf_k, mask=mask_n[None, :] & mask_d[:, None], other=0.0)

        # QK = Q @ K^T, shape [BLOCK_M, BLOCK_N]
        qk = tl.dot(q.to(k.dtype), k) * sm_scale
        qk = tl.where(final_mask, qk, float("-inf"))

        # Online softmax update
        row_max = tl.max(qk, 1)
        # Avoid -inf in exp: clamp to a large negative value
        row_max_safe = tl.where(row_max == float("-inf"), -1e20, row_max)
        m_new = tl.maximum(m_i, row_max_safe)
        re_scale = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * re_scale + tl.sum(p, 1)

        # Load V: same 4D layout as K
        offs_buf_v = (
            kv_block_ids[:, None] * stride_vb
            + cur_kv_head * stride_vh
            + kv_offsets[:, None] * stride_vt
            + offs_dv[None, :]
        )
        v = tl.load(V_Buffer + offs_buf_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0)

        # Accumulate: rescale old acc, add new P @ V
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)
        m_i = m_new

    # Final output = acc / l_i
    offs_o = (
        (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    # Avoid division by zero for fully masked rows
    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    tl.store(O + offs_o, acc / safe_l[:, None], mask=mask_m[:, None] & mask_dv[None, :])


def extend_attention_fwd_unified(
    q,
    o,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    prefix_lens,
    num_q_heads,
    num_kv_heads,
    head_dim,
    max_len_extend,
    is_causal=True,
    sm_scale=None,
):
    """
    Launch the unified extend attention kernel.

    Args:
        q: [num_tokens, num_q_heads, head_dim]
        o: [num_tokens, num_q_heads, head_dim] (output, will be written)
        k_buffer: KV cache key buffer [num_blocks, kv_heads, block_size, head_dim]
        v_buffer: KV cache value buffer [num_blocks, kv_heads, block_size, head_dim]
        qo_indptr: [bs+1] query/output CSR indptr
        kv_indptr: [bs+1] unified KV CSR indptr
        kv_indices: [total_kv_len] flat token indices into paged cache
        prefix_lens: [bs] prefix length per sequence
        num_q_heads: number of query heads
        num_kv_heads: number of KV heads
        head_dim: head dimension
        max_len_extend: max extend length (for grid sizing)
        is_causal: whether to apply causal mask
        sm_scale: softmax scale, defaults to 1/sqrt(head_dim)
    """
    Lq = head_dim
    Lv = head_dim
    BLOCK_DMODEL = triton.next_power_of_2(Lq)
    BLOCK_DV = triton.next_power_of_2(Lv)

    # Choose block sizes based on head_dim
    if Lq <= 128:
        BLOCK_M, BLOCK_N = 64, 128
    elif Lq <= 256:
        BLOCK_M, BLOCK_N = 64, 64
    else:
        BLOCK_M, BLOCK_N = 32, 32
    num_warps = 4 if Lq <= 64 else 8

    sm_scale = sm_scale or (1.0 / (Lq**0.5))
    batch_size = qo_indptr.shape[0] - 1
    kv_group_num = num_q_heads // num_kv_heads

    # KV cache block_size: k_buffer shape is [num_blocks, kv_heads, block_size, head_dim]
    kv_block_size = k_buffer.shape[2]

    grid = (batch_size, num_q_heads, triton.cdiv(max_len_extend, BLOCK_M))

    _fwd_kernel_unified[grid](
        q,
        o,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        sm_scale,
        kv_group_num,
        q.strides[0],
        q.strides[1],
        o.strides[0],
        o.strides[1],
        # K strides: [num_blocks, kv_heads, block_size, head_dim]
        k_buffer.strides[0],
        k_buffer.strides[1],
        k_buffer.strides[2],
        # V strides: [num_blocks, kv_heads, block_size, head_dim]
        v_buffer.strides[0],
        v_buffer.strides[1],
        v_buffer.strides[2],
        Lq=Lq,
        Lv=Lv,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        KV_BLOCK_SIZE=kv_block_size,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=1,
    )

    return o
