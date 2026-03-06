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

# ---------------------------------------------------------------------------
# Index building utilities
# ---------------------------------------------------------------------------


@triton.jit
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
):
    """
    Build unified KV indices from prefix and extend parts.
    Uses paddle.cumsum for indptr (host-side), Triton kernel for index copy.

    Returns:
        (unified_kv_indptr, unified_kv_indices, prefix_lens)
    """
    prefix_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]
    unified_lens = prefix_lens + extend_seq_lens[:bs]
    unified_kv_indptr = paddle.concat(
        [
            paddle.zeros([1], dtype="int32"),
            paddle.cumsum(unified_lens).astype("int32"),
        ]
    )

    total_len = prefix_kv_indices.shape[0] + extend_kv_indices.shape[0]
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


def build_kv_indices_from_block_tables(block_tables, seq_lens, block_size, bs):
    """
    Convert FastDeploy's block_tables to flat token-level KV indices.

    Args:
        block_tables: [bs, max_blocks_per_seq], maps (seq, block_idx) -> physical_block_id
        seq_lens: [bs], total KV length per sequence (prefix + extend)
        block_size: tokens per block
        bs: batch size

    Returns:
        kv_indptr: [bs+1] int32, CSR indptr
        kv_indices: [total_kv_len] int32, flat token indices into paged cache
    """
    kv_indptr = paddle.concat(
        [
            paddle.zeros([1], dtype="int32"),
            paddle.cumsum(seq_lens[:bs]).astype("int32"),
        ]
    )
    total_kv_len = int(paddle.sum(seq_lens[:bs]).item())
    kv_indices = paddle.empty([max(total_kv_len, 1)], dtype="int32")

    # Build flat indices: for token at position t in sequence s,
    # its physical location = block_tables[s, t // block_size] * block_size + t % block_size
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
# Triton attention kernel (unified, deterministic)
# ---------------------------------------------------------------------------


@triton.jit
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
