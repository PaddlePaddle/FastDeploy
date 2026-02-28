# Copyright 2023-2024 SGLang Team
# Copyright 2025 PaddlePaddle Authors. All Rights Reserved.
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
Adapted from SGLang extend_attention.py:
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/extend_attention.py

Modifications by FastDeploy team:
- Adapted for PaddlePaddle framework
- Modified for deterministic mode with prefix caching support
- Simplified to focus on unified 1-stage extend attention kernel

Unified 1-stage extend attention for deterministic inference with prefix caching.

Key idea: Build unified KV indices that combine prefix KV and extend KV,
then process all KV in a single kernel pass. This ensures deterministic
behavior regardless of cache hit/miss status.
"""

import paddle
import triton
import triton.language as tl


def _get_device_capability():
    """Get CUDA device capability."""
    if paddle.device.is_compiled_with_cuda():
        prop = paddle.device.cuda.get_device_properties()
        return (prop.major, prop.minor)
    return (0, 0)


def _get_block_sizes_for_extend_attention(Lq: int, Lv: int):
    """
    Get block sizes and configuration for extend attention kernels.

    Args:
        Lq: Query head dimension
        Lv: Value head dimension

    Returns:
        tuple: (BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps)
    """
    # Determine BLOCK_DMODEL and BLOCK_DPE based on head dimension
    if Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lq == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    elif Lq == 192:
        BLOCK_DMODEL = 128
        BLOCK_DPE = 64
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lq)
        BLOCK_DPE = 0

    BLOCK_DV = triton.next_power_of_2(Lv)

    # Determine BLOCK_M, BLOCK_N, and num_warps based on hardware
    capability = _get_device_capability()

    if capability[0] == 12:
        # sm120 workstation Blackwell architecture
        if Lq <= 128:
            BLOCK_M, BLOCK_N = (64, 128)
        elif Lq <= 256:
            BLOCK_M, BLOCK_N = (64, 64)
        else:
            BLOCK_M, BLOCK_N = (32, 32)
    elif capability[0] >= 9:
        # Hopper architecture (H100, etc.)
        if Lq <= 256:
            BLOCK_M, BLOCK_N = (128, 64)
        else:
            BLOCK_M, BLOCK_N = (32, 64)
    elif capability[0] >= 8:
        # Ampere architecture (A100, etc.)
        if capability[1] == 9 or capability[1] == 6:
            # sm86/sm89 has smaller shared memory
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (64, 128)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 32)
        else:
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (128, 128)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
    else:
        # Older architectures
        BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 32)

    num_warps = 4 if Lq <= 64 else 8

    return BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps


@triton.jit
def _copy_unified_indices_kernel(
    # Input buffers
    prefix_kv_indptr,
    prefix_kv_indices,
    extend_start_loc,
    extend_seq_lens,
    extend_kv_indices,
    unified_kv_indptr,
    # Output buffer
    unified_kv_indices,
    # Size
    bs,
):
    """
    Triton kernel to copy indices to unified buffer (parallel per sequence).
    Each thread block processes one sequence with vectorized loads/stores.
    """
    pid = tl.program_id(0)

    if pid >= bs:
        return

    # Load sequence info
    prefix_start = tl.load(prefix_kv_indptr + pid)
    prefix_end = tl.load(prefix_kv_indptr + pid + 1)
    extend_start = tl.load(extend_start_loc + pid)
    extend_len = tl.load(extend_seq_lens + pid)

    prefix_len = prefix_end - prefix_start
    unified_start = tl.load(unified_kv_indptr + pid)

    # Copy indices in vectorized chunks
    BLOCK_SIZE: tl.constexpr = 128

    # Process prefix indices
    for block_start in range(0, prefix_len, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < prefix_len

        src_idx = prefix_start + offs
        dst_idx = unified_start + offs

        vals = tl.load(prefix_kv_indices + src_idx, mask=mask, other=0)
        tl.store(unified_kv_indices + dst_idx, vals, mask=mask)

    # Process extend indices
    for block_start in range(0, extend_len, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < extend_len

        src_idx = extend_start + offs
        dst_idx = unified_start + prefix_len + offs

        vals = tl.load(extend_kv_indices + src_idx, mask=mask, other=0)
        tl.store(unified_kv_indices + dst_idx, vals, mask=mask)


def build_unified_kv_indices(
    prefix_kv_indptr: paddle.Tensor,
    prefix_kv_indices: paddle.Tensor,
    extend_start_loc: paddle.Tensor,
    extend_seq_lens: paddle.Tensor,
    extend_kv_indices: paddle.Tensor,
    bs: int,
) -> tuple:
    """
    Build unified KV indices efficiently:
    - Use Paddle's optimized cumsum for indptr
    - Use Triton kernel for parallel index copying

    Args:
        prefix_kv_indptr: [bs+1] prefix KV indptr
        prefix_kv_indices: prefix KV indices (block ids)
        extend_start_loc: [bs] extend start location in extend_kv_indices
        extend_seq_lens: [bs] extend sequence lengths
        extend_kv_indices: extend KV indices (block ids)
        bs: batch size

    Returns:
        (unified_kv_indptr, unified_kv_indices, prefix_lens)
    """
    # Compute prefix lengths
    prefix_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]

    # Create unified_kv_indptr
    unified_lens = prefix_lens + extend_seq_lens[:bs]
    # Use prefix_kv_indptr[:1] * 0 to create zeros on the same device
    zeros_tensor = prefix_kv_indptr[:1] * 0
    unified_kv_indptr = paddle.concat(
        [
            zeros_tensor,
            paddle.cumsum(unified_lens, axis=0).astype("int32"),
        ]
    )

    max_unified_len = len(prefix_kv_indices) + len(extend_kv_indices)
    # Use paddle.empty and it will be on the same device as the input
    unified_kv_indices = paddle.empty([max_unified_len], dtype="int64")
    # Ensure it's on the same device
    unified_kv_indices = unified_kv_indices._copy_to(prefix_kv_indptr.place, False)

    # Launch Triton kernel for parallel index copying
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
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Unified 1-stage kernel for deterministic extend attention.
    Both prefix and extend KV are accessed through the unified kv_indices.

    Key insight for determinism:
    - All KV (prefix + extend) are accessed via unified indices
    - Same kernel configuration regardless of cache hit/miss
    - Causal mask: prefix region has no mask, extend region has standard causal mask
    """
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    # Load sequence information
    cur_seq_q_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_q_len = tl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_kv_len = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_prefix_len = tl.load(prefix_lens + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_q_len
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    # Load Q
    offs_q = (
        (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q + offs_qpe, mask=mask_m[:, None], other=0.0)

    # Initialize accumulators for online softmax
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # Unified loop: process all KV tokens (prefix + extend)
    for start_n in range(0, cur_seq_kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_kv_len

        # Compute mask
        final_mask = mask_m[:, None] & mask_n[None, :]

        # Apply causal mask for extend part
        if IS_CAUSAL:
            # Determine if current KV block is in extend region
            q_idx = cur_block_m * BLOCK_M + offs_m[:, None]
            k_idx_in_total = start_n + offs_n[None, :]

            # Causal mask logic:
            # - For prefix region (k_idx < prefix_len): no causal mask (all visible)
            # - For extend region (k_idx >= prefix_len): standard causal mask
            k_is_extend = k_idx_in_total >= cur_seq_prefix_len
            k_idx_in_extend = k_idx_in_total - cur_seq_prefix_len
            causal_mask = tl.where(
                k_is_extend,
                q_idx >= k_idx_in_extend,  # extend region: need causal
                True,  # prefix region: no causal mask
            )
            final_mask &= causal_mask

        # Check if we can skip this tile
        if IS_CAUSAL:
            # For causal case, we can skip if all mask values are False
            pass  # Continue processing, let the mask handle it

        # Load KV indices
        offs_kv_loc = tl.load(
            kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
            mask=mask_n,
            other=0,
        )

        # Load K (transposed access for efficient dot product)
        offs_buf_k = offs_kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[:, None]
        k = tl.load(
            K_Buffer + offs_buf_k,
            mask=(mask_n[None, :]) & (mask_d[:, None]),
            other=0.0,
        )

        # Compute QK
        qk = tl.dot(q.to(k.dtype), k)
        if BLOCK_DPE > 0:
            offs_kpe = offs_kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_dpe[:, None]
            kpe = tl.load(
                K_Buffer + offs_kpe,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe.to(kpe.dtype), kpe)

        qk *= sm_scale

        if logit_cap > 0:
            # Logit capping for stability
            qk = logit_cap * tl.sigmoid(2 * qk / logit_cap) - logit_cap

        qk = tl.where(final_mask, qk, float("-inf"))

        # Online softmax
        row_max = tl.max(qk, 1)
        row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
        n_e_max = tl.maximum(row_max_fixed, e_max)

        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        # Load V
        offs_buf_v = offs_kv_loc[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_dv[None, :]
        v = tl.load(
            V_Buffer + offs_buf_v,
            mask=mask_n[:, None] & mask_dv[None, :],
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)

        e_max = n_e_max

    # Store output
    offs_o = (
        (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O + offs_o,
        acc / deno[:, None],
        mask=mask_m[:, None] & mask_dv[None, :],
    )


def extend_attention_fwd_unified(
    q: paddle.Tensor,
    o: paddle.Tensor,
    k_buffer: paddle.Tensor,
    v_buffer: paddle.Tensor,
    qo_indptr: paddle.Tensor,
    kv_indptr: paddle.Tensor,
    kv_indices: paddle.Tensor,
    prefix_lens: paddle.Tensor,
    max_len_extend: int,
    sm_scale: float = None,
    logit_cap: float = 0.0,
    is_causal: bool = True,
):
    """
    Unified 1-stage extend attention for deterministic inference.

    This kernel processes both prefix KV and extend KV in a single pass,
    ensuring deterministic behavior regardless of cache hit/miss status.

    Args:
        q: Query tensor [num_tokens, num_heads, head_dim]
        o: Output tensor [num_tokens, num_heads, head_dim]
        k_buffer: Key cache buffer [max_blocks, kv_num_heads, block_size, head_dim]
        v_buffer: Value cache buffer [max_blocks, kv_num_heads, block_size, head_dim]
        qo_indptr: Query offsets [batch_size + 1]
        kv_indptr: KV offsets [batch_size + 1] (unified prefix + extend)
        kv_indices: Unified KV indices (block ids for both prefix and extend)
        prefix_lens: Prefix length for each sequence [batch_size]
        max_len_extend: Maximum extend length in the batch
        sm_scale: Softmax scale (default: 1/sqrt(head_dim))
        logit_cap: Logit capping value (0 for no capping)
        is_causal: Whether to apply causal mask
    """
    Lq, Lv = q.shape[-1], v_buffer.shape[-1]

    # Get block sizes and configuration
    BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = _get_block_sizes_for_extend_attention(Lq, Lv)

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

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
        q.stride(0),
        q.stride(1),
        o.stride(0),
        o.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        logit_cap=logit_cap,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        Lq=Lq,
        Lv=Lv,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )
