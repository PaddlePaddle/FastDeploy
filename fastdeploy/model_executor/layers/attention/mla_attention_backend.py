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

from __future__ import annotations

import paddle

paddle.enable_compat(scope={"flash_mla"})  # Enable paddle.enable_compat before importing flash_mla
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import paddle
from paddle.nn.functional.flash_attention import flash_attn_unpadded
from paddleformers.utils.log import logger

try:
    from paddle.nn.functional.flash_attention import flash_attention_v3_varlen
except Exception as e:
    logger.debug(f"flash_attention_v3_varlen not available: {e}")
    flash_attention_v3_varlen = None

from fastdeploy.model_executor.layers.attention.ops import (
    get_block_shape_and_split_kv_block,
    init_kv_signal_per_query,
    init_signal_layerwise,
    open_shm_and_get_meta_signal,
)
from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import (
        decode_mla_write_cache,
        multi_head_latent_attention,
        prefill_mla_write_cache,
    )

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

import triton
import triton.language as tl

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id
from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)
from fastdeploy.spec_decode import SpecMethod

# ============================================================================
# Latent Cache Read Kernel for Prefix Cache Support
# ============================================================================


def read_latent_from_cache_naive(
    latent_cache: paddle.Tensor,
    block_tables: paddle.Tensor,
    cache_kv_lens: paddle.Tensor,
    cu_seqlens_cached_kv: paddle.Tensor,
    total_cached_tokens: int,
    block_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    Read latent vectors (kv_c and k_pe) from paged latent cache for prefix cache support.

    Args:
        latent_cache: [num_blocks, 1, block_size, kv_lora_rank + qk_rope_head_dim]
        block_tables: [batch_size, max_blocks_per_seq]
        cache_kv_lens: [batch_size] - cached KV length for each request
        cu_seqlens_cached_kv: [batch_size + 1] - cumulative sequence lengths for cached KV
        total_cached_tokens: Total number of cached tokens across all requests
        block_size: Block size for paged attention
        kv_lora_rank: LoRA rank for KV compression
        qk_rope_head_dim: Dimension of RoPE part in key

    Returns:
        cached_kv_c: [total_cached_tokens, kv_lora_rank]
        cached_k_pe: [total_cached_tokens, qk_rope_head_dim]
    """
    if total_cached_tokens == 0:
        return None, None

    # latent_dim = kv_lora_rank + qk_rope_head_dim

    # Allocate output tensors
    cached_kv_c = paddle.empty([total_cached_tokens, kv_lora_rank], dtype=latent_cache.dtype)
    cached_k_pe = paddle.empty([total_cached_tokens, qk_rope_head_dim], dtype=latent_cache.dtype)

    # Use a simpler approach: iterate through each batch and gather latent vectors
    # IMPORTANT: Only process batches that have prefix cache (use cu_seqlens_cached_kv)
    bsz = cu_seqlens_cached_kv.shape[0] - 1
    output_idx = 0

    for batch_id in range(bsz):
        # Get the number of cached tokens for this batch from cu_seqlens_cached_kv
        cu_start = (
            cu_seqlens_cached_kv[batch_id].item()
            if hasattr(cu_seqlens_cached_kv[batch_id], "item")
            else cu_seqlens_cached_kv[batch_id]
        )
        cu_end = (
            cu_seqlens_cached_kv[batch_id + 1].item()
            if hasattr(cu_seqlens_cached_kv[batch_id + 1], "item")
            else cu_seqlens_cached_kv[batch_id + 1]
        )
        cache_len = cu_end - cu_start

        if cache_len <= 0:
            continue

        # Read tokens from multiple blocks if cache_len > block_size
        local_idx = 0
        while local_idx < cache_len:
            block_idx = local_idx // block_size
            block_offset = local_idx % block_size
            tokens_to_read = min(block_size - block_offset, cache_len - local_idx)

            physical_block_id = block_tables[batch_id, block_idx].item()

            # Load latent vectors from this block
            for offset in range(tokens_to_read):
                latent_vec = latent_cache[physical_block_id, 0, block_offset + offset, :]

                # Split into kv_c and k_pe
                cached_kv_c[output_idx] = latent_vec[:kv_lora_rank]
                cached_k_pe[output_idx] = latent_vec[kv_lora_rank:]
                output_idx += 1

            local_idx += tokens_to_read

    assert (
        output_idx == total_cached_tokens
    ), f"read_latent_from_cache_naive: wrote {output_idx} tokens, expected {total_cached_tokens}"
    return cached_kv_c, cached_k_pe


def interleave_cached_and_new_latent_naive(
    cached_kv_c: paddle.Tensor,
    cached_k_pe: paddle.Tensor,
    new_compressed_kv: paddle.Tensor,
    new_k_pe: paddle.Tensor,
    cu_seqlens_cached_kv: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    Interleave cached and new latent vectors per batch for correct FlashAttention layout.

    The key insight is that FlashAttention expects tokens to be ordered by batch:
    [batch0_tokens, batch1_tokens, ...]
    where each batch's tokens should be [cached_tokens, new_tokens].

    Args:
        cached_kv_c: [total_cached_tokens, kv_lora_rank] - cached KV latent (from read_latent_from_cache)
        cached_k_pe: [total_cached_tokens, qk_rope_head_dim] - cached K RoPE (from read_latent_from_cache)
        new_compressed_kv: [total_new_tokens, kv_lora_rank] - new KV latent from current forward
        new_k_pe: [total_new_tokens, qk_rope_head_dim] - new K RoPE from current forward (already with RoPE applied)
        cu_seqlens_cached_kv: [batch_size + 1] - cumulative sequence lengths for cached KV
        seq_lens_encoder: [batch_size] - number of new tokens per batch
        cu_seqlens_q: [batch_size + 1] - cumulative sequence lengths for new tokens
        kv_lora_rank: LoRA rank for KV compression
        qk_rope_head_dim: Dimension of RoPE part in key

    Returns:
        full_compressed_kv: [total_tokens, kv_lora_rank] - properly interleaved KV latent
        full_k_pe: [total_tokens, qk_rope_head_dim] - properly interleaved K RoPE
    """
    bsz = cu_seqlens_cached_kv.shape[0] - 1

    # Calculate total output size
    total_cached = cached_kv_c.shape[0] if cached_kv_c is not None and cached_kv_c.numel() > 0 else 0
    total_new = new_compressed_kv.shape[0]
    total_tokens = total_cached + total_new

    # Allocate output tensors
    full_compressed_kv = paddle.empty([total_tokens, kv_lora_rank], dtype=new_compressed_kv.dtype)
    full_k_pe = paddle.empty([total_tokens, qk_rope_head_dim], dtype=new_k_pe.dtype)

    # Track indices into cached and new tensors
    cached_idx = 0
    new_idx = 0
    out_position = 0  # Track output position for each batch

    for batch_id in range(bsz):
        # Number of cached tokens for this batch
        cu_cached_start = (
            cu_seqlens_cached_kv[batch_id].item()
            if hasattr(cu_seqlens_cached_kv[batch_id], "item")
            else cu_seqlens_cached_kv[batch_id]
        )
        cu_cached_end = (
            cu_seqlens_cached_kv[batch_id + 1].item()
            if hasattr(cu_seqlens_cached_kv[batch_id + 1], "item")
            else cu_seqlens_cached_kv[batch_id + 1]
        )
        num_cached = cu_cached_end - cu_cached_start

        # Number of new tokens for this batch
        cu_new_start = (
            cu_seqlens_q[batch_id].item() if hasattr(cu_seqlens_q[batch_id], "item") else cu_seqlens_q[batch_id]
        )
        cu_new_end = (
            cu_seqlens_q[batch_id + 1].item()
            if hasattr(cu_seqlens_q[batch_id + 1], "item")
            else cu_seqlens_q[batch_id + 1]
        )
        num_new = cu_new_end - cu_new_start

        # Output position for this batch (sequential, no gaps)
        out_start = out_position

        # Copy cached tokens first (if any)
        if num_cached > 0 and cached_kv_c is not None:
            full_compressed_kv[out_start : out_start + num_cached] = cached_kv_c[cached_idx : cached_idx + num_cached]
            full_k_pe[out_start : out_start + num_cached] = cached_k_pe[cached_idx : cached_idx + num_cached]
            cached_idx += num_cached

        # Then copy new tokens
        if num_new > 0:
            full_compressed_kv[out_start + num_cached : out_start + num_cached + num_new] = new_compressed_kv[
                new_idx : new_idx + num_new
            ]
            full_k_pe[out_start + num_cached : out_start + num_cached + num_new] = new_k_pe[
                new_idx : new_idx + num_new
            ]
            new_idx += num_new

        # Update output position for next batch
        out_position += num_cached + num_new

    assert (
        cached_idx == total_cached
    ), f"interleave_cached_and_new_latent_naive: cached_idx={cached_idx} != total_cached={total_cached}"
    assert new_idx == total_new, f"interleave_cached_and_new_latent_naive: new_idx={new_idx} != total_new={total_new}"
    assert (
        out_position == total_tokens
    ), f"interleave_cached_and_new_latent_naive: out_position={out_position} != total_tokens={total_tokens}"
    return full_compressed_kv, full_k_pe


# ----------------------------------------------------------------------------
# Public dispatchers. Default to the naive Python implementations; Task 3/4 will
# swap these to high-performance Triton kernels, controlled by FD_MLA_USE_NAIVE.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Triton implementation of read_latent_from_cache.
# ----------------------------------------------------------------------------


@enable_compat_on_triton_kernel
@triton.jit()
def _read_latent_triton_kernel(
    latent_cache_ptr,  # [num_blocks, 1, block_size, latent_dim]
    block_tables_ptr,  # [bsz, max_blocks_per_seq]
    batch_id_per_token_ptr,  # [total_cached_tokens] int32
    local_offset_per_token_ptr,  # [total_cached_tokens] int32
    output_kv_c_ptr,  # [total_cached_tokens, kv_lora_rank]
    output_k_pe_ptr,  # [total_cached_tokens, qk_rope_head_dim]
    max_blocks_per_seq: tl.constexpr,
    block_size: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    LATENT_DIM: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)

    batch_id = tl.load(batch_id_per_token_ptr + token_idx)
    local_off = tl.load(local_offset_per_token_ptr + token_idx)

    block_idx = local_off // block_size
    block_offset = local_off % block_size

    physical_block_id = tl.load(block_tables_ptr + batch_id * max_blocks_per_seq + block_idx)

    latent_base = latent_cache_ptr + physical_block_id * block_size * LATENT_DIM + block_offset * LATENT_DIM

    kv_c_offs = tl.arange(0, kv_lora_rank)
    kv_c_val = tl.load(latent_base + kv_c_offs)
    tl.store(output_kv_c_ptr + token_idx * kv_lora_rank + kv_c_offs, kv_c_val)

    k_pe_offs = tl.arange(0, qk_rope_head_dim)
    k_pe_val = tl.load(latent_base + kv_lora_rank + k_pe_offs)
    tl.store(output_k_pe_ptr + token_idx * qk_rope_head_dim + k_pe_offs, k_pe_val)


def read_latent_from_cache_triton(
    latent_cache: paddle.Tensor,
    block_tables: paddle.Tensor,
    cache_kv_lens: paddle.Tensor,
    cu_seqlens_cached_kv: paddle.Tensor,
    total_cached_tokens: int,
    block_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
):
    """Triton-accelerated version of read_latent_from_cache_naive.

    Returns (cached_kv_c, cached_k_pe) with the same semantics as the naive
    implementation.
    """
    if total_cached_tokens == 0:
        return None, None

    bsz = cu_seqlens_cached_kv.shape[0] - 1
    max_blocks_per_seq = block_tables.shape[1]

    # --- host-side build of batch_id_per_token / local_offset_per_token ---
    # O(bsz) CPU work; much cheaper than the in-kernel linear scan used by the
    # legacy read_latent_from_cache_kernel.
    cu_list = cu_seqlens_cached_kv.tolist() if hasattr(cu_seqlens_cached_kv, "tolist") else list(cu_seqlens_cached_kv)
    batch_id_host = [0] * total_cached_tokens
    local_off_host = [0] * total_cached_tokens
    for b in range(bsz):
        start = int(cu_list[b])
        end = int(cu_list[b + 1])
        for t in range(start, end):
            batch_id_host[t] = b
            local_off_host[t] = t - start
    batch_id_per_token = paddle.to_tensor(batch_id_host, dtype="int32", place=latent_cache.place)
    local_offset_per_token = paddle.to_tensor(local_off_host, dtype="int32", place=latent_cache.place)

    cached_kv_c = paddle.empty([total_cached_tokens, kv_lora_rank], dtype=latent_cache.dtype)
    cached_k_pe = paddle.empty([total_cached_tokens, qk_rope_head_dim], dtype=latent_cache.dtype)

    grid = (total_cached_tokens,)
    _read_latent_triton_kernel[grid](
        latent_cache,
        block_tables,
        batch_id_per_token,
        local_offset_per_token,
        cached_kv_c,
        cached_k_pe,
        max_blocks_per_seq=max_blocks_per_seq,
        block_size=block_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        LATENT_DIM=kv_lora_rank + qk_rope_head_dim,
    )

    assert (
        cached_kv_c.shape[0] == total_cached_tokens
    ), f"read_latent_from_cache_triton: output shape mismatch {cached_kv_c.shape[0]} vs {total_cached_tokens}"
    return cached_kv_c, cached_k_pe


def read_latent_from_cache(*args, **kwargs):
    if os.environ.get("FD_MLA_USE_NAIVE", "0") == "1":
        return read_latent_from_cache_naive(*args, **kwargs)
    return read_latent_from_cache_triton(*args, **kwargs)


# ----------------------------------------------------------------------------
# Triton implementation of interleave_cached_and_new_latent.
# ----------------------------------------------------------------------------


@enable_compat_on_triton_kernel
@triton.jit()
def _interleave_latent_kernel(
    cached_kv_c_ptr,  # [total_cached, kv_lora_rank] (may be invalid when total_cached==0)
    cached_k_pe_ptr,  # [total_cached, qk_rope_head_dim]
    new_kv_c_ptr,  # [total_new, kv_lora_rank]
    new_k_pe_ptr,  # [total_new, qk_rope_head_dim]
    src_is_cached_ptr,  # [total_tokens] int32 (1 == cached, 0 == new)
    src_idx_ptr,  # [total_tokens] int32 (index within the chosen source tensor)
    out_kv_c_ptr,  # [total_tokens, kv_lora_rank]
    out_k_pe_ptr,  # [total_tokens, qk_rope_head_dim]
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    is_cached = tl.load(src_is_cached_ptr + token_idx)
    src_idx = tl.load(src_idx_ptr + token_idx)

    kv_c_offs = tl.arange(0, kv_lora_rank)
    k_pe_offs = tl.arange(0, qk_rope_head_dim)

    if is_cached != 0:
        kv_c_val = tl.load(cached_kv_c_ptr + src_idx * kv_lora_rank + kv_c_offs)
        k_pe_val = tl.load(cached_k_pe_ptr + src_idx * qk_rope_head_dim + k_pe_offs)
    else:
        kv_c_val = tl.load(new_kv_c_ptr + src_idx * kv_lora_rank + kv_c_offs)
        k_pe_val = tl.load(new_k_pe_ptr + src_idx * qk_rope_head_dim + k_pe_offs)

    tl.store(out_kv_c_ptr + token_idx * kv_lora_rank + kv_c_offs, kv_c_val)
    tl.store(out_k_pe_ptr + token_idx * qk_rope_head_dim + k_pe_offs, k_pe_val)


def interleave_cached_and_new_latent_triton(
    cached_kv_c: paddle.Tensor,
    cached_k_pe: paddle.Tensor,
    new_compressed_kv: paddle.Tensor,
    new_k_pe: paddle.Tensor,
    cu_seqlens_cached_kv: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
):
    """Triton-accelerated version of interleave_cached_and_new_latent_naive."""
    bsz = cu_seqlens_cached_kv.shape[0] - 1
    total_cached = cached_kv_c.shape[0] if cached_kv_c is not None and cached_kv_c.numel() > 0 else 0
    total_new = new_compressed_kv.shape[0]
    total_tokens = total_cached + total_new

    full_compressed_kv = paddle.empty([total_tokens, kv_lora_rank], dtype=new_compressed_kv.dtype)
    full_k_pe = paddle.empty([total_tokens, qk_rope_head_dim], dtype=new_k_pe.dtype)

    if total_tokens == 0:
        return full_compressed_kv, full_k_pe

    # ---- host-side mapping: for each output position, pick source ----
    cu_cached = (
        cu_seqlens_cached_kv.tolist() if hasattr(cu_seqlens_cached_kv, "tolist") else list(cu_seqlens_cached_kv)
    )
    cu_new = cu_seqlens_q.tolist() if hasattr(cu_seqlens_q, "tolist") else list(cu_seqlens_q)

    src_is_cached = [0] * total_tokens
    src_idx = [0] * total_tokens
    out_pos = 0
    for b in range(bsz):
        nc = int(cu_cached[b + 1]) - int(cu_cached[b])
        nn = int(cu_new[b + 1]) - int(cu_new[b])
        cached_base = int(cu_cached[b])
        new_base = int(cu_new[b])
        for t in range(nc):
            src_is_cached[out_pos] = 1
            src_idx[out_pos] = cached_base + t
            out_pos += 1
        for t in range(nn):
            src_is_cached[out_pos] = 0
            src_idx[out_pos] = new_base + t
            out_pos += 1

    assert (
        out_pos == total_tokens
    ), f"interleave_cached_and_new_latent_triton: host map out_pos={out_pos} != total_tokens={total_tokens}"

    dev = new_compressed_kv.place
    src_is_cached_t = paddle.to_tensor(src_is_cached, dtype="int32", place=dev)
    src_idx_t = paddle.to_tensor(src_idx, dtype="int32", place=dev)

    # Provide non-null pointers for the cached tensors even when total_cached == 0;
    # Triton kernel path is still gated by src_is_cached flag.
    cached_kv_c_ptr = cached_kv_c if total_cached > 0 else full_compressed_kv
    cached_k_pe_ptr = cached_k_pe if total_cached > 0 else full_k_pe

    grid = (total_tokens,)
    _interleave_latent_kernel[grid](
        cached_kv_c_ptr,
        cached_k_pe_ptr,
        new_compressed_kv,
        new_k_pe,
        src_is_cached_t,
        src_idx_t,
        full_compressed_kv,
        full_k_pe,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    )

    return full_compressed_kv, full_k_pe


def interleave_cached_and_new_latent(*args, **kwargs):
    if os.environ.get("FD_MLA_USE_NAIVE", "0") == "1":
        return interleave_cached_and_new_latent_naive(*args, **kwargs)
    return interleave_cached_and_new_latent_triton(*args, **kwargs)


# ============================================================================


@enable_compat_on_triton_kernel
@triton.jit()
def extract_kernel(
    q,
    cu_seqlens_q,
    seq_lens_encoder,
    seq_lens_decoder,
    output,
    cache_seqlens,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    batch_id = tl.program_id(axis=0)
    cache_kv_len = tl.load(seq_lens_decoder + batch_id)

    # 这个batch不是decoder，所以不需要动弹
    if cache_kv_len <= 0:
        return

    cu_len_this_batch = tl.load(cu_seqlens_q + batch_id)

    read_offsets = tl.arange(0, BLOCK_SIZE)
    q += cu_len_this_batch * HIDDEN_DIM

    row_data = tl.load(q + read_offsets, mask=read_offsets < HIDDEN_DIM)

    output += batch_id * HIDDEN_DIM

    tl.store(output + read_offsets, row_data, mask=read_offsets < HIDDEN_DIM)

    tl.store(cache_seqlens + batch_id, cache_kv_len + 1)


def extract_decoder_token_from_q(
    q: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
):
    assert len(q.shape) == 2
    assert len(cu_seqlens_q.shape) == 1
    assert len(seq_lens_encoder.shape) == 1
    assert len(seq_lens_decoder.shape) == 1

    max_bsz = seq_lens_decoder.shape[0]

    hidden_dim = q.shape[-1]
    out = paddle.empty([max_bsz, hidden_dim], dtype=q.dtype)

    cache_seqlens = paddle.zeros_like(seq_lens_decoder)

    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)

    grid = (max_bsz,)

    extract_kernel[grid](
        q,
        cu_seqlens_q,
        seq_lens_encoder,
        seq_lens_decoder,
        out,
        cache_seqlens,
        hidden_dim,
        BLOCK_SIZE,
    )

    return out, cache_seqlens


@enable_compat_on_triton_kernel
@triton.jit()
def insert_kernel(
    decoder_res,
    cu_seqlens_q,
    seq_lens_encoder,
    seq_lens_decoder,
    output,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    batch_id = tl.program_id(axis=0)
    cache_kv_len = tl.load(seq_lens_decoder + batch_id)

    # 这个batch不是decoder，所以不需要动弹
    if cache_kv_len <= 0:
        return

    cu_len_this_batch = tl.load(cu_seqlens_q + batch_id)

    read_offsets = tl.arange(0, BLOCK_SIZE)

    decoder_res += batch_id * HIDDEN_DIM

    row_data = tl.load(decoder_res + read_offsets, mask=read_offsets < HIDDEN_DIM)

    output += cu_len_this_batch * HIDDEN_DIM

    tl.store(output + read_offsets, row_data, mask=read_offsets < HIDDEN_DIM)


def insert_decoder_result_back(
    decoder_result: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    mixed_token_num,
):
    assert len(decoder_result.shape) == 4
    assert len(cu_seqlens_q.shape) == 1
    assert len(seq_lens_encoder.shape) == 1

    max_bsz = seq_lens_encoder.shape[0]

    hidden_dim = decoder_result.shape[-2] * decoder_result.shape[-1]
    out = paddle.zeros([mixed_token_num, hidden_dim], dtype=decoder_result.dtype)

    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)

    grid = (max_bsz,)

    insert_kernel[grid](
        decoder_result,
        cu_seqlens_q,
        seq_lens_encoder,
        seq_lens_decoder,
        out,
        hidden_dim,
        BLOCK_SIZE,
    )

    return out


def yarn_get_mscale(scale=1, mscale=1):
    """ """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@dataclass
class MLAAttentionMetadata(AttentionMetadata):
    """
    MLAAttentionMetadata for Multi-Layer Attention
    """

    _dtype: paddle.dtype = paddle.bfloat16
    encoder_max_partition_size: int = 32768
    max_partition_size: int = 32768
    block_tables: Optional[paddle.Tensor] = None
    rotary_embs: Optional[paddle.Tensor] = None
    attn_mask: Optional[paddle.Tensor] = None
    _fuse_kernel_compute_dtype: str = "bf16"

    # pd_disaggregation
    kv_signal_metadata: Optional[paddle.Tensor] = None
    kv_signal_data_list: List[Optional[paddle.Tensor]] = field(default_factory=list)

    max_enc_len_this_time: Optional[paddle.Tensor] = None
    max_dec_len_this_time: Optional[paddle.Tensor] = None
    max_kv_len_this_time: Optional[paddle.Tensor] = None

    # For prefix cache and chunked prefill support
    # Indicates whether any request has prefix cache (cached KV from previous requests)
    has_prefix_cache: bool = False
    # Total number of cached KV tokens across all requests with prefix cache
    total_cached_kv_tokens: int = 0
    # cu_seqlens for cached KV tokens (similar to cu_seqlens_k but for cached portion)
    cu_seqlens_cached_kv: Optional[paddle.Tensor] = None
    # Maximum cached KV length across all requests
    max_cached_kv_len: int = 0
    # cu_seqlens_k that includes cached tokens (for FlashAttention when prefix cache is present)
    cu_seqlens_k_with_cache: Optional[paddle.Tensor] = None
    # Maximum total KV length (cached + new) across all requests for FlashAttention
    max_total_kv_len: int = 0


class MLAAttentionBackend(AttentionBackend):
    """
    MLA Attention Backend implementation.
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: MLAAttentionMetadata
    flash_attn_func: callable = None

    def __init__(
        self,
        fd_config: FDConfig,
        kv_num_heads: int,
        num_heads: int,
        head_dim: int,
        encoder_block_shape_q: int = -1,
        decoder_block_shape_q: int = -1,
    ) -> None:
        """
        MLAAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: MLAAttentionMetadata = None

        # 基础配置
        self.block_size: int = fd_config.cache_config.block_size
        self.max_seq_len: int = fd_config.model_config.max_model_len
        self.rope_theta: float = (
            10000.0 if fd_config.model_config.rope_theta is None else fd_config.model_config.rope_theta
        )
        self.rope_3d: bool = fd_config.enable_rope_3d_runtime
        self.causal: bool = getattr(fd_config.model_config, "causal", True)
        self.speculative_method = fd_config.speculative_config.method
        self.use_speculate: bool = self.speculative_method is not None
        self.speculate_max_draft_token_num: int = fd_config.speculative_config.num_speculative_tokens
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method == SpecMethod.MTP)

        self.num_heads: int = num_heads
        self.heads_need_padding = False
        if self.num_heads < 64 and fd_config.parallel_config.tensor_parallel_size > 1:
            self.padding_num_heads = 64 - self.num_heads
            self.heads_need_padding = True
            logger.warning(
                f"MLA num attention heads is less than 64, force to use 64 num heads. "
                f"current num_heads={self.num_heads}, tp_size={fd_config.parallel_config.tensor_parallel_size}"
            )
        self.head_dim: int = fd_config.model_config.head_dim
        self.num_layers: int = fd_config.model_config.num_hidden_layers

        # For Multi Head Latent Attention
        self.kv_lora_rank: int = fd_config.model_config.kv_lora_rank
        self.qk_rope_head_dim: int = fd_config.model_config.qk_rope_head_dim
        self.qk_head_dim: int = fd_config.model_config.qk_nope_head_dim + fd_config.model_config.qk_rope_head_dim
        self.attn_softmax_scale: float = self.qk_head_dim**-0.5
        self.rope_scaling = getattr(fd_config.model_config, "rope_scaling", None)
        if self.rope_scaling and "factor" in self.rope_scaling:
            # if fd_config.model_config.rope_scaling:
            mscale_all_dim = fd_config.model_config.rope_scaling.get("mscale_all_dim", False)  # 1.0
            scaling_factor = fd_config.model_config.rope_scaling["factor"]  # 40
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.attn_softmax_scale = self.attn_softmax_scale * mscale * mscale

        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index
        self.device_id: int = os.getenv("CUDA_VISIBLE_DEVICES", None)

        self.rank, self.device_id = init_rank_and_device_id(fd_config)

        self.useless_tensor = paddle.randn([1]).cast("int32")

        if self.flash_attn_func is None:
            prop = paddle.device.cuda.get_device_properties()
            cc = prop.major * 10 + prop.minor
            is_current_sm_supported = cc >= 90
            is_paddle_supported = any(num >= 90 for num in paddle.version.cuda_archs())
            if is_current_sm_supported and is_paddle_supported:
                self.flash_attn_func = flash_attention_v3_varlen
                logger.info("The current platform supports Flash Attention V3.")
                self.flash_attn_kwargs = {"softmax_scale": self.attn_softmax_scale}
            else:
                self.flash_attn_func = flash_attn_unpadded
                self.flash_attn_kwargs = {"scale": self.attn_softmax_scale, "training": False}
                logger.info(
                    "The current platform does not support Flash Attention V3, so Flash Attention V2 will be used instead."
                )

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attention metadata hence all layers in the forward pass can reuse it."""
        metadata = MLAAttentionMetadata()
        metadata.max_partition_size = 32768
        metadata.encoder_max_partition_size = self.max_seq_len
        metadata._dtype = paddle.get_default_dtype()
        if metadata._dtype == "bfloat16":
            metadata._fuse_kernel_compute_dtype = "bf16"
        elif metadata._dtype == "float16":
            metadata._fuse_kernel_compute_dtype = "fp16"
        elif metadata._dtype == "float32":
            metadata._fuse_kernel_compute_dtype = "fp32"

        metadata.block_tables = forward_meta.block_tables
        metadata.rotary_embs = forward_meta.rotary_embs
        metadata.attn_mask = forward_meta.attn_mask
        metadata.pre_caches_length = forward_meta.pre_caches_length

        get_block_shape_and_split_kv_block(
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.decoder_batch_ids,
            forward_meta.decoder_tile_ids_per_batch,
            self.useless_tensor,  # not used in mla
            forward_meta.decoder_num_blocks_device,
            forward_meta.decoder_chunk_size_device,
            forward_meta.max_len_tensor_cpu,
            self.useless_tensor,  # not used in mla
            self.useless_tensor,  # not used in mla
            self.useless_tensor,  # not used in mla
            forward_meta.kv_batch_ids,
            forward_meta.kv_tile_ids_per_batch,
            forward_meta.kv_num_blocks_x_cpu,
            -1,  # not need.
            -1,  # not need.
            -1,  # not need.
            self.block_size,
        )
        # MLA
        metadata.max_enc_len_this_time = forward_meta.max_len_tensor_cpu[1]
        metadata.max_dec_len_this_time = forward_meta.max_len_tensor_cpu[2]
        metadata.max_kv_len_this_time = forward_meta.max_len_tensor_cpu[5]

        # Compute prefix cache metadata
        # For MLA, prefix cache is indicated by seq_lens_decoder > 0 when there are encoder tokens
        # This means the request has cached KV from a previous prefix
        bsz = forward_meta.seq_lens_this_time.shape[0]
        has_prefix_cache = False
        total_cached_kv_tokens = 0
        max_cached_kv_len = 0
        max_total_kv_len = 0  # max(dec_len + enc_len) for FlashAttention

        # Check if any request has prefix cache
        # Prefix cache exists when seq_lens_decoder > 0
        # seq_lens_decoder stores the cached KV length for chunked prefill/prefix cache
        for i in range(bsz):
            # enc_len = (
            #     forward_meta.seq_lens_encoder[i].item()
            #     if hasattr(forward_meta.seq_lens_encoder[i], "item")
            #     else forward_meta.seq_lens_encoder[i]
            # )
            dec_len = (
                forward_meta.seq_lens_decoder[i].item()
                if hasattr(forward_meta.seq_lens_decoder[i], "item")
                else forward_meta.seq_lens_decoder[i]
            )
            seq_this_time = (
                forward_meta.seq_lens_this_time[i].item()
                if hasattr(forward_meta.seq_lens_this_time[i], "item")
                else forward_meta.seq_lens_this_time[i]
            )
            # Per-batch K length after interleave = dec_len (cached) + seq_this_time (new).
            # seq_this_time equals enc_len for prefill/chunked-prefill, and 1 for decode.
            # Using enc_len here drops the decode-batch new token and yields a wrong
            # max_seqlen_k for FlashAttention, which then tile-truncates the last K row.
            per_batch_k = dec_len + seq_this_time
            if dec_len > 0:
                has_prefix_cache = True
                total_cached_kv_tokens += dec_len
                max_cached_kv_len = max(max_cached_kv_len, dec_len)
            if per_batch_k > 0:
                max_total_kv_len = max(max_total_kv_len, per_batch_k)

        metadata.has_prefix_cache = has_prefix_cache
        metadata.total_cached_kv_tokens = total_cached_kv_tokens
        metadata.max_cached_kv_len = max_cached_kv_len
        metadata.max_total_kv_len = max_total_kv_len

        # Compute cu_seqlens_cached_kv if there's prefix cache
        if has_prefix_cache and total_cached_kv_tokens > 0:
            cu_seqlens_cached_kv = paddle.zeros([bsz + 1], dtype=paddle.int32)
            cu_seqlens_k_with_cache = paddle.zeros([bsz + 1], dtype=paddle.int32)
            cumsum_cached = 0
            cumsum_total = 0
            # cu_seqlens layout must stay CONSISTENT with the input tensor layout used
            # by the prefill FlashAttention call. The input `compressed_kv`/`key_pe` contains
            # ALL tokens in the batch (prefill + decode), in seq_lens_this_time order.
            #
            # For each batch i, the interleaved layout writes: [cached_tokens, new_tokens_of_this_batch]
            #   - new_tokens_of_this_batch = seq_lens_this_time[i] (== enc_len for prefill, == 1 for decode)
            #   - cached_tokens = dec_len if dec_len > 0 else 0
            # cu_seqlens_k_with_cache must reflect this sum per batch.
            # cu_seqlens_cached_kv tracks only the cached portion for read_latent_from_cache().
            for i in range(bsz):
                # enc_len = (
                #     forward_meta.seq_lens_encoder[i].item()
                #     if hasattr(forward_meta.seq_lens_encoder[i], "item")
                #     else forward_meta.seq_lens_encoder[i]
                # )
                dec_len = (
                    forward_meta.seq_lens_decoder[i].item()
                    if hasattr(forward_meta.seq_lens_decoder[i], "item")
                    else forward_meta.seq_lens_decoder[i]
                )
                seq_this_time = (
                    forward_meta.seq_lens_this_time[i].item()
                    if hasattr(forward_meta.seq_lens_this_time[i], "item")
                    else forward_meta.seq_lens_this_time[i]
                )
                if dec_len > 0:
                    cumsum_cached += dec_len
                    cumsum_total += dec_len
                # Add this batch's new tokens to cumsum_total. Use seq_lens_this_time to cover
                # both prefill (== enc_len) and decode (== 1) correctly.
                cumsum_total += seq_this_time
                cu_seqlens_cached_kv[i + 1] = cumsum_cached
                cu_seqlens_k_with_cache[i + 1] = cumsum_total
            # Consistency checks: starts at 0, monotonic non-decreasing, final equals cumulative.
            assert cu_seqlens_cached_kv[0].item() == 0, "cu_seqlens_cached_kv must start at 0"
            assert cu_seqlens_k_with_cache[0].item() == 0, "cu_seqlens_k_with_cache must start at 0"
            assert (
                cu_seqlens_cached_kv[bsz].item() == cumsum_cached
            ), f"cu_seqlens_cached_kv[-1]={cu_seqlens_cached_kv[bsz].item()} != cumsum_cached={cumsum_cached}"
            assert (
                cu_seqlens_k_with_cache[bsz].item() == cumsum_total
            ), f"cu_seqlens_k_with_cache[-1]={cu_seqlens_k_with_cache[bsz].item()} != cumsum_total={cumsum_total}"
            metadata.cu_seqlens_cached_kv = cu_seqlens_cached_kv
            metadata.cu_seqlens_k_with_cache = cu_seqlens_k_with_cache
            # Cross-check: max_total_kv_len MUST be >= max per-batch K length implied by
            # cu_seqlens_k_with_cache. Otherwise FlashAttention will tile-truncate the
            # last K rows of the longest batch and silently corrupt attention output.
            if bsz > 0:
                cu_k_list = cu_seqlens_k_with_cache.tolist()
                observed_max = max(cu_k_list[i + 1] - cu_k_list[i] for i in range(bsz))
                assert max_total_kv_len >= observed_max, (
                    f"max_total_kv_len={max_total_kv_len} < observed per-batch max "
                    f"K length={observed_max} from cu_seqlens_k_with_cache={cu_k_list}. "
                    f"FlashAttention will truncate K tiles."
                )

        # pd_disaggregation
        metadata.kv_signal_data_list = [None] * self.num_layers
        if self.pd_disaggregation_mode == "per_chunk":
            if not self.keep_pd_step_flag and not forward_meta.is_dummy_or_profile_run:
                init_kv_signal_per_query(
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_this_time,
                    forward_meta.seq_lens_decoder,
                    self.rank,
                    self.num_layers + self.num_layers_draft_model,
                )
        elif self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_metadata = open_shm_and_get_meta_signal(
                self.rank, int(self.device_id), self.keep_pd_step_flag
            )

        self.attention_metadata: AttentionMetadata = metadata

    def get_attention_meta(self) -> AttentionMetadata:
        """get_attention_meta"""
        return self.attention_metadata

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ) -> Tuple[int, int, int, int]:
        """
        Calculate kv cache shape for MLA
        """
        key_cache_shape = [max_num_blocks, 1, self.block_size, self.kv_lora_rank + self.qk_rope_head_dim]
        value_cache_shape = []
        return key_cache_shape, value_cache_shape

    def forward_extend(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: Attention,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """
        Prefill阶段的前向传播，支持 prefix cache

        对于 MLA 模型的 prefix cache 支持：
        1. 如果存在 prefix cache (metadata.has_prefix_cache = True)
           - k 和 v 应该已经包含了 cached KV 和 new KV 的拼接
           - cu_seqlens_k 应该已经调整为包含 cached tokens
        2. 如果不存在 prefix cache，行为与之前相同
        """
        metadata = self.attention_metadata

        if self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        latent_cache = forward_meta.caches[layer.layer_id] if hasattr(forward_meta, "caches") else None

        # 写入新的 KV 到缓存 (只写入新 tokens，不写入 cached 部分)
        prefill_mla_write_cache(
            compressed_kv,
            k_pe,
            latent_cache,
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
            metadata.block_tables,
            metadata.kv_signal_data_list[layer.layer_id],
            "none",
            getattr(forward_meta, "max_input_length", -1),
        )

        # Flash注意力计算
        # 对于 prefix cache 场景：
        # - k 和 v 应该已经包含了 cached + new tokens
        # - cu_seqlens_k 应该已经调整
        # - max_seqlen_k 应该是 cached_len + new_len 的最大值

        # 获取正确的 cu_seqlens_k 和 max_seqlen_k
        if metadata.has_prefix_cache and metadata.cu_seqlens_k_with_cache is not None:
            # When prefix cache is present, use cu_seqlens_k that includes cached tokens
            cu_seqlens_k = metadata.cu_seqlens_k_with_cache
            max_seqlen_k = metadata.max_total_kv_len  # max(dec_len + enc_len)
        else:
            cu_seqlens_k = forward_meta.cu_seqlens_k
            max_seqlen_k = metadata.max_enc_len_this_time

        fmha_out = self.flash_attn_func(
            q,
            k,
            v,
            forward_meta.cu_seqlens_q,
            cu_seqlens_k,
            metadata.max_enc_len_this_time,  # max_seqlen_q - only new tokens
            max_seqlen_k,  # max_seqlen_k - may include cached tokens
            causal=self.causal,
            **self.flash_attn_kwargs,
        )[0]

        return fmha_out

    def forward_decode(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: Attention,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """
        Decode阶段的前向传播
        """
        metadata = self.attention_metadata

        if self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        latent_cache = forward_meta.caches[layer.layer_id] if hasattr(forward_meta, "caches") else None

        # 获取推测解码参数
        speculate_decoder = self.speculative_method is not None
        speculate_max_tokens = self.speculate_max_draft_token_num

        # 写入缓存
        decode_mla_write_cache(
            compressed_kv,
            k_pe,
            latent_cache,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_encoder,
            forward_meta.batch_id_per_token,
            forward_meta.cu_seqlens_q,
            metadata.block_tables,
            "none",
            self.max_seq_len,
            speculate_decoder,
        )

        # 多头潜在注意力计算
        fmha_out = multi_head_latent_attention(
            q,
            latent_cache,
            latent_cache,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.cu_seqlens_q,
            forward_meta.batch_id_per_token,
            metadata.block_tables,
            forward_meta.kv_batch_ids,
            forward_meta.kv_tile_ids_per_batch,
            forward_meta.kv_num_blocks_x_cpu,
            forward_meta.decoder_batch_ids,
            forward_meta.decoder_tile_ids_per_batch,
            forward_meta.decoder_num_blocks_device,
            forward_meta.decoder_chunk_size_device,
            metadata.max_dec_len_this_time,
            metadata.max_kv_len_this_time,
            None,  # attn_mask
            None,  # qkv_bias
            None,  # qkv_out_scales
            None,  # cache_k_quant_scales
            None,  # cache_v_quant_scales
            None,  # cache_k_dequant_scales
            None,  # cache_v_dequant_scales
            None,  # cache_k_zp
            None,  # cache_v_zp
            None,  # out_shifts
            None,  # out_smooths
            metadata._fuse_kernel_compute_dtype,
            "none",  # cache_quant_type
            self.kv_lora_rank,
            self.max_seq_len,
            self.attn_softmax_scale,
            0.0,  # quant_max_bound
            0.0,  # quant_min_bound
            0.0,  # out_linear_in_scale
            speculate_max_tokens,
            True,  # causal
            speculate_decoder,
        )

        return fmha_out

    def forward_mixed(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: Attention,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        """
        Mixed模式的前向传播，支持 prefix cache

        对于 MLA 模型的 prefix cache 支持：
        1. Prefill 分支：k 和 v 应该已包含 cached + new tokens
        2. Decode 分支：保持原有 latent attention 逻辑
        """
        metadata = self.attention_metadata
        speculate_decoder = self.speculative_method is not None
        speculate_max_tokens = self.speculate_max_draft_token_num

        if self.pd_disaggregation_mode == "per_query":
            metadata.kv_signal_data_list[layer.layer_id] = init_signal_layerwise(
                metadata.kv_signal_metadata,
                layer.layer_id + self.start_layer_index,
            )

        latent_cache = forward_meta.caches[layer.layer_id] if hasattr(forward_meta, "caches") else None

        # Prefill branch: k is not None
        if k is not None:
            bsz = forward_meta.cu_seqlens_q.shape[0] - 1

            # Write cache only for new tokens of prefill/chunked-prefill batches.
            # Decode batches (seq_lens_encoder == 0) are intentionally skipped here — they
            # are handled later by decode_mla_write_cache() via the k=None branch when
            # need_do_decode is True. Using seq_lens_encoder keeps that separation correct
            # and avoids double-writing decode tokens.
            prefill_mla_write_cache(
                compressed_kv,
                k_pe,
                latent_cache,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.batch_id_per_token,
                forward_meta.cu_seqlens_q,
                metadata.block_tables,
                metadata.kv_signal_data_list[layer.layer_id],
                "none",
                self.max_seq_len,
            )

            # Determine cu_seqlens_k and max_seqlen_k considering prefix cache
            if metadata.has_prefix_cache and metadata.cu_seqlens_k_with_cache is not None:
                cu_seqlens_k = metadata.cu_seqlens_k_with_cache
                max_seqlen_k = metadata.max_total_kv_len  # max(dec_len + enc_len)
            else:
                cu_seqlens_k = forward_meta.cu_seqlens_k
                max_seqlen_k = metadata.max_enc_len_this_time

            # Shape consistency: k/v token count must match cu_seqlens_k terminal value
            _expected_k = cu_seqlens_k[bsz].item() if hasattr(cu_seqlens_k[bsz], "item") else int(cu_seqlens_k[bsz])
            assert (
                k.shape[0] == _expected_k
            ), f"forward_mixed: k.shape[0]={k.shape[0]} != cu_seqlens_k[-1]={_expected_k}"
            assert (
                v.shape[0] == _expected_k
            ), f"forward_mixed: v.shape[0]={v.shape[0]} != cu_seqlens_k[-1]={_expected_k}"

            # FlashAttention for prefill
            fmha_out = self.flash_attn_func(
                q,
                k,
                v,
                forward_meta.cu_seqlens_q,
                cu_seqlens_k,
                metadata.max_enc_len_this_time,  # max_seqlen_q - only new tokens
                max_seqlen_k,  # max_seqlen_k - may include cached tokens
                causal=self.causal,
                **self.flash_attn_kwargs,
            )[0]

            return fmha_out

        # Decode branch: k is None
        if k is None:
            decode_mla_write_cache(
                compressed_kv,
                k_pe,
                latent_cache,
                forward_meta.seq_lens_decoder,
                forward_meta.seq_lens_encoder,
                forward_meta.batch_id_per_token,
                forward_meta.cu_seqlens_q,
                metadata.block_tables,
                "none",
                self.max_seq_len,
                speculate_decoder,
            )

            if int(os.getenv("USE_FLASH_MLA", "0")) == 0:
                assert self.num_heads <= 64, "paddle mla attention support failed"
                if self.heads_need_padding:
                    q = paddle.nn.functional.pad(
                        q, [0, (self.padding_num_heads) * (self.kv_lora_rank + self.qk_rope_head_dim)], value=0.0
                    ).contiguous()
                # 多头潜在注意力计算
                fmha_out = multi_head_latent_attention(
                    q,
                    latent_cache,
                    latent_cache,
                    forward_meta.seq_lens_decoder,
                    forward_meta.seq_lens_this_time,
                    forward_meta.cu_seqlens_q,
                    forward_meta.batch_id_per_token,
                    metadata.block_tables,
                    forward_meta.kv_batch_ids,
                    forward_meta.kv_tile_ids_per_batch,
                    forward_meta.kv_num_blocks_x_cpu,
                    forward_meta.decoder_batch_ids,
                    forward_meta.decoder_tile_ids_per_batch,
                    forward_meta.decoder_num_blocks_device,
                    forward_meta.decoder_chunk_size_device,
                    metadata.max_dec_len_this_time,
                    metadata.max_kv_len_this_time,
                    None,  # attn_mask
                    None,  # qkv_bias
                    None,  # qkv_out_scales
                    None,  # cache_k_quant_scales
                    None,  # cache_v_quant_scales
                    None,  # cache_k_dequant_scales
                    None,  # cache_v_dequant_scales
                    None,  # cache_k_zp
                    None,  # cache_v_zp
                    None,  # out_shifts
                    None,  # out_smooths
                    metadata._fuse_kernel_compute_dtype,
                    "none",  # cache_quant_type
                    self.kv_lora_rank,
                    self.max_seq_len,
                    self.attn_softmax_scale,
                    0.0,  # quant_max_bound
                    0.0,  # quant_min_bound
                    0.0,  # out_linear_in_scale
                    speculate_max_tokens,
                    True,  # causal
                    speculate_decoder,
                )
                if self.heads_need_padding:
                    fmha_out = fmha_out[:, : self.num_heads * self.kv_lora_rank].contiguous()

                return fmha_out
            else:
                import flash_mla

                decoder_q, cache_seqlens = extract_decoder_token_from_q(
                    q,
                    forward_meta.cu_seqlens_q,
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_decoder,
                )

                tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata()
                token_num = q.shape[0]
                decoder_q.reshape_([-1, 1, self.num_heads, 576])
                if self.heads_need_padding:
                    padded_q = paddle.zeros(
                        [decoder_q.shape[0], decoder_q.shape[1], 64, decoder_q.shape[3]], dtype=decoder_q.dtype
                    )
                    padded_q[:, :, : self.num_heads, :] = decoder_q
                    decoder_q = padded_q

                new_cache_shape = latent_cache.shape
                assert new_cache_shape[1] == 1
                new_cache_shape[1], new_cache_shape[2] = new_cache_shape[2], new_cache_shape[1]

                decoder_res, _ = flash_mla.flash_mla_with_kvcache(
                    decoder_q,
                    # 外面的开源仓库的kv cache存储格式和FD的不同
                    # 幸好这里缓存的头是1，直接view即可，否则上上下下要改很多！
                    latent_cache.view(new_cache_shape),
                    metadata.block_tables,
                    cache_seqlens,
                    512,  # t.dv,
                    tile_scheduler_metadata,
                    num_splits,
                    softmax_scale=self.attn_softmax_scale,
                    causal=True,
                )
                if self.heads_need_padding:
                    decoder_res = decoder_res[:, :, : self.num_heads, :].contiguous()

                final_res = insert_decoder_result_back(
                    decoder_res,
                    forward_meta.cu_seqlens_q,
                    forward_meta.seq_lens_encoder,
                    forward_meta.seq_lens_decoder,
                    token_num,
                )

                return final_res

    @staticmethod
    def flashmla_baseline(decoder_q, latent_cache, block_table, cache_seqlens, attn_softmax_scale):
        page_size = 64
        q_num_heads = decoder_q.shape[2]
        assert decoder_q.shape[1:] == [1, q_num_heads, 576]
        assert latent_cache.shape[1:] == [1, page_size, 576]

        res_baseline = paddle.zeros([decoder_q.shape[0], 1, q_num_heads, 512])
        for batch_id in range(decoder_q.shape[0]):
            kv_len = cache_seqlens[batch_id].item()
            extract_k = paddle.zeros([kv_len, 576], dtype=decoder_q.dtype)
            extract_v = paddle.zeros([kv_len, 512], dtype=decoder_q.dtype)

            for local_seq_id in range(0, kv_len, page_size):
                start = local_seq_id
                end = min(local_seq_id + page_size, kv_len)
                physical_id = block_table[batch_id, local_seq_id // page_size].item()

                page_end = page_size if end % page_size == 0 else end % page_size
                extract_k[start:end, :] = latent_cache[physical_id, 0, :page_end, :]
                extract_v[start:end, :] = latent_cache[physical_id, 0, :page_end, :512]

            this_batch_q = decoder_q[batch_id, 0, :, :]
            p = paddle.matmul(this_batch_q, extract_k.transpose([1, 0]).contiguous())
            p = p * attn_softmax_scale
            p = paddle.nn.functional.softmax(p, -1)
            res_baseline[batch_id, 0, :, :] = paddle.matmul(p, extract_v).contiguous()

        return res_baseline
