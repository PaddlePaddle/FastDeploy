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
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/utils.py
# Licensed under Apache License 2.0
#
# Triton kernel for writing MLA compressed KV cache into paged buffer.
"""

import paddle
import triton
import triton.language as tl

from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)


@enable_compat_on_triton_kernel
@triton.jit
def _mla_write_cache_kernel(
    compressed_kv_ptr,
    k_pe_ptr,
    cache_ptr,
    slot_mapping_ptr,
    stride_ckv_token,
    stride_kpe_token,
    stride_cache_block,
    stride_cache_bs,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    """
    Write [compressed_kv || k_pe] into paged cache at slot_mapping positions.

    cache layout: [num_blocks, 1, block_size, head_dim]
    slot_mapping[i] = block_id * block_size + offset
    """
    pid_token = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = kv_lora_rank + qk_rope_head_dim
    mask = offs < total_dim

    slot = tl.load(slot_mapping_ptr + pid_token).to(tl.int64)
    block_id = slot // KV_BLOCK_SIZE
    offset_in_block = slot % KV_BLOCK_SIZE

    # Bounds check: skip if block_id is out of range
    if block_id >= NUM_BLOCKS or block_id < 0:
        return

    dst_ptr = cache_ptr + block_id * stride_cache_block + offset_in_block * stride_cache_bs + offs

    if base + BLOCK <= kv_lora_rank:
        src = tl.load(compressed_kv_ptr + pid_token * stride_ckv_token + offs, mask=mask)
    elif base >= kv_lora_rank:
        offs_rope = offs - kv_lora_rank
        src = tl.load(k_pe_ptr + pid_token * stride_kpe_token + offs_rope, mask=mask)
    else:
        is_nope = offs < kv_lora_rank
        is_rope = (offs >= kv_lora_rank) & (offs < total_dim)
        src_nope = tl.load(
            compressed_kv_ptr + pid_token * stride_ckv_token + offs,
            mask=mask & is_nope,
            other=0,
        )
        src_rope = tl.load(
            k_pe_ptr + pid_token * stride_kpe_token + (offs - kv_lora_rank),
            mask=mask & is_rope,
            other=0,
        )
        src = tl.where(is_nope, src_nope, src_rope)

    tl.store(dst_ptr, src, mask=mask)


def mla_write_cache_triton(
    compressed_kv: paddle.Tensor,
    k_pe: paddle.Tensor,
    latent_cache: paddle.Tensor,
    slot_mapping: paddle.Tensor,
):
    """
    Write [compressed_kv || k_pe] into paged latent_cache at slot_mapping positions.

    Args:
        compressed_kv: [num_tokens, kv_lora_rank]
        k_pe: [num_tokens, 1, qk_rope_head_dim] or [num_tokens, qk_rope_head_dim]
        latent_cache: [num_blocks, 1, block_size, kv_lora_rank + qk_rope_head_dim]
        slot_mapping: [num_tokens] int64
    """
    num_tokens = compressed_kv.shape[0]
    if num_tokens == 0:
        return

    kv_lora_rank = compressed_kv.shape[-1]
    k_pe_flat = k_pe.reshape([num_tokens, -1])
    qk_rope_head_dim = k_pe_flat.shape[-1]
    total_dim = kv_lora_rank + qk_rope_head_dim

    kv_block_size = latent_cache.shape[2]

    BLOCK = 128
    grid = (num_tokens, triton.cdiv(total_dim, BLOCK))

    # stride for cache: [num_blocks, 1, block_size, head_dim]
    # stride_cache_block = 1 * block_size * head_dim
    # stride_cache_bs = head_dim (stride along block_size dim)
    stride_cache_block = latent_cache.strides[0]
    stride_cache_bs = latent_cache.strides[2]

    num_blocks = latent_cache.shape[0]

    _mla_write_cache_kernel[grid](
        compressed_kv,
        k_pe_flat,
        latent_cache,
        slot_mapping,
        compressed_kv.strides[0],
        k_pe_flat.strides[0],
        stride_cache_block,
        stride_cache_bs,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        KV_BLOCK_SIZE=kv_block_size,
        BLOCK=BLOCK,
        NUM_BLOCKS=num_blocks,
    )
