"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import triton
import triton.language as tl

from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)


@enable_compat_on_triton_kernel
@triton.jit
def update_attn_mask_offsets_kernel(
    seq_lens_this_time_ptr,
    seq_lens_encoder_ptr,
    cu_seqlens_k_ptr,
    attn_mask_offsets_ptr,
    BLOCK_M: tl.constexpr,
):
    """
    seq_lens_this_time: [bsz]
    seq_lens_encoder: [bsz]
    cu_seqlens_k: [bsz+1]
    attn_mask_offsets: [num_tokens * 2], even indices = start, odd indices = end
    """
    batch_id = tl.program_id(0)

    seq_len_encoder = tl.load(seq_lens_encoder_ptr + batch_id)

    # Skip decode requests (seq_lens_encoder == 0); offsets remain 0
    if seq_len_encoder <= 0:
        return

    seq_len_this_time = tl.load(seq_lens_this_time_ptr + batch_id)
    token_start_k = tl.load(cu_seqlens_k_ptr + batch_id)  # start offset in flattened token dim

    for block_start in range(0, seq_len_this_time, BLOCK_M):
        offsets = block_start + tl.arange(0, BLOCK_M)
        mask = offsets < seq_len_this_time

        global_token_idx = token_start_k + offsets  # [BLOCK_M]

        # causal window: start = batch k start (same for all tokens), end = current token + 1
        ks = tl.full((BLOCK_M,), token_start_k, dtype=tl.int32)
        ke = (global_token_idx + 1).to(tl.int32)

        tl.store(attn_mask_offsets_ptr + global_token_idx * 2, ks, mask=mask)
        tl.store(attn_mask_offsets_ptr + global_token_idx * 2 + 1, ke, mask=mask)


def update_indexer_attn_mask_offsets(
    ids_remove_padding,
    seq_lens_this_time,
    seq_lens_encoder,
    cu_seqlens_k,
):
    assert ids_remove_padding.ndim == 1
    assert seq_lens_this_time.ndim == 1
    assert seq_lens_encoder.ndim == 1
    assert cu_seqlens_k.ndim == 1
    num_tokens = ids_remove_padding.shape[0]
    bsz = seq_lens_this_time.shape[0]
    attention_mask_offset = paddle.zeros((num_tokens * 2), dtype=paddle.int32)

    BLOCK_M = 128
    grid = (bsz,)

    update_attn_mask_offsets_kernel[grid](
        seq_lens_this_time,
        seq_lens_encoder,
        cu_seqlens_k,
        attention_mask_offset,
        BLOCK_M,
    )
    return attention_mask_offset
