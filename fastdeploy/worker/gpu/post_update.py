"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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


@triton.jit
def _post_update_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
):
    batch_id = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_id)

    total_len = tl.load(total_len_ptr + req_state_idx)
    num_sampled = tl.load(num_sampled_ptr + batch_id)
    if num_sampled > 0:
        token_id = tl.load(sampled_tokens_ptr + batch_id * sampled_tokens_stride + num_sampled - 1)
        tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)
        tl.store(total_len_ptr + req_state_idx, total_len + num_sampled)

    for i in range(num_sampled):
        token_id = tl.load(sampled_tokens_ptr + batch_id * sampled_tokens_stride + i)
        tl.store(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + total_len + i,
            token_id,
        )

        if output_bin_counts_ptr is not None:
            token_ptr = output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + token_id
            count = tl.load(token_ptr)
            tl.store(token_ptr, count + 1)

    query_start = tl.load(query_start_loc_ptr + batch_id)
    query_end = tl.load(query_start_loc_ptr + batch_id + 1)
    query_len = query_end - query_start
    num_rejected = tl.load(num_rejected_ptr + batch_id)

    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    num_computed += query_len - num_rejected
    tl.store(num_computed_tokens_ptr + req_state_idx, num_computed)


def post_update(
    # [num_seqs]
    idx_mapping: paddle.Tensor,
    # [max_num_seqs]
    num_computed_tokens: paddle.Tensor,
    # [max_num_seqs]
    last_sampled_tokens: paddle.Tensor,
    # [max_num_seqs, vocab_size]
    output_bin_counts: paddle.Tensor | None,
    # [num_seqs, num_speculative_steps + 1]
    sampled_tokens: paddle.Tensor,
    # [num_seqs]
    num_sampled: paddle.Tensor,
    # [num_seqs]
    num_rejected: paddle.Tensor,
    # [num_seqs + 1]
    query_start_loc: paddle.Tensor,
    # [max_num_seqs, max_model_len]
    all_token_ids: paddle.Tensor,
    # [max_num_seqs]
    total_len: paddle.Tensor,
) -> None:
    num_seqs = idx_mapping.shape[0]
    _post_update_kernel[(num_seqs,)](
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        output_bin_counts.stride(0) if output_bin_counts is not None else 0,
        sampled_tokens,
        sampled_tokens.stride(0),
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        num_warps=1,
    )


@triton.jit
def _post_update_pool_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
):
    batch_id = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + batch_id)
    query_end = tl.load(query_start_loc_ptr + batch_id + 1)
    query_len = query_end - query_start

    req_state_idx = tl.load(idx_mapping_ptr + batch_id)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + query_len)


def post_update_pool(
    # [num_seqs]
    idx_mapping: paddle.Tensor,
    # [max_num_seqs]
    num_computed_tokens: paddle.Tensor,
    # [num_seqs + 1]
    query_start_loc: paddle.Tensor,
) -> None:
    num_seqs = idx_mapping.shape[0]
    _post_update_pool_kernel[(num_seqs,)](
        idx_mapping,
        num_computed_tokens,
        query_start_loc,
    )
