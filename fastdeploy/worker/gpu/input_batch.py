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

from dataclasses import dataclass, field
from typing import Optional

import paddle
import numpy as np
import triton
import triton.language as tl

class InputBuffers:
    def __init__(
        self,
        max_num_seqs: int,
        max_num_tokens: int,
        max_position_embeddings: int,
        rotary_dim: int,
    ):
        self.input_ids = paddle.full(max_num_tokens, -1, dtype=paddle.int32)
        self.positions = paddle.full(max_num_tokens, -1, dtype=paddle.int64)
        
        self.query_start_loc = paddle.zeros(max_num_seqs + 1, dtype=paddle.int32)
        self.seq_lens = paddle.zeros(max_num_seqs, dtype=paddle.int32)

        # fp16只用来存储，实际计算流程仍然用float32
        self.cos_sin_buffer = paddle.empty((max_position_embeddings, rotary_dim), dtype=paddle.float16)


@dataclass
class InputBatch:
    num_seqs: int

    # batch_idx -> req_state_idx
    idx_mapping: paddle.Tensor
    idx_mapping_np: np.ndarray

    expanded_idx_mapping: paddle.Tensor
    expanded_local_pos: paddle.Tensor

    num_tokens: int
    num_decode_tokens: int
    num_prefill_tokens: int
    num_draft_tokens: int

    num_decodes: int
    num_prefills: int

    query_start_loc: paddle.Tensor     # [num_seqs + 1]
    query_start_loc_np: np.ndarray     # [num_seqs + 1]

    seq_lens: paddle.Tensor            # [num_seqs]
    seq_lens_np: np.ndarray            # [num_seqs]

    input_ids: paddle.Tensor            # [num_tokens]
    positions: paddle.Tensor

    # [total_num_logits]
    logits_indices: paddle.Tensor
    # [num_seqs + 1]
    cu_num_logits: paddle.Tensor
    cu_num_logits_np: np.ndarray

@triton.jit
def _prepare_prefill_inputs_kernel(
    input_ids_ptr,
    next_prefill_tokens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    if num_computed >= prefill_len:
        # Not prefill.
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    request_ptr = all_token_ids_ptr + req_state_idx * all_token_ids_stride
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        tokens = tl.load(request_ptr + num_computed + block, mask=mask)
        tl.store(input_ids_ptr + query_start + block, tokens, mask=mask)

    next_pos = num_computed + query_len
    if next_pos < prefill_len:
        next_token = tl.load(request_ptr + next_pos)
        tl.store(next_prefill_tokens_ptr + req_state_idx, next_token)


def prepare_prefill_inputs(
    input_ids: paddle.Tensor,
    next_prefill_tokens: paddle.Tensor,
    idx_mapping: paddle.Tensor,
    query_start_loc: paddle.Tensor,
    all_token_ids: paddle.Tensor,
    prefill_len: paddle.Tensor,
    num_computed_tokens: paddle.Tensor,
) -> None:
    num_seqs = idx_mapping.shape[0]
    _prepare_prefill_inputs_kernel[(num_seqs,)](
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        all_token_ids.stride(0),
        prefill_len,
        num_computed_tokens,
        BLOCK_SIZE=1024,
    )


@triton.jit
def _prepare_pos_seq_lens_kernel(
    pos_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_computed_tokens_ptr,
    max_num_seqs,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    num_seqs = tl.num_programs(0) - 1
    if batch_id == num_seqs:
        # Pad unused seq_lens as 0 for full CUDA graphs.
        for i in tl.range(num_seqs, max_num_seqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_seqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    req_state_idx = tl.load(idx_mapping_ptr + batch_id)
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)

    start = tl.load(query_start_loc_ptr + batch_id)
    end = tl.load(query_start_loc_ptr + batch_id + 1)
    query_len = end - start

    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + batch_id, seq_len)

    for i in tl.range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        pos = num_computed_tokens + block
        tl.store(pos_ptr + start + block, pos, mask=mask)


def prepare_pos_seq_lens(
    idx_mapping: paddle.Tensor,
    query_start_loc: paddle.Tensor,
    num_computed_tokens: paddle.Tensor,
    pos: paddle.Tensor,
    seq_lens: paddle.Tensor,
) -> None:
    num_seqs = idx_mapping.shape[0]
    # +1 for the thread block used to pad unused seq_lens.
    _prepare_pos_seq_lens_kernel[(num_seqs + 1,)](
        pos,
        seq_lens,
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        seq_lens.shape[0],
        BLOCK_SIZE=1024,
    )


@triton.jit
def _combine_sampled_and_draft_tokens_kernel(
    input_ids_ptr,
    idx_mapping_ptr,
    last_sampled_tokens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    cu_num_logits_ptr,
    logits_indices_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    # Get the number of logits and draft tokens.
    cu_num_logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    cu_num_logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = cu_num_logits_end - cu_num_logits_start
    num_draft_tokens = num_logits - 1

    # Compute the logits indices.
    block = tl.arange(0, BLOCK_SIZE)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    logits_start = query_end - num_logits
    tl.store(
        logits_indices_ptr + cu_num_logits_start + block,
        logits_start + block,
        mask=block < num_logits,
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if seq_len <= prefill_len:
        # Handling prefill tokens. No sampled or draft tokens.
        return

    # Write the last sampled token ID to input_ids.
    last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
    tl.store(input_ids_ptr + query_end - num_logits, last_token_id)

    # Write the draft tokens (if any) to input_ids.
    if num_draft_tokens > 0:
        mask = block < num_draft_tokens
        draft_tokens = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + block,
            mask=mask,
        )
        tl.store(
            input_ids_ptr + query_end - num_draft_tokens + block,
            draft_tokens,
            mask=mask,
        )


def combine_sampled_and_draft_tokens(
    input_ids: paddle.Tensor,
    idx_mapping: paddle.Tensor,
    last_sampled_tokens: paddle.Tensor,
    query_start_loc: paddle.Tensor,
    seq_lens: paddle.Tensor,
    prefill_len: paddle.Tensor,
    draft_tokens: paddle.Tensor,
    cu_num_logits: paddle.Tensor,
    num_logits: int,
) -> paddle.Tensor:
    # use idx_mapping.shape[0] for actual request count
    num_seqs = idx_mapping.shape[0]
    num_speculative_steps = draft_tokens.shape[-1]

    logits_indices = paddle.empty(
        num_logits,
        dtype=paddle.int64,
    )
    _combine_sampled_and_draft_tokens_kernel[(num_seqs,)](
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        draft_tokens.stride(0),
        cu_num_logits,
        logits_indices,
        BLOCK_SIZE=triton.next_power_of_2(num_speculative_steps + 1),
    )
    return logits_indices


@triton.jit
def _get_num_sampled_and_rejected_kernel(
    num_sampled_ptr,
    num_rejected_ptr,
    seq_lens_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    prefill_len_ptr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    is_chunked_prefilling = seq_len < prefill_len

    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    num_sampled = tl.where(is_chunked_prefilling, 0, num_sampled)
    tl.store(num_sampled_ptr + batch_idx, num_sampled)

    logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = logits_end - logits_start

    num_rejected = num_logits - num_sampled
    num_rejected = tl.where(is_chunked_prefilling, 0, num_rejected)
    tl.store(num_rejected_ptr + batch_idx, num_rejected)


def get_num_sampled_and_rejected(
    num_sampled: paddle.Tensor,
    seq_lens: paddle.Tensor,
    cu_num_logits: paddle.Tensor,
    idx_mapping: paddle.Tensor,
    prefill_len: paddle.Tensor,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    num_seqs = idx_mapping.shape[0]
    num_rejected = paddle.empty_like(num_sampled)
    _get_num_sampled_and_rejected_kernel[(num_seqs,)](
        num_sampled,
        num_rejected,
        seq_lens,
        cu_num_logits,
        idx_mapping,
        prefill_len,
    )
    return num_sampled, num_rejected


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
        token_id = tl.load(
            sampled_tokens_ptr + batch_id * sampled_tokens_stride + num_sampled - 1
        )
        tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)
        tl.store(total_len_ptr + req_state_idx, total_len + num_sampled)

    for i in range(num_sampled):
        token_id = tl.load(sampled_tokens_ptr + batch_id * sampled_tokens_stride + i)
        tl.store(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + total_len + i,
            token_id,
        )

        if output_bin_counts_ptr is not None:
            token_ptr = (
                output_bin_counts_ptr
                + req_state_idx * output_bin_counts_stride
                + token_id
            )
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


@triton.jit
def _expand_idx_mapping_kernel(
    idx_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    cu_num_logits_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + batch_idx)
    end_idx = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_tokens = end_idx - start_idx

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < num_tokens
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    tl.store(expanded_idx_mapping_ptr + start_idx + block, req_state_idx, mask=mask)
    tl.store(expanded_local_pos_ptr + start_idx + block, block, mask=mask)


def expand_idx_mapping(
    idx_mapping: paddle.Tensor,
    total_num_logits: int,
    cu_num_logits: paddle.Tensor,
    max_expand_len: int,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    num_seqs = idx_mapping.shape[0]
    expanded_idx_mapping = idx_mapping.new_empty(total_num_logits)
    expanded_local_pos = paddle.empty(
        total_num_logits, dtype=paddle.int32, device=idx_mapping.device
    )
    _expand_idx_mapping_kernel[(num_seqs,)](
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        cu_num_logits,
        BLOCK_SIZE=triton.next_power_of_2(max_expand_len),
    )
    return expanded_idx_mapping, expanded_local_pos
