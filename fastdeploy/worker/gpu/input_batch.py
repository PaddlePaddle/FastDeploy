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

from dataclasses import dataclass

import numpy as np
import paddle
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

        # bf16 for storage; the fused_rotary_position_encoding op converts to fp32 internally.
        self.cos_sin_buffer = paddle.empty((max_position_embeddings, rotary_dim), dtype=paddle.bfloat16)


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

    query_start_loc: paddle.Tensor  # [num_seqs + 1]
    query_start_loc_np: np.ndarray  # [num_seqs + 1]

    seq_lens: paddle.Tensor  # [num_seqs]
    seq_lens_np: np.ndarray  # [num_seqs]

    input_ids: paddle.Tensor  # [num_tokens]
    positions: paddle.Tensor

    # [total_num_logits]
    logits_indices: paddle.Tensor
    # [num_seqs + 1]
    cu_num_logits: paddle.Tensor
    cu_num_logits_np: np.ndarray


@triton.jit
def _prepare_query_start_loc_and_seq_lens_kernel(
    sorted_num_tokens_per_seq_ptr,  # [num_seqs] int32, sorted P-first / D-after
    idx_mapping_ptr,  # [num_seqs] int32, batch_idx -> req_state_idx
    num_computed_tokens_ptr,  # [max_num_seqs] int32, indexed by req_state_idx
    query_start_loc_ptr,  # [max_num_seqs + 1] int32, output
    seq_lens_ptr,  # [max_num_seqs] int32, output
    num_seqs,
    max_num_seqs,
    draft_tokens_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-block kernel that computes per-seq query length, its cumulative
    sum (query_start_loc) and full sequence length (seq_lens) in one pass.

    For each batch slot:
        query_len = num_tokens                    if prefill (num_tokens > 1)
                  = num_tokens + draft_tokens_len if decode  (num_tokens == 1)
        query_start_loc[0]   = 0
        query_start_loc[i+1] = sum(query_len[:i+1])
        seq_lens[i]          = num_computed_tokens[idx_mapping[i]] + query_len[i]

    Inactive slots (>= num_seqs, < max_num_seqs) in seq_lens are zeroed
    for full CUDA-graph compatibility.
    """
    offs = tl.arange(0, BLOCK_SIZE)
    active_mask = offs < num_seqs

    num_tokens = tl.load(sorted_num_tokens_per_seq_ptr + offs, mask=active_mask, other=0)
    is_decode = num_tokens == 1
    query_len = tl.where(is_decode, num_tokens + draft_tokens_len, num_tokens)
    # Force inactive lanes to 0 so they don't contaminate the cumsum.
    query_len = tl.where(active_mask, query_len, 0)

    cum = tl.cumsum(query_len, axis=0)

    # query_start_loc[0] = 0; query_start_loc[1..num_seqs] = cum[0..num_seqs-1]
    tl.store(query_start_loc_ptr, 0)
    tl.store(query_start_loc_ptr + offs + 1, cum, mask=active_mask)

    # seq_lens for active rows
    req_state_idx = tl.load(idx_mapping_ptr + offs, mask=active_mask, other=0)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx, mask=active_mask, other=0)
    seq_lens_val = num_computed + query_len
    tl.store(seq_lens_ptr + offs, seq_lens_val, mask=active_mask)

    # Pad inactive seq_lens to 0 for CUDA-graph friendliness.
    pad_mask = (offs >= num_seqs) & (offs < max_num_seqs)
    tl.store(seq_lens_ptr + offs, 0, mask=pad_mask)


def prepare_query_start_loc_and_seq_lens(
    sorted_num_tokens_per_seq: paddle.Tensor,  # [num_seqs] int32
    idx_mapping: paddle.Tensor,  # [num_seqs] int32
    num_computed_tokens: paddle.Tensor,  # [max_num_seqs] int32
    query_start_loc: paddle.Tensor,  # [max_num_seqs + 1] int32 (InputBuffer)
    seq_lens: paddle.Tensor,  # [max_num_seqs] int32 (InputBuffer)
    draft_tokens_len: int,
) -> None:
    num_seqs = idx_mapping.shape[0]
    max_num_seqs = seq_lens.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(max_num_seqs)
    _prepare_query_start_loc_and_seq_lens_kernel[(1,)](
        sorted_num_tokens_per_seq,
        idx_mapping,
        num_computed_tokens,
        query_start_loc,
        seq_lens,
        num_seqs,
        max_num_seqs,
        draft_tokens_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def _prepare_input_ids_kernel(
    # Outputs
    input_ids_ptr,  # [max_num_tokens] int32
    positions_ptr,  # [max_num_tokens] int64
    logits_indices_ptr,  # [total_num_logits] int64
    expanded_idx_mapping_ptr,  # [total_num_logits] int32
    expanded_local_pos_ptr,  # [total_num_logits] int32
    # Per-seq metadata
    idx_mapping_ptr,  # [num_seqs] int32, batch_idx -> req_state_idx
    query_start_loc_ptr,  # [num_seqs + 1] int32
    cu_num_logits_ptr,  # [num_seqs + 1] int32
    # Req-state indexed inputs
    num_computed_tokens_ptr,  # [max_num_seqs] int32
    prefill_len_ptr,  # [max_num_seqs] int32
    last_sampled_tokens_ptr,  # [max_num_seqs] int32
    # Token-id source buffers
    all_token_ids_ptr,  # [max_num_seqs, max_model_len] int32
    all_token_ids_stride,
    draft_tokens_ptr,  # [max_num_seqs, num_spec_tokens] int32
    draft_tokens_stride,
    BLOCK_SIZE_Q: tl.constexpr,  # covers max query_len
    BLOCK_SIZE_L: tl.constexpr,  # covers max num_logits (== num_spec_tokens + 1)
):
    """
    Per-seq fused kernel. For each batch slot it writes:
      - positions[query_start:query_end] = num_computed + arange(query_len)
      - input_ids[query_start:query_end]:
          * if prefill (num_computed < prefill_len): copy from all_token_ids
          * else (decode/draft): tail = last_sampled_token followed by draft_tokens
            (the leading prefix, if any, was written elsewhere; for pure decode
             query_len == 1 + num_draft_tokens so the tail covers everything)
      - logits_indices[cu_lo:cu_hi]       = (query_end - num_logits) + arange(num_logits)
      - expanded_idx_mapping[cu_lo:cu_hi] = req_state_idx
      - expanded_local_pos[cu_lo:cu_hi]   = arange(num_logits)
    """
    batch_idx = tl.program_id(0)

    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)

    is_prefill = num_computed < prefill_len

    # ---- positions + (prefill) input_ids over [query_start, query_end) ----
    request_token_ptr = all_token_ids_ptr + req_state_idx * all_token_ids_stride
    for i in tl.range(0, query_len, BLOCK_SIZE_Q):
        block = i + tl.arange(0, BLOCK_SIZE_Q)
        mask = block < query_len

        # positions: always written
        pos = num_computed + block
        tl.store(positions_ptr + query_start + block, pos, mask=mask)

        # input_ids prefill copy (decode/draft tail handled below)
        if is_prefill:
            tokens = tl.load(request_token_ptr + num_computed + block, mask=mask)
            tl.store(input_ids_ptr + query_start + block, tokens, mask=mask)

    # ---- logits_indices / expanded_idx_mapping / expanded_local_pos ----
    cu_lo = tl.load(cu_num_logits_ptr + batch_idx)
    cu_hi = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = cu_hi - cu_lo

    block_l = tl.arange(0, BLOCK_SIZE_L)
    mask_l = block_l < num_logits

    logits_start = query_end - num_logits
    tl.store(logits_indices_ptr + cu_lo + block_l, logits_start + block_l, mask=mask_l)
    tl.store(expanded_idx_mapping_ptr + cu_lo + block_l, req_state_idx, mask=mask_l)
    tl.store(expanded_local_pos_ptr + cu_lo + block_l, block_l, mask=mask_l)

    # ---- decode/draft tail of input_ids ----
    # Skip when this seq is still in (chunked) prefill.
    if not is_prefill:
        # Last sampled token sits at query_end - num_logits.
        last_token = tl.load(last_sampled_tokens_ptr + req_state_idx)
        tl.store(input_ids_ptr + query_end - num_logits, last_token)

        num_draft_tokens = num_logits - 1
        if num_draft_tokens > 0:
            mask_d = block_l < num_draft_tokens
            draft = tl.load(
                draft_tokens_ptr + req_state_idx * draft_tokens_stride + block_l,
                mask=mask_d,
            )
            tl.store(
                input_ids_ptr + query_end - num_draft_tokens + block_l,
                draft,
                mask=mask_d,
            )


def prepare_input_ids(
    # Outputs (InputBuffer slices for the first three; freshly-allocated for the last two)
    input_ids: paddle.Tensor,  # [max_num_tokens] int32
    positions: paddle.Tensor,  # [max_num_tokens] int64
    logits_indices: paddle.Tensor,  # [total_num_logits] int64
    expanded_idx_mapping: paddle.Tensor,  # [total_num_logits] int32
    expanded_local_pos: paddle.Tensor,  # [total_num_logits] int32
    # Per-seq / req-state metadata
    idx_mapping: paddle.Tensor,  # [num_seqs] int32
    query_start_loc: paddle.Tensor,  # [num_seqs + 1] int32
    cu_num_logits: paddle.Tensor,  # [num_seqs + 1] int32
    num_computed_tokens: paddle.Tensor,  # [max_num_seqs] int32
    prefill_len: paddle.Tensor,  # [max_num_seqs] int32
    last_sampled_tokens: paddle.Tensor,  # [max_num_seqs] int32
    # Token-id sources
    all_token_ids: paddle.Tensor,  # [max_num_seqs, max_model_len] int32
    draft_tokens: paddle.Tensor,  # [max_num_seqs, num_spec_tokens] int32
    # Sizing constants for inner loops
    max_query_len: int,
    max_num_logits: int,
) -> None:
    """
    Fused per-seq kernel that produces input_ids, positions, logits_indices,
    expanded_idx_mapping, expanded_local_pos in a single launch. Must be
    called AFTER prepare_query_start_loc_and_seq_lens (depends on the
    populated query_start_loc).
    """
    num_seqs = idx_mapping.shape[0]
    BLOCK_SIZE_Q = triton.next_power_of_2(max(max_query_len, 1))
    BLOCK_SIZE_L = triton.next_power_of_2(max(max_num_logits, 1))
    _prepare_input_ids_kernel[(num_seqs,)](
        input_ids,
        positions,
        logits_indices,
        expanded_idx_mapping,
        expanded_local_pos,
        idx_mapping,
        query_start_loc,
        cu_num_logits,
        num_computed_tokens,
        prefill_len,
        last_sampled_tokens,
        all_token_ids,
        all_token_ids.stride(0),
        draft_tokens,
        draft_tokens.stride(0),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_L=BLOCK_SIZE_L,
    )


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
