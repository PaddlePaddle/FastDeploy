"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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
"""Triton post-processing kernels for the GPU sampler.

Each kernel is single-purpose and only iterates the current batch
(``num_seqs``) — never ``max_num_seqs``.  Per-request hyper-parameters
live in ``SamplingStates`` / ``RequestState`` buffers indexed by the
original request slot, which is looked up through ``idx_mapping``:

    batch_idx (0..num_seqs)  --idx_mapping-->  req_idx (0..max_num_seqs)

The Sampler is responsible for orchestrating the call order.
"""

import paddle
import triton
import triton.language as tl


NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# check_stop
# ---------------------------------------------------------------------------
@triton.jit
def _check_stop_kernel(
    sampled_tokens_ptr,        # [sum(num_sampled)]
    num_sampled_ptr,            # [num_seqs]
    num_sampled_cu_ptr,         # [num_seqs + 1] prefix-sum of num_sampled
    idx_mapping_ptr,            # [num_seqs]
    eos_token_ids_ptr,          # [num_eos]
    stop_token_ids_ptr,         # [max_num_seqs, max_stop_len]
    stop_token_stride,
    num_stop_token_ids_ptr,     # [max_num_seqs]
    min_dec_len_ptr,            # [max_num_seqs]
    max_dec_len_ptr,            # [max_num_seqs]
    step_idx_ptr,               # [max_num_seqs]  -- tokens emitted so far
    stop_flags_ptr,             # [num_seqs] bool, output
    num_eos: tl.constexpr,
    MAX_STOP_LEN: tl.constexpr,
):
    bid = tl.program_id(0)
    req_idx = tl.load(idx_mapping_ptr + bid)

    step = tl.load(step_idx_ptr + req_idx)
    min_dl = tl.load(min_dec_len_ptr + req_idx)
    max_dl = tl.load(max_dec_len_ptr + req_idx)

    # Range of sampled tokens for this sequence.
    tok_start = tl.load(num_sampled_cu_ptr + bid)
    tok_end = tl.load(num_sampled_cu_ptr + bid + 1)

    num_stop = tl.load(num_stop_token_ids_ptr + req_idx)

    stop = False
    for t in range(tok_start, tok_end):
        token = tl.load(sampled_tokens_ptr + t)
        step_t = step + (t - tok_start) + 1

        # Force stop on max length.
        if step_t >= max_dl:
            stop = True

        # Honor min_dec_len: only allow early stop once min length is hit.
        if step_t >= min_dl:
            # eos match
            eos_offsets = tl.arange(0, num_eos)
            eos_vals = tl.load(eos_token_ids_ptr + eos_offsets)
            if tl.sum((eos_vals == token).to(tl.int32)) > 0:
                stop = True
            # per-request stop token match
            if num_stop > 0:
                stop_offsets = tl.arange(0, MAX_STOP_LEN)
                mask = stop_offsets < num_stop
                stop_vals = tl.load(
                    stop_token_ids_ptr + req_idx * stop_token_stride + stop_offsets,
                    mask=mask,
                    other=-1,
                )
                if tl.sum(((stop_vals == token) & mask).to(tl.int32)) > 0:
                    stop = True

    tl.store(stop_flags_ptr + bid, stop)


def check_stop(
    sampled_tokens: paddle.Tensor,        # [sum(num_sampled)] int32/int64
    num_sampled: paddle.Tensor,           # [num_seqs] int32
    num_sampled_cu: paddle.Tensor,        # [num_seqs + 1] int32
    idx_mapping: paddle.Tensor,           # [num_seqs] int32
    eos_token_ids: paddle.Tensor,         # [num_eos] int64
    stop_token_ids: paddle.Tensor,        # [max_num_seqs, max_stop_len] int32 (staged)
    num_stop_token_ids: paddle.Tensor,    # [max_num_seqs] int32 (staged)
    min_dec_len: paddle.Tensor,           # [max_num_seqs] int32
    max_dec_len: paddle.Tensor,           # [max_num_seqs] int32
    step_idx: paddle.Tensor,              # [max_num_seqs] int32
) -> paddle.Tensor:
    """Return ``stop_flags`` of length ``num_seqs`` (no max_num_seqs loop)."""
    num_seqs = idx_mapping.shape[0]
    stop_flags = paddle.zeros([num_seqs], dtype=paddle.bool)
    if num_seqs == 0:
        return stop_flags

    max_stop_len = stop_token_ids.shape[1]
    num_eos = eos_token_ids.shape[0]

    _check_stop_kernel[(num_seqs,)](
        sampled_tokens,
        num_sampled,
        num_sampled_cu,
        idx_mapping,
        eos_token_ids,
        stop_token_ids,
        stop_token_ids.stride(0),
        num_stop_token_ids,
        min_dec_len,
        max_dec_len,
        step_idx,
        stop_flags,
        num_eos=triton.next_power_of_2(max(1, num_eos)),
        MAX_STOP_LEN=triton.next_power_of_2(max(1, max_stop_len)),
    )
    return stop_flags


# ---------------------------------------------------------------------------
# apply_repetition_penalty
# ---------------------------------------------------------------------------
@triton.jit
def _apply_repetition_penalty_kernel(
    logits_ptr,              # [num_seqs, vocab]
    logits_stride,
    output_bin_counts_ptr,   # [max_num_seqs, vocab] int32
    bin_stride,
    repetition_penalty_ptr,  # [max_num_seqs] float32
    idx_mapping_ptr,         # [num_seqs]
    vocab_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    col = tl.program_id(1)
    req_idx = tl.load(idx_mapping_ptr + bid)
    penalty = tl.load(repetition_penalty_ptr + req_idx)

    offs = col * BLOCK + tl.arange(0, BLOCK)
    mask = offs < vocab_size

    lg_ptr = logits_ptr + bid * logits_stride + offs
    cnt_ptr = output_bin_counts_ptr + req_idx * bin_stride + offs

    logits = tl.load(lg_ptr, mask=mask, other=0.0).to(tl.float32)
    counts = tl.load(cnt_ptr, mask=mask, other=0)

    has_appeared = counts > 0
    # logits > 0: divide; logits <= 0: multiply
    penalized = tl.where(logits > 0, logits / penalty, logits * penalty)
    logits = tl.where(has_appeared, penalized, logits)

    tl.store(lg_ptr, logits, mask=mask)


def apply_repetition_penalty(
    logits: paddle.Tensor,              # [num_seqs, vocab] fp32 (inplace)
    output_bin_counts: paddle.Tensor,   # [max_num_seqs, vocab] int32
    repetition_penalty: paddle.Tensor,  # [max_num_seqs] fp32
    idx_mapping: paddle.Tensor,         # [num_seqs] int32
    block: int = 1024,
) -> paddle.Tensor:
    num_seqs, vocab = logits.shape
    if num_seqs == 0:
        return logits
    grid = (num_seqs, triton.cdiv(vocab, block))
    _apply_repetition_penalty_kernel[grid](
        logits,
        logits.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        repetition_penalty,
        idx_mapping,
        vocab_size=vocab,
        BLOCK=block,
    )
    return logits


# ---------------------------------------------------------------------------
# apply_bad_words_mask
# ---------------------------------------------------------------------------
@triton.jit
def _apply_bad_words_mask_kernel(
    logits_ptr,              # [num_seqs, vocab]
    logits_stride,
    bad_word_token_ids_ptr,  # [max_num_seqs, MAX_BAD_WORDS_TOTAL_TOKENS] int32
    bad_word_token_stride,
    bad_word_offsets_ptr,    # [max_num_seqs, MAX_NUM_BAD_WORDS + 1] int32
    bad_word_offsets_stride,
    num_bad_words_ptr,       # [max_num_seqs] int32
    idx_mapping_ptr,         # [num_seqs]
    vocab_size,
    MAX_NUM_BAD_WORDS: tl.constexpr,
):
    bid = tl.program_id(0)
    req_idx = tl.load(idx_mapping_ptr + bid)
    n_bad = tl.load(num_bad_words_ptr + req_idx)
    if n_bad <= 0:
        return

    # Each bad word is a single token (sampler_state enforces this): the
    # token id is stored directly at bad_word_token_ids[req_idx, k] for
    # k in [0, n_bad).  bad_word_offsets encodes cumulative lengths so we
    # rely on offsets[k+1]-offsets[k] == 1 and mask logits[:, token_id].
    lg_base = logits_ptr + bid * logits_stride
    ks = tl.arange(0, MAX_NUM_BAD_WORDS)
    mask = ks < n_bad
    token_ids = tl.load(
        bad_word_token_ids_ptr + req_idx * bad_word_token_stride + ks,
        mask=mask,
        other=-1,
    )
    in_vocab = (token_ids >= 0) & (token_ids < vocab_size) & mask
    tl.store(lg_base + token_ids, NEG_INF, mask=in_vocab)


def apply_bad_words_mask(
    logits: paddle.Tensor,              # [num_seqs, vocab] (inplace)
    bad_word_token_ids: paddle.Tensor,  # [max_num_seqs, MAX_BAD_WORDS_TOTAL_TOKENS]
    bad_word_offsets: paddle.Tensor,    # [max_num_seqs, MAX_NUM_BAD_WORDS + 1]
    num_bad_words: paddle.Tensor,       # [max_num_seqs]
    idx_mapping: paddle.Tensor,         # [num_seqs]
) -> paddle.Tensor:
    num_seqs, vocab = logits.shape
    if num_seqs == 0:
        return logits
    max_num_bad_words = bad_word_offsets.shape[1] - 1
    _apply_bad_words_mask_kernel[(num_seqs,)](
        logits,
        logits.stride(0),
        bad_word_token_ids,
        bad_word_token_ids.stride(0),
        bad_word_offsets,
        bad_word_offsets.stride(0),
        num_bad_words,
        idx_mapping,
        vocab,
        MAX_NUM_BAD_WORDS=triton.next_power_of_2(max(1, max_num_bad_words)),
    )
    return logits


# ---------------------------------------------------------------------------
# apply_temperature
# ---------------------------------------------------------------------------
@triton.jit
def _apply_temperature_kernel(
    logits_ptr,        # [num_seqs, vocab]
    logits_stride,
    temperature_ptr,   # [max_num_seqs]
    idx_mapping_ptr,   # [num_seqs]
    vocab_size,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    col = tl.program_id(1)
    req_idx = tl.load(idx_mapping_ptr + bid)
    temp = tl.load(temperature_ptr + req_idx)

    offs = col * BLOCK + tl.arange(0, BLOCK)
    mask = offs < vocab_size
    lg_ptr = logits_ptr + bid * logits_stride + offs
    logits = tl.load(lg_ptr, mask=mask, other=0.0).to(tl.float32)
    # Guard against temperature==0 (greedy) — caller typically sets 1.0 then
    # uses argmax, but keep this safe: treat 0 as 1.0.
    temp = tl.where(temp > 0.0, temp, 1.0)
    logits = logits / temp
    tl.store(lg_ptr, logits, mask=mask)


def apply_temperature(
    logits: paddle.Tensor,              # [num_seqs, vocab] fp32 (inplace)
    temperature: paddle.Tensor,         # [max_num_seqs] fp32
    idx_mapping: paddle.Tensor,         # [num_seqs] int32
    block: int = 1024,
) -> paddle.Tensor:
    num_seqs, vocab = logits.shape
    if num_seqs == 0:
        return logits
    grid = (num_seqs, triton.cdiv(vocab, block))
    _apply_temperature_kernel[grid](
        logits,
        logits.stride(0),
        temperature,
        idx_mapping,
        vocab,
        BLOCK=block,
    )
    return logits
