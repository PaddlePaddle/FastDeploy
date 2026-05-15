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
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
# Licensed under Apache License 2.0
#
# Memory-efficient split-KV attention for decoding, adapted for paged KV cache.
"""

import paddle
import triton
import triton.language as tl

from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)

_MIN_BLOCK_KV = 32


@enable_compat_on_triton_kernel
@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kb,
    stride_buf_kh,
    stride_buf_kt,
    stride_buf_vb,
    stride_buf_vh,
    stride_buf_vt,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    """
    Split-KV decode attention stage 1 for grouped query attention on paged cache.
    Each program handles (batch, head_group, kv_split).
    """
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]

    kv_len_per_split = tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        if BLOCK_DPE > 0:
            qpe = tl.load(Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            # Decompose flat index into (block_id, offset_in_block) for paged cache
            kv_block_id = kv_loc // KV_BLOCK_SIZE
            kv_offset = kv_loc % KV_BLOCK_SIZE

            # Load K: cache shape [num_blocks, kv_heads, block_size, head_dim]
            offs_buf_k = (
                kv_block_id[None, :] * stride_buf_kb
                + cur_kv_head * stride_buf_kh
                + kv_offset[None, :] * stride_buf_kt
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = (
                    kv_block_id[None, :] * stride_buf_kb
                    + cur_kv_head * stride_buf_kh
                    + kv_offset[None, :] * stride_buf_kt
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf"))

            # Load V from paged cache
            offs_buf_v = (
                kv_block_id[:, None] * stride_buf_vb
                + cur_kv_head * stride_buf_vh
                + kv_offset[:, None] * stride_buf_vt
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


@enable_compat_on_triton_kernel
@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    Mid_O_1,
    O,
    kv_indptr,
    num_kv_splits,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    """
    Stage 2: reduce across KV splits to produce final output.
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    # Guard against e_sum==0 (empty sequences from CUDAGraph padding) to avoid NaN
    safe_e_sum = tl.where(e_sum == 0.0, 1.0, e_sum)
    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        tl.where(e_sum == 0.0, 0.0, acc / safe_e_sum),
        mask=mask_d,
    )


@enable_compat_on_triton_kernel
@triton.jit
def _compute_num_kv_splits_kernel(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq: tl.constexpr,
    max_kv_splits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compute number of KV splits per sequence based on seq_len."""
    idx = tl.arange(0, BLOCK)
    mask = idx < num_seq
    seq_len = tl.load(seq_lens_ptr + idx, mask=mask, other=0)
    splits = (seq_len + 255) // 256
    splits = tl.minimum(splits, max_kv_splits)
    splits = tl.maximum(splits, 1)
    tl.store(num_kv_splits_ptr + idx, splits, mask=mask)


def compute_num_kv_splits(seq_lens, num_seq, max_kv_splits, out_buf=None):
    """
    Compute number of KV splits per sequence. CUDA Graph compatible.

    Args:
        seq_lens: [num_seq] int32 tensor of sequence lengths
        num_seq: number of sequences
        max_kv_splits: maximum number of splits
        out_buf: Optional pre-allocated buffer. If provided, writes into it.

    Returns:
        num_kv_splits: [num_seq] int32 tensor (or out_buf if provided)
    """
    if out_buf is not None:
        num_kv_splits = out_buf
    else:
        num_kv_splits = paddle.empty([num_seq], dtype="int32")
    if num_seq == 0:
        return num_kv_splits
    BLOCK = triton.next_power_of_2(num_seq)
    _compute_num_kv_splits_kernel[(1,)](
        num_kv_splits, seq_lens, num_seq=num_seq, max_kv_splits=max_kv_splits, BLOCK=BLOCK
    )
    return num_kv_splits


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    kv_block_size,
):
    """Launch stage 1 grouped decode attention kernel."""
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK_H = 16
    MAX_KV_SPLITS = max_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.strides[0],
        q.strides[1],
        k_buffer.strides[0],
        k_buffer.strides[1],
        k_buffer.strides[2],
        v_buffer.strides[0],
        v_buffer.strides[1],
        v_buffer.strides[2],
        att_out.strides[0],
        att_out.strides[1],
        att_out.strides[2],
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        KV_BLOCK_SIZE=kv_block_size,
        num_warps=4,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


def _decode_softmax_reducev_fwd(
    logits,
    lse,
    q,
    o,
    v_buffer,
    kv_indptr,
    num_kv_splits,
    max_kv_splits,
):
    """Launch stage 2 reduce kernel."""
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    MAX_KV_SPLITS = max_kv_splits

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        lse,
        o,
        kv_indptr,
        num_kv_splits,
        logits.strides[0],
        logits.strides[1],
        logits.strides[2],
        o.strides[0],
        o.strides[1],
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    kv_block_size,
):
    """
    Triton decode attention for paged KV cache (split-KV approach).

    Args:
        q: [batch, num_heads, Lk] query tensor
        k_buffer: [num_blocks, kv_heads, block_size, Lk] paged key cache
        v_buffer: [num_blocks, kv_heads, block_size, Lv] paged value cache
        o: [batch, num_heads, Lv] output tensor
        kv_indptr: [batch+1] CSR indptr for KV
        kv_indices: [total_kv_len] flat token indices (block_id * block_size + offset)
        attn_logits: [batch, num_heads, max_kv_splits, Lv] intermediate buffer
        attn_lse: [batch, num_heads, max_kv_splits] intermediate lse buffer
        num_kv_splits: [batch] number of splits per sequence
        max_kv_splits: int, maximum number of splits
        sm_scale: float, softmax scale
        kv_block_size: int, the page block size
    """
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        kv_block_size,
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
    )
