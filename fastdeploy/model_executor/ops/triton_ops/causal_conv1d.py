# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py
# Original: Copyright (c) 2024, Tri Dao (Apache License 2.0)
# Adapted for FastDeploy (PaddlePaddle) by PaddlePaddle Authors, 2025.
"""
Causal Conv1d Triton Kernels — FastDeploy 版（GDN Prefill/Decode 路径）。

移植说明:
  - Triton kernel 代码完全不变
  - Python wrapper: torch → paddle
    * torch.empty_like → paddle.empty_like
    * tensor.stride() → tensor.strides  (list, 通过下标取值)
    * .size(0) → .shape[0]
    * torch.empty(x) → paddle.empty([...])
  - 仅保留 GDN 需要的两个接口:
    * causal_conv1d_fn     — Prefill varlen 路径（pool + slot_ids + has_initial_state）
    * causal_conv1d_update — Decode 单 token 路径（pool + slot_ids）
  - 移除 speculative decoding / Eagle tree attention 相关参数（GDN 不需要）
  - 移除 @torch.compiler.disable

公开 API:
    causal_conv1d_fn(x, weight, bias, conv_states, query_start_loc,
                     seq_lens_cpu, cache_indices, has_initial_state, activation)
        x: (dim, cu_seqlen) - 所有序列拼接
        weight: (dim, width)
        bias: (dim,) or None
        conv_states: [max_seqs, dim, width-1]  (pool, in-place 更新)
        query_start_loc: [N+1] int32
        seq_lens_cpu: List[int]
        cache_indices: [N] int32 (slot 索引)
        has_initial_state: [N] bool
        activation: "silu" or None
        → out: (dim, cu_seqlen)

    causal_conv1d_update(x, conv_state, weight, bias, activation, conv_state_indices)
        x: (batch, dim)
        conv_state: [max_seqs, dim, state_len]  (pool, in-place 更新)
        weight: (dim, width)
        bias: (dim,) or None
        activation: "silu" or None
        conv_state_indices: [batch] int32 (slot 索引)
        → out: (batch, dim)
"""

from typing import List, Optional, Union

import paddle
import triton
import triton.language as tl

PAD_SLOT_ID = -1


# ============================================================
# Prefill kernel (unchanged from SGLang)
# ============================================================


@triton.jit()
def _causal_conv1d_fwd_kernel(
    x_ptr,  # (dim, cu_seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr: [max_seqs, dim, width-1]
    cache_indices_ptr,  # conv_state_indices_ptr: [N]
    has_initial_states_ptr,  # [N] bool
    query_start_loc_ptr,  # [N+1]
    o_ptr,  # (dim, cu_seqlen)
    # dimensions
    dim: tl.constexpr,
    seqlen: tl.int32,
    num_cache_lines: tl.constexpr,
    # strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # meta
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1

    idx_seq = tl.program_id(0)
    chunk_offset = tl.program_id(1)
    idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    if segment_len <= 0:
        return

    x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim

    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            return
    conv_states_base = (
        conv_states_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (idx_feats * stride_conv_state_dim)
    )

    w_base = w_ptr + (idx_feats * stride_w_dim)

    if chunk_offset == 0:
        load_init_state = False
        if HAS_INITIAL_STATES:
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            if KERNEL_WIDTH >= 2:
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        if state_len <= seqlen:
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x = (idx_tokens_last >= 0)[:, None] & (idx_tokens_last < seqlen)[:, None] & (idx_feats < dim)[None, :]
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            conv_states_ptrs_target = conv_states_base[None, :] + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, new_conv_state, mask)
        else:
            if load_init_state:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_state_batch_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )
                mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)
                VAL = state_len - seqlen
                x_ptrs = x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                tl.debug_barrier()
                new_conv_state = tl.where(mask, conv_state, loaded_x)
                conv_states_ptrs_target = conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                VAL = state_len - seqlen
                x_ptrs = x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
                conv_states_ptrs_target = conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token

    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload
        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            acc += matrix_x * matrix_w

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (idx_feats < dim)
        o_ptrs = (
            o_ptr + (sequence_start_index + token_offset + idx_token) * stride_o_token + (idx_feats * stride_o_dim)
        )
        tl.store(o_ptrs, acc, mask=mask_1d)


# ============================================================
# Decode kernel (simplified from SGLang: seqlen=1, no spec decoding)
# ============================================================


@triton.jit()
def _causal_conv1d_update_kernel(
    x_ptr,  # (batch, dim)  — seqlen=1 decode token
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,  # [max_seqs, dim, state_len]
    conv_state_indices_ptr,  # [batch]
    o_ptr,  # (batch, dim)
    # dimensions
    batch: int,
    dim: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    # strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # meta
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # seqlen == 1 for single-token decode
    seqlen = 1

    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_state_indices).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            return

    conv_states_base = (
        conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    # STEP 1: READ old conv_state (sliding window history)
    prior_tokens = conv_states_base  # start at index 0
    if KERNEL_WIDTH >= 2:
        col0 = tl.load(prior_tokens, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        col1 = tl.load(prior_tokens + 1 * stride_conv_state_tok, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        col2 = tl.load(prior_tokens + 2 * stride_conv_state_tok, mask_w, 0.0)

    # STEP 2: Shift-left conv_state and append new x (sliding window update)
    idx_tokens = tl.arange(0, NP2_STATELEN)
    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # [BLOCK_N]

    # Load old state shifted by seqlen=1 (elements [1..state_len-1])
    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + seqlen) * stride_conv_state_tok)[:, None]
    )
    mask_old = (
        (conv_state_batch_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    old_conv_state = tl.load(conv_state_ptrs_source, mask_old, other=0.0)

    # Load new x (only the last slot, VAL = state_len - 1)
    VAL = state_len - seqlen
    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_x_dim)[:, None]
    )  # stride_x_dim used for token offset in dim-contiguous layout
    mask_x = (idx_tokens - VAL >= 0)[:, None] & (idx_tokens - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :]
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)

    tl.debug_barrier()
    new_conv_state = tl.where(mask_old, old_conv_state, loaded_x)

    # Write back new conv_state
    conv_state_ptrs_target = conv_states_base + (idx_tokens * stride_conv_state_tok)[:, None]
    mask_store = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask_store)

    # STEP 3: Load weights and compute convolution output
    if HAS_BIAS:
        acc = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)
    else:
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    w_base = w_ptr + (idx_feats * stride_w_dim)
    if KERNEL_WIDTH >= 2:
        w_col0 = tl.load(w_base + 0 * stride_w_width, mask_w, other=0.0)
        w_col1 = tl.load(w_base + 1 * stride_w_width, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + 2 * stride_w_width, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + 3 * stride_w_width, mask_w, other=0.0)

    x_now = tl.load(x_base, mask_w, 0.0)
    if KERNEL_WIDTH == 2:
        acc += col0 * w_col0 + x_now * w_col1
    elif KERNEL_WIDTH == 3:
        acc += col0 * w_col0 + col1 * w_col1 + x_now * w_col2
    elif KERNEL_WIDTH == 4:
        acc += col0 * w_col0 + col1 * w_col1 + col2 * w_col2 + x_now * w_col3

    if SILU_ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))

    o_ptrs = o_ptr + idx_seq * stride_o_seq + idx_feats * stride_o_dim
    tl.store(o_ptrs, acc, mask=mask_w)


# ============================================================
# Python Wrappers (paddle 版)
# ============================================================


def causal_conv1d_fn(
    x: paddle.Tensor,
    weight: paddle.Tensor,
    bias: Optional[paddle.Tensor],
    conv_states: paddle.Tensor,
    query_start_loc: paddle.Tensor,
    seq_lens_cpu: List[int],
    cache_indices: Optional[paddle.Tensor] = None,
    has_initial_state: Optional[paddle.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
) -> paddle.Tensor:
    """
    Causal conv1d 前向（Prefill varlen 路径）。

    Args:
        x: (dim, cu_seqlen) — 所有序列拼接
        weight: (dim, width) — 卷积核
        bias: (dim,) or None
        conv_states: [max_seqs, dim, width-1] — conv 状态池（in-place 更新）
        query_start_loc: [N+1] int32 — 每个序列在 x 中的起始位置
        seq_lens_cpu: List[int] — 每个序列的长度（host side）
        cache_indices: [N] int32 — 每个序列对应的 pool slot 索引
        has_initial_state: [N] bool — 是否有初始状态（从 pool 读取）
        activation: "silu" or None
        pad_slot_id: padding slot 标记（跳过处理）

    Returns:
        out: (dim, cu_seqlen)
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    out = paddle.empty_like(x)

    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    stride_x_seq = 0
    stride_x_dim = x.strides[0]
    stride_x_token = x.strides[1]
    stride_w_dim = weight.strides[0]
    stride_w_width = weight.strides[1]

    num_cache_lines = 0
    stride_istate_seq = stride_istate_dim = stride_istate_token = 0
    if conv_states is not None:
        num_cache_lines = conv_states.shape[0]
        stride_istate_seq = conv_states.strides[0]
        stride_istate_dim = conv_states.strides[1]
        stride_istate_token = conv_states.strides[2]

    stride_o_seq = 0
    stride_o_dim = out.strides[0]
    stride_o_token = out.strides[1]

    def grid(META):
        max_seq_len = max(seq_lens_cpu)
        return (
            len(seq_lens_cpu),
            (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_fwd_kernel[grid](
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        out,
        dim,
        cu_seqlen,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_INITIAL_STATES=has_initial_state is not None,
        HAS_CACHE=conv_states is not None,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        BLOCK_M=8,
        BLOCK_N=256,
        num_stages=2,
    )
    return out


def causal_conv1d_update(
    x: paddle.Tensor,
    conv_state: paddle.Tensor,
    weight: paddle.Tensor,
    bias: Optional[paddle.Tensor] = None,
    activation: Union[bool, str, None] = None,
    conv_state_indices: Optional[paddle.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
) -> paddle.Tensor:
    """
    Causal conv1d 单 token 更新（Decode 路径）。

    Args:
        x: (batch, dim) — 当前 token
        conv_state: [max_seqs, dim, state_len] — conv 状态池（in-place 更新）
        weight: (dim, width) — 卷积核
        bias: (dim,) or None
        activation: "silu" or None
        conv_state_indices: [batch] int32 — pool slot 索引
        pad_slot_id: padding slot 标记

    Returns:
        out: (batch, dim)
    """
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    batch, dim = x.shape
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.shape

    out = paddle.empty_like(x)

    stride_w_dim, stride_w_width = weight.strides[0], weight.strides[1]
    stride_x_seq, stride_x_dim = x.strides[0], x.strides[1]
    stride_o_seq, stride_o_dim = out.strides[0], out.strides[1]
    stride_istate_seq = conv_state.strides[0]
    stride_istate_dim = conv_state.strides[1]
    stride_istate_token = conv_state.strides[2]
    stride_state_indices = conv_state_indices.strides[0] if conv_state_indices is not None else 0

    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (batch, triton.cdiv(dim, META["BLOCK_N"]))

    _causal_conv1d_update_kernel[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        out,
        batch,
        dim,
        state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=256,
    )
    return out
