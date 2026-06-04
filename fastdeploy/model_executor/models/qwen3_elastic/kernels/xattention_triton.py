"""Triton @jit kernels lifted from ``elasticattn.src.Xattention``.

Triton accepts any tensor exposing ``data_ptr()`` (paddle Tensor does), so
the kernels themselves are framework-agnostic; only the Python wrappers
(``flat_group_gemm_fuse_reshape`` / ``softmax_fuse_block_sum``) need to
target paddle.
"""

from __future__ import annotations

import math

import paddle
import triton
import triton.language as tl


@triton.jit
def softmax_fuse_block_sum_kernel_causal(
    In, Out, scale,
    input_stride_0, input_stride_1, input_stride_2,
    output_stride_0, output_stride_1, output_stride_2,
    real_q_len, k_len, chunk_start, chunk_end,
    segment_size: tl.constexpr, block_size: tl.constexpr,
):
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    offs_q = tl.arange(0, block_size) + chunk_start + block_id * block_size
    offs_k = tl.arange(0, segment_size)

    num_iters = k_len // segment_size
    num_iters_before_causal = (chunk_start + (block_id + 1) * block_size - 1) // segment_size

    m_i = tl.zeros([block_size], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_size], dtype=tl.float32) + 1.0

    input_ptr = In + batch_id * input_stride_0 + head_id * input_stride_1 + block_id * block_size * input_stride_2
    input_ptr = input_ptr + tl.arange(0, segment_size) + tl.arange(0, block_size)[:, None] * input_stride_2

    output_ptr = Out + batch_id * output_stride_0 + head_id * output_stride_1 + block_id * output_stride_2
    output_ptr = output_ptr + tl.arange(0, segment_size // block_size)

    for it in range(0, num_iters_before_causal):
        X = tl.load(input_ptr + it * segment_size).to(tl.float32) * scale
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)
        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local
        m_i = m_new

    for it in range(num_iters_before_causal, num_iters_before_causal + 1):
        X = tl.load(input_ptr + it * segment_size).to(tl.float32) * scale
        mask = offs_q[:, None] >= (offs_k[None, :] + it * segment_size)
        X = tl.where(mask, X, -1.0e6)
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)
        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local
        m_i = m_new

    l_i_inv = 1.0 / l_i
    sum_mask = offs_q[:, None] < real_q_len

    for it in range(0, num_iters_before_causal):
        X = tl.load(input_ptr + it * segment_size).to(tl.float32) * scale
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(output_ptr + it * segment_size // block_size, X.to(Out.type.element_ty))

    for it in range(num_iters_before_causal, num_iters_before_causal + 1):
        X = tl.load(input_ptr + it * segment_size).to(tl.float32) * scale
        mask = offs_q[:, None] >= (offs_k[None, :] + it * segment_size)
        X = tl.where(mask, X, -1.0e6)
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(output_ptr + it * segment_size // block_size, X.to(Out.type.element_ty))

    for it in range(num_iters_before_causal + 1, num_iters):
        X = tl.zeros([segment_size // block_size], dtype=tl.float32)
        tl.store(output_ptr + it * segment_size // block_size, X.to(Out.type.element_ty))


@triton.jit
def softmax_fuse_block_sum_kernel_non_causal(
    In, Out, scale,
    input_stride_0, input_stride_1, input_stride_2,
    output_stride_0, output_stride_1, output_stride_2,
    real_q_len, k_len, chunk_start, chunk_end,
    segment_size: tl.constexpr, block_size: tl.constexpr,
):
    block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    offs_q = tl.arange(0, block_size) + chunk_start + block_id * block_size
    num_iters = k_len // segment_size

    m_i = tl.zeros([block_size], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_size], dtype=tl.float32) + 1.0

    input_ptr = In + batch_id * input_stride_0 + head_id * input_stride_1 + block_id * block_size * input_stride_2
    input_ptr = input_ptr + tl.arange(0, segment_size) + tl.arange(0, block_size)[:, None] * input_stride_2

    output_ptr = Out + batch_id * output_stride_0 + head_id * output_stride_1 + block_id * output_stride_2
    output_ptr = output_ptr + tl.arange(0, segment_size // block_size)

    for it in range(0, num_iters):
        X = tl.load(input_ptr + it * segment_size).to(tl.float32) * scale
        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)
        X = X - m_new[:, None]
        l_local = tl.sum(tl.math.exp2(X), 1)
        l_i = l_i * alpha + l_local
        m_i = m_new

    l_i_inv = 1.0 / l_i
    sum_mask = offs_q[:, None] < real_q_len

    for it in range(0, num_iters):
        X = tl.load(input_ptr + it * segment_size).to(tl.float32) * scale
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(sum_mask, X, 0)
        X = tl.reshape(X, (block_size, segment_size // block_size, block_size))
        X = tl.sum(X, 2)
        X = tl.sum(X, 0)
        tl.store(output_ptr + it * segment_size // block_size, X.to(Out.type.element_ty))


@triton.jit
def flat_group_gemm_fuse_reshape_kernel(
    Q, K, Out,
    stride_qz, stride_qh, stride_qn,
    stride_kz, stride_kh, stride_kn,
    stride_oz, stride_oh, stride_on,
    chunk_start, chunk_end,
    H: tl.constexpr, STRIDE: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, is_causal: tl.constexpr,
):
    block_m = tl.program_id(0).to(tl.int64)
    block_n = tl.program_id(1).to(tl.int64)
    batch_id = tl.program_id(2).to(tl.int64) // H
    head_id = tl.program_id(2).to(tl.int64) % H

    if is_causal:
        if chunk_start + (block_m + 1) * BLOCK_M <= block_n * BLOCK_N:
            return

    Q_ptrs = Q + batch_id * stride_qz + head_id * stride_qh + block_m * BLOCK_M * STRIDE * stride_qn
    K_ptrs = K + batch_id * stride_kz + head_id * stride_kh + block_n * BLOCK_N * STRIDE * stride_kn

    Q_ptrs = (
        Q_ptrs
        + tl.arange(0, BLOCK_M)[:, None] * (stride_qn * STRIDE)
        + tl.arange(0, HEAD_DIM)[None, :]
        + stride_qn * (STRIDE - 1)
    )
    K_ptrs = (
        K_ptrs
        + tl.arange(0, BLOCK_N)[None, :] * (stride_kn * STRIDE)
        + tl.arange(0, HEAD_DIM)[:, None]
    )

    o = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for it in range(STRIDE):
        q = tl.load(Q_ptrs - it * stride_qn)
        k = tl.load(K_ptrs + it * stride_kn)
        o += tl.dot(q, k)

    O_ptrs = Out + batch_id * stride_oz + head_id * stride_oh + block_m * BLOCK_M * stride_on + block_n * BLOCK_N
    O_ptrs = O_ptrs + tl.arange(0, BLOCK_M)[:, None] * stride_on + tl.arange(0, BLOCK_N)[None, :]
    tl.store(O_ptrs, o.to(Out.type.element_ty))


def _ensure_contig(t: paddle.Tensor) -> paddle.Tensor:
    return t if t.is_contiguous() else t.contiguous()


def softmax_fuse_block_sum(
    attn_weights_slice: paddle.Tensor,
    reshaped_block_size: int,
    segment_size: int,
    chunk_start: int,
    chunk_end: int,
    real_q_len: int,
    scale: float,
    is_causal: bool = True,
) -> paddle.Tensor:
    batch_size, num_heads, q_len, k_len = attn_weights_slice.shape
    assert q_len % reshaped_block_size == 0
    assert k_len % segment_size == 0
    assert segment_size % reshaped_block_size == 0
    attn_weights_slice = _ensure_contig(attn_weights_slice)

    output = paddle.empty(
        [batch_size, num_heads, q_len // reshaped_block_size, k_len // reshaped_block_size],
        dtype=attn_weights_slice.dtype,
    )
    grid = (q_len // reshaped_block_size, num_heads, batch_size)

    s0, s1, s2, s3 = attn_weights_slice.strides
    o0, o1, o2, _ = output.strides

    if is_causal:
        softmax_fuse_block_sum_kernel_causal[grid](
            attn_weights_slice, output, scale,
            s0, s1, s2, o0, o1, o2,
            real_q_len, k_len, chunk_start, chunk_end,
            segment_size, reshaped_block_size,
        )
    else:
        softmax_fuse_block_sum_kernel_non_causal[grid](
            attn_weights_slice, output, scale,
            s0, s1, s2, o0, o1, o2,
            real_q_len, k_len, chunk_start, chunk_end,
            segment_size, reshaped_block_size,
        )
    return output


def flat_group_gemm_fuse_reshape(
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    stride: int,
    chunk_start: int,
    chunk_end: int,
    is_causal: bool = True,
) -> paddle.Tensor:
    batch_size, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[2]

    query_states = _ensure_contig(query_states)
    key_states = _ensure_contig(key_states)

    output = paddle.empty(
        [batch_size, num_heads, q_len // stride, kv_len // stride],
        dtype=query_states.dtype,
    )
    BLOCK_M = 64
    BLOCK_N = 64
    assert q_len % (stride * BLOCK_M) == 0
    assert kv_len % (stride * BLOCK_N) == 0

    grid = (q_len // stride // BLOCK_M, kv_len // stride // BLOCK_N, batch_size * num_heads)

    qs0, qs1, qs2, _ = query_states.strides
    ks0, ks1, ks2, _ = key_states.strides
    os0, os1, os2, _ = output.strides

    flat_group_gemm_fuse_reshape_kernel[grid](
        query_states, key_states, output,
        qs0, qs1, qs2, ks0, ks1, ks2, os0, os1, os2,
        chunk_start, chunk_end,
        num_heads, stride, head_dim, BLOCK_M, BLOCK_N, is_causal,
    )
    return output
