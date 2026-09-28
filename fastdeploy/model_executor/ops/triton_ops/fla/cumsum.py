# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/cumsum.py
# Original: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang (MIT License)
# Adapted for FastDeploy (PaddlePaddle) by PaddlePaddle Authors, 2025.
"""
FLA chunk-local prefix cumulative sum Triton Kernel.

Porting notes:
  - torch.empty_like(g, dtype=...) → paddle.empty_like(g, dtype=...)
  - autotune left commented out (consistent with SGLang)
  - check_shared_mem() and is_nvidia_hopper moved to local utils
  - Triton kernel code is unchanged
"""

from typing import Optional

import paddle
import triton
import triton.language as tl

from fastdeploy.model_executor.ops.triton_ops.fla.index import prepare_chunk_indices
from fastdeploy.model_executor.ops.triton_ops.fla.utils import input_guard

# BS_LIST depends on shared memory size; use conservative defaults for inference
BS_LIST = [32, 64]


# ============================================================
# Triton Kernel — Scalar cumsum (3D tensor)
# ============================================================


# @triton.autotune(
#     configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
#     key=["B", "H", "BT", "IS_VARLEN", "REVERSE"],
# )
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


# ============================================================
# Triton Kernel — Vector cumsum (4D tensor)
# ============================================================


@triton.autotune(
    configs=[triton.Config({"BS": BS}, num_warps=num_warps) for BS in BS_LIST for num_warps in [2, 4, 8]],
    key=["B", "H", "S", "BT", "IS_VARLEN", "REVERSE", "HAS_SCALE"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    if REVERSE:
        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    else:
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(
            s + (bos * H + i_h * T) * S,
            (T, S),
            (S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_o = tl.make_block_ptr(
            o + (bos * H + i_h * T) * S,
            (T, S),
            (S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
    else:
        p_s = tl.make_block_ptr(
            s + (bos * H + i_h) * S,
            (T, S),
            (H * S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_o = tl.make_block_ptr(
            o + (bos * H + i_h) * S,
            (T, S),
            (H * S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# ============================================================
# Python Wrappers (paddle edition)
# ============================================================


def chunk_local_cumsum_scalar(
    g: paddle.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[paddle.Tensor] = None,
    head_first: bool = False,
    output_dtype=None,
) -> paddle.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    out_dtype = output_dtype or g.dtype
    g_org = g
    g = paddle.empty_like(g, dtype=out_dtype)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        HAS_SCALE=scale is not None,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=8,
        num_stages=3,
    )
    return g


def chunk_local_cumsum_vector(
    g: paddle.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[paddle.Tensor] = None,
    head_first: bool = False,
    output_dtype=None,
) -> paddle.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    out_dtype = output_dtype or g.dtype
    g_org = g
    g = paddle.empty_like(g, dtype=out_dtype)

    def grid(meta):
        return (triton.cdiv(meta["S"], meta["BS"]), NT, B * H)

    chunk_local_cumsum_vector_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        HAS_SCALE=scale is not None,
        IS_VARLEN=cu_seqlens is not None,
    )
    return g


@input_guard
def chunk_local_cumsum(
    g: paddle.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[paddle.Tensor] = None,
    head_first: bool = False,
    output_dtype=None,
    **kwargs,
) -> paddle.Tensor:
    """
    Chunk-local prefix cumulative sum (forward).

    Args:
        g: [B, T, H] or [B, T, H, S]
        chunk_size: chunk size (must be a power of 2)
        reverse: whether to compute reverse cumsum
        scale: optional scale factor
        cu_seqlens: cumulative sequence lengths [N+1] for varlen mode
        head_first: whether layout is head-first
        output_dtype: output dtype (defaults to same as input)

    Returns:
        cumsum tensor with the same shape as g
    """
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "batch_size must be 1 in varlen mode"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    elif len(g.shape) == 4:
        return chunk_local_cumsum_vector(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}, "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise"
        )
