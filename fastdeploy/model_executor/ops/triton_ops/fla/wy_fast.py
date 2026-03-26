# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/wy_fast.py
# Original: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang (MIT License)
# Adapted for FastDeploy (PaddlePaddle) by PaddlePaddle Authors, 2025.
"""
WY decomposition W/U matrix recomputation Triton Kernel.

Porting notes:
  - torch.empty_like(v) → paddle.empty_like(v)
  - k.new_empty(B, T, H, K) → paddle.empty([B, T, H, K], dtype=k.dtype)
  - Triton kernel code is unchanged
"""

from typing import Optional, Tuple

import paddle
import triton
import triton.language as tl

from fastdeploy.model_executor.ops.triton_ops.fla.index import prepare_chunk_indices

# ============================================================
# Triton Kernel (unchanged from SGLang)
# ============================================================


@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


# ============================================================
# Python Wrapper (paddle edition)
# ============================================================


def recompute_w_u_fwd(
    k: paddle.Tensor,
    v: paddle.Tensor,
    beta: paddle.Tensor,
    g_cumsum: paddle.Tensor,
    A: paddle.Tensor,
    cu_seqlens: Optional[paddle.Tensor],
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    Recompute W and U matrices from the WY decomposition.

    Args:
        k: [B, T, Hg, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g_cumsum: [B, T, H] cumsum of log decay
        A: [B, T, H, BT] lower-triangular matrix inverse (from solve_tril)
        cu_seqlens: cumulative sequence lengths [N+1] for varlen mode

    Returns:
        w: [B, T, H, K]
        u: [B, T, H, V] — updated value vectors (new value tensor)
    """
    B, T, Hg, K = k.shape
    V = v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = 64
    BV = 64
    u = paddle.empty_like(v)
    w = paddle.empty([B, T, H, K], dtype=k.dtype)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
        num_stages=3,
    )
    return w, u


# Alias for SGLang API compatibility
fwd_recompute_w_u = recompute_w_u_fwd
