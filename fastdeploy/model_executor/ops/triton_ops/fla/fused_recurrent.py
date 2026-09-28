# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py
# Original: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang (MIT License)
# Adapted for FastDeploy (PaddlePaddle) by PaddlePaddle Authors, 2025.
"""
GDN Fused Recurrent Kernel — Decode path core implementation.

Provides two public functions:
  1. fused_recurrent_gated_delta_rule
       Standard interface: accepts initial_state / outputs final_state Tensor
       Suitable for state saving after Prefill, single-pass inference
       State layout: [N, HV, K, V]  (K-first)

  2. fused_recurrent_gated_delta_rule_update
       Pool-index interface: in-place read/write of state at pool[indices]
       Suitable for serving Decode phase (no external gather/scatter needed)
       Pool layout: [max_seqs, HV, K, V]

Notes:
  - Triton kernel code is identical to SGLang, no modifications needed
  - Python wrapper replaces torch.Tensor → paddle.Tensor, removes torch.autograd.Function
  - FD inference does not require backpropagation; fwd functions are called directly
"""

from typing import Optional, Tuple

import paddle
import triton
import triton.language as tl

from fastdeploy.model_executor.ops.triton_ops.fla.op import exp
from fastdeploy.model_executor.ops.triton_ops.fla.utils import input_guard

# ============================================================
# Triton Kernel — Standard fused recurrent (full state in/out)
# Source: SGLang fused_recurrent.py lines 15-121
# Triton code is unchanged
# ============================================================


@triton.jit(do_not_specialize=["T"])
def _fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    if not IS_KDA:
        p_g = g + bos * HV + i_hv
    else:
        p_gk = g + (bos * HV + i_hv) * K + o_k

    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale

        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)
        else:
            b_gk = tl.load(p_gk).to(tl.float32)
            b_h *= exp(b_gk[:, None])

        b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


# ============================================================
# Triton Kernel — Pool-index fused recurrent (in-place state read/write)
# Source: SGLang fused_recurrent.py fused_recurrent_gated_delta_rule_update_fwd_kernel
# Key feature: reads/writes state directly at h0_source[h0_indices[i]], no external gather/scatter
# Triton code is unchanged
# ============================================================


@triton.jit(do_not_specialize=["T"])
def _fused_recurrent_gated_delta_rule_update_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DISABLE_STATE_UPDATE: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    """
    Pool-index variant: reads initial state from h0_source[h0_indices[i_n]],
    and writes the final state back in-place to the same location after computation.
    Requests with PAD_SLOT_ID=-1 skip state read/write automatically (safe for CUDA Graph padding).
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    if not IS_KDA:
        p_g = g + bos * HV + i_hv
    else:
        p_gk = g + (bos * HV + i_hv) * K + o_k

    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:  # skip when PAD_SLOT_ID=-1
            p_h0 = h0_source + idx * HV * K * V + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale

        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)
        else:
            b_gk = tl.load(p_gk).to(tl.float32)
            b_h *= exp(b_gk[:, None])

        b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)

    # In-place write-back to pool
    if not DISABLE_STATE_UPDATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:  # skip write-back when PAD_SLOT_ID=-1
            p_h0 = h0_source + idx * HV * K * V + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


# ============================================================
# Python Wrapper — Standard interface (paddle edition)
# ============================================================


@input_guard
def fused_recurrent_gated_delta_rule_fwd(
    q: paddle.Tensor,  # [B, T, H, K]
    k: paddle.Tensor,  # [B, T, H, K]
    v: paddle.Tensor,  # [B, T, HV, V]
    g: paddle.Tensor,  # [B, T, HV]
    beta: paddle.Tensor,  # [B, T, HV] or [B, T, HV, V]
    scale: float,
    initial_state: Optional[paddle.Tensor],  # [N, HV, K, V]
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[paddle.Tensor] = None,  # [N+1] int64
) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
    """
    Standard fused recurrent forward.

    Args:
        q, k: [B, T, H, K]  (H = num_k_heads)
        v:    [B, T, HV, V] (HV = num_v_heads, HV >= H for GVA)
        g:    [B, T, HV]    log decay (negative values)
        beta: [B, T, HV]    write gate [0, 1]
        scale: float        Q scale factor (typically 1/sqrt(K))
        initial_state: [N, HV, K, V]  initial SSM state (K-first layout)
        output_final_state: whether to output the final state
        use_qk_l2norm_in_kernel: whether to apply L2 norm inside the kernel
        cu_seqlens: [N+1] int64, cumulative sequence lengths for varlen mode

    Returns:
        o: [B, T, HV, V]
        final_state: [N, HV, K, V] if output_final_state else None
    """
    B, T, H, K = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    HV, V = v.shape[2], v.shape[3]
    N = B if cu_seqlens is None else cu_seqlens.shape[0] - 1

    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, f"NK > 1 is not supported yet (K={K}, BK={BK})"

    num_stages = 3
    num_warps = 1

    # output Tensor (NK=1, squeezed)
    o = paddle.empty([NK, B, T, HV, V], dtype=v.dtype)
    final_state = None
    if output_final_state:
        final_state = paddle.empty([N, HV, K, V], dtype=paddle.float32)

    grid = (NK, NV, N * HV)
    _fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)  # [B, T, HV, V]
    return o, final_state


def fused_recurrent_gated_delta_rule(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    g: paddle.Tensor,
    beta: Optional[paddle.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[paddle.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[paddle.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
    """
    GDN Fused Recurrent public interface (standard).

    For use in Prefill phase or test comparison scenarios.
    For Decode phase, prefer fused_recurrent_gated_delta_rule_update (pool-index variant).

    Args:
        q, k: [B, T, H, K]
        v:    [B, T, HV, V]
        g:    [B, T, HV]       log decay
        beta: [B, T, HV]       write gate; all-ones when None
        scale: Q scale; defaults to 1/sqrt(K) when None
        initial_state: [N, HV, K, V]
        output_final_state: whether to return final state
        cu_seqlens: [N+1] varlen mode

    Returns:
        o: [B, T, HV, V]
        final_state: [N, HV, K, V] or None
    """
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"batch_size must be 1 in varlen mode, but got {q.shape[0]}. "
            "Please concatenate variable-length inputs before passing in."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = paddle.ones(q.shape[:-1], dtype=q.dtype)  # [B, T, HV]

    return fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
    )


# ============================================================
# Python Wrapper — Pool-index interface (Decode core)
# ============================================================


@input_guard
def fused_recurrent_gated_delta_rule_update_fwd(
    q: paddle.Tensor,  # [B, T, H, K]
    k: paddle.Tensor,  # [B, T, H, K]
    v: paddle.Tensor,  # [B, T, HV, V]
    g: paddle.Tensor,  # [B, T, HV]
    beta: paddle.Tensor,  # [B, T, HV]
    scale: float,
    ssm_pool: paddle.Tensor,  # [max_seqs, HV, K, V] in-place read/write
    ssm_indices: paddle.Tensor,  # [N] int32/int64, PAD_SLOT_ID=-1 safe
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[paddle.Tensor] = None,  # [N+1]
    disable_state_update: bool = False,
) -> paddle.Tensor:
    """
    Pool-index fused recurrent forward (Decode phase core).

    Reads initial state from ssm_pool[ssm_indices[i]] and writes back in-place,
    avoiding external gather/scatter operations, compatible with CUDA Graph.

    Args:
        q, k: [B, T, H, K]
        v:    [B, T, HV, V]
        g:    [B, T, HV]  log decay
        beta: [B, T, HV]  write gate
        scale: float
        ssm_pool: [max_seqs, HV, K, V]  full SSM state pool (K-first layout)
        ssm_indices: [N]  pool slot index per request in this step;
                          requests with PAD_SLOT_ID=-1 skip state read/write
        disable_state_update: when True, only computes output without updating pool state

    Returns:
        o: [B, T, HV, V]
    """
    B, T, H, K = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    HV, V = v.shape[2], v.shape[3]
    N = B if cu_seqlens is None else cu_seqlens.shape[0] - 1

    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, f"NK > 1 is not supported yet (K={K}, BK={BK})"

    num_stages = 3
    num_warps = 1

    o = paddle.empty([NK, B, T, HV, V], dtype=v.dtype)

    grid = (NK, NV, N * HV)
    _fused_recurrent_gated_delta_rule_update_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0_source=ssm_pool,
        h0_indices=ssm_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=ssm_pool is not None,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        DISABLE_STATE_UPDATE=disable_state_update,
        IS_KDA=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)  # [B, T, HV, V]
    return o


def fused_recurrent_gated_delta_rule_update(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    g: paddle.Tensor,
    beta: Optional[paddle.Tensor] = None,
    scale: Optional[float] = None,
    ssm_pool: Optional[paddle.Tensor] = None,
    ssm_indices: Optional[paddle.Tensor] = None,
    cu_seqlens: Optional[paddle.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    disable_state_update: bool = False,
) -> paddle.Tensor:
    """
    GDN Pool-index Fused Recurrent public interface (Decode core).

    Recommended interface for serving Decode phase.
    Operates directly on the SSM Pool, no external gather/scatter needed.

    Args:
        q, k: [B, T, H, K]  (T=1 for Decode)
        v:    [B, T, HV, V]
        g:    [B, T, HV]    log decay
        beta: [B, T, HV]    write gate; all-ones when None
        scale: defaults to 1/sqrt(K) when None
        ssm_pool: [max_seqs, HV, K, V]  SSM state pool (K-first)
        ssm_indices: [N] int  pool slot index per request
        disable_state_update: read-only when True (for debugging)

    Returns:
        o: [B, T, HV, V]
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = paddle.ones(q.shape[:-1], dtype=q.dtype)

    return fused_recurrent_gated_delta_rule_update_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        ssm_pool=ssm_pool,
        ssm_indices=ssm_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        disable_state_update=disable_state_update,
    )
