# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
# Original: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang (MIT License)
# Adapted for FastDeploy (PaddlePaddle) by PaddlePaddle Authors, 2025.
"""
GDN Chunk Algorithm Coordinator — Prefill path core implementation.

Executes the standard 6-step chunk GDN algorithm:
  1. chunk_local_cumsum(g)             — compute local decay cumulative sum
  2. chunk_scaled_dot_kkt_fwd(k,beta)  — compute A = beta * K * K^T
  3. solve_tril(A)                     — compute (I+A)^{-1}
  4. recompute_w_u_fwd(k,v,beta,A)     — compute W, U (WY decomposition)
  5. chunk_gated_delta_rule_fwd_h      — state propagation
  6. chunk_fwd_o                       — compute output

Porting notes:
  - Removed torch.autograd.Function (no backprop needed for inference)
  - Removed @torch.compiler.disable (not applicable to paddle)
  - Removed einops rearrange (head_first=False is the only supported layout)
  - Removed SUPPRESS_LEVEL / autocast_custom_fwd (not relevant for inference)
  - assert q.dtype != torch.float32 → assert q.dtype != paddle.float32
  - .to(q.dtype) → .cast(q.dtype)
"""

from typing import Optional, Tuple

import paddle

from fastdeploy.model_executor.ops.triton_ops.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from fastdeploy.model_executor.ops.triton_ops.fla.chunk_o import chunk_fwd_o
from fastdeploy.model_executor.ops.triton_ops.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from fastdeploy.model_executor.ops.triton_ops.fla.cumsum import chunk_local_cumsum
from fastdeploy.model_executor.ops.triton_ops.fla.l2norm import l2norm_fwd
from fastdeploy.model_executor.ops.triton_ops.fla.solve_tril import solve_tril
from fastdeploy.model_executor.ops.triton_ops.fla.utils import input_guard
from fastdeploy.model_executor.ops.triton_ops.fla.wy_fast import recompute_w_u_fwd


def chunk_gated_delta_rule_fwd(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    g: paddle.Tensor,
    beta: paddle.Tensor,
    scale: float,
    initial_state: paddle.Tensor,
    initial_state_indices: paddle.Tensor,
    cu_seqlens: Optional[paddle.Tensor] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    GDN 6-step chunk algorithm forward (internal implementation).

    Args:
        q, k: [B, T, H, K]
        v:    [B, T, H, V]
        g:    [B, T, H]  log decay (negative values)
        beta: [B, T, H]  write gate
        scale: Q scale factor
        initial_state: [N, H, K, V]  initial SSM state pool
        initial_state_indices: [N]   pool slot index per sequence
        cu_seqlens: [N+1] varlen mode (optional)

    Returns:
        o: [B, T, H, V]
        h: [B, NT, H, K, V]  initial state at each chunk (for debugging/testing)
    """
    # Step 1: compute chunk-local cumulative sum of g (force float32; safe_exp requires fp32/fp64)
    g = chunk_local_cumsum(g, chunk_size=64, output_dtype=paddle.float32, cu_seqlens=cu_seqlens)

    # Step 2: compute A = beta * K * K^T (strictly lower-triangular)
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        output_dtype=paddle.float32,
    )

    # Step 3: compute (I + A)^{-1}
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)

    # Step 4: compute W, U (WY decomposition)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )

    # Step 5: state propagation
    # The kernel always loads initial_state_indices even when USE_INITIAL_STATE=False,
    # so dummy values are needed to avoid NoneType errors when initial_state is None.
    B, T, H, K = k.shape
    V = u.shape[-1]
    _initial_state = initial_state
    _initial_state_indices = initial_state_indices
    if _initial_state is None:
        # dummy: zero state, indices pointing to slot 0
        _initial_state = paddle.zeros([B, H, K, V], dtype=k.dtype)
        _initial_state_indices = paddle.arange(B, dtype=paddle.int32)
    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=_initial_state,
        initial_state_indices=_initial_state_indices,
        cu_seqlens=cu_seqlens,
    )

    # Step 6: compute output
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    return o, h


@input_guard
def chunk_gated_delta_rule(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    g: paddle.Tensor,
    beta: paddle.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[paddle.Tensor] = None,
    initial_state_indices: Optional[paddle.Tensor] = None,
    cu_seqlens: Optional[paddle.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
    """
    GDN Chunk Algorithm public interface (Prefill path).

    Only supports head_first=False (batch-first) layout: [B, T, H, ...].

    Args:
        q, k: [B, T, H, K]
        v:    [B, T, H, V]
        g:    [B, T, H]   log decay (negative values)
        beta: [B, T, H]   write gate
        scale: Q scale factor; defaults to 1/sqrt(K) when None
        initial_state: [N, H, K, V]  initial state (from SSM pool)
        initial_state_indices: [N]   pool slot indices
        cu_seqlens: [N+1] varlen mode
        use_qk_l2norm_in_kernel: whether to apply L2 norm to Q/K inside the kernel

    Returns:
        o: [B, T, H, V]
        h: [B, NT, H, K, V]  initial state at each chunk (can be used for debugging)
    """
    assert q.dtype == k.dtype == v.dtype, "q, k, v must have the same dtype"
    assert q.dtype != paddle.float32, "chunk_gated_delta_rule does not support float32; use bfloat16 or float16."
    assert len(beta.shape) == 3, "beta must have shape [B, T, H] (head_first=False)"

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"batch_size must be 1 in varlen mode, but got {q.shape[0]}. "
            "Please concatenate variable-length inputs before passing in."
        )
    if (
        cu_seqlens is not None
        and initial_state_indices is not None
        and initial_state_indices.shape[0] != cu_seqlens.shape[0] - 1
    ):
        raise ValueError(
            f"initial_state_indices length must equal the number of sequences "
            f"{cu_seqlens.shape[0] - 1}, but got {initial_state_indices.shape[0]}."
        )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    o, h = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
    )
    return o.cast(q.dtype), h
