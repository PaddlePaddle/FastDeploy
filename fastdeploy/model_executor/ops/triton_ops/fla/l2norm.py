# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
# Original: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang (MIT License)
# Adapted for FastDeploy (PaddlePaddle) by PaddlePaddle Authors, 2025.
"""
L2 Norm Triton Kernel.

Porting notes:
  - Removed torch.autograd.Function and nn.Module (no backprop needed for inference)
  - torch.empty_like(x) → paddle.empty_like(x)
  - Retained both Triton kernels unchanged (pure GPU instructions)
  - Exposed l2norm_fwd directly as the main entry point
"""


import paddle
import triton
import triton.language as tl

from fastdeploy.model_executor.ops.triton_ops.fla.utils import input_guard

# ============================================================
# Triton Kernels (unchanged from SGLang)
# ============================================================

BT_LIST = [8, 16, 32, 64, 128]


@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    # Normalize and apply linear transformation
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


@triton.jit
def l2norm_fwd_kernel(
    x,
    y,
    eps,
    NB: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


# ============================================================
# Python Wrapper (paddle edition)
# ============================================================


@input_guard
def l2norm_fwd(
    x: paddle.Tensor,
    eps: float = 1e-6,
    output_dtype=None,
) -> paddle.Tensor:
    """
    L2 normalization forward (Triton-accelerated).

    Args:
        x: arbitrary shape, last dimension is feature dim D
        eps: numerical stability term
        output_dtype: output dtype; None means same as input

    Returns:
        L2 normalized tensor, same shape as x
    """
    x_shape_og = x.shape
    x = x.reshape([-1, x.shape[-1]])

    if output_dtype is None:
        y = paddle.empty_like(x)
    else:
        y = paddle.empty_like(x, dtype=output_dtype)

    assert y.strides[-1] == 1 if hasattr(y, "strides") else True
    T, D = x.shape[0], x.shape[-1]

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if D <= 512:
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        l2norm_fwd_kernel[grid](
            x,
            y,
            eps,
            NB=NB,
            T=T,
            D=D,
            BD=BD,
            BT=16,
            num_warps=8,
            num_stages=3,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            x,
            y,
            eps=eps,
            D=D,
            BD=BD,
            num_warps=8,
            num_stages=3,
        )

    return y.reshape(x_shape_og)


# Aliases for SGLang API compatibility
l2norm = l2norm_fwd
l2_norm = l2norm_fwd
