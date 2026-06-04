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
"""

# Adapt from SGLang's layernorm_gated.py implementation
# Reference: https://github.com/sglang-ai/sglang/blob/main/python/sglang/srt/layers/attention/fla/layernorm_gated.py

import paddle
import triton
import triton.language as tl

from fastdeploy.utils import ceil_div

# Maximum rows per Triton block for layernorm gated kernel
MAX_ROWS_PER_BLOCK = 4


def next_power_of_2(n: int):
    return 1 << (n - 1).bit_length() if n > 0 else 1


def calc_rows_per_block(M: int, BLOCK_N: int, num_warps: int) -> int:
    """Calculate optimal rows per block based on input size and warp count.

    The goal is to keep all threads in a block busy. A Triton block has
    `num_warps * 32` threads and processes a 2D tile of shape
    [ROWS_PER_BLOCK, BLOCK_N].  When BLOCK_N < num_warps * 32, processing a
    single row leaves some threads idle; increasing ROWS_PER_BLOCK fills those
    idle threads with work from additional rows.

    rows_per_block = ceil(num_warps * 32 / BLOCK_N), rounded up to the next
    power of two, capped at MAX_ROWS_PER_BLOCK and at M itself.
    """
    min_rows = max(1, ceil_div(num_warps * 32, BLOCK_N))
    rows = next_power_of_2(min_rows)
    return min(rows, MAX_ROWS_PER_BLOCK, max(1, M))


@triton.jit
def rms_norm_gated_fwd_kernel(
    x_ptr,  # pointer to the input
    y_ptr,  # pointer to the output
    w_ptr,  # pointer to the weights
    b_ptr,  # pointer to the biases
    z_ptr,  # pointer to the gate
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Map the program id to the starting row of X and Y it should compute.
    row_start = tl.program_id(0) * ROWS_PER_BLOCK

    # Create 2D tile: [ROWS_PER_BLOCK, BLOCK_N]
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    # Compute offsets for 2D tile
    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :]

    # Base pointers
    X_base = x_ptr + row_offsets + col_offsets
    Y_base = y_ptr + rows[:, None] * stride_y_row + col_offsets

    # Create mask for valid rows and columns
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    # Load input data with 2D tile
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm: compute variance per row (reduce along axis 1)
    var = tl.sum(x * x, axis=1) / N

    # Load weights and biases (broadcast across rows)
    w = tl.load(w_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b = tl.load(b_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)

    # Normalize and apply linear transformation (RMSNorm: no mean subtraction)
    x_hat = x * tl.rsqrt(var[:, None] + eps)

    y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]

    if HAS_Z:
        Z_base = z_ptr + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == 0:  # swish/silu
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == 1:  # sigmoid
            y *= tl.sigmoid(z)

    # Write output
    tl.store(Y_base, y, mask=mask)


def rmsnorm_gated(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    activation: str = "swish",
):
    """
    Fused RMSNorm with gate activation.

    Args:
        x: Input tensor of shape [M, N]
        weight: Weight tensor of shape [N]
        bias: Bias tensor of shape [N] or None
        eps: Epsilon value for numerical stability
        z: Gate tensor of shape [M, N] or None
        out: Output tensor of shape [M, N] or None
        activation: Activation function type ("swish" or "sigmoid")

    Returns:
        Output tensor of shape [M, N]
    """
    M, N = x.shape
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)

    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = paddle.empty_like(x)
    assert out.stride(-1) == 1

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This RMSNorm doesn't support feature dim >= 64KB.")

    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)

    # Calculate rows per block based on BLOCK_N and num_warps
    rows_per_block = calc_rows_per_block(M, BLOCK_N, num_warps)

    # Update grid to use rows_per_block
    grid = (ceil_div(M, rows_per_block),)

    # Map activation string to integer constant
    activation_map = {"swish": 0, "silu": 0, "sigmoid": 1}
    activation_const = activation_map.get(activation.lower(), 0)

    # Use x as a placeholder for z_ptr when z is None to avoid passing a null
    # pointer, which can cause issues in some Triton versions. The kernel will
    # not access z_ptr when HAS_Z=False.
    z_ref = z if z is not None else x
    rms_norm_gated_fwd_kernel[grid](
        x,
        out,
        weight,
        bias,
        z_ref,
        x.stride(0),
        out.stride(0),
        z_ref.stride(0),
        M,
        N,
        eps,
        BLOCK_N=BLOCK_N,
        ROWS_PER_BLOCK=rows_per_block,
        HAS_BIAS=bias is not None,
        HAS_Z=z is not None,
        ACTIVATION=activation_const,
        num_warps=num_warps,
    )
    return out
