# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/batch_invariant_ops.py

import contextlib
import os
from collections import namedtuple
from collections.abc import Callable
from typing import Any, Dict

from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)
from fastdeploy.utils import get_logger

logger = get_logger("worker_process", "worker_process.log")

import paddle
import triton
import triton.language as tl

__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "disable_batch_invariant_mode",
    "enable_batch_invariant_mode",
]


def _matmul_launch_metadata(grid: Callable[..., Any], kernel: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"
    if "tiles_per_update" in args:
        ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}, tiles_per_update={args['tiles_per_update']:02}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@enable_compat_on_triton_kernel
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@enable_compat_on_triton_kernel
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


def get_compute_units():
    """
    Returns the number of streaming multiprocessors (SMs) or equivalent compute units
    for the available accelerator. Assigns the value to NUM_SMS.
    """
    NUM_SMS = None

    if paddle.is_compiled_with_cuda():
        try:
            paddle.device.get_device()  # Triton + Paddle may can't get the device
            device_properties = paddle.cuda.get_device_properties(0)
            NUM_SMS = device_properties.multi_processor_count
        except Exception as e:
            logger.warning(f"Could not get CUDA device properties ({e}), falling back to CPU core count")
            # TODO(liujundong): Paddle lacks a torch.get_num_threads() equivalent for the *configured* thread count.
            # Using os.cpu_count() (total logical cores) as a fallback, which may not be correct.
            # Must check downstream logic to determine if this impacts correctness.
            NUM_SMS = os.cpu_count()
    else:
        logger.warning("No CUDA device available. Using CPU.")
        # For CPU, use the number of CPU cores
        NUM_SMS = os.cpu_count()

    return NUM_SMS


def matmul_persistent(a: paddle.Tensor, b: paddle.Tensor, bias: paddle.Tensor | None = None):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, f"Incompatible dtypes: a={a.dtype}, b={b.dtype}"
    assert bias is None or bias.dim() == 1, "Currently assuming bias is 1D, let Horace know if you run into this"

    NUM_SMS = get_compute_units()
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output. In PaddlePaddle, we create on the same device as input tensor
    # Simply create the tensor without specifying device, Paddle will handle it
    c = paddle.empty((M, N), dtype=dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),)

    configs = {
        paddle.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        paddle.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        paddle.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }
    # print(a.device, b.device, c.device)
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        bias,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
        # Use M*K, K*N, M*N instead of numel() to avoid cudaErrorStreamCaptureImplicit
        # during CUDA Graph capture
        A_LARGE=int(M * K > 2**31),
        B_LARGE=int(K * N > 2**31),
        C_LARGE=int(M * N > 2**31),
        HAS_BIAS=int(bias is not None),
        # The Triton compiler (when used with Paddle) cannot handle these variables as booleans. Explicitly cast to int so the compiler can process them.
        **configs[dtype],
    )
    return c


@enable_compat_on_triton_kernel
@triton.jit
def _log_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute log_softmax along the last dimension of a 2D tensor.
    Each block handles one row of the input tensor.
    """
    # Get the row index for this block
    row_idx = tl.program_id(0).to(tl.int64)

    # Compute base pointers for input and output rows
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Find maximum value in the row for numerical stability
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))

        # Update maximum
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)

        # Compute exp(x - max_val) and accumulate
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)

    # Step 3: Compute final log_softmax values: x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask)

        # Compute log_softmax
        output = vals - max_val - log_sum_exp

        # Store results
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax(input: paddle.Tensor, axis: int = -1) -> paddle.Tensor:
    """
    Compute log_softmax using Triton kernel.

    Args:
        input: Input tensor
        axis: Dimension along which to compute log_softmax (only -1 or last dim supported)
    Returns:
        Tensor with log_softmax applied along the specified dimension
    """
    # print("You are using triton impl for log_softmax")
    if axis != -1 and axis != input.ndim - 1:
        raise ValueError("This implementation only supports log_softmax along the last dimension")

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()

    n_rows, n_cols = input_2d.shape

    # Allocate output tensor
    output = paddle.empty_like(input_2d)

    # Choose block size based on the number of columns
    BLOCK_SIZE = 1024

    # Launch kernel with one block per row
    grid = (n_rows,)
    _log_softmax_kernel[grid](
        input_2d,
        output,
        input_2d.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # Reshape output back to original shape
    return output.reshape(original_shape)


@enable_compat_on_triton_kernel
@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing mean along a single dimension.
    Input is viewed as (M, N, K) where N is the dimension being reduced.
    """
    # Program ID gives us which output element we're computing
    pid = tl.program_id(0)

    # Compute output indices
    m_idx = pid // K
    k_idx = pid % K

    # Bounds check
    if m_idx >= M or k_idx >= K:
        return

    # Accumulate sum across reduction dimension
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N

        # Calculate input indices
        input_idx = m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2

        # Load and accumulate
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Compute mean and store
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)


def mean_dim(
    input: paddle.Tensor, dim: int, keepdim: bool = False, dtype: paddle.dtype | None = None
) -> paddle.Tensor:
    """
    Triton implementation of paddle.mean with single dimension reduction.

    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype. If None, uses input dtype (or float32 for integer inputs)

    Returns:
        Tensor with mean values along specified dimension
    """
    # Validate inputs
    assert input.is_cuda, "Input must be a CUDA tensor"
    assert -input.ndim <= dim < input.ndim, f"Invalid dimension {dim} for tensor with {input.ndim} dimensions"

    # Handle negative dim
    if dim < 0:
        dim = dim + input.ndim

    # Handle dtype
    if dtype is None:
        if input.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64]:
            dtype = paddle.float32
        else:
            dtype = input.dtype

    # Convert input to appropriate dtype if needed
    if input.dtype != dtype:
        input = input.to(dtype)

    # Get input shape and strides
    shape = list(input.shape)

    # Calculate dimensions for kernel
    M = 1
    for i in range(dim):
        M *= shape[i]

    N = shape[dim]

    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]

    # Reshape input to 3D view (M, N, K)
    input_3d = input.reshape(M, N, K)

    # Create output shape
    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    # Create output tensor
    output = paddle.empty(output_shape, dtype=dtype)

    # Reshape output for kernel
    if keepdim:
        output_2d = output.reshape(M, 1, K).squeeze(1)
    else:
        output_2d = output.reshape(M, K)

    # Launch kernel
    grid = (M * K,)
    BLOCK_SIZE = 1024

    mean_kernel[grid](
        input_3d,
        output_2d,
        input_3d.stride(0),
        input_3d.stride(1),
        input_3d.stride(2),
        output_2d.stride(0),
        output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M,
        N,
        K,
        BLOCK_SIZE,
    )

    return output


# The bmm_kernel_persistent kernel and bmm_persistent wrapper below are adapted from
# SGLang (https://github.com/sgl-project/sglang), licensed under Apache License 2.0.
# Original source:
#   sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
# which itself was adapted from:
#   https://github.com/thinking-machines-lab/batch_invariant_ops
# We thank the SGLang authors and the Thinking Machines Lab for their contributions.


@enable_compat_on_triton_kernel
@triton.jit  # pragma: no cover
def bmm_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    B,
    M,
    N,
    K,  #
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
):
    """
    Batched matrix multiplication kernel that processes batches in parallel.
    Each tile processes a (BLOCK_SIZE_M, BLOCK_SIZE_N) output block for a specific batch.
    Uses persistent kernel approach with fixed tile traversal order for determinism.
    """
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles_per_batch = num_pid_m * num_pid_n
    num_tiles_total = B * num_tiles_per_batch

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Process tiles in a deterministic order: batch-major ordering
    for tile_id in tl.range(start_pid, num_tiles_total, NUM_SMS, flatten=True):
        # Decompose tile_id into batch and within-batch tile
        batch_idx = tile_id // num_tiles_per_batch
        tile_in_batch = tile_id % num_tiles_per_batch

        pid_m, pid_n = _compute_pid(tile_in_batch, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Add batch offset
        if A_LARGE or B_LARGE:
            batch_idx_typed = batch_idx.to(tl.int64)
        else:
            batch_idx_typed = batch_idx

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

            a_ptrs = a_ptr + (batch_idx_typed * stride_ab + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (batch_idx_typed * stride_bb + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + batch_idx_typed * stride_cb + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


def bmm_persistent(a: paddle.Tensor, b: paddle.Tensor) -> paddle.Tensor:
    """Batch-invariant batched matrix multiply: (B, M, K) x (B, K, N) -> (B, M, N)"""
    assert a.ndim == 3 and b.ndim == 3, f"bmm_persistent expects 3D tensors, got shapes {a.shape} and {b.shape}"
    assert a.shape[0] == b.shape[0], "Batch sizes must match"
    assert a.shape[2] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, f"Incompatible dtypes: a={a.dtype}, b={b.dtype}"

    B = a.shape[0]
    M = a.shape[1]
    K = a.shape[2]
    N = b.shape[2]
    dtype = a.dtype

    NUM_SMS = get_compute_units()
    c = paddle.empty((B, M, N), dtype=dtype)

    configs = {
        paddle.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        paddle.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        paddle.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }

    config = configs.get(dtype)
    if config is None:
        raise ValueError(
            f"Unsupported dtype {dtype} for bmm_persistent. " f"Supported dtypes are: {list(configs.keys())}"
        )

    num_tiles_per_batch = triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"])
    num_tiles_total = B * num_tiles_per_batch
    grid = (min(NUM_SMS, num_tiles_total),)

    bmm_kernel_persistent[grid](
        a,
        b,
        c,  #
        B,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),
        a.stride(2),  #
        b.stride(0),
        b.stride(1),
        b.stride(2),  #
        c.stride(0),
        c.stride(1),
        c.stride(2),  #
        NUM_SMS=NUM_SMS,  #
        # Use element counts instead of numel() to avoid cudaErrorStreamCaptureImplicit
        # during CUDA Graph capture
        A_LARGE=int(B * M * K > 2**31),
        B_LARGE=int(B * K * N > 2**31),
        C_LARGE=int(B * M * N > 2**31),
        **config,
    )
    return c


def bmm_batch_invariant(x, y):
    """Drop-in replacement for paddle._C_ops.bmm"""
    return bmm_persistent(x, y)


def mm_batch_invariant(a, b, transpose_x=False, transpose_y=False, out=None):
    if transpose_x:
        a = a.T
    if transpose_y:
        b = b.T
    result = matmul_persistent(a, b)
    if out is not None:
        out.copy_(result, False)
        return out
    return result


def addmm_batch_invariant(
    input: paddle.Tensor, x: paddle.Tensor, y: paddle.Tensor, beta: float = 1.0, alpha: float = 1.0
) -> paddle.Tensor:
    """ "
    We need achieve `Out = alpha * (x @ y) + beta * input`
    But matmul_persistent only achieve `x @ y + input`(according to aten::addmm in torch,paddle._C_ops.addmm have more parameters)
    So we use `alpha * (x @ y) + beta * input  =  alpha * [ (x @ y) + (beta / alpha) * input ]`
    to minimize the effection on performance
    """
    if alpha == 0:
        return paddle.broadcast_to(beta * input, [x.shape[0], y.shape[1]])
    matmul_result = matmul_persistent(a=x, b=y, bias=input * beta / alpha)
    result = alpha * matmul_result
    return result


def _log_softmax_batch_invariant(x: paddle.Tensor, axis: int = -1, out=None) -> paddle.Tensor:
    result = log_softmax(input=x, axis=axis)
    # Handle out parameter if provided
    if out is not None:
        out.copy_(result)
        return out
    return result


def mean_batch_invariant(
    x: paddle.Tensor, axis: list[int] = [], keepdim: bool = False, dtype: paddle.dtype | None = None, out=None
) -> paddle.Tensor:
    assert dtype is None or dtype == paddle.float32, f"unsupported dtype: {dtype}"
    if axis is None:  # Global mean (no axis specified)
        # Avoid x.numel() to prevent cudaErrorStreamCaptureImplicit during CUDA Graph capture
        n_elems = 1
        for s in x.shape:
            n_elems *= s
        result = paddle.sum(x, keepdim=keepdim, dtype=paddle.float32) / n_elems
    elif type(axis) is int:
        result = mean_dim(x, axis, keepdim=keepdim)
    elif len(axis) == 1:  # axis: int | Sequence[int]
        result = mean_dim(x, axis[0], keepdim=keepdim)
    else:
        assert x.dtype in {paddle.float16, paddle.bfloat16, paddle.float32}, "only float types supported for now"
        n_elems = 1
        for d in axis:
            n_elems *= x.shape[d]
        result = paddle.sum(x, axis=axis, keepdim=keepdim, dtype=paddle.float32) / n_elems

    # Handle out parameter if provided
    if out is not None:
        out.copy_(result)
        return out
    return result


# ---------------------------------------------------------------------------
# Batch-invariant RMSNorm (Triton): one program per row, fixed reduction order
# ---------------------------------------------------------------------------


@enable_compat_on_triton_kernel
@triton.jit
def _rms_norm_kernel(  # pragma: no cover
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-row RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight.
    Each program handles exactly one row → M-invariant."""
    row_idx = tl.program_id(0).to(tl.int64)
    row_start = input_ptr + row_idx * input_row_stride
    out_start = output_ptr + row_idx * output_row_stride

    # Pass 1: sum of squares in float32
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(row_start + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(tl.where(mask, x * x, 0.0))

    inv_rms = 1.0 / tl.sqrt(sum_sq / n_cols + eps)

    # Pass 2: normalize and scale
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(row_start + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y = x * inv_rms * w
        tl.store(out_start + cols, y.to(out_start.dtype.element_ty), mask=mask)


def rms_norm_batch_invariant(x: paddle.Tensor, weight: paddle.Tensor, eps: float = 1e-6) -> paddle.Tensor:
    """M-invariant RMSNorm: each row computed independently via Triton."""
    orig_shape = x.shape
    x_2d = x.reshape([-1, x.shape[-1]]).contiguous()
    weight = weight.contiguous()
    n_rows, n_cols = x_2d.shape
    out = paddle.empty_like(x_2d)
    BLOCK_SIZE = 1024
    _rms_norm_kernel[(n_rows,)](
        x_2d,
        weight,
        out,
        x_2d.stride(0),
        out.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.reshape(orig_shape)


_original_ops = {"mm": None, "addmm": None, "_log_softmax": None, "mean_dim": None, "bmm": None}

_batch_invariant_MODE = False


def is_batch_invariant_mode_enabled():
    return _batch_invariant_MODE


def enable_batch_invariant_mode():
    global _batch_invariant_MODE, _original_ops
    if _batch_invariant_MODE:
        return

    if hasattr(paddle, "compat") and hasattr(paddle.compat, "enable_torch_proxy"):
        paddle.compat.enable_torch_proxy()
        # TODO(liujundong): Enabling torch proxy here has a global effect.
        # Do NOT call this function from module import time,
        # otherwise it may affect other test cases during pytest collection.
        # (ex: Could not import module 'PretrainedTokenizer' or No module named 'paddle.distributed.tensor')
        # Other side effects have not been observed yet, but they should be watched out for in the future.
    else:
        raise RuntimeError(
            "Unable to enable batch-invariant mode: Paddle version is too old. " "Please upgrade PaddlePaddle."
        )

    _original_ops["mm"] = paddle._C_ops.matmul
    _original_ops["addmm"] = paddle._C_ops.addmm
    _original_ops["log_softmax"] = paddle._C_ops.log_softmax
    _original_ops["mean"] = paddle._C_ops.mean
    _original_ops["bmm"] = paddle._C_ops.bmm

    paddle._C_ops.matmul = mm_batch_invariant
    paddle._C_ops.addmm = addmm_batch_invariant
    paddle._C_ops.log_softmax = _log_softmax_batch_invariant
    paddle._C_ops.mean = mean_batch_invariant
    paddle._C_ops.bmm = bmm_batch_invariant

    _batch_invariant_MODE = True


def init_deterministic_mode():
    """One-stop initialization for deterministic mode.

    Call after worker creation but before model loading.
    """
    if not is_batch_invariant_mode_enabled():
        enable_batch_invariant_mode()


def disable_batch_invariant_mode():
    global _batch_invariant_MODE, _original_ops
    if not _batch_invariant_MODE:
        return

    if _original_ops["mm"]:
        paddle._C_ops.matmul = _original_ops["mm"]
    if _original_ops["addmm"]:
        paddle._C_ops.addmm = _original_ops["addmm"]
    if _original_ops["log_softmax"]:
        paddle._C_ops.log_softmax = _original_ops["log_softmax"]
    if _original_ops["mean"]:
        paddle._C_ops.mean = _original_ops["mean"]
    if _original_ops["bmm"]:
        paddle._C_ops.bmm = _original_ops["bmm"]

    _batch_invariant_MODE = False


@contextlib.contextmanager
def set_batch_invariant_mode(enabled: bool = True):
    global _batch_invariant_MODE, _original_ops
    old_mode = _batch_invariant_MODE
    if enabled:
        enable_batch_invariant_mode()
    else:
        disable_batch_invariant_mode()
    yield
    if old_mode:
        enable_batch_invariant_mode()
    else:
        disable_batch_invariant_mode()


AttentionBlockSize = namedtuple("AttentionBlockSize", ["block_m", "block_n"])


def get_batch_invariant_attention_block_size() -> AttentionBlockSize:
    return AttentionBlockSize(block_m=16, block_n=16)
