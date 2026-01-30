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

import functools
from typing import Tuple, Union

import numpy as np
import paddle
from paddle import Tensor, nn
from paddle.framework import in_dynamic_mode
from scipy.linalg import block_diag

from fastdeploy.config import FDConfig
from fastdeploy.platforms import current_platform

if current_platform.is_cuda() and current_platform.available():
    try:
        from fastdeploy.model_executor.ops.gpu import get_padding_offset
    except Exception:
        raise ImportError(
            "Verify environment consistency between compilation and FastDeploy installation. "
            "And ensure the Paddle version supports FastDeploy's custom operators"
        )


from fastdeploy import envs

cache_params = envs.FD_CACHE_PARAMS
if cache_params != "none":
    c8_state_dict = paddle.load(cache_params, return_numpy=True)


DEFAULT_VOCAB_PADDING_SIZE = 64


def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    paddle.Tensor: An orthogonal matrix of the specified size.
    """
    paddle.device.cuda.empty_cache()
    if device == "cuda":
        random_matrix = paddle.randn(size, size, dtype="float32").to("gpu")
    q, r = paddle.linalg.qr(random_matrix)
    q *= paddle.sign(paddle.diag(r)).unsqueeze(0)
    return q


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)


def get_hadK(n, transpose=False):
    hadK, K = None, None
    assert is_pow2(n)
    K = 1
    return hadK, K


def matmul_hadU_int4(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().reshape((-1, n, 1))
    output = input.clone()
    while input.shape[1] > K:
        input = input.reshape((input.shape[0], input.shape[1] // 2, 2, input.shape[2]))
        output = output.reshape(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.reshape((input.shape[0], input.shape[1], -1))
        (input, output) = (output, input)
    del output

    if K > 1:
        input = hadK.reshape((1, K, K)).to(input) @ input

    return input.reshape(X.shape) / paddle.to_tensor(n, dtype="float32").sqrt()


def random_hadamard_matrix_int4(size, device=None, ffn2=False):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    if not ffn2:
        Q = paddle.randint(low=0, high=2, shape=(size,)).cast("float32")
        Q = paddle.ones_like(Q, dtype="float32")
        Q = Q * 2 - 1
        Q = paddle.diag(Q)
        return matmul_hadU_int4(Q), None

    else:
        num_blocks = size
        while not (num_blocks % 2):
            num_blocks = num_blocks // 2
        block_size = size // num_blocks
        Q = paddle.diag(paddle.ones((block_size,), dtype="float32"))
        block = matmul_hadU_int4(Q)
        large_matrix = paddle.zeros([size, size])

        for i in range(num_blocks):
            start_row = i * block_size
            start_col = i * block_size
            large_matrix[start_row : start_row + block_size, start_col : start_col + block_size] = block
        return large_matrix.cast("float32"), block_size


def get_orthogonal_matrix(size, mode="hadamard", device="cuda"):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix_int4(size, device)
    elif mode == "hadamard_ffn2":
        return random_hadamard_matrix_int4(size, device, True)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_model(
    state_dict, prefix_layer_name, layer_idx, moe_num_experts, hidden_size, moe_intermediate_size, ep_rank=0
):
    with paddle.no_grad():
        # collect hadamard rotation matrix [moe_intermediate_size, moe_intermediate_size]
        Q_ffn2, moe_block_size = get_orthogonal_matrix(size=moe_intermediate_size, mode="hadamard_ffn2")
        # down_proj.weight: [moe_intermediate_size, hidden_size]
        expert_list = [
            get_tensor(
                state_dict[
                    f"ernie.{prefix_layer_name}.{layer_idx}.mlp.experts.{ep_rank * moe_num_experts + expert_idx}.down_proj.weight"
                ]
            )
            for expert_idx in range(moe_num_experts)
        ]
        moe_weight = paddle.concat(expert_list, axis=-1)  # [moe_intermediate_size, hidden_size * moe_num_experts]
        new_moe_weight = Q_ffn2.cast("float32").T @ moe_weight.to(Q_ffn2.place)
        for expert_idx in range(moe_num_experts):
            rotated_weight = new_moe_weight[:, expert_idx * hidden_size : (expert_idx + 1) * hidden_size]
            expert_idx_local = ep_rank * moe_num_experts + expert_idx
            state_dict[f"ernie.{prefix_layer_name}.{layer_idx}.mlp.experts.{expert_idx_local}.down_proj.weight"] = (
                rotated_weight.cpu()
            )
        del moe_weight, new_moe_weight, rotated_weight
        paddle.device.cuda.empty_cache()
    return Q_ffn2.cpu()


def pack(src, bits=4):
    pack_num = 8 // bits
    shift_bits = (paddle.arange(0, pack_num) * bits).cast("uint8")
    src = paddle.to_tensor(src).cast("uint8")

    if len(src.shape) == 2:
        row, col = src.shape
        src = src.reshape((row, col // pack_num, pack_num))
    else:
        src = src.reshape((src.shape[0] // pack_num, pack_num))

    src[..., 0] = paddle.bitwise_and(src[..., 0], paddle.to_tensor(15, dtype="uint8"))
    src = paddle.to_tensor(src.numpy() << shift_bits.numpy())

    return src.sum(axis=-1).transpose((1, 0)).cast("int8")


def group_wise_int4_weight_quantize(weight: paddle.Tensor, group_size: int = 128):
    """
    Block-wise int4 weight quantization.

    Args
        weight: paddle.Tensor
        group_size: int

    Returns
        weight_quant: paddle.Tensor, int8 weight after quantization and pack
        weight_scale: paddle.Tensor, fp32 weight scale with group_size
    """
    if weight.dtype == paddle.bfloat16:
        weight = weight.astype(paddle.float32)
    assert weight.dim() == 2
    weight = weight.transpose((1, 0))
    out_features, in_features = weight.shape
    q_max, q_min = 7, -8

    # [out_features, in_features] -> [out_features, in_features // group_size, group_size]
    assert (
        in_features % group_size == 0
    ), f"in_features must be divisible by group_size: {group_size}, but got in_features: {in_features}"
    weight = weight.reshape((out_features, in_features // group_size, group_size))

    # calculate weight_scale
    abs_max = paddle.max(paddle.abs(weight), axis=-1, keepdim=False).astype(paddle.float32)
    weight_scale = paddle.clip(abs_max, min=1e-8)

    quant_weight = paddle.round(weight / weight_scale.unsqueeze(-1) * q_max)
    quant_weight = paddle.clip(quant_weight, min=q_min, max=q_max)
    quant_weight = quant_weight.reshape((out_features, in_features)).transpose((1, 0))

    return quant_weight.astype(paddle.int8), weight_scale


def scale_wrapper(x_amax: paddle.Tensor, eps: float = 0.0) -> paddle.Tensor:
    """
    Paddle implementation of CUDA ScaleWrapper logic.
    Args:
        x_amax (paddle.Tensor): amax tensor (float32 recommended)
        eps (float): epsilon to avoid division by zero
    Returns:
        paddle.Tensor: scale tensor, same shape as x_amax
    """
    fp8_max = 448.0
    float_max = paddle.finfo(paddle.float32).max
    amax_mod = paddle.maximum(
        x_amax,
        paddle.full_like(x_amax, eps),
    )
    scale = fp8_max / amax_mod
    scale = paddle.where(
        amax_mod == 0,
        paddle.ones_like(scale),
        scale,
    )
    scale = paddle.where(
        paddle.isinf(scale),
        paddle.full_like(scale, float_max),
        scale,
    )
    return scale


def per_block_cast_to_fp8(x: Tensor, block_size: list = [128, 128]) -> Tuple[Tensor, Tensor]:
    """
    Only used in deep_gemm block wise quant weight.
    copy from FastDeploy/custom_ops/gpu_ops/fp8_deep_gemm/tests/test_core.py.
    """
    from fastdeploy.model_executor.ops.gpu.deep_gemm import ceil_div

    assert x.dim() == 2
    m, n = x.shape
    x_padded = paddle.zeros(
        (
            ceil_div(m, block_size[0]) * block_size[0],
            ceil_div(n, block_size[1]) * block_size[1],
        ),
        dtype=x.dtype,
    )
    x_padded[:m, :n] = x
    x_view = paddle.view(
        x_padded,
        (-1, block_size[0], x_padded.shape[1] // block_size[1], block_size[1]),
    )

    x_abs = paddle.abs(x_view).astype(paddle.float32)
    x_amax = paddle.amax(x_abs, axis=(1, 3), keepdim=True)
    scale = scale_wrapper(x_amax)
    x_scaled = (x_view * scale).astype(paddle.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (
        paddle.view(1.0 / scale, (x_view.shape[0], x_view.shape[2]))
    )


def per_token_cast_to_fp8(x: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Per token cast to float8_e4m3fn used in wfp8apf8
    """
    x_abs = paddle.abs(x).astype(paddle.float32)
    x_max = x_abs.max(axis=-1, keepdim=True).clip_(min=1e-4)
    x_s = x_max / 448.0
    x_q = paddle.clip(x / x_s, -448.0, 448.0).astype(paddle.float8_e4m3fn)
    return x_q, x_s


# for distributed tensor model parallel
def _set_var_distributed(var: Tensor, split_axis: int):
    """
    Set whether the variable is distributed. If the variable is None, no operation will be performed.

    Args:
        var (Tensor): A Variable object, which can be None. The default value is None.
            The Variable object should have an attribute 'is_distributed' to indicate whether
            the variable has been processed in a distributed manner.
        split_axis (int): the sharding dimension of dist tensors.

    Returns:
    None. No return value.

    """
    if var is None:
        return

    var.is_distributed = True
    var.split_axis = split_axis

    if not in_dynamic_mode():
        # NOTE: use current_block and find_var_recursive to support while_loop
        startup_block = paddle.static.default_startup_program().current_block()
        main_block = paddle.static.default_main_program().current_block()
        startup_block._find_var_recursive(var.name).is_distributed = True
        main_block._find_var_recursive(var.name).is_distributed = True


def get_tensor(input: Union[paddle.Tensor, np.ndarray, str], model_path=None) -> paddle.Tensor:
    """
    Return a corresponding PaddlePaddle tensor based on the type and content of the input.

    Args:
        input (Union[paddle.Tensor, np.ndarray, str]): The input data.

    Returns:
        paddle.Tensor: Returns a PaddlePaddle tensor.

    """
    if "PySafeSlice" in str(type(input)):
        input = input.get()

    if isinstance(input, paddle.Tensor):
        if input.place.is_cpu_place():
            if current_platform.is_cuda():
                return input.cuda()
            else:
                return input.to(paddle.device.get_device())
        return input
    elif isinstance(input, np.ndarray):
        return paddle.to_tensor(input)
    elif isinstance(input, str):
        from fastdeploy.model_executor.load_weight_utils import load_reordered_experts

        return load_reordered_experts(model_path, input)
    else:
        return input


def matmul_hadU(X: Tensor) -> paddle.Tensor:
    """
    Perform matrix multiplication using the Hadamard matrix.

    Args:
        X (Tensor): The tensor to be multiplied.

    Returns:
        Tensor: The tensor after Hadamard matrix multiplication, with the same shape as the input tensor X.

    """
    input = X.clone().reshape((-1, X.shape[-1], 1))
    output = input.clone()
    while input.shape[1] > 1:
        input = input.reshape((input.shape[0], input.shape[1] // 2, 2, input.shape[2]))
        output = output.reshape(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.reshape((input.shape[0], input.shape[1], -1))
        (input, output) = (output, input)
    del output
    return input.reshape(X.shape)


def random_hadamard_matrix(block_size: int, dtype: Union[paddle.dtype, str]) -> paddle.Tensor:
    """
    Generate a random Hadamard matrix.

    Args:
        block_size (int): The size of the block, i.e., the number of rows and columns of the matrix.
        dtype (str): The data type, for example 'float32'.

    Returns:
        paddle.Tensor: The generated random Hadamard matrix.

    """
    Q = paddle.diag(paddle.ones((block_size), dtype=dtype))
    block = matmul_hadU(Q)
    return block


def create_hadamard_matrix(hidden_size: int) -> paddle.Tensor:
    """
    Generate a Hadamard matrix.

    Args:
        hidden_size (int): The size of the hidden layer.

    Returns:
        paddle.Tensor: The generated Hadamard matrix.

    """
    hadamard_block_size = 32
    h = random_hadamard_matrix(hadamard_block_size, "float32")
    block_num = hidden_size // hadamard_block_size
    hadamard_matrix = paddle.to_tensor(block_diag(*[h for i in range(block_num)]))
    return hadamard_matrix


def ensure_divisibility(numerator, denominator):
    """
    Ensure the numerator is divisible by the denominator.

    Args:
        numerator (int): The numerator.
        denominator (int): The denominator.

    Returns:
        None

    Raises:
        AssertionError: If the numerator cannot be evenly divided by the denominator, an assertion error is raised.

    """
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"


def divide(numerator: int, denominator: int):
    """
    Calculate the division result of two numbers.

    Args:
        numerator (int): The dividend.
        denominator (int): The divisor.

    Returns:
        int: The result of the division, which is the quotient of the dividend divided by the divisor.

    """
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def remove_padding(
    max_len: paddle.Tensor,
    input_ids: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    Remove padded sequences from the input.

    Args:
        max_len (paddle.Tensor): The maximum length of the input sequences.
        input_ids (paddle.Tensor): The IDs of the input sequences.
        seq_lens_this_time (paddle.Tensor): The actual length of each sequence.

    Returns:
        tuple: A tuple containing:
            - The sequence IDs with padding removed (paddle.Tensor).
            - The padding offsets (paddle.Tensor).
            - The cumulative offsets (paddle.Tensor).
            - The query sequence lengths (paddle.Tensor).
            - The key sequence lengths (paddle.Tensor).
    """
    if current_platform.is_cuda():
        cum_offsets_now = paddle.cumsum(max_len - seq_lens_this_time, dtype="int32")
        token_num = paddle.sum(seq_lens_this_time)
        (
            ids_remove_padding,
            cum_offsets,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = get_padding_offset(input_ids, cum_offsets_now, token_num, seq_lens_this_time)
        return (
            ids_remove_padding,
            padding_offset,
            cum_offsets,
            cu_seqlens_q,
            cu_seqlens_k,
        )


class CpuGuard:
    """CpuGuard"""

    def __init__(self):
        """init"""
        pass

    def __enter__(self):
        """enter"""
        self.ori_device = paddle.device.get_device()
        paddle.device.set_device("cpu")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """exit"""
        paddle.device.set_device(self.ori_device)


def create_and_set_parameter(layer: nn.Layer, name: str, tensor: paddle.Tensor):
    """
    Create a parameter for a specified layer and set its value to the given tensor.

    Args:
        layer (nn.Layer): The layer object to which the parameter will be added.
        name (str): The name of the parameter to be created.
        tensor (paddle.Tensor): The tensor to set as the value of the parameter.

    Returns:
        None
    """
    setattr(
        layer,
        name,
        layer.create_parameter(
            shape=tensor.shape,
            dtype=tensor.dtype,
            default_initializer=paddle.nn.initializer.Constant(0),
        ),
    )
    getattr(layer, name).set_value(tensor)


@functools.cache
def create_empty_tensor(shape: Tuple[int, ...], dtype: Union[paddle.dtype, str]) -> paddle.Tensor:
    """
    Creates and caches an empty tensor with the specified shape and data type.

    Args:
        shape (Tuple[int, ...]): A tuple representing the dimensions of the tensor.
        dtype (Union[paddle.dtype, str]): The data type for the tensor, such as 'bfloat16', 'float16', etc.

    Returns:
        paddle.Tensor: An empty tensor with the specified shape and data type.
    """
    return paddle.empty(list(shape), dtype=dtype)


def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size: int, rank: int, offset: int = 0):
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f + offset, index_l + offset


def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int, offset: int = 0):
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, offset=offset)


def modules_to_convert(prefix: str, fd_config: FDConfig):
    import fnmatch

    if (
        hasattr(fd_config.model_config, "quantization_config")
        and fd_config.model_config.quantization_config is not None
    ):
        if "modules_to_not_convert" in fd_config.model_config.quantization_config:
            patterns = fd_config.model_config.quantization_config["modules_to_not_convert"]
            for p in patterns:
                if fnmatch.fnmatch(prefix, p) or fnmatch.fnmatch(prefix, p + ".*"):
                    return False
        return True
    else:
        return True
