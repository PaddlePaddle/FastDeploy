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

from typing import Tuple

import numpy as np
import paddle
from paddle import Tensor
from paddle.framework import in_dynamic_mode
from fastdeploy.platforms import current_platform
if current_platform.is_cuda() and current_platform.available():
    try:
        from fastdeploy.model_executor.ops.gpu import (
            get_padding_offset,
            speculate_get_padding_offset,
        )
    except Exception:
        raise ImportError(
            f"Verify environment consistency between compilation and FastDeploy installation. "
            f"And ensure the Paddle version supports FastDeploy's custom operators"
        )
import re

import os
cache_params = os.getenv("CACHE_PARAMS", "none")
if cache_params != "none":
    c8_state_dict = paddle.load(cache_params, return_numpy=True)

def per_block_cast_to_fp8(x: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Only used in deep_gemm block wise quant weight.
    copy from FastDeploy/custom_ops/gpu_ops/fp8_deep_gemm/tests/test_core.py.
    """
    from fastdeploy.model_executor.ops.gpu.deep_gemm import ceil_div

    assert x.dim() == 2
    m, n = x.shape
    x_padded = paddle.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128),
                            dtype=x.dtype)
    x_padded[:m, :n] = x
    x_view = paddle.view(x_padded, (-1, 128, x_padded.shape[1] // 128, 128))

    x_abs = paddle.abs(x_view).astype(paddle.float32)
    x_amax = paddle.amax(x_abs, axis=(1, 3), keepdim=True)
    x_amax = paddle.clip(x_amax, min=1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).astype(paddle.float8_e4m3fn)

    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (paddle.view(
        x_amax / 448.0, (x_view.shape[0], x_view.shape[2])))


# for distributed tensor model parallel
def _set_var_distributed(var, split_axis):
    """
    Set whether the variable is distributed. If the variable is None, no operation will be performed.

    Args:
    var (Variable, Optional): A Variable object, which can be None. The default value is None.
    The Variable object should have an attribute 'is_distributed' to indicate whether
    the variable has been processed in a distributed manner.
    split_axis (Integer): the sharding dimension of dist tensors

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


def get_tensor(input):
    """
    EP并行中，权重按层分布式存储，为了节省峰值显存，在state_dict处理部分仅保存
    层名与对应权重的路径，因此需要将权重的类型转换为paddle.Tensor
    """
    if isinstance(input, paddle.Tensor):
        if input.place.is_cpu_place():
            return input.to(paddle.device.get_device())
        return input
    elif isinstance(input, np.ndarray):
        return paddle.to_tensor(input)
    elif isinstance(input, str):
        if ".safetensors" in input:

            match = re.match(r"\[(.*?)\](.*)", input)
            if match:
                key_name = match.group(1)
                model_path = match.group(2)
            from safetensors import safe_open

            with safe_open(model_path, framework="np", device="cpu") as f:
                if key_name in f.keys():
                    weight = f.get_tensor(key_name)
                    weight = paddle.Tensor(weight, zero_copy=True)
                    weight = weight._copy_to(
                        paddle.framework._current_expected_place(), False
                    )
                    return weight
                else:
                    return None
        else:   
            if cache_params != "none":
                tmp_key = input.split("/")[-1]
                if tmp_key in c8_state_dict:
                    print(f"Loading {tmp_key} in extra C8_state_dict")
                    return paddle.to_tensor(c8_state_dict.pop(tmp_key))
            return paddle.load(input)
    else:
        # 理论上不会命中这个分支
        return input


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def remove_padding(max_len, input_ids, seq_lens_this_time):
    """
    remove_padding
    """
    if current_platform.is_cuda():
        cum_offsets_now = paddle.cumsum(max_len - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        (
            ids_remove_padding,
            cum_offsets,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = get_padding_offset(input_ids, cum_offsets_now, token_num,
                                seq_lens_this_time)
        return (
            ids_remove_padding,
            padding_offset,
            cum_offsets,
            cu_seqlens_q,
            cu_seqlens_k,
        )

def speculate_remove_padding(max_len, input_ids, seq_lens_this_time,
                                    draft_tokens, seq_lens_encoder):
    """
    remove_padding
    """
    if current_platform.is_cuda():
        cum_offsets_now = paddle.cumsum(max_len - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        (
            ids_remove_padding,
            cum_offsets,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = speculate_get_padding_offset(
            input_ids,
            draft_tokens,
            cum_offsets_now,
            token_num,
            seq_lens_this_time,
            seq_lens_encoder,
        )
        return (
            ids_remove_padding,
            padding_offset,
            cum_offsets,
            cu_seqlens_q,
            cu_seqlens_k,
        )
