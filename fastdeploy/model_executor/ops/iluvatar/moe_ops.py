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

from typing import Optional

import paddle
from paddle.incubate.nn.functional import swiglu
from paddle.nn.quant import weight_only_linear

try:
    from fastdeploy.model_executor.ops.iluvatar import w8a16_group_gemm
except ImportError:
    w8a16_group_gemm = None


def group_gemm(
    input: paddle.Tensor,
    tokens_expert_prefix_sum: paddle.Tensor,
    weight: paddle.Tensor,
    scale: paddle.Tensor,
    output: paddle.Tensor,
):
    assert (
        input.dim() == 2
        and tokens_expert_prefix_sum.dim() == 1
        and weight.dim() == 3
        and scale.dim() == 2
        and output.dim() == 2
    )
    num_tokens = input.shape[0]
    dim_in = input.shape[1]
    dim_out = weight.shape[1]
    num_experts = weight.shape[0]

    # check shape
    assert tokens_expert_prefix_sum.shape == [
        num_experts,
    ]
    assert weight.shape == [num_experts, dim_out, dim_in]
    assert scale.shape == [num_experts, dim_out]
    assert output.shape == [num_tokens, dim_out]

    # check dtype
    assert input.dtype in (paddle.float16, paddle.bfloat16)
    assert scale.dtype == input.dtype and output.dtype == input.dtype
    assert tokens_expert_prefix_sum.dtype == paddle.int64
    assert weight.dtype == paddle.int8

    # check others
    assert tokens_expert_prefix_sum.place.is_cpu_place()
    assert tokens_expert_prefix_sum[-1] == num_tokens
    for i in range(num_experts):
        expert_start = 0 if i == 0 else tokens_expert_prefix_sum[i - 1]
        expert_end = tokens_expert_prefix_sum[i]
        if expert_start == expert_end:
            continue
        input_i = input[expert_start:expert_end]
        weight_i = weight[i]
        scale_i = scale[i]
        # avoid d2d?
        output[expert_start:expert_end] = weight_only_linear(
            input_i, weight_i, weight_scale=scale_i, weight_dtype="int8", group_size=-1
        )


def iluvatar_moe_expert_ffn(
    permute_input: paddle.Tensor,
    tokens_expert_prefix_sum: paddle.Tensor,
    up_gate_proj_weight: paddle.Tensor,
    down_proj_weight: paddle.Tensor,
    up_gate_proj_bias: Optional[paddle.Tensor],
    up_gate_proj_scale: Optional[paddle.Tensor],
    down_proj_scale: Optional[paddle.Tensor],
    down_proj_in_scale: Optional[paddle.Tensor],
    expert_idx_per_token: Optional[paddle.Tensor],
    quant_method: str,
    used_in_ep_low_latency: bool,
):
    assert up_gate_proj_bias is None
    assert up_gate_proj_scale is not None
    assert down_proj_scale is not None
    assert down_proj_in_scale is None
    assert expert_idx_per_token is None
    assert quant_method in ("weight_only_int8")
    assert not used_in_ep_low_latency
    tokens_expert_prefix_sum_cpu = tokens_expert_prefix_sum.to("cpu")
    ffn1_output = w8a16_group_gemm(
        permute_input, up_gate_proj_weight, up_gate_proj_scale, tokens_expert_prefix_sum_cpu, -1
    )
    act_out = swiglu(ffn1_output)
    output = w8a16_group_gemm(act_out, down_proj_weight, down_proj_scale, tokens_expert_prefix_sum_cpu, -1)
    return output
