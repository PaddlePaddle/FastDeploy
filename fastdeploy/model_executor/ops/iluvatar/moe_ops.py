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
from paddle.nn.functional import swiglu

try:
    from fastdeploy.model_executor.ops.iluvatar import (
        restore_tokens_per_expert,
        w8a16_group_gemm,
        w8a16_group_gemv,
        wi4a16_group_gemm,
        wi4a16_group_gemv,
    )
except:
    w8a16_group_gemm = None
    w8a16_group_gemv = None
    wi4a16_group_gemm = None
    wi4a16_group_gemv = None
    restore_tokens_per_expert = None


def _pre_process_expert_ffn(
    moe_phase: str,
    quant_method: str,
    tokens_expert_prefix_sum: paddle.Tensor,
):
    if quant_method == "weight_only_int8":
        if moe_phase == "decode":
            group_gemm_func = w8a16_group_gemv
            tokens_per_expert = restore_tokens_per_expert(tokens_expert_prefix_sum).to("int32")
        else:
            group_gemm_func = w8a16_group_gemm
            tokens_per_expert = tokens_expert_prefix_sum
    else:
        if moe_phase == "decode":
            group_gemm_func = wi4a16_group_gemv
            tokens_per_expert = restore_tokens_per_expert(tokens_expert_prefix_sum).to("int32")
        else:
            group_gemm_func = wi4a16_group_gemm
            tokens_per_expert = tokens_expert_prefix_sum
    return group_gemm_func, tokens_per_expert


def iluvatar_moe_expert_ffn(
    permute_input: paddle.Tensor,
    tokens_expert_prefix_sum: paddle.Tensor,
    up_gate_proj_weight: paddle.Tensor,
    down_proj_weight: paddle.Tensor,
    up_gate_proj_bias: Optional[paddle.Tensor],
    up_gate_proj_scale: Optional[paddle.Tensor],
    up_gate_proj_zeros: Optional[paddle.Tensor],
    down_proj_scale: Optional[paddle.Tensor],
    down_proj_zeros: Optional[paddle.Tensor],
    quant_method: str,
    group_size: int,
    moe_phase: str,
):
    assert up_gate_proj_bias is None
    assert up_gate_proj_scale is not None
    assert down_proj_scale is not None
    if quant_method == "weight_only_int4":
        assert up_gate_proj_zeros is not None
        assert down_proj_zeros is not None
    group_gemm_func, tokens_per_expert = _pre_process_expert_ffn(moe_phase, quant_method, tokens_expert_prefix_sum)
    ffn1_output = group_gemm_func(
        permute_input, up_gate_proj_weight, up_gate_proj_scale, up_gate_proj_zeros, tokens_per_expert, group_size
    )
    act_out = swiglu(ffn1_output)
    output = group_gemm_func(
        act_out, down_proj_weight, down_proj_scale, down_proj_zeros, tokens_per_expert, group_size
    )
    return output
