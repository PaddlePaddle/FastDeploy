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

import paddle

from fastdeploy.model_executor.ops.gpu import (
    fused_cast_sigmoid_bias as _fused_cast_sigmoid_bias_cuda,
)


def fused_cast_sigmoid_bias(
    gate_out: paddle.Tensor,
    e_score_correction_bias: paddle.Tensor,
    cast_type: str = "float32",
) -> tuple:
    """
    融合操作：将gate_out转换为指定类型，应用sigmoid函数，并添加偏置。

    该函数融合了以下三个独立操作：
      1. gate_out = gate_out.cast(cast_type)
      2. scores = sigmoid(gate_out)
      3. scores_with_bias = scores + e_score_correction_bias

    Args:
        gate_out: [num_tokens, num_experts]，bf16/fp16/fp32类型 - 原始gate输出
        e_score_correction_bias: [num_experts]，fp32类型 - 修正偏置
        cast_type: 输出数据类型字符串，支持"float32"、"float16"、"bfloat16"

    Returns:
        scores: [num_tokens, num_experts]，cast_type类型 - sigmoid(gate_out)的结果
        scores_with_bias: [num_tokens, num_experts]，cast_type类型 - 加上偏置后的分数
    """
    return _fused_cast_sigmoid_bias_cuda(gate_out, e_score_correction_bias, cast_type)
