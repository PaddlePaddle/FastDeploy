"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

try:
    from fastdeploy.model_executor.ops.gpu import (
        fused_cast_sigmoid_bias as _fused_cast_sigmoid_bias_cuda,
    )
except ImportError as e:
    raise ImportError(
        "fused_cast_sigmoid_bias is not available. " "Please ensure the GPU custom ops are compiled."
    ) from e


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

    Precision:
        所有中间计算（cast、sigmoid、bias加法）均在float32精度下进行，
        仅在最终存储时转换为cast_type。当cast_type为"float32"时，结果与
        以下参考实现完全一致：
            gate_fp32 = gate_out.cast("float32")
            scores = sigmoid(gate_fp32)
            scores_with_bias = scores + bias
        当cast_type为"float16"/"bfloat16"时，精度损失仅来自最终的类型转换，
        等价于在float32计算后调用.cast(cast_type)。
    """
    return _fused_cast_sigmoid_bias_cuda(gate_out, e_score_correction_bias, cast_type)
