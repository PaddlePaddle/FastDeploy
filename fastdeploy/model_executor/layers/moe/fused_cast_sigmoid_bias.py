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
) -> tuple:
    """
    Fused operation: cast gate_out to float32, apply sigmoid, and add bias.

    This fuses three separate operations:
      1. gate_out = gate_out.cast("float32")
      2. scores = sigmoid(gate_out)
      3. scores_with_bias = scores + e_score_correction_bias

    Args:
        gate_out: [num_tokens, num_experts], bf16/fp16/fp32 - raw gate output
        e_score_correction_bias: [num_experts], fp32 - correction bias

    Returns:
        scores: [num_tokens, num_experts], fp32 - sigmoid(gate_out)
        scores_with_bias: [num_tokens, num_experts], fp32 - scores + bias
    """
    return _fused_cast_sigmoid_bias_cuda(gate_out, e_score_correction_bias)
