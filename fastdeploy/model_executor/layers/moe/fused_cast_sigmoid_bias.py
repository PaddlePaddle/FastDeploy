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

_FUSED_CAST_SIGMOID_BIAS_IMPORT_ERROR = None

try:
    from fastdeploy.model_executor.ops.gpu import (
        fused_cast_sigmoid_bias as _fused_cast_sigmoid_bias_cuda,
    )
except ImportError as e:
    _fused_cast_sigmoid_bias_cuda = None
    _FUSED_CAST_SIGMOID_BIAS_IMPORT_ERROR = e


def is_available() -> bool:
    """Return whether the fused GPU custom op is available."""
    return _fused_cast_sigmoid_bias_cuda is not None


def fused_cast_sigmoid_bias(
    gate_out: paddle.Tensor,
    e_score_correction_bias: paddle.Tensor,
    cast_type: str = "float32",
) -> tuple:
    """
    Fused operation: cast gate_out to the specified type, apply sigmoid, and add bias.

    This function fuses the following three separate operations:
      1. gate_out = gate_out.cast(cast_type)
      2. scores = sigmoid(gate_out)
      3. scores_with_bias = scores + e_score_correction_bias

    Args:
        gate_out: [num_tokens, num_experts], bf16/fp16/fp32 dtype - raw gate output
        e_score_correction_bias: [num_experts], fp32 dtype - correction bias
        cast_type: output dtype string, supports "float32", "float16", "bfloat16"

    Returns:
        scores: [num_tokens, num_experts], cast_type dtype - result of sigmoid(gate_out)
        scores_with_bias: [num_tokens, num_experts], cast_type dtype - scores with bias added

    Precision:
        All intermediate computations (cast, sigmoid, bias addition) are performed
        in float32 precision; conversion to cast_type happens only at the final store.
        When cast_type is "float32", the result is bit-exact with the following
        reference implementation:
            gate_fp32 = gate_out.cast("float32")
            scores = sigmoid(gate_fp32)
            scores_with_bias = scores + bias
        When cast_type is "float16"/"bfloat16", the only precision loss comes from
        the final type conversion, equivalent to calling .cast(cast_type) after
        computing in float32.
    """
    if _fused_cast_sigmoid_bias_cuda is None:
        raise ImportError(
            "fused_cast_sigmoid_bias is not available. " "Please ensure the GPU custom ops are compiled."
        ) from _FUSED_CAST_SIGMOID_BIAS_IMPORT_ERROR
    return _fused_cast_sigmoid_bias_cuda(gate_out, e_score_correction_bias, cast_type)
