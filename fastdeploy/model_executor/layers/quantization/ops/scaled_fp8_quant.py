"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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


def scaled_fp8_quant(
    input: paddle.Tensor,
    scale: Optional[paddle.Tensor] = None,
    num_token_padding: Optional[int] = None,
    scale_ub: float = 0,
    use_per_token_if_dynamic: bool = False,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.

    Returns:
        tuple[paddle.Tensor, paddle.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    shape = input.shape
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    output = paddle.empty(shape, dtype=paddle.float8_e4m3fn)

    if scale is None:
        if use_per_token_if_dynamic:
            scale = paddle.empty([shape[0], 1], dtype=paddle.float32)
            from fastdeploy.model_executor.ops.gpu import (
                dynamic_per_token_scaled_fp8_quant,
            )

            dynamic_per_token_scaled_fp8_quant(output, input, scale, scale_ub)
        else:
            scale = paddle.zeros([1], dtype=paddle.float32)
            from fastdeploy.model_executor.ops.gpu import dynamic_scaled_fp8_quant

            dynamic_scaled_fp8_quant(output, input, scale)
    else:
        # num_token_padding not implemented for this case
        # assert (scale.numel() == 1 or num_token_padding is None)
        from fastdeploy.model_executor.ops.gpu import static_scaled_fp8_quant

        static_scaled_fp8_quant(output, input, scale)

    return output, scale
