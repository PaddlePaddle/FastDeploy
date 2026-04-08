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

 !! This file will be deleted after the platform is fully functional
"""

from typing import Tuple

import numpy as np
import paddle


def xpu_clip_and_round(x: np.ndarray) -> np.ndarray:
    """
    Clip and round the input array to the range [-127, 127] and convert to int8.

    Args:
        x (numpy.ndarray): The input array to be clipped and rounded.

    Returns:
        numpy.ndarray: The clipped and rounded array with dtype int8.
    """
    return np.clip(np.around(x), -127, 127).astype("int8")


def xpu_quant_qkv_weight(
    weight_np: np.ndarray,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    Quantize the query, key, and value weights for the Transformer model.

    Args:
        weight_np (numpy.ndarray): The original weights of query, key, and value in numpy format.
            It should be a 2D or higher dimensional tensor, where the last dimension represents the
            embedding dimension.

    Returns:
        tuple: A tuple containing:
            quanted_weight (paddle.Tensor): The quantized weights in paddle tensor format,
                with the same shape as the input weight_np.
            weight_scales (paddle.Tensor): The scaling factors for each element in the last dimension
                of the input, used to recover the original value range from the quantized weights.
    """
    dim_embed = weight_np.shape[-1]
    weight = np.reshape(weight_np, [-1, dim_embed])
    max_value = np.max(np.abs(weight), axis=1).reshape(-1, 1)
    quanted_weight = xpu_clip_and_round(weight / max_value * 127.0)
    quanted_weight = np.reshape(quanted_weight, weight_np.shape)
    quanted_weight = paddle.to_tensor(quanted_weight, place=paddle.CPUPlace())
    weight_scales = (max_value / 127.0).astype(weight_np.dtype).reshape(-1)
    weight_scales = paddle.to_tensor(weight_scales, place=paddle.CPUPlace())
    weight_scales = paddle.cast(weight_scales, paddle.get_default_dtype())
    return quanted_weight, weight_scales


def xpu_quant_weight(
    weight_np: np.ndarray,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    Quantize the weight tensor for XPU devices.

    Args:
        weight_np (numpy.ndarray): The original weight tensor in numpy format,
            expected to be a 2D array.

    Returns:
        tuple: A tuple containing two elements:
            quanted_weight (paddle.Tensor): The quantized weight tensor,
                converted to a Paddle Tensor on CPU.
            weight_scales (paddle.Tensor): The corresponding scales for the quantized
                weights, also converted to a Paddle Tensor on CPU and cast to the
                default data type.
    """
    weight = np.transpose(weight_np, [1, 0])
    max_value = np.max(np.abs(weight), axis=1).reshape(-1, 1)
    quanted_weight = xpu_clip_and_round(weight / max_value * 127.0)
    quanted_weight = paddle.to_tensor(quanted_weight, place=paddle.CPUPlace())
    weight_scales = (max_value / 127.0).astype(weight_np.dtype).reshape(-1)
    weight_scales = paddle.to_tensor(weight_scales, place=paddle.CPUPlace())
    weight_scales = paddle.cast(weight_scales, paddle.get_default_dtype())
    return quanted_weight, weight_scales
