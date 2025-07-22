"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle


def convert_to_npu_dequant_scale(deq_scale):
    """
    Convert dequantization scale for NPU.

    Args:
        deq_scale (paddle.Tensor): The original dequantization scale tensor.

    Returns:
        paddle.Tensor: Converted dequantization scale tensor for NPU.
        If NPU is not available, the original tensor is returned.

    This function is designed to prepare the dequantization scale tensor for NPU.
    It first checks if NPU is available. If not, it simply returns the original tensor.
    If NPU is available, it converts the tensor into a specific format required by NPU
    by stacking the original scale values with zeros, reshaping, and converting the data
    type to int64 before returning it as a paddle tensor.
    """
    if not paddle.is_compiled_with_custom_device("npu"):
        return deq_scale
    arr = deq_scale.numpy()
    new_deq_scale = np.stack([arr.reshape(-1, 1), np.zeros_like(arr).reshape(-1, 1)], axis=-1).reshape(-1)
    return paddle.to_tensor(np.frombuffer(new_deq_scale.tobytes(), dtype=np.int64))
