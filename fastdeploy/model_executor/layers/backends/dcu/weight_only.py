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
from paddle.nn.quant import weight_dequantize

from fastdeploy.model_executor.layers.quantization.weight_only import (
    GPUWeightOnlyLinearMethod,
    WeightOnlyConfig,
)


class DCUWeightOnlyLinearMethod(GPUWeightOnlyLinearMethod):
    """
    Weight only quantization method for linear layer on GPU
    The weights are loaded in the BF16 numerical format. After loading, the quantization coefficients will be computed,
    and the weights will be quantized to int8 or int4.
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__(quant_config)

    def apply(self, layer, x):
        dequant_out = weight_dequantize(
            x=layer.weight,
            scale=layer.weight_scale,
            algo=self.quant_config.algo,
            out_dtype=paddle.get_default_dtype(),
        )
        linear_out = paddle.matmul(x, dequant_out)
        if layer.bias is not None:
            linear_out = paddle.add(linear_out, layer.bias)
        return linear_out
