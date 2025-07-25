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
from paddle import nn

from fastdeploy.model_executor.layers.quantization.weight_only import (
    WeightOnlyConfig,
    WeightOnlyLinearMethod,
)
from fastdeploy.model_executor.ops.xpu import weight_quantize_xpu


class XPUWeightOnlyLinearMethod(WeightOnlyLinearMethod):
    """
    Weight only quantization method for linear layer on XPU
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__(quant_config)

    def create_weights(self, layer: nn.Layer) -> None:
        """
        Create weights for linear layer on XPU
        """
        # The scale shape should be equal to the output dim of weight using Per-Channel Quantization.
        weight_scale_shape = [layer.weight_shape[1]]
        layer.weight_shape.reverse()
        if self.quant_config.name() == "weight_only_int4":
            layer.weight_shape[0] //= 2
        layer.weight_dtype = "int8"
        layer.weight_scale = layer.create_parameter(
            shape=weight_scale_shape,
            dtype="float32",
            is_bias=False,
        )

    def process_loaded_weights(self, layer: nn.Layer, weight: paddle.Tensor) -> None:
        """
        loaded_weights using xpu special quantization
        """
        quanted_weight_tensor, weight_scale_tensor = weight_quantize_xpu(weight, self.quant_config.algo, -1, -1)
        layer.weight.set_value(paddle.transpose(quanted_weight_tensor, [1, 0]))
        layer.weight_scale.set_value(weight_scale_tensor)
