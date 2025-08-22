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

from fastdeploy.model_executor.layers.quantization.weight_only import (
    WeightOnlyConfig,
    WeightOnlyLinearMethod,
)
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.ops.gcu import linear_quant, weight_quantize_rtn


class GCUWeightOnlyLinearMethod(WeightOnlyLinearMethod):
    """
    Weight only quantization method for linear layer on GCU
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__(quant_config)
        self.quant_config = quant_config
        self.group_size = -1

    def create_weights(self, layer, **extra_weight_attrs):
        # The scale shape should be equal to the output dim of weight using Per-Channel Quantization.
        weight_scale_shape = [layer.weight_shape[1]]

        layer.weight_shape.reverse()
        if self.quant_config.name() == "wint4":
            layer.weight_shape[0] //= 2
        layer.weight_dtype = "int8"

        layer.weight = layer.create_parameter(
            shape=layer.weight_shape,
            dtype=layer.weight_dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        layer.weight_scale = layer.create_parameter(
            shape=weight_scale_shape,
            dtype=layer._dtype,
            is_bias=False,
        )

    def process_prequanted_weights(self, layer, state_dict, is_rearrange: bool = False) -> None:
        """
        Process pre-quantized weights before applying them to the model
        Args:
            layer: The layer that owns the weights
            quant_weight: The quantized weights
            weight_scale: The scale of the quantized weights
        """
        quant_weight = get_tensor(state_dict.pop(layer.weight_key))
        weight_scale = get_tensor(state_dict.pop(layer.weight_scale_key))
        layer.weight.set_value(quant_weight)
        layer.weight_scale.set_value(weight_scale.astype(paddle.get_default_dtype()))

    def process_loaded_weights(self, layer, weight) -> None:
        quanted_weight_tensor, weight_scale_tensor = weight_quantize_rtn(
            weight,
            self.quant_config.algo,
            self.group_size,  # group_size
        )

        layer.weight.set_value(quanted_weight_tensor)
        layer.weight_scale.set_value(weight_scale_tensor.astype(paddle.get_default_dtype()))

    @paddle.no_grad()
    def apply(self, layer, x):
        linear_out = linear_quant(
            lhs=x,
            rhs=layer.weight,
            scale=layer.weight_scale,
            bias=None,
            group_size=self.group_size,
        )
        return linear_out
