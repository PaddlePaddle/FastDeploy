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
    WeightOnlyConfig, WeightOnlyLinearMethod)
from fastdeploy.model_executor.ops.npu import fused_linear_op as weight_only_linear
from fastdeploy.model_executor.ops.npu import npu_quant_weight
# import inspect

class NPUWeightOnlyLinearMethod(WeightOnlyLinearMethod):
    """
    Weight only quantization method for linear layer on NPU
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__(quant_config)

    def create_weights(self, layer):
        """
        Create weights for linear layer on NPU
        """
        
        linear_weight_scale_shape = [layer.embed_dim]
        if hasattr(layer, "linear_weight_shape"):
            if isinstance(layer.linear_weight_shape, list):
                layer_weight_shape = layer.linear_weight_shape
                linear_weight_scale_shape = layer_weight_shape[:1]

        layer.linear_weight_scale = layer.create_parameter(
            shape=linear_weight_scale_shape,
            dtype="bfloat16",
            is_bias=False,
        )

    def process_loaded_weights(self, layer, weight) -> None:
        """
        loaded_weights using npu special quantization
        """

        quanted_weight_tensor, weight_scale_tensor = npu_quant_weight(weight)
        layer.linear_weight.set_value(quanted_weight_tensor.T)  
        layer.linear_weight_scale.set_value(
            weight_scale_tensor.astype(paddle.get_default_dtype())
        )

    def apply(self, layer, x):
        linear_out = weight_only_linear(
                x,
                weight=layer.linear_weight.T,
                weight_scale=layer.linear_weight_scale,
            )
        return linear_out
