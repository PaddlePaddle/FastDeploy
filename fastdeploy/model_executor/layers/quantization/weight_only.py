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
from abc import abstractmethod
from typing import Optional

import paddle
from paddle.nn.quant import weight_only_linear, weight_quantize

from fastdeploy.platforms import current_platform

from .quant_base import QuantConfigBase, QuantMethodBase


class WeightOnlyConfig(QuantConfigBase):
    """
    Quantization config for weight only
    Args:
        weight_only_linear_arch: The architecture of weight only linear layer
        algo: The quant algorithm("weight_only_int8" or "weight_only_int4") used for weight only linear layer
    """

    def __init__(
        self,
        weight_only_linear_arch: int,
        algo: str,
    ) -> None:
        super().__init__()
        self.weight_only_linear_arch = weight_only_linear_arch
        self.algo = algo

    def get_name(self) -> str:
        return "weight_only"

    @classmethod
    def from_config(cls, config: dict) -> "WeightOnlyConfig":
        weight_only_linear_arch = config["weight_only_linear_arch"]
        algo = config["algo"]
        return cls(weight_only_linear_arch, algo)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        if current_platform.is_xpu():
            from fastdeploy.model_executor.layers.backends import XPUWeightOnlyLinearMethod
            return XPUWeightOnlyLinearMethod(self)
        else:
            return GPUWeightOnlyLinearMethod(self)


class WeightOnlyLinearMethod(QuantMethodBase):
    """
    Weight only quantization method for linear layer
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__()
        self.quant_config = quant_config

    def create_weights(self, layer):
        weight_only_scale_name = layer.prefix + ".weight_only_scale"
        linear_weight_scale_shape = [layer.embed_dim]
        if hasattr(layer, "linear_weight_shape"):
            if isinstance(layer.linear_weight_shape, list):
                layer_weight_shape = layer.linear_weight_shape
                linear_weight_scale_shape = layer_weight_shape[:1]

        layer.linear_weight_scale = layer.create_parameter(
            shape=linear_weight_scale_shape,
            dtype=layer._dtype,
            is_bias=False,
        )

    @abstractmethod
    def process_loaded_weights(self, layer, weights) -> None:
        raise NotImplementedError

    def apply(self, layer, x):
        linear_out = weight_only_linear(
            x,
            weight=layer.linear_weight,
            bias=layer.linear_bias if layer.add_bias else None,
            weight_scale=layer.linear_weight_scale,
            weight_dtype=layer.weight_dtype,
            arch=self.quant_config.weight_only_linear_arch,
        )
        return linear_out


class GPUWeightOnlyLinearMethod(WeightOnlyLinearMethod):
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

    def process_loaded_weights(self, layer, weight) -> None:
        quanted_weight_tensor, weight_scale_tensor = weight_quantize(
            weight,
            algo=self.quant_config.algo,
            arch=self.quant_config.weight_only_linear_arch,
        )

        layer.linear_weight.set_value(quanted_weight_tensor)
        layer.linear_weight_scale.set_value(
            weight_scale_tensor.astype(paddle.get_default_dtype()))
