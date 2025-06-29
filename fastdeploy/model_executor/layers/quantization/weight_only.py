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
import os
from abc import abstractmethod
from typing import Optional

import paddle
from paddle.nn.quant import weight_only_linear, weight_quantize

from fastdeploy.platforms import current_platform

from ..moe import FusedMoE
from ..utils import get_tensor
from .quant_base import QuantConfigBase, QuantMethodBase


class WeightOnlyConfig(QuantConfigBase):
    """
    Quantization config for weight only
    Args:
        algo: The quant algorithm("weight_only_int8" or "weight_only_int4") used for weight only linear layer
    """

    def __init__(
        self,
        algo: str,
    ) -> None:
        super().__init__()
        self.algo = algo
        # arch (int): The compute arch for target device. For example, A100 is 80, v100 is 70,
        # if you do not assign arch, we will get arch from your device, default: None.
        self.weight_only_linear_arch = os.getenv(
            "FLAGS_weight_only_linear_arch")
        if self.weight_only_linear_arch is not None:
            self.weight_only_linear_arch = int(self.weight_only_linear_arch)
        self.quant_max_bound = 0
        self.quant_min_bound = 0
        self.quant_round_type = 0

    def name(self) -> str:
        return "weight_only"

    @classmethod
    def from_config(cls, config: dict) -> "WeightOnlyConfig":
        algo = config["algo"]
        return cls(algo)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        if current_platform.is_xpu():
            from fastdeploy.model_executor.layers.backends import (
                XPUWeightOnlyLinearMethod, XPUWeightOnlyMoEMethod)
            if isinstance(layer, FusedMoE):
                return XPUWeightOnlyMoEMethod(self)
            else:
                return XPUWeightOnlyLinearMethod(self)
        else:
            if isinstance(layer, FusedMoE):
                if layer.use_method == "cutlass":
                    from fastdeploy.model_executor.layers.moe.fused_moe_cutlass_backend import \
                        CutlassWeightOnlyMoEMethod
                    return CutlassWeightOnlyMoEMethod(self)
                elif layer.use_method == "triton":
                    from fastdeploy.model_executor.layers.moe.fused_moe_triton_backend import \
                        TritonWeightOnlyMoEMethod
                    return TritonWeightOnlyMoEMethod(self)
                elif layer.use_method == "marlin":
                    from fastdeploy.model_executor.layers.moe.fused_moe_marlin_backend import \
                        MarlinWeightOnlyMoEMethod
                    return MarlinWeightOnlyMoEMethod(self)
                else:
                    raise ValueError(
                        f"Unsupported MOE backend {layer.use_method}")
            else:
                return GPUWeightOnlyLinearMethod(self)


class WINT8Config(WeightOnlyConfig):
    """
    weight only int8 config
    """

    def __init__(self, ) -> None:
        super().__init__("weight_only_int8")

    @classmethod
    def from_config(cls, config: dict) -> "WINT8Config":
        return cls()

    def name(self) -> str:
        return "wint8"


class WINT4Config(WeightOnlyConfig):
    """
    weight only int4 config
    """

    def __init__(self, ) -> None:
        super().__init__("weight_only_int4")

    @classmethod
    def from_config(cls, config: dict) -> "WINT4Config":
        return cls()

    def name(self) -> str:
        return "wint4"


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
        layer.linear_weight_shape.reverse()
        if self.quant_config.name() == "wint4":
            layer.linear_weight_shape[0] //= 2
        layer.weight_dtype = "int8"
        linear_weight_scale_shape = [layer.embed_dim]
        if hasattr(layer, "linear_weight_shape"):
            if isinstance(layer.linear_weight_shape, list):
                layer_weight_shape = layer.linear_weight_shape
                linear_weight_scale_shape = layer_weight_shape[:1]
            if self.quant_config.name() == "wint4":
                linear_weight_scale_shape[0] *= 2

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
            weight_dtype="int8"
            if self.quant_config.name() == "wint8" else "int4",
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

    def process_prequanted_weights(self, layer, state_dict) -> None:
        """
        Process pre-quantized weights before applying them to the model
        Args:
            layer: The layer that owns the weights
            quant_weight: The quantized weights
            weight_scale: The scale of the quantized weights
        """
        quant_weight = get_tensor(state_dict.pop(layer.weight_key))
        weight_scale = get_tensor(state_dict.pop(layer.weight_scale_key))
        layer.linear_weight.set_value(quant_weight)
        layer.linear_weight_scale.set_value(
            weight_scale.astype(paddle.get_default_dtype()))

    def process_loaded_weights(self, layer, weight) -> None:
        quanted_weight_tensor, weight_scale_tensor = weight_quantize(
            weight,
            algo=self.quant_config.algo,
            arch=self.quant_config.weight_only_linear_arch,
        )

        layer.linear_weight.set_value(quanted_weight_tensor)
        layer.linear_weight_scale.set_value(
            weight_scale_tensor.astype(paddle.get_default_dtype()))
