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

from typing import Optional

import paddle

import fastdeploy

from ..moe import FusedMoE
from .quant_base import QuantConfigBase, QuantMethodBase

QUANT_SCALING_FACTOR = 448


class W4AFP8Config(QuantConfigBase):
    """
    quantization config for weight 4bits and activation fp8
    """

    def __init__(self, weight_scale_dict, act_scale_dict, is_permuted, hadamard_block_size) -> None:
        super().__init__()
        self.weight_scale_dict = weight_scale_dict
        self.act_scale_dict = act_scale_dict
        self.quant_max_bound = 448
        self.quant_min_bound = -448
        self.quant_round_type = 1
        self.is_permuted = is_permuted
        self.hadamard_block_size = hadamard_block_size

    def name(self) -> str:
        return "w4afp8"

    @classmethod
    def from_config(cls, config: dict) -> "W4AFP8Config":
        weight_scale_dict = config.get("weight_scale_dict", None)
        act_scale_dict = config.get("act_scale_dict", None)
        is_permuted = config.get("is_permuted", True)
        hadamard_block_size = config.get("hadamard_block_size", 128)
        return cls(weight_scale_dict, act_scale_dict, is_permuted, hadamard_block_size)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        if isinstance(layer, FusedMoE):
            from fastdeploy.model_executor.layers.moe.fused_moe_cutlass_backend import (
                CutlassW4AFP8MoEMethod,
            )

            return CutlassW4AFP8MoEMethod(self)
        return W4AFP8LinearMethod(self)


class W4AFP8LinearMethod(QuantMethodBase):
    """
    W4 AFP8 quant method for linear
    """

    def __init__(
        self,
        quant_config: W4AFP8Config,
    ) -> None:
        super().__init__()
        self.quant_config = quant_config

    def create_weights(self, layer, **extra_weight_attrs):
        layer.weight_shape.reverse()
        layer.weight_shape[0] //= 2
        layer.weight_dtype = "int8"

        layer.weight = layer.create_parameter(
            shape=layer.weight_shape,
            dtype=layer.weight_dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

    def process_loaded_weights(self, layer, weights) -> None:
        (
            quanted_weight_tensor,
            weight_scale_tensor,
        ) = fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16_weight_quantize(
            paddle.cast(weights, "float32").cpu(),
            groupsize=-1,
            scale_dtype="float16",
        )
        weight_scale_tensor = paddle.view(weight_scale_tensor, layer._dtype)
        layer.weight.set_value(quanted_weight_tensor)
        layer.weight_scale.set_value(weight_scale_tensor)

    def apply(self, layer, x):
        linear_out = fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16(
            x,
            layer.weight,
            layer.weight_scale,
            zero_points=None,
            bias=layer.bias if layer.add_bias else None,
            out_scale=self.quant_config.weight_scale_dict.get(layer.prefix + ".weight_scale")
            / (
                self.quant_config.act_scale_dict.get(layer.prefix + ".activation_scale")
                * QUANT_SCALING_FACTOR
                * QUANT_SCALING_FACTOR
            ),
            groupsize=0,
            out_dtype=layer._dtype,
        )
        return linear_out
