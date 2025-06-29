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

from .quant_base import QuantConfigBase, QuantMethodBase

QUANT_SCALING_FACTOR = 448


class W4AFP8Config(QuantConfigBase):
    """
    quantization config for weight 4bits and activation fp8
    """

    def __init__(self, weight_scale_dict, act_scale_dict) -> None:
        super().__init__()
        self.weight_scale_dict = weight_scale_dict
        self.act_scale_dict = act_scale_dict
        self.quant_max_bound = 448
        self.quant_min_bound = -448
        self.quant_round_type = 1

    def name(self) -> str:
        return "w4afp8"

    @classmethod
    def from_config(cls, config: dict) -> "W4AFP8Config":
        weight_scale_dict = config["weight_scale_dict"]
        act_scale_dict = config["act_scale_dict"]
        return cls(weight_scale_dict, act_scale_dict)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
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

    def create_weights(self, layer):
        layer.linear_weight_shape.reverse()
        layer.linear_weight_shape[0] //= 2
        layer.weight_dtype = "int8"
        pass

    def process_loaded_weights(self, layer, weights) -> None:
        quanted_weight_tensor, weight_scale_tensor = (
            fastdeploy.model_executor.ops.gpu.
            scaled_gemm_f8_i4_f16_weight_quantize(
                paddle.cast(weights, "float32").cpu(),
                groupsize=-1,
                scale_dtype="float16",
            ))
        weight_scale_tensor = paddle.view(weight_scale_tensor, layer._dtype)
        layer.linear_weight.set_value(quanted_weight_tensor)
        layer.linear_weight_scale.set_value(weight_scale_tensor)

    def apply(self, layer, x):
        linear_out = fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16(
            x,
            layer.linear_weight,
            layer.linear_weight_scale,
            zero_points=None,
            bias=layer.linear_bias if layer.add_bias else None,
            out_scale=self.quant_config.weight_scale_dict.get(layer.prefix +
                                                              ".weight_scale")
            / (self.quant_config.act_scale_dict.get(layer.prefix +
                                                    ".activation_scale") *
               QUANT_SCALING_FACTOR * QUANT_SCALING_FACTOR),
            groupsize=0,
            out_dtype=layer._dtype,
        )
        return linear_out
