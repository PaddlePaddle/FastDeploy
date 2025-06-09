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
from fastdeploy.platforms.utils import convert_to_npu_dequant_scale

from .quant_base import QuantConfigBase, QuantMethodBase


class WFP8AFP8Config(QuantConfigBase):
    """
    Quantization config for weight and activation with FP8.
    """

    def __init__(self, weight_scale_dict, act_scale_dict) -> None:
        super().__init__()
        self.weight_scale_dict = weight_scale_dict
        self.act_scale_dict = act_scale_dict

    def get_name(self) -> str:
        return "wfp8afp8"

    @classmethod
    def from_config(cls, config: dict) -> "WFP8AFP8Config":
        weight_scale_dict = config["weight_scale_dict"]
        act_scale_dict = config["act_scale_dict"]
        return cls(weight_scale_dict, act_scale_dict)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        return WFP8AFP8LinearMethod(self)


class WFP8AFP8LinearMethod(QuantMethodBase):
    """
    Weight and activation quantization method for linear layer with FP8
    """

    def __init__(
        self,
        quant_config: WFP8AFP8Config,
    ) -> None:
        super().__init__()
        self.quant_config = quant_config

    def create_weights(self, layer):
        # TODO(YuanRisheng): set weight logic should be moved to process_loaded_weights func
        weight_scale = self.quant_config.weight_scale_dict.get(
            layer.prefix + ".weight_quanter")
        in_scale = self.quant_config.act_scale_dict.get(layer.prefix +
                                                        ".activation_quanter")
        self.skip_quant = False
        # we will skip quant if weight_scale is not found or in_scale is not found
        if weight_scale is None or in_scale is None:
            self.skip_quant = True
        else:
            max_range = 448.0
            layer.scalar_scale_name = layer.prefix + ".scalar_weight_quanter"
            layer.scalar_scale = layer.create_parameter(
                shape=([1]),
                dtype="float32",
            )
            layer.scalar_scale.set_value(
                paddle.to_tensor([1.0 / (max_range * in_scale)],
                                 dtype="float32"))
            linear_out_scale = paddle.to_tensor(weight_scale /
                                                max_range).astype("float32")
            layer.linear_out_scale = layer.create_parameter(
                shape=[layer.embed_dim],
                dtype="float32",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            layer.linear_out_scale.set_value(
                convert_to_npu_dequant_scale(linear_out_scale))

    def process_loaded_weights(self, layer, weights) -> None:
        # TODO(YuanRisheng): We should abstract the ‌skip_quant‌ logic to adapt to more quant methods
        if self.skip_quant:
            weight_tensor = weights.cast(layer._dtype)
            layer.linear_weight.set_value(weight_tensor)
            return
        weight_tensor = weights.transpose([1, 0])
        weight_tensor = paddle.cast(weight_tensor, self.weight_dtype)
        self.linear_weight.copy_(weight_tensor, False)

    def apply(self, layer, x):
        if self.skip_quant:
            linear_out = paddle.matmul(x, layer.linear_weight, False, True)
            return linear_out
        linear_out = fastdeploy.model_executor.ops.gpu.per_channel_fp8_fp8_half_gemm_fused(
            x,
            layer.linear_weight,
            bias=layer.linear_bias if layer.add_bias else None,
            scalar_scale=layer.scalar_scale,
            channel_scale=layer.linear_out_scale,
            transpose_x=False,
            transpose_y=True,
            output_dtype=layer._dtype,
        )
        return linear_out
