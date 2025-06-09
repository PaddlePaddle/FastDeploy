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
from paddlenlp.utils.log import logger

import fastdeploy
from fastdeploy.platforms.utils import convert_to_npu_dequant_scale

from .quant_base import QuantConfigBase, QuantMethodBase


class W8A8Config(QuantConfigBase):
    """
    quantization config for weight 8bits and activation 8bits
    """

    def __init__(self, weight_scale_dict, act_scale_dict,
                 use_gemm_dequant) -> None:
        super().__init__()
        self.weight_scale_dict = weight_scale_dict
        self.act_scale_dict = act_scale_dict
        self.use_gemm_dequant = use_gemm_dequant

    def get_name(self) -> str:
        return "w8a8"

    @classmethod
    def from_config(cls, config: dict) -> "W8A8Config":
        weight_scale_dict = config["weight_scale_dict"]
        act_scale_dict = config["act_scale_dict"]
        use_gemm_dequant = config["use_gemm_dequant"]
        return cls(weight_scale_dict, act_scale_dict, use_gemm_dequant)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        return W8A8LinearMethod(self)


class W8A8LinearMethod(QuantMethodBase):
    """
    quantization method for weight 8bits and activation 8bits of linear layer
    """

    def __init__(
        self,
        quant_config: W8A8Config,
    ) -> None:
        super().__init__()
        self.quant_config = quant_config

    def create_weights(self, layer):
        weight_scale = self.quant_config.weight_scale_dict.get(
            layer.prefix + ".weight_quanter")
        in_scale = self.quant_config.act_scale_dict.get(layer.prefix +
                                                        ".activation_quanter")
        self.skip_quant = False
        if weight_scale is None or in_scale is None:
            self.skip_quant = True
            return

        max_range = 127.0
        linear_out_scale = paddle.to_tensor(
            weight_scale /
            (max_range * max_range * in_scale)).astype("float32")
        layer.linear_out_scale = layer.create_parameter(
            shape=[layer.embed_dim],
            dtype="float32",
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.linear_out_scale.set_value(
            convert_to_npu_dequant_scale(linear_out_scale))

    def process_loaded_weights(self, layer, weights) -> None:
        if self.skip_quant:
            logger.debug(f"{layer.prefix} skip quant")
            weight_tensor = weights.cast(layer._dtype)
            layer.linear_weight.set_value(weight_tensor)
        else:
            weight_tensor = weights.transpose([1, 0])
            weight_tensor = paddle.cast(weight_tensor, layer.weight_dtype)
            layer.linear_weight.set_value(weight_tensor)

    def apply(self, layer, x):
        if self.skip_quant:
            linear_out = paddle.matmul(x, layer.linear_weight, False, True)
            return linear_out
        if self.quant_config.use_gemm_dequant:
            linear_out = fastdeploy.model_executor.ops.gpu.gemm_dequant(
                x, layer.linear_weight, layer.linear_out_scale, layer._dtype)
        else:
            linear_out = paddle.matmul(x, layer.linear_weight, False, True)
            linear_out = fastdeploy.model_executor.ops.gpu.dequant_int8(
                linear_out, layer.linear_out_scale, layer._dtype)
        return linear_out
