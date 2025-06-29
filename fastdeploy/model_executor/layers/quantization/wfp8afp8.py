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

from fastdeploy.model_executor.layers.quantization.ops import (
    cutlass_scaled_mm, scaled_fp8_quant)
from fastdeploy.model_executor.layers.quantization.quant_base import (
    QuantConfigBase, QuantMethodBase)


class WFP8AFP8Config(QuantConfigBase):
    """
    Quantization config for weight and activation with FP8.
    """

    def __init__(self, weight_scale_dict, act_scale_dict) -> None:
        super().__init__()
        self.weight_scale_dict = weight_scale_dict
        self.act_scale_dict = act_scale_dict
        self.quant_max_bound = 448
        self.quant_min_bound = -448
        self.quant_round_type = 1

    def name(self) -> str:
        """
        """
        return "wfp8afp8"

    @classmethod
    def from_config(cls, config: dict) -> "WFP8AFP8Config":
        """
        """
        weight_scale_dict = config.get("weight_scale_dict", None)
        act_scale_dict = config.get("act_scale_dict", None)
        return cls(weight_scale_dict, act_scale_dict)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        """
        """
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
        """
        """
        layer.linear_weight_shape.reverse()
        layer.weight_dtype = "float8_e4m3fn"
        # TODO(YuanRisheng): set weight logic should be moved to process_loaded_weights func
        self.skip_quant = False
        layer.linear_weight_scale = layer.create_parameter(
            shape=[1],
            dtype="float32",
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

    def process_loaded_weights(self, layer, weights) -> None:
        """
        """
        if self.skip_quant:
            weight_tensor = weights.cast(layer._dtype)
            layer.linear_weight.set_value(weight_tensor)
            return
        if weights.dtype != paddle.float8_e4m3fn:
            self.use_per_token_if_dynamic = True
        weight_tensor = weights.transpose([1, 0]).contiguous()
        qweight, weight_scale = scaled_fp8_quant(
            weight_tensor,
            use_per_token_if_dynamic=False,
        )
        layer.linear_weight.copy_(qweight, False)
        layer.linear_weight_scale.set_value(weight_scale)

    def apply(self, layer, x):
        """
        """
        if self.skip_quant:
            linear_out = paddle.matmul(x, layer.linear_weight, False, True)
            return linear_out
        if self.use_per_token_if_dynamic:
            out_type = x.dtype
            a_q, a_scales = scaled_fp8_quant(
                x, use_per_token_if_dynamic=self.use_per_token_if_dynamic)
            linear_out = cutlass_scaled_mm(a_q, layer.linear_weight, a_scales,
                                           layer.linear_weight_scale, out_type,
                                           layer.linear_bias)
        else:
            raise NotImplementedError
        return linear_out
