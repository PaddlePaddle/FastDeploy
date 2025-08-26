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

from fastdeploy.model_executor.layers.moe import FusedMoE

from ..utils import get_tensor
from .quant_base import QuantConfigBase, QuantMethodBase


class TensorWiseFP8Config(QuantConfigBase):
    """
    Quantization config for weight and activation with FP8.
    """

    def __init__(self) -> None:
        """
        Nothing else to do!
        """
        super().__init__()

    def name(self) -> str:
        """
        Nothing else to do!
        """
        return "tensor_wise_fp8"

    @classmethod
    def from_config(cls, config: dict) -> "TensorWiseFP8Config":
        """
        Nothing else to do!
        """
        return cls()

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        """
        return method according to this config!
        """
        if isinstance(layer, FusedMoE):
            from fastdeploy.model_executor.layers.moe.fused_moe_triton_backend import (
                TensorWiseFP8MoEMethod,
            )

            return TensorWiseFP8MoEMethod(self)
        else:
            return TensorWiseFP8LinearMethod(self)


class TensorWiseFP8LinearMethod(QuantMethodBase):
    """
    Weight and activation quantization method for linear layer with per tensor FP8
    """

    def __init__(
        self,
        quant_config: TensorWiseFP8Config,
    ) -> None:
        """
        Nothing special to do!
        """
        super().__init__()
        self.quant_config = quant_config
        self.quant_max_bound = 448
        self.quant_min_bound = -448
        self.quant_round_type = 1
        self.weight_dtype = "float8_e4m3fn"

    def create_weights(self, layer, **extra_weight_attrs):
        layer.weight_dtype = "float8_e4m3fn"
        layer.weight = layer.create_parameter(
            shape=layer.weight_shape,
            dtype=layer.weight_dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
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
        act_scale = get_tensor(state_dict.pop(layer.act_scale_key))

        quant_weight = quant_weight.transpose([1, 0]).contiguous()
        layer.weight.copy_(quant_weight.view("float8_e4m3fn"), False)

        self.act_scale = act_scale.item()
        self.total_scale = (act_scale * weight_scale).item()

    def process_loaded_weights(self, layer, weights, state_dict) -> None:
        """
        Read fp8 weight, act scale, weight scale
        """
        pass

    def apply(self, layer, x):
        """
        compute!
        """
        from fastdeploy.model_executor.ops.gpu import (
            cutlass_fp8_fp8_half_gemm_fused,
            fused_hadamard_quant_fp8,
        )

        fp8_x = fused_hadamard_quant_fp8(x, scale=self.act_scale)

        linear_out = cutlass_fp8_fp8_half_gemm_fused(
            fp8_x,
            layer.weight,
            transpose_x=False,
            transpose_y=True,
            bias=None,
            scale=self.total_scale,
            output_dtype="bfloat16",
            activation_type="identity",
        )
        return linear_out
