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
from fastdeploy import envs
from fastdeploy.model_executor.layers.moe import FusedMoE

from ..utils import get_tensor, per_block_cast_to_fp8
from .quant_base import QuantConfigBase, QuantMethodBase


class BlockWiseFP8Config(QuantConfigBase):
    """
    block wise quantization config, only support fp8 quant and only supports loading weights in BF16 format.
    After loading the weights, it will automatically compute quantization sparsity and dynamically perform
    per-token quantization of activations during inference.
    """

    def __init__(self, weight_block_size: list = [-1, -1]) -> None:
        super().__init__()
        self.weight_block_size = weight_block_size
        self.quant_max_bound = 448
        self.quant_min_bound = -448
        self.quant_round_type = 1
        self.use_deep_gemm = bool(envs.FD_USE_DEEP_GEMM)

    def name(self) -> str:
        return "block_wise_fp8"

    @classmethod
    def from_config(cls, config: dict) -> "BlockWiseFP8Config":
        weight_block_size = config.get("weight_block_size", [128, 128])
        return cls(weight_block_size)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        """
        Get quantization method.
        """
        if isinstance(layer, FusedMoE):
            if self.use_deep_gemm:
                from fastdeploy.model_executor.layers.moe.fused_moe_deepgemm_backend import (
                    DeepGemmFusedMoeMethod,
                )

                return DeepGemmFusedMoeMethod(self)
            else:
                from fastdeploy.model_executor.layers.moe.fused_moe_triton_backend import (
                    BlockWiseFP8MoEMethod,
                )
            return BlockWiseFP8MoEMethod(self)
        else:
            return BlockWiseFP8LinearMethod(self)


class BlockWiseFP8LinearMethod(QuantMethodBase):
    """
    block wise quantization method for linear
    """

    def __init__(
        self,
        quant_config: BlockWiseFP8Config,
    ) -> None:
        super().__init__()
        self.quant_config = quant_config

    def create_weights(self, layer):
        layer.weight_shape.reverse()
        layer.weight_scale = layer.create_parameter(
            shape=[
                (layer.output_size + self.quant_config.weight_block_size[0] - 1)
                // self.quant_config.weight_block_size[0],
                (layer.input_size + self.quant_config.weight_block_size[1] - 1)
                // self.quant_config.weight_block_size[1],
            ],
            dtype="float32",
            is_bias=False,
        )
        layer.weight_dtype = "float8_e4m3fn"

    def process_loaded_weights(self, layer, weights) -> None:
        weight_tensor = weights.transpose([1, 0])
        quanted_weight_tensor, weight_block_scale_tensor = per_block_cast_to_fp8(weight_tensor)
        layer.weight.copy_(quanted_weight_tensor, False)
        layer.weight_scale.set_value(weight_block_scale_tensor)

    def process_prequanted_weights(self, layer, state_dict):
        """
        process_prequanted_weights
        """
        quant_weight = get_tensor(state_dict.pop(layer.weight_key))
        weight_scale = get_tensor(state_dict.pop(layer.weight_scale_key))

        quant_weight = quant_weight.transpose([1, 0]).contiguous()
        layer.weight.copy_(quant_weight.view("float8_e4m3fn"), False)

        weight_scale = weight_scale.transpose([1, 0])
        layer.weight_scale.set_value(weight_scale)

    def apply(self, layer, x):
        x, x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant_padding(
            x, self.quant_config.weight_block_size[0]
        )
        linear_out = paddle.empty((x.shape[0], layer.output_size), dtype=paddle.bfloat16)
        from fastdeploy.model_executor.ops.gpu import deep_gemm

        deep_gemm.gemm_fp8_fp8_bf16_nt(
            (x, x_scale_tensor),
            (layer.weight, layer.weight_scale),
            linear_out,
        )
        if layer.with_bias:
            linear_out = paddle.add(linear_out, layer.bias)
        return linear_out
