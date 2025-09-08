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
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from fastdeploy.model_executor.layers.moe import FusedMoE
from fastdeploy.model_executor.utils import TensorTracker, set_weight_attrs

from ..utils import get_tensor, per_block_cast_to_fp8
from .quant_base import QuantConfigBase, QuantMethodBase


class BlockWiseFP8Config(QuantConfigBase):
    """
    block wise quantization config, only support fp8 quant and only supports loading weights in BF16 format.
    After loading the weights, it will automatically compute quantization sparsity and dynamically perform
    per-token quantization of activations during inference.
    """

    def __init__(self, weight_block_size: list = [-1, -1], is_checkpoint_bf16: bool = False) -> None:
        super().__init__()
        self.weight_block_size = weight_block_size
        self.quant_max_bound = 448
        self.quant_min_bound = -448
        self.quant_round_type = 1
        self.use_deep_gemm = bool(envs.FD_USE_DEEP_GEMM)
        self.is_checkpoint_bf16 = is_checkpoint_bf16

    def name(self) -> str:
        return "block_wise_fp8"

    @classmethod
    def from_config(cls, config: dict) -> "BlockWiseFP8Config":
        weight_block_size = config.get("weight_block_size", [128, 128])
        is_checkpoint_bf16 = config.get("is_checkpoint_bf16", False)
        return cls(weight_block_size, is_checkpoint_bf16)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        """
        Get quantization method.
        """
        if isinstance(layer, FusedMoE):
            if layer.ep_size > 1 or self.use_deep_gemm:
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

    def create_weights(self, layer, **extra_weight_attrs):
        if self.quant_config.is_checkpoint_bf16:
            layer.weight = layer.create_parameter(
                shape=layer.weight_shape,
                dtype=layer.weight_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            quant_attrs = extra_weight_attrs
            if isinstance(layer, MergedColumnParallelLinear) or isinstance(layer, QKVParallelLinear):
                quant_attrs = {
                    **extra_weight_attrs,
                    "tensor_track": TensorTracker(
                        shape=layer.weight_shape, output_dim=extra_weight_attrs.get("output_dim")
                    ),
                }
            set_weight_attrs(
                layer.weight,
                quant_attrs,
            )
        else:
            layer.weight_shape.reverse()
            layer.weight_dtype = "float8_e4m3fn"
            layer.weight = layer.create_parameter(
                shape=layer.weight_shape,
                dtype=layer.weight_dtype,
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            layer.weight_scale_inv = layer.create_parameter(
                shape=[
                    (layer.output_size + self.quant_config.weight_block_size[0] - 1)
                    // self.quant_config.weight_block_size[0],
                    (layer.input_size + self.quant_config.weight_block_size[1] - 1)
                    // self.quant_config.weight_block_size[1],
                ],
                dtype="float32",
                is_bias=False,
            )

    def process_weights_after_loading(self, layer) -> None:
        if not self.quant_config.is_checkpoint_bf16:
            return
        weight_tensor = layer.weight.transpose([1, 0])
        quanted_weight_tensor, weight_block_scale_tensor = per_block_cast_to_fp8(weight_tensor)

        if hasattr(layer.weight, "tensor_track"):
            layer.weight.tensor_track = None
        layer.weight.value().get_tensor()._clear()
        del layer.weight

        layer.weight = layer.create_parameter(
            shape=quanted_weight_tensor.shape,
            dtype="float8_e4m3fn",
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.weight_scale_inv = layer.create_parameter(
            shape=weight_block_scale_tensor.shape,
            dtype="float32",
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        layer.weight.copy_(quanted_weight_tensor, False)
        layer.weight_scale_inv.copy_(weight_block_scale_tensor, False)

    def process_loaded_weights(self, layer, weights) -> None:
        weight_tensor = weights.transpose([1, 0])
        quanted_weight_tensor, weight_block_scale_tensor = per_block_cast_to_fp8(weight_tensor)
        layer.weight.copy_(quanted_weight_tensor, False)
        layer.weight_scale_inv.set_value(weight_block_scale_tensor)

    def process_prequanted_weights(self, layer, state_dict, is_rearrange: bool = False):
        """
        process_prequanted_weights
        """
        quant_weight = get_tensor(state_dict.pop(layer.weight_key))
        weight_scale = get_tensor(state_dict.pop(layer.weight_scale_key))

        quant_weight = quant_weight.transpose([1, 0]).contiguous()
        layer.weight.copy_(quant_weight.view("float8_e4m3fn"), False)

        weight_scale = weight_scale.transpose([1, 0])
        layer.weight_scale_inv.set_value(weight_scale)

    def apply(self, layer, x):
        x, x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant_padding(
            x, self.quant_config.weight_block_size[0]
        )
        linear_out = paddle.empty((x.shape[0], layer.output_size), dtype=paddle.bfloat16)
        from fastdeploy.model_executor.ops.gpu import deep_gemm

        deep_gemm.gemm_fp8_fp8_bf16_nt(
            (x, x_scale_tensor),
            (layer.weight, layer.weight_scale_inv),
            linear_out,
        )
        if layer.with_bias:
            linear_out = paddle.add(linear_out, layer.bias)
        return linear_out
