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

import paddle
from paddle import nn

from fastdeploy.model_executor.layers.quantization.tensor_wise_fp8 import (
    TensorWiseFP8Config,
    TensorWiseFP8LinearMethod,
)
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.ops.intel_hpu import fused_quant
from fastdeploy.model_executor.utils import set_weight_attrs


class HpuTensorWiseFP8LinearMethod(TensorWiseFP8LinearMethod):
    """
    Tensor wise fp8 quantization method for linear layer on HPU
    """

    def __init__(
        self,
        quant_config: TensorWiseFP8Config,
    ) -> None:
        super().__init__(quant_config)
        self.max_bound = 240.0

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

        # these activation_scale will fall in, but only quant for self_attn
        # mlp.shared_experts.up_gate_proj / down_proj
        # self_attn.qkv_proj / o_proj
        if "self_attn" in layer.act_scale_key:
            act_scale_inv = act_scale / self.max_bound
            act_scale = self.max_bound / act_scale
        else:
            act_scale_inv = act_scale
            act_scale = 1.0 / act_scale

        layer.weight.copy_(quant_weight.view("float8_e4m3fn"), False)
        layer.weight_scale.set_value(weight_scale.astype(paddle.get_default_dtype()))
        layer.act_scale.set_value(act_scale.astype(paddle.get_default_dtype()))
        layer.act_scale_inv.set_value(act_scale_inv.astype(paddle.get_default_dtype()))

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs) -> None:
        """
        Create weights for linear layer on HPU
        """
        layer.weight_dtype = "float8_e4m3fn"
        layer.weight = layer.create_parameter(
            shape=layer.weight_shape,
            dtype=layer.weight_dtype,
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.weight_scale = layer.create_parameter(
            shape=[1],
            dtype="bfloat16",
            is_bias=False,
        )
        layer.act_scale = layer.create_parameter(
            shape=[1],
            dtype="bfloat16",
            is_bias=False,
        )
        layer.act_scale_inv = layer.create_parameter(
            shape=[1],
            dtype="bfloat16",
            is_bias=False,
        )

        self.model_format = extra_weight_attrs.get("model_format")
        if self.model_format == "torch" and "output_dim" in extra_weight_attrs:
            extra_weight_attrs["output_dim"] = not extra_weight_attrs["output_dim"]
        set_weight_attrs(
            layer.weight,
            extra_weight_attrs,
        )

    def process_loaded_weights(self, layer: nn.Layer, weight: paddle.Tensor) -> None:
        """
        loaded_weights using HPU specific quantization
        """
        quanted_weight_tensor, weight_scale_tensor = fused_quant(weight)
        layer.weight.set_value(quanted_weight_tensor)
        layer.weight_scale.set_value(weight_scale_tensor)

    def process_weights_after_loading(self, layer: nn.Layer):
        """
        use for loader v1
        """
        # these activation_scale will fall in, but only quant for self_attn
        # mlp.shared_experts.up_gate_proj / down_proj
        # self_attn.qkv_proj / o_proj
        if layer.act_scale._is_initialized():
            if "self_attn" in layer.act_scale_key:
                act_scale_inv = layer.act_scale / self.max_bound
                act_scale = self.max_bound / layer.act_scale
            else:
                act_scale_inv = layer.act_scale
                act_scale = 1.0 / layer.act_scale
        layer.act_scale.set_value(act_scale.astype(paddle.get_default_dtype()))
        layer.act_scale_inv.set_value(act_scale_inv.astype(paddle.get_default_dtype()))
