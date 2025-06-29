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

from typing import Dict

import paddle
from paddle import nn

from fastdeploy.model_executor.layers.quantization.quant_base import \
    QuantMethodBase
from fastdeploy.model_executor.layers.quantization.weight_only import (
    WeightOnlyConfig, WeightOnlyLinearMethod)
from fastdeploy.model_executor.ops.xpu import weight_quantize_xpu


class XPUWeightOnlyLinearMethod(WeightOnlyLinearMethod):
    """
    Weight only quantization method for linear layer on XPU
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__(quant_config)

    def create_weights(self, layer: nn.Layer) -> None:
        """
        Create weights for linear layer on XPU
        """
        layer.linear_weight_shape.reverse()
        if self.quant_config.name() == "weight_only_int4":
            layer.linear_weight_shape[0] //= 2
        layer.weight_dtype = "int8"
        linear_weight_scale_shape = [layer.embed_dim]
        if hasattr(layer, "linear_weight_shape"):
            if isinstance(layer.linear_weight_shape, list):
                layer_weight_shape = layer.linear_weight_shape
                linear_weight_scale_shape = layer_weight_shape[:1]

        layer.linear_weight_scale = layer.create_parameter(
            shape=linear_weight_scale_shape,
            dtype="float32",
            is_bias=False,
        )

    def process_loaded_weights(self, layer: nn.Layer,
                               weight: paddle.Tensor) -> None:
        """
        loaded_weights using xpu special quantization
        """
        quanted_weight_tensor, weight_scale_tensor = weight_quantize_xpu(
            weight, self.quant_config.algo, -1, -1)
        layer.linear_weight.set_value(
            paddle.transpose(quanted_weight_tensor, [1, 0]))
        layer.linear_weight_scale.set_value(weight_scale_tensor)


class XPUWeightOnlyMoEMethod(QuantMethodBase):
    """
    XPU Fused MoE Method.
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__()
        self.quant_config = quant_config
        self.moe_quant_type = self.quant_config.algo

    def create_weights(self, layer: nn.Layer, state_dict: Dict[str,
                                                               paddle.Tensor]):
        """
        Paddle cutlass create weight process.
        """
        ffn1_weights, ffn2_weights = layer.extract_moe_ffn_weights(state_dict)
        assert len(ffn1_weights) == layer.num_local_experts
        assert len(ffn2_weights) == layer.num_local_experts
        assert ffn1_weights[0].shape == [
            layer.hidden_size, layer.moe_intermediate_size * 2
        ]
        assert ffn2_weights[0].shape == [
            layer.moe_intermediate_size, layer.hidden_size
        ]

        added_weight_attrs = ["moe_ffn1_weight", "moe_ffn2_weight"]
        added_scale_attrs = ["moe_ffn1_weight_scale", "moe_ffn2_weight_scale"]

        for idx, weight_tensor in enumerate([ffn1_weights, ffn2_weights]):
            weight_name = added_weight_attrs[idx]
            scale_name = added_scale_attrs[idx]

            weight_list = []
            weight_scale_list = []
            for i in range(layer.num_local_experts):
                quant_weight, scale = weight_quantize_xpu(
                    weight_tensor[i], self.moe_quant_type, -1,
                    -1)  # weight is [k,n]
                weight_list.append(quant_weight.transpose(
                    [1, 0]))  # transpose weight to [n,k]
                weight_scale_list.append(scale)
            quanted_weight = paddle.stack(weight_list, axis=0)
            setattr(
                layer, weight_name,
                layer.create_parameter(
                    shape=quanted_weight.shape,
                    dtype=quanted_weight.dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ))
            getattr(layer, weight_name).set_value(quanted_weight)

            quanted_weight_scale = paddle.stack(weight_scale_list, axis=0)
            setattr(
                layer, scale_name,
                layer.create_parameter(
                    shape=quanted_weight_scale.shape,
                    dtype=quanted_weight_scale.dtype,
                ))
            getattr(layer, scale_name).set_value(quanted_weight_scale)

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        XPU compute Fused MoE.
        """
        from fastdeploy.model_executor.ops.xpu import xpu_moe_layer

        fused_moe_out = xpu_moe_layer(
            x,
            layer.gate_weight.transpose([1, 0]),
            layer.gate_correction_bias,
            layer.moe_ffn1_weight,
            layer.moe_ffn2_weight,
            None,  # ffn1 bias
            None,  # ffn2 bias
            (layer.moe_ffn1_weight_scale
             if hasattr(layer, "moe_ffn1_weight_scale") else None),
            (layer.moe_ffn2_weight_scale
             if hasattr(layer, "moe_ffn2_weight_scale") else None),
            (layer.moe_ffn2_in_scale
             if hasattr(layer, "moe_ffn2_in_scale") else None),
            self.moe_quant_type,
            layer.top_k,
            False,  # moe group, used in deepseek
        )
        if layer.tp_size > 1:
            from fastdeploy.distributed.communication_op import \
                tensor_model_parallel_all_reduce
            tensor_model_parallel_all_reduce(fused_moe_out)

        return fused_moe_out
