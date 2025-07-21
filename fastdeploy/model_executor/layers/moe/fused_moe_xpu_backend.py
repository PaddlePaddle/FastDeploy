"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from fastdeploy.model_executor.layers.quantization.quant_base import QuantMethodBase
from fastdeploy.model_executor.layers.quantization.weight_only import WeightOnlyConfig
from fastdeploy.model_executor.ops.xpu import weight_quantize_xpu

from .fused_moe_backend_base import MoEMethodBase


class XPUMoEMethod(MoEMethodBase):
    """
    XPU MOE
    """

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass create weight process.
        """
        # bf16
        up_gate_proj_weights, down_proj_weights = layer.extract_moe_ffn_weights(state_dict)
        for weights in [up_gate_proj_weights, down_proj_weights]:
            for idx, weight in enumerate(weights):
                weights[idx] = weight.transpose([1, 0])
        stacked_up_gate_proj_weights = paddle.stack(up_gate_proj_weights, axis=0)
        stacked_down_proj_weights = paddle.stack(down_proj_weights, axis=0)
        for idx, weight_tensor in enumerate([stacked_up_gate_proj_weights, stacked_down_proj_weights]):
            weight_name = self.added_weight_attrs[idx]
            setattr(
                layer,
                weight_name,
                layer.create_parameter(
                    shape=weight_tensor.shape,
                    dtype=weight_tensor.dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            getattr(layer, weight_name).set_value(weight_tensor)

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Paddle Cutlass compute Fused MoE.
        """
        from fastdeploy.model_executor.ops.xpu import xpu_moe_layer

        fused_moe_out = xpu_moe_layer(
            x,
            layer.gate_weight.transpose([1, 0]),
            layer.gate_correction_bias,
            layer.up_gate_proj_weight,
            layer.down_proj_weight,
            None,  # up_gate_proj bias
            None,  # down_proj bias
            None,  # up_gate_proj scale
            None,  # down_proj scale
            None,  # up_gate_proj_in_scale
            "",  # moe_quant_type
            layer.top_k,
            False,  # moe group, used in deepseek
        )
        if layer.tp_size > 1:
            from fastdeploy.distributed.communication_op import (
                tensor_model_parallel_all_reduce,
            )

            tensor_model_parallel_all_reduce(fused_moe_out)

        return fused_moe_out

    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply the EP prefill method.
        """
        raise NotImplementedError

    def apply_ep_decode(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply the EP decoder method.
        """
        raise NotImplementedError


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

    def create_weights(self, layer: nn.Layer, state_dict: Dict[str, paddle.Tensor]):
        """
        Paddle cutlass create weight process.
        """
        up_gate_proj_weights, down_proj_weights = layer.extract_moe_ffn_weights(state_dict)
        assert len(up_gate_proj_weights) == layer.num_local_experts
        assert len(down_proj_weights) == layer.num_local_experts
        assert up_gate_proj_weights[0].shape == [
            layer.hidden_size,
            layer.moe_intermediate_size * 2,
        ]
        assert down_proj_weights[0].shape == [
            layer.moe_intermediate_size,
            layer.hidden_size,
        ]

        added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
        added_scale_attrs = [
            "up_gate_proj_weight_scale",
            "down_proj_weight_scale",
        ]

        for idx, weight_tensor in enumerate([up_gate_proj_weights, down_proj_weights]):
            weight_name = added_weight_attrs[idx]
            scale_name = added_scale_attrs[idx]

            weight_list = []
            weight_scale_list = []
            for i in range(layer.num_local_experts):
                quant_weight, scale = weight_quantize_xpu(
                    weight_tensor[i], self.moe_quant_type, -1, -1
                )  # weight is [k,n]
                weight_list.append(quant_weight.transpose([1, 0]))  # transpose weight to [n,k]
                weight_scale_list.append(scale)
            quanted_weight = paddle.stack(weight_list, axis=0)
            setattr(
                layer,
                weight_name,
                layer.create_parameter(
                    shape=quanted_weight.shape,
                    dtype=quanted_weight.dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            getattr(layer, weight_name).set_value(quanted_weight)

            quanted_weight_scale = paddle.stack(weight_scale_list, axis=0)
            setattr(
                layer,
                scale_name,
                layer.create_parameter(
                    shape=quanted_weight_scale.shape,
                    dtype=quanted_weight_scale.dtype,
                ),
            )
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
            layer.up_gate_proj_weight,
            layer.down_proj_weight,
            None,  # up_gate_proj bias
            None,  # down_proj bias
            (layer.up_gate_proj_weight_scale if hasattr(layer, "up_gate_proj_weight_scale") else None),
            (layer.down_proj_weight_scale if hasattr(layer, "down_proj_weight_scale") else None),
            (layer.down_proj_in_scale if hasattr(layer, "down_proj_in_scale") else None),
            self.moe_quant_type,
            layer.top_k,
            False,  # moe group, used in deepseek
        )
        if layer.tp_size > 1:
            from fastdeploy.distributed.communication_op import (
                tensor_model_parallel_all_reduce,
            )

            tensor_model_parallel_all_reduce(fused_moe_out)

        return fused_moe_out
