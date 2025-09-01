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

from fastdeploy.model_executor.layers.moe.fused_moe_backend_base import (
    UnquantizedFusedMoEMethod,
)
from fastdeploy.model_executor.layers.quantization.quant_base import QuantMethodBase
from fastdeploy.model_executor.layers.quantization.weight_only import WeightOnlyConfig
from fastdeploy.model_executor.ops.npu import npu_quant_weight


class NPUMoEMethod(UnquantizedFusedMoEMethod):
    """
    NPU MOE
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_quant_type = "wint8"
        self.added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
        self.added_scale_attrs = [
            "up_gate_proj_weight_scale",
            "down_proj_weight_scale",
        ]

    def process_loaded_weights(self, layer: nn.Layer, state_dict):

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

        up_gate_proj_tensor = paddle.stack(up_gate_proj_weights, axis=0)
        down_proj_tensor = paddle.stack(down_proj_weights, axis=0)

        if self.moe_quant_type == "wint8":
            max_bound = 127
        elif self.moe_quant_type == "wint4":
            max_bound = 7

        for idx, weight_tensor in enumerate([up_gate_proj_tensor, down_proj_tensor]):
            print("<><><><><>run quant moe")
            weight_name = self.added_weight_attrs[idx]
            scale_name = self.added_scale_attrs[idx]

            quanted_weight_scale = weight_tensor.abs().max(axis=1)
            quanted_weight = weight_tensor / quanted_weight_scale[:,
                                                                  None, :] * max_bound
            """
            RuntimeError: (NotFound) The kernel with key (npu, Undefined(AnyLayout), bfloat16) of kernel `round` is not registered and fail to fallback to CPU one. Selected wrong DataType `bfloat16`. Paddle support following DataTypes: float32, float64, float16.
            """
            quanted_weight = paddle.cast(quanted_weight, paddle.float16)
            quanted_weight = paddle.round(quanted_weight).astype("int8")
            quanted_weight_scale = quanted_weight_scale / max_bound

            # quanted_weight, quanted_weight_scale = npu_quant_weight(weight_tensor) # FIXME: 这个地方应该看看这个函数和的功能是不是重复的

            
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

            setattr(
                layer,
                scale_name,
                layer.create_parameter(
                    shape=quanted_weight_scale.shape,
                    dtype="bfloat16",
                ),
            )

            getattr(layer, scale_name).set_value(quanted_weight_scale)

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Paddle Cutlass compute Fused MoE.
        """
        from fastdeploy.model_executor.ops.npu import fused_sparse_moe
        fused_moe_out = fused_sparse_moe(
            x,
            gate.weight.transpose([1, 0]),
            layer.up_gate_proj_weight,
            layer.down_proj_weight,
            None,  # ffn1_bias
            None,  # ffn1_scale
            None,  # ffn2_bias
            None,  # ffn2_scale
            self.moe_quant_type,
            layer.top_k,
            layer.tp_size
        )
        if layer.tp_size > 1:
            from fastdeploy.distributed.communication import (
                tensor_model_parallel_all_reduce,
            )

            tensor_model_parallel_all_reduce(fused_moe_out)

        return fused_moe_out

    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Apply the EP prefill method.
        """
        raise NotImplementedError

    def apply_ep_decode(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Apply the EP decoder method.
        """
        raise NotImplementedError


class NPUWeightOnlyMoEMethod(QuantMethodBase):
    """
    NPU Fused MoE Method.
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
                quant_weight, scale = npu_quant_weight(
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
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        NPU compute Fused MoE.
        """
        from fastdeploy.model_executor.ops.npu import fused_sparse_moe
        fused_moe_out = fused_sparse_moe(
            x,
            gate.weight.transpose([1, 0]),
            layer.up_gate_proj_weight,
            layer.down_proj_weight,
            None,  # ffn1_bias
            (layer.up_gate_proj_weight_scale if hasattr(layer, "up_gate_proj_weight_scale") else None),
            None,  # ffn2_bias
            (layer.down_proj_weight_scale if hasattr(layer, "down_proj_weight_scale") else None),
            self.moe_quant_type,
            layer.top_k,
            layer.tp_size
        )
        if layer.tp_size > 1:
            from fastdeploy.distributed.communication import (
                tensor_model_parallel_all_reduce,
            )

            tensor_model_parallel_all_reduce(fused_moe_out)

        return fused_moe_out