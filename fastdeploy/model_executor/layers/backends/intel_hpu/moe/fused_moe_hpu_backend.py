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

from typing import Callable

import paddle
from paddle import nn

from fastdeploy import envs
from fastdeploy.model_executor.layers.moe.fused_moe_backend_base import (
    UnquantizedFusedMoEMethod,
)


class HpuMoEMethod(UnquantizedFusedMoEMethod):
    """
    Implements Fused Mixture-of-Experts (MoE) computation using HPU-optimized operations.
    This method leverages the HPU backend's fused_gate_moe function for efficient expert routing and computation.
    Designed specifically for PaddlePaddle execution on Habana Processing Units (HPU).
    """

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle HPU load weight process.
        """
        # bf16
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)

        stacked_up_gate_proj_weights = paddle.stack(up_gate_proj_weights, axis=0)
        stacked_down_proj_weights = paddle.stack(down_proj_weights, axis=0)

        layer.up_gate_proj_weight.set_value(stacked_up_gate_proj_weights)
        layer.down_proj_weight.set_value(stacked_down_proj_weights)

    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
        topk_ids_hookfunc: Callable = None,
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
        topk_ids_hookfunc: Callable = None,
    ) -> paddle.Tensor:
        """
        Apply the EP decoder method.
        """
        raise NotImplementedError

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
    ) -> paddle.Tensor:
        """
        Paddle hpu Fused MoE.
        """
        if layer.topk_method == "noaux_tc":
            raise NotImplementedError

        # norm_topk_prob = False if layer.topk_method == "noaux_tc" else True
        chunk_size = envs.FD_HPU_CHUNK_SIZE
        from fastdeploy.model_executor.ops.intel_hpu import fused_gate_moe

        fused_moe_out = fused_gate_moe(
            x,
            gate.weight,
            layer.gate_correction_bias,
            layer.up_gate_proj_weight,
            layer.down_proj_weight,
            layer.top_k,
            norm_topk_prob=True,
            permuted_weights=False,
            activation="silu",
            experts_min=layer.expert_id_offset,
            experts_max=layer.expert_id_offset + layer.num_local_experts - 1,
            chunk_size=chunk_size,
        )

        return fused_moe_out


class HpuTensorWiseFP8MoEMethod(HpuMoEMethod):
    """
    Use Cutlass Group Gemm to compute Fused MoE.
    This method is the oldest way to compute MoE in Paddle.
    """

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        # TODO: split create_parameter from process_loaded_weights
        return NotImplemented

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle HPU load weight process.
        """
        # bf16
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)

        from fastdeploy.model_executor.ops.intel_hpu import fused_quant

        self.quant_fn = fused_quant
        self.moe_quant_type = "tensor_wise_fp8"

        for idx, weights_tensor in enumerate([up_gate_proj_weights, down_proj_weights]):
            weights_name = self.added_weight_attrs[idx]
            scales_name = self.added_scale_attrs[idx]

            weights_list = []
            scales_list = []

            for i in range(layer.num_local_experts):
                # quantize loaded weights
                quant_weight, scale = self.quant_fn(weights_tensor[i])
                weights_list.append(quant_weight)
                scales_list.append(scale)

            setattr(layer, weights_name, weights_list)
            setattr(layer, scales_name, scales_list)

    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
        topk_ids_hookfunc: Callable = None,
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
        topk_ids_hookfunc: Callable = None,
    ) -> paddle.Tensor:
        """
        Apply the EP decoder method.
        """
        raise NotImplementedError

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
    ) -> paddle.Tensor:
        """
        Paddle hpu Fused MoE.
        """
        if layer.topk_method == "noaux_tc":
            raise NotImplementedError

        # norm_topk_prob = False if layer.topk_method == "noaux_tc" else True

        chunk_size = envs.FD_HPU_CHUNK_SIZE
        from fastdeploy.model_executor.ops.intel_hpu import fused_gate_moe_fp8

        fused_moe_out = fused_gate_moe_fp8(
            x,
            gate.weight,
            layer.gate_correction_bias,
            layer.up_gate_proj_weight,
            layer.down_proj_weight,
            layer.up_gate_proj_in_scale,
            layer.down_proj_in_scale,
            layer.up_gate_proj_weight_scale,
            layer.down_proj_weight_scale,
            layer.top_k,
            norm_topk_prob=True,
            permuted_weights=False,
            activation="silu",
            experts_min=layer.expert_id_offset,
            experts_max=layer.expert_id_offset + layer.num_local_experts - 1,
            chunk_size=chunk_size,
        )

        return fused_moe_out
