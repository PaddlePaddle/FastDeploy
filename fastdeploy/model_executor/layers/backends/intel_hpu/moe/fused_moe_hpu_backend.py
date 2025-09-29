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

import paddle
from paddle import nn

from fastdeploy.distributed.communication import tensor_model_parallel_all_reduce_custom
from fastdeploy.model_executor.layers.moe.fused_moe_backend_base import MoEMethodBase


class HpuMoEMethod(MoEMethodBase):
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

        for idx, weights_tensor in enumerate([up_gate_proj_weights, down_proj_weights]):
            weights_list = []
            for i in range(layer.num_local_experts):
                weight_tensor = weights_tensor[i]
                weight = layer.create_parameter(
                    shape=weight_tensor.shape,
                    dtype=weight_tensor.dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                )
                weight.set_value(weight_tensor)
                weights_list.append(weight)
            weights_name = self.added_weight_attrs[idx]
            setattr(layer, weights_name, weights_list)

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

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Paddle hpu Fused MoE.
        """
        if layer.topk_method == "noaux_tc":
            raise NotImplementedError

        # norm_topk_prob = False if layer.topk_method == "noaux_tc" else True
        """
        weights = paddle.nn.functional.softmax(gate_out, axis=-1)
        if layer.moe_use_gate_correction_bias:
            scores = weights + layer.gate_correction_bias
            _, selected_experts = paddle.topk(scores, layer.top_k, axis=-1)
            routing_weights = paddle.index_sample(weights, selected_experts)
        else:
            routing_weights, selected_experts = paddle.topk(weights, layer.top_k, axis=-1)
        routing_weights /= paddle.sum(routing_weights, axis=-1, keepdim=True)

        common_inputs = (x, selected_experts, routing_weights.cast("bfloat16"))

        common_params = (
            False,  #permuted_weights
            "silu", #activation,
            0,
            layer.num_experts - 1,
        )

        weights = (
            layer.moe_ffn1_weight,
            layer.moe_ffn2_weight,
        )

        fused_moe_out, _ = mixture_of_experts(
            *common_inputs, *weights, *common_params, False
        )

        # if norm_topk_prob:
        #     routing_weights_norm = paddle.sum(routing_weights, axis=-1, keepdim=True).cast("bfloat16")
        #     fused_moe_out = fused_moe_out / routing_weights_norm
        """
        chunk_size = 64
        from fastdeploy.model_executor.ops.intel_hpu import fused_gate_moe

        # TODO: fuse matmul to gate_moe
        gate_out = paddle.matmul(x.cast("float32"), gate.weight)
        fused_moe_out = fused_gate_moe(
            x,
            gate_out,
            layer.gate_correction_bias,
            layer.up_gate_proj_weight,
            layer.down_proj_weight,
            layer.top_k,
            layer.moe_use_gate_correction_bias,
            norm_topk_prob=True,
            permuted_weights=False,
            activation="silu",
            experts_min=layer.expert_id_offset,
            experts_max=layer.expert_id_offset + layer.num_local_experts - 1,
            chunk_size=chunk_size,
        )
        if layer.reduce_results and layer.tp_size > 1:
            tensor_model_parallel_all_reduce_custom(fused_moe_out)

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

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Paddle hpu Fused MoE.
        """
        if layer.topk_method == "noaux_tc":
            raise NotImplementedError

        # norm_topk_prob = False if layer.topk_method == "noaux_tc" else True

        chunk_size = 64
        from fastdeploy.model_executor.ops.intel_hpu import fused_gate_moe_fp8

        # TODO: fuse matmul to gate_moe
        gate_out = paddle.matmul(x.cast("float32"), gate.weight)
        fused_moe_out = fused_gate_moe_fp8(
            x,
            gate_out,
            layer.gate_correction_bias,
            layer.up_gate_proj_weight,
            layer.down_proj_weight,
            None,  # intermediate_hidden_states_scales
            layer.up_gate_proj_weight_scale,
            layer.down_proj_weight_scale,
            layer.top_k,
            layer.moe_use_gate_correction_bias,
            norm_topk_prob=True,
            permuted_weights=False,
            activation="silu",
            experts_min=layer.expert_id_offset,
            experts_max=layer.expert_id_offset + layer.num_local_experts - 1,
            chunk_size=chunk_size,
        )

        if layer.reduce_results and layer.tp_size > 1:
            tensor_model_parallel_all_reduce_custom(fused_moe_out)

        return fused_moe_out
