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
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.utils import set_weight_attrs


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

        # for measurement mode
        up_gate_proj_expert_weight_key = layer.weight_key_map.get("up_gate_proj_expert_weight_key", None)
        down_proj_expert_weight_key = layer.weight_key_map.get("down_proj_expert_weight_key", None)
        self.up_gate_proj_act_scale_key = up_gate_proj_expert_weight_key.replace("{}.", "").replace(
            "weight", "activation_scale"
        )
        self.down_proj_expert_act_scale_key = down_proj_expert_weight_key.replace("weight", "activation_scale")

    def init_ep(self, layer: nn.Layer) -> None:
        """
        Initialize EP (Expert Parallel) related modules.
        """
        return

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
    ) -> paddle.Tensor:
        """
        Apply the TP prefill method.
        """
        raise NotImplementedError

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

    def apply(
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
        measurement_mode = getattr(layer, "measurement_mode", False)
        if measurement_mode:
            from fastdeploy.model_executor.ops.intel_hpu import fused_gate_moe_ref

            fused_moe_out = fused_gate_moe_ref(
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
                measurement_mode=True,
                up_gate_act_scale_key=self.up_gate_proj_act_scale_key,
                down_act_scale_key=self.down_proj_expert_act_scale_key,
            )
        else:
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

    def process_prequanted_weights(self, layer: nn.Layer, state_dict, is_rearrange: bool = False):
        """
        Paddle HPU process prequanted weights.
        """

        def _extract_scale_tensor(key_template, logical_expert_ids):
            result = []
            for i in logical_expert_ids:
                result.append(get_tensor(state_dict.pop(key_template.format(i))))
            return result  # bf16 tensor list

        def _extract_descale_tensor(key_template, logical_expert_ids):
            if key_template.format(0) in state_dict:
                # Extract scale tensors for all logical_expert_ids
                scale_tensors = []
                for i in logical_expert_ids:
                    scale_tensor = get_tensor(state_dict.pop(key_template.format(i)))
                    scale_tensors.append(scale_tensor)
                # Stack all scale tensors into one tensor
                stacked = paddle.stack(scale_tensors)
                reciprocal = 1.0 / stacked
                # Take min over all logical_expert_ids (axis=0)
                min_tensor = paddle.min(reciprocal, axis=0)
                return min_tensor.cast(paddle.get_default_dtype())
            else:
                key = key_template.replace("{}.", "")
                scale_tensor = get_tensor(state_dict.pop(key))
                reciprocal = 1.0 / scale_tensor
                return reciprocal.cast(paddle.get_default_dtype())

        up_gate_proj_weight, down_proj_weight, logical_expert_ids, _ = layer.extract_moe_ffn_weights(state_dict)
        up_gate_proj_weights = [t.view(paddle.float8_e4m3fn) for t in up_gate_proj_weight]
        down_proj_weights = [t.view(paddle.float8_e4m3fn) for t in down_proj_weight]

        weight_key_map = layer.weight_key_map

        up_gate_proj_expert_weight_scale_key = weight_key_map.get("up_gate_proj_expert_weight_scale_key", None)
        down_proj_expert_weight_scale_key = weight_key_map.get("down_proj_expert_weight_scale_key", None)
        up_gate_proj_expert_in_scale_key = weight_key_map.get("up_gate_proj_expert_in_scale_key", None)
        down_proj_expert_in_scale_key = weight_key_map.get("down_proj_expert_in_scale_key", None)

        up_gate_proj_weight_scale = _extract_scale_tensor(up_gate_proj_expert_weight_scale_key, logical_expert_ids)
        down_proj_weight_scale = _extract_scale_tensor(down_proj_expert_weight_scale_key, logical_expert_ids)
        up_gate_proj_in_scale = _extract_descale_tensor(up_gate_proj_expert_in_scale_key, logical_expert_ids)
        down_proj_in_scale = _extract_scale_tensor(down_proj_expert_in_scale_key, logical_expert_ids)

        up_gate_proj_weight = paddle.stack(up_gate_proj_weights, axis=0)
        down_proj_weight = paddle.stack(down_proj_weights, axis=0)
        up_gate_proj_weight_scale = paddle.stack(up_gate_proj_weight_scale, axis=0)
        down_proj_weight_scale = paddle.stack(down_proj_weight_scale, axis=0)
        down_proj_in_scale = paddle.stack(down_proj_in_scale, axis=0)

        name_tensor_map = {
            "up_gate_proj_weight": up_gate_proj_weight,
            "down_proj_weight": down_proj_weight,
            "up_gate_proj_weight_scale": up_gate_proj_weight_scale,
            "down_proj_weight_scale": down_proj_weight_scale,
            "up_gate_proj_in_scale": up_gate_proj_in_scale,
            "down_proj_in_scale": down_proj_in_scale,
        }
        for name, tensor in name_tensor_map.items():
            getattr(layer, name).set_value(tensor)

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        Paddle HPU create weight process.
        """
        self.weight_dtype = "float8_e4m3fn"
        self.up_gate_proj_weight_shape = [
            layer.num_local_experts,
            layer.hidden_size,
            layer.moe_intermediate_size * 2,
        ]
        self.down_proj_weight_shape = [
            layer.num_local_experts,
            layer.moe_intermediate_size,
            layer.hidden_size,
        ]
        setattr(
            layer,
            self.added_weight_attrs[0],
            layer.create_parameter(
                shape=self.up_gate_proj_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            self.added_weight_attrs[1],
            layer.create_parameter(
                shape=self.down_proj_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )

        self.default_dtype = layer._helper.get_default_dtype()
        # in_scales
        setattr(
            layer,
            "up_gate_proj_in_scale",
            layer.create_parameter(
                shape=[1],
                dtype=self.default_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            "down_proj_in_scale",
            layer.create_parameter(
                shape=[layer.num_local_experts, 1],
                dtype=self.default_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )

        # weight_scales
        setattr(
            layer,
            "up_gate_proj_weight_scale",
            layer.create_parameter(
                shape=[layer.num_local_experts, layer.moe_intermediate_size * 2],
                dtype=self.default_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            "down_proj_weight_scale",
            layer.create_parameter(
                shape=[layer.num_local_experts, layer.hidden_size],
                dtype=self.default_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        extra_weight_attrs = {
            **(extra_weight_attrs or {}),
            "SHARD_ID_TO_SHARDED_DIM": {"gate": 1, "down": 0, "up": 1},
        }
        set_weight_attrs(layer.up_gate_proj_weight, extra_weight_attrs)
        set_weight_attrs(layer.down_proj_weight, extra_weight_attrs)
        extra_scale_attrs = {
            **(extra_weight_attrs or {}),
            "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "up": 0, "down": None},
        }
        set_weight_attrs(layer.down_proj_in_scale, extra_scale_attrs)
        set_weight_attrs(layer.up_gate_proj_weight_scale, extra_scale_attrs)
        set_weight_attrs(layer.down_proj_weight_scale, extra_scale_attrs)

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

    def process_weights_after_loading(self, layer):
        return

    def init_ep(self, layer: nn.Layer) -> None:
        """
        Initialize EP (Expert Parallel) related modules.
        """
        return

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
    ) -> paddle.Tensor:
        """
        Apply the TP decoder method.
        """
        raise NotImplementedError

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

    def apply(
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
