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

import fastdeploy

from ..quantization.quant_base import QuantMethodBase
from ..utils import create_and_set_parameter, get_tensor


class Wint2MoeMethod(QuantMethodBase):
    """
    Use  compute Fused MoE.
    """

    def __init__(self, quant_config):
        super().__init__()
        self.moe_quant_type = quant_config.moe_quant_type

    def process_loaded_weights(self, layer, weights) -> None:
        """
        process_loaded_weights
        """
        pass

    def check(self, layer: nn.Layer, ffn1_weights, ffn2_weights):
        """
        check layer is valid for this method
        """
        assert len(
            ffn1_weights
        ) == layer.num_local_experts, "ffn1_weights length should be equal to num_local_experts."
        assert len(
            ffn2_weights
        ) == layer.num_local_experts, "ffn2_weights length should be equal to num_local_experts."

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass create weight process.
        """
        pass


class TritonWint2FusedMoeMethod(Wint2MoeMethod):
    """
    Use Triton Group Gemm to compute Fused MoE.
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.moe_quant_type = quant_config.moe_quant_type

    def process_loaded_weights(self, layer, weights) -> None:
        """
        process_loaded_weights
        """
        pass

    def process_prequanted_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass process prequanted weights.
        """
        ffn1_expert_weight_key = layer.weight_key_map.get(
            "ffn1_expert_weight_key", None)
        ffn2_expert_weight_key = layer.weight_key_map.get(
            "ffn2_expert_weight_key", None)
        ffn1_expert_weight_scale_key = layer.weight_key_map.get(
            "ffn1_expert_weight_scale_key", None)
        ffn2_expert_weight_scale_key = layer.weight_key_map.get(
            "ffn2_expert_weight_scale_key", None)
        ffn1_expert_super_scales_key = layer.weight_key_map.get(
            "ffn1_expert_super_scales_key", None)
        ffn2_expert_super_scales_key = layer.weight_key_map.get(
            "ffn2_expert_super_scales_key", None)
        ffn1_expert_code_scale_key = layer.weight_key_map.get(
            "ffn1_expert_code_scale_key", None)
        ffn2_expert_code_scale_key = layer.weight_key_map.get(
            "ffn2_expert_code_scale_key", None)
        ffn1_expert_code_zp_key = layer.weight_key_map.get(
            "ffn1_expert_code_zp_key", None)
        ffn2_expert_code_zp_key = layer.weight_key_map.get(
            "ffn2_expert_code_zp_key", None)

        ffn1_weights, ffn2_weights = layer.load_experts_weight(
            state_dict, ffn1_expert_weight_key, ffn2_expert_weight_key)
        # self.check(layer, ffn1_weights, ffn2_weights)

        ffn1_weight_scale = []
        ffn2_weight_scale = []
        ffn1_super_scales = []
        ffn2_super_scales = []
        ffn1_code_scale = []
        ffn2_code_scale = []
        ffn1_code_zp = []
        ffn2_code_zp = []
        for i in range(layer.num_experts):
            expert_idx = layer.expert_id_offset + i
            ffn1_weight_scale.append(
                get_tensor(
                    state_dict.pop(
                        ffn1_expert_weight_scale_key.format(expert_idx))))
            ffn2_weight_scale.append(
                get_tensor(
                    state_dict.pop(
                        ffn2_expert_weight_scale_key.format(expert_idx))))
            ffn1_super_scales.append(
                get_tensor(
                    state_dict.pop(
                        ffn1_expert_super_scales_key.format(expert_idx))))
            ffn2_super_scales.append(
                get_tensor(
                    state_dict.pop(
                        ffn2_expert_super_scales_key.format(expert_idx))))
            ffn1_code_scale.append(
                get_tensor(
                    state_dict.pop(
                        ffn1_expert_code_scale_key.format(expert_idx))))
            ffn2_code_scale.append(
                get_tensor(
                    state_dict.pop(
                        ffn2_expert_code_scale_key.format(expert_idx))))
            ffn1_code_zp.append(
                get_tensor(
                    state_dict.pop(
                        ffn1_expert_code_zp_key.format(expert_idx))))
            ffn2_code_zp.append(
                get_tensor(
                    state_dict.pop(
                        ffn2_expert_code_zp_key.format(expert_idx))))

        ffn1_weight = paddle.stack(ffn1_weights, axis=0)
        ffn2_weight = paddle.stack(ffn2_weights, axis=0)
        ffn1_weight_scale = paddle.stack(ffn1_weight_scale, axis=0)
        ffn2_weight_scale = paddle.stack(ffn2_weight_scale, axis=0)
        ffn1_super_scales = paddle.stack(ffn1_super_scales, axis=0)
        ffn2_super_scales = paddle.stack(ffn2_super_scales, axis=0)
        ffn1_code_scale = paddle.stack(ffn1_code_scale, axis=0)
        ffn2_code_scale = paddle.stack(ffn2_code_scale, axis=0)
        ffn1_code_zp = paddle.stack(ffn1_code_zp, axis=0)
        ffn2_code_zp = paddle.stack(ffn2_code_zp, axis=0)

        name_tensor_map = {
            "moe_ffn1_weight": ffn1_weight,
            "moe_ffn2_weight": ffn2_weight,
            "moe_ffn1_weight_scale": ffn1_weight_scale,
            "moe_ffn2_weight_scale": ffn2_weight_scale,
            "moe_ffn1_super_scales": ffn1_super_scales,
            "moe_ffn2_super_scales": ffn2_super_scales,
            "moe_ffn1_code_scale": ffn1_code_scale,
            "moe_ffn2_code_scale": ffn2_code_scale,
            "moe_ffn1_code_zp": ffn1_code_zp,
            "moe_ffn2_code_zp": ffn2_code_zp
        }
        for name, tensor in name_tensor_map.items():
            create_and_set_parameter(layer, name, tensor)

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass create weight process.
        """
        pass

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Use Wint2 Triton Fusedmoe compute Fused MoE.
        """

        from fastdeploy.model_executor.ops.gpu import moe_expert_dispatch
        (
            permute_input,
            token_nums_per_expert,
            permute_indices_per_token,
            topk_weights,
            topk_idx,
            expert_idx_per_token,
        ) = moe_expert_dispatch(
            x,
            gate_out,
            layer.gate_correction_bias,
            (layer.moe_ffn1_in_scale if hasattr(layer, "moe_ffn1_in_scale")
             else None),  # if set, permute_input will be int8_t
            layer.top_k,
            False,
            topk_only_mode=False,
        )

        ffn_out = fastdeploy.model_executor.ops.gpu.moe_expert_ffn_wint2(
            permute_input,
            token_nums_per_expert,
            layer.moe_ffn1_weight,
            layer.moe_ffn2_weight,
            None,
            layer.moe_ffn1_super_scales,
            layer.moe_ffn2_super_scales,
            layer.moe_ffn1_weight_scale,
            layer.moe_ffn1_code_scale,
            layer.moe_ffn1_code_zp,
            layer.moe_ffn2_weight_scale,
            layer.moe_ffn2_code_scale,
            layer.moe_ffn2_code_zp,
            False,
        )

        from fastdeploy.model_executor.ops.gpu import moe_expert_reduce

        fused_moe_out = moe_expert_reduce(
            ffn_out,
            topk_weights,
            permute_indices_per_token,
            topk_idx,
            None,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )

        return fused_moe_out
