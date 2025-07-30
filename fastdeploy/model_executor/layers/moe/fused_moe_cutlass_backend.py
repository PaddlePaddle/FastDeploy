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
from paddle.nn.quant import weight_quantize
from paddleformers.utils.log import logger

import fastdeploy
from fastdeploy.distributed.communication import tensor_model_parallel_all_reduce
from fastdeploy.platforms import current_platform

from ..utils import create_and_set_parameter, get_tensor
from .fused_moe_backend_base import MoEMethodBase

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import (
        moe_expert_dispatch,
        moe_expert_reduce,
        noaux_tc,
    )
elif current_platform.is_iluvatar():
    from fastdeploy.model_executor.ops.iluvatar import (
        moe_expert_dispatch,
        moe_expert_reduce,
    )


# used for deepseek_v3
def get_moe_scores(
    gating_output: paddle.Tensor,
    n_group,
    topk_group,
    top_k,
    routed_scaling_factor,
    e_score_correction_bias,
) -> paddle.Tensor:
    """
    compute moe scores using e_score_correction_bias.
    """
    scores = paddle.nn.functional.sigmoid(gating_output)
    scores_with_bias = scores + e_score_correction_bias.unsqueeze(0)
    scores = noaux_tc(
        scores,
        scores_with_bias,
        n_group,
        topk_group,
        top_k,
        routed_scaling_factor,
    )
    return scores


class CutlassMoEMethod(MoEMethodBase):
    """
    Use Cutlass Group Gemm to compute Fused MoE.
    This method is the oldest way to compute MoE in Paddle.
    """

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass create weight process.
        """
        # bf16
        up_gate_proj_weights, down_proj_weights = layer.extract_moe_ffn_weights(state_dict)
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

    def compute_ffn(
        self,
        layer: nn.Layer,
        permute_input: paddle.Tensor,
        token_nums_per_expert: paddle.Tensor,
        expert_idx_per_token: paddle.Tensor,
        used_in_ep_low_latency: bool = False,
    ):
        """
        Paddle Cutlass compute Fused MoE.
        """
        if current_platform.is_iluvatar():
            return fastdeploy.model_executor.ops.iluvatar.moe_expert_ffn(
                permute_input,
                token_nums_per_expert,
                layer.up_gate_proj_weight,
                layer.down_proj_weight,
                None,
                (layer.up_gate_proj_weight_scale if hasattr(layer, "up_gate_proj_weight_scale") else None),
                (layer.down_proj_weight_scale if hasattr(layer, "down_proj_weight_scale") else None),
                (layer.down_proj_in_scale if hasattr(layer, "down_proj_in_scale") else None),
                expert_idx_per_token,
                self.moe_quant_type,
                used_in_ep_low_latency,
            )
        return fastdeploy.model_executor.ops.gpu.moe_expert_ffn(
            permute_input,
            token_nums_per_expert,
            layer.up_gate_proj_weight,
            layer.down_proj_weight,
            None,
            (layer.up_gate_proj_weight_scale if hasattr(layer, "up_gate_proj_weight_scale") else None),
            (layer.down_proj_weight_scale if hasattr(layer, "down_proj_weight_scale") else None),
            (layer.down_proj_in_scale if hasattr(layer, "down_proj_in_scale") else None),
            expert_idx_per_token,
            self.moe_quant_type,
            used_in_ep_low_latency,
        )

    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply the EP prefill method.
        """
        # 1. Select topk experts and weights
        topk_idx, topk_weights = self.ep_prefill_runner.moe_select(layer, gate_out)
        # 2. EP Dispatch
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_num_tokens_per_expert_list,
            handle,
            _,
        ) = self.ep_prefill_runner.dispatch(x, topk_idx, topk_weights)
        token_all_num = sum(recv_num_tokens_per_expert_list)

        # 3. Compute ffn
        if token_all_num > 0:
            logger.info(f"token_all_num {token_all_num}")
            (
                permute_input,
                permute_indices_per_token,
                recv_num_tokens_per_expert_list_cumsum,
                dst_weights,
                dst_indices,
                cumsum_idx_gpu,
                expert_idx_per_token,
            ) = fastdeploy.model_executor.ops.gpu.ep_moe_expert_dispatch(
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                (self.up_gate_proj_in_scale if hasattr(self, "up_gate_proj_in_scale") else None),
                recv_num_tokens_per_expert_list,
                token_all_num,
                self.moe_quant_type,
            )
            if self.moe_quant_type != "w4a8":
                # only w4a8 need expert_idx_per_token
                # Other need not this tensor, so we make it None.
                expert_idx_per_token = None
            else:
                expert_idx_per_token = expert_idx_per_token.cast("int64")

            ffn_out = self.compute_ffn(
                layer,
                permute_input,
                recv_num_tokens_per_expert_list_cumsum,
                expert_idx_per_token,
            )

            # prmt back per rank
            tmp_ffn_out = fastdeploy.model_executor.ops.gpu.ep_moe_expert_combine(
                ffn_out,
                dst_weights,
                permute_indices_per_token,
                dst_indices,
                None,  # down_proj_bias,
                False,  # norm_topk_prob
                1.0,
            )[0]
        else:
            tmp_ffn_out = recv_x

        # 4. EP combine
        return self.ep_prefill_runner.combine(tmp_ffn_out, handle, recv_topk_weights)

    def apply_ep_decode(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply the EP decoder method.
        """
        # 1. Select topk experts and weights
        topk_idx, topk_weights = self.ep_decoder_runner.moe_select(layer, gate_out)
        expertwise_scale = getattr(layer, "up_gate_proj_in_scale_all_experts")
        # 2. EP Dispatch
        permute_input, token_nums_per_expert, handle = self.ep_decoder_runner.dispatch(
            x, topk_idx, topk_weights, expertwise_scale=expertwise_scale
        )
        # 3. Compute ffn
        if self.moe_quant_type == "w4a8":
            num_local_experts, max_num, _ = permute_input.shape
            expert_idx_per_token = paddle.arange(num_local_experts)[:, None].tile([1, max_num])
        elif self.moe_quant_type in ["weight_only_int8", "weight_only_int4"]:
            expert_idx_per_token = None
        else:
            raise NotImplementedError

        ffn_out = self.compute_ffn(
            layer,
            permute_input,
            token_nums_per_expert.cast("int64"),
            expert_idx_per_token,
            True,
        )

        # 4. EP combine
        return self.ep_decoder_runner.combine(ffn_out, topk_idx, topk_weights, handle)

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Paddle Cutlass compute Fused MoE.
        """
        if layer.topk_method == "noaux_tc":
            gate_out = get_moe_scores(
                gate_out,
                layer.n_group,
                layer.topk_group,
                layer.top_k,
                layer.routed_scaling_factor,
                layer.gate_correction_bias,
            )

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
                None,  # Use layer.gate_correction_bias in get_moe_scores.
                (
                    layer.up_gate_proj_in_scale if hasattr(layer, "up_gate_proj_in_scale") else None
                ),  # if set, permute_input will be int8_t
                layer.top_k,
                False,
                topk_only_mode=True,
            )
        else:
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
                (
                    layer.up_gate_proj_in_scale if hasattr(layer, "up_gate_proj_in_scale") else None
                ),  # if set, permute_input will be int8_t
                layer.top_k,
                False,
                topk_only_mode=False,
            )

        if self.moe_quant_type != "w4a8":
            # only w4a8 need expert_idx_per_token
            # Other need not this tensor, so we make it None.
            expert_idx_per_token = None
        else:
            expert_idx_per_token = expert_idx_per_token.cast("int64")

        ffn_out = self.compute_ffn(layer, permute_input, token_nums_per_expert, expert_idx_per_token)

        # reduce 中会做 topk 个 weight 的 norm 和 routed_scaling_factor
        fused_moe_out = moe_expert_reduce(
            ffn_out,
            topk_weights,
            permute_indices_per_token,
            topk_idx,
            None,
            norm_topk_prob=False if layer.topk_method == "noaux_tc" else True,
            routed_scaling_factor=1.0,
        )

        if layer.reduce_results and layer.tp_size > 1:
            tensor_model_parallel_all_reduce(fused_moe_out)

        return fused_moe_out


class CutlassW4A8MoEMethod(CutlassMoEMethod):
    """
    w4a8 MoE Method
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.quant_config = quant_config
        self.moe_quant_type = "w4a8"
        self.pack_num = 2

    def process_prequanted_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass process prequanted weights.
        """
        up_gate_proj_expert_weight_key = layer.weight_key_map.get("up_gate_proj_expert_weight_key", None)
        down_proj_expert_weight_key = layer.weight_key_map.get("down_proj_expert_weight_key", None)
        up_gate_proj_expert_weight_scale_key = layer.weight_key_map.get("up_gate_proj_expert_weight_scale_key", None)
        down_proj_expert_weight_scale_key = layer.weight_key_map.get("down_proj_expert_weight_scale_key", None)
        up_gate_proj_expert_in_scale_key = layer.weight_key_map.get("up_gate_proj_expert_in_scale_key", None)
        down_proj_expert_in_scale_key = layer.weight_key_map.get("down_proj_expert_in_scale_key", None)

        up_gate_proj_weights, down_proj_weights, logical_expert_ids, ep_rank_to_expert_id_list = (
            layer.load_experts_weight(
                state_dict,
                up_gate_proj_expert_weight_key,
                down_proj_expert_weight_key,
            )
        )

        up_gate_proj_weight_scale = []
        down_proj_weight_scale = []
        up_gate_proj_in_scale_all_experts = []
        up_gate_proj_in_scale = []
        down_proj_in_scale = []

        if layer.ep_size > 1:
            for expert_idx in ep_rank_to_expert_id_list:
                scale_tensor = get_tensor(state_dict[up_gate_proj_expert_in_scale_key.format(expert_idx)])
                up_gate_proj_in_scale_all_experts.append(scale_tensor)

        for expert_idx in logical_expert_ids:
            up_gate_proj_weight_scale.append(
                get_tensor(state_dict.pop(up_gate_proj_expert_weight_scale_key.format(expert_idx)))
            )
            down_proj_weight_scale.append(
                get_tensor(state_dict.pop(down_proj_expert_weight_scale_key.format(expert_idx)))
            )
            up_gate_proj_in_scale.append(
                get_tensor(state_dict.pop(up_gate_proj_expert_in_scale_key.format(expert_idx)))
            )
            down_proj_in_scale.append(get_tensor(state_dict.pop(down_proj_expert_in_scale_key.format(expert_idx))))

        up_gate_proj_weight = paddle.stack(up_gate_proj_weights, axis=0)
        down_proj_weight = paddle.stack(down_proj_weights, axis=0)
        up_gate_proj_weight_scale = paddle.stack(up_gate_proj_weight_scale, axis=0).cast(paddle.get_default_dtype())
        down_proj_weight_scale = paddle.stack(down_proj_weight_scale, axis=0).cast(paddle.get_default_dtype())
        up_gate_proj_in_scale_all_experts = paddle.stack(up_gate_proj_in_scale_all_experts, axis=0)
        up_gate_proj_in_scale = paddle.stack(up_gate_proj_in_scale, axis=0)
        down_proj_in_scale = paddle.stack(down_proj_in_scale, axis=0)

        name_tensor_map = {
            "up_gate_proj_weight": up_gate_proj_weight,
            "down_proj_weight": down_proj_weight,
            "up_gate_proj_weight_scale": up_gate_proj_weight_scale,
            "down_proj_weight_scale": down_proj_weight_scale,
            "up_gate_proj_in_scale_all_experts": up_gate_proj_in_scale_all_experts,
            "up_gate_proj_in_scale": up_gate_proj_in_scale,
            "down_proj_in_scale": down_proj_in_scale,
        }
        for name, tensor in name_tensor_map.items():
            create_and_set_parameter(layer, name, tensor)

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass create weight process.
        """
        up_gate_proj_weights, down_proj_weights = layer.extract_moe_ffn_weights(state_dict)
        self.check(layer, up_gate_proj_weights, down_proj_weights)
        for idx, weight_tensor in enumerate([up_gate_proj_weights, down_proj_weights]):
            weight_name = self.added_weight_attrs[idx]
            weight_list = []
            for i in range(layer.num_local_experts):
                quant_weight, scale = weight_quantize(weight_tensor[i], algo=self.moe_quant_type, arch=80)
                weight_list.append(quant_weight)
            quanted_weight = paddle.stack(weight_list, axis=0)
            create_and_set_parameter(layer, weight_name, quanted_weight)

        self.create_w4a8_scale_weights(layer, layer.weight_key_map, state_dict)

    def create_w4a8_scale_weights(self, layer: nn.Layer, weight_key_map: dict, state_dict: dict):
        """
        Get w4a8 weights from state dict and process them.
        Args:
            layer (nn.Layer): The layer to add parameters to.
            weight_key_map (dict): The weight key map.
            state_dict (dict): The state dict.
        """

        def _extract_scale_tensor(state_dict, key_template, expert_idx):
            return get_tensor(state_dict.pop(key_template.format(expert_idx)))

        def _process_in_scale(name: str, in_scales: list[paddle.Tensor]):
            processed_in_scale = 1 / paddle.concat(in_scales)
            create_and_set_parameter(layer, name, processed_in_scale)
            return processed_in_scale

        def _process_weight_scale(
            name: str,
            weight_scales: list[paddle.Tensor],
            processed_in_scale: paddle.Tensor,
        ):
            processed_weight_scale = (
                paddle.stack(weight_scales, axis=0) / (127 * 112) / processed_in_scale[:, None]
            ).cast(paddle.get_default_dtype())
            create_and_set_parameter(layer, name, processed_weight_scale)

        # 1. Init scale containers and maps
        up_gate_proj_weight_scales = []
        down_proj_weight_scales = []
        up_gate_proj_in_scales_all_experts = []
        up_gate_proj_in_scales = []
        down_proj_in_scales = []

        scale_weight_map = {
            "up_gate_proj_weight_scale": up_gate_proj_weight_scales,
            "down_proj_weight_scale": down_proj_weight_scales,
            "up_gate_proj_in_scale": up_gate_proj_in_scales,
            "down_proj_in_scale": down_proj_in_scales,
        }
        scale_key_map = {
            "up_gate_proj_weight_scale": weight_key_map.get("up_gate_proj_expert_weight_scale_key", None),
            "down_proj_weight_scale": weight_key_map.get("down_proj_expert_weight_scale_key", None),
            "up_gate_proj_in_scale": weight_key_map.get("up_gate_proj_expert_in_scale_key", None),
            "down_proj_in_scale": weight_key_map.get("down_proj_expert_in_scale_key", None),
        }
        for name, value in scale_key_map.items():
            if value is None:
                raise ValueError(f"scale {name} should not be none in w4a8 mode.")

        # 2. Extract scale tensor from state dict
        if layer.ep_size > 1:
            for expert_idx in range(layer.num_experts):
                scale_tensor = get_tensor(state_dict[scale_key_map["up_gate_proj_in_scale"].format(expert_idx)])
                up_gate_proj_in_scales_all_experts.append(1 / scale_tensor)
            create_and_set_parameter(
                layer, "up_gate_proj_in_scale_all_experts", paddle.concat(up_gate_proj_in_scales_all_experts)
            )

        for local_expert_idx in range(layer.num_local_experts):
            expert_idx = local_expert_idx + layer.expert_id_offset
            for name, scale_key_template in scale_key_map.items():
                scale_tensor = _extract_scale_tensor(state_dict, scale_key_template, expert_idx)
                scale_weight_map[name].append(scale_tensor)

        # 3. Process scale tensor and set to layer
        in_scales = []
        for in_scale_name in ["up_gate_proj_in_scale", "down_proj_in_scale"]:
            in_scales.append(_process_in_scale(in_scale_name, scale_weight_map[in_scale_name]))

        for i, weight_scale_name in enumerate(["up_gate_proj_weight_scale", "down_proj_weight_scale"]):
            _process_weight_scale(
                weight_scale_name,
                scale_weight_map[weight_scale_name],
                in_scales[i],
            )


class CutlassWeightOnlyMoEMethod(CutlassMoEMethod):
    """
    weight only for moe
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.quant_config = quant_config
        self.moe_quant_type = self.quant_config.algo
        self.pack_num = 1

    def process_prequanted_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass process prequanted weights.
        """
        up_gate_proj_expert_weight_key = layer.weight_key_map.get("up_gate_proj_expert_weight_key", None)
        down_proj_expert_weight_key = layer.weight_key_map.get("down_proj_expert_weight_key", None)
        up_gate_proj_expert_weight_scale_key = layer.weight_key_map.get("up_gate_proj_expert_weight_scale_key", None)
        down_proj_expert_weight_scale_key = layer.weight_key_map.get("down_proj_expert_weight_scale_key", None)

        up_gate_proj_weights, down_proj_weights, logical_expert_ids, _ = layer.load_experts_weight(
            state_dict,
            up_gate_proj_expert_weight_key,
            down_proj_expert_weight_key,
        )
        # self.check(layer, up_gate_proj_weights, down_proj_weights)
        up_gate_proj_weight_scale = []
        down_proj_weight_scale = []
        for expert_idx in logical_expert_ids:
            up_gate_proj_weight_scale.append(
                get_tensor(state_dict.pop(up_gate_proj_expert_weight_scale_key.format(expert_idx)))
            )
            down_proj_weight_scale.append(
                get_tensor(state_dict.pop(down_proj_expert_weight_scale_key.format(expert_idx)))
            )

        up_gate_proj_weight = paddle.stack(up_gate_proj_weights, axis=0)
        down_proj_weight = paddle.stack(down_proj_weights, axis=0)
        up_gate_proj_weight_scale = paddle.stack(up_gate_proj_weight_scale, axis=0)
        down_proj_weight_scale = paddle.stack(down_proj_weight_scale, axis=0)

        name_tensor_map = {
            "up_gate_proj_weight": up_gate_proj_weight,
            "down_proj_weight": down_proj_weight,
            "up_gate_proj_weight_scale": up_gate_proj_weight_scale,
            "down_proj_weight_scale": down_proj_weight_scale,
        }
        for name, tensor in name_tensor_map.items():
            create_and_set_parameter(layer, name, tensor)

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass create weight process.
        """
        up_gate_proj_weights, down_proj_weights = layer.extract_moe_ffn_weights(state_dict)
        self.check(layer, up_gate_proj_weights, down_proj_weights)

        for idx, weight_tensor in enumerate([up_gate_proj_weights, down_proj_weights]):
            weight_name = self.added_weight_attrs[idx]
            scale_name = self.added_scale_attrs[idx]

            weight_list = []
            weight_scale_list = []
            for i in range(layer.num_local_experts):
                quant_weight, scale = weight_quantize(weight_tensor[i], algo=self.moe_quant_type)
                weight_list.append(quant_weight)
                weight_scale_list.append(scale)
            quanted_weight = paddle.stack(weight_list, axis=0)
            create_and_set_parameter(layer, weight_name, quanted_weight)

            quanted_weight_scale = paddle.stack(weight_scale_list, axis=0)
            create_and_set_parameter(layer, scale_name, quanted_weight_scale)
