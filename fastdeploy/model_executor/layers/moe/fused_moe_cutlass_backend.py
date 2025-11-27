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
from fastdeploy.platforms import current_platform

from ..utils import get_tensor
from .fused_moe_backend_base import UnquantizedFusedMoEMethod

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import moe_expert_dispatch, moe_expert_reduce

    try:
        from fastdeploy.model_executor.ops.gpu import (
            w4afp8_gemm_scale_permute,
            w4afp8_gemm_weight_convert,
        )
    except:
        logger.warning("import w4afp8_gemm_scale_permute Failed!")
elif current_platform.is_iluvatar():
    from fastdeploy.model_executor.ops.iluvatar import (
        moe_expert_dispatch,
        moe_expert_reduce,
    )

from fastdeploy.model_executor.layers.moe.moe import get_moe_scores
from fastdeploy.model_executor.utils import (
    TensorTracker,
    free_tensor,
    process_weight_transpose,
    set_weight_attrs,
    weight_fully_copied,
)


class CutlassMoEMethod(UnquantizedFusedMoEMethod):
    """
    Use Cutlass Group Gemm to compute Fused MoE.
    This method is the oldest way to compute MoE in Paddle.
    """

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        up_gate_proj_weights, down_proj_weights, logical_expert_ids, ep_rank_to_expert_id_list = (
            layer.extract_moe_ffn_weights(state_dict)
        )
        stacked_up_gate_proj_weights = paddle.stack(up_gate_proj_weights, axis=0)
        stacked_down_proj_weights = paddle.stack(down_proj_weights, axis=0)

        layer.up_gate_proj_weight.set_value(stacked_up_gate_proj_weights)
        layer.down_proj_weight.set_value(stacked_down_proj_weights)

        if layer.with_bias:
            up_gate_proj_bias, down_proj_bias = layer.extract_moe_ffn_bias(state_dict)
            stacked_up_gate_proj_bias = paddle.stack(up_gate_proj_bias, axis=0)
            stacked_down_proj_bias = paddle.stack(down_proj_bias, axis=0)

            layer.up_gate_proj_bias.set_value(stacked_up_gate_proj_bias)
            layer.down_proj_bias.set_value(stacked_down_proj_bias)

    def compute_ffn(
        self,
        layer: nn.Layer,
        permute_input: paddle.Tensor,
        token_nums_per_expert: paddle.Tensor,
        expert_idx_per_token: paddle.Tensor,
        used_in_ep_low_latency: bool = False,
        estimate_total_token_nums: int = -1,
        dequant_scale: paddle.Tensor = None,
    ):
        """
        Paddle Cutlass compute Fused MoE.
        """
        if current_platform.is_iluvatar():
            ffn_out_without_down_proj_bias = fastdeploy.model_executor.ops.iluvatar.moe_expert_ffn(
                permute_input,
                token_nums_per_expert,
                getattr(layer, self.added_weight_attrs[0]),
                getattr(layer, self.added_weight_attrs[1]),
                # None,
                (layer.up_gate_proj_bias if hasattr(layer, "up_gate_proj_bias") else None),
                (layer.up_gate_proj_weight_scale if hasattr(layer, "up_gate_proj_weight_scale") else None),
                (layer.down_proj_weight_scale if hasattr(layer, "down_proj_weight_scale") else None),
                (layer.down_proj_in_scale if hasattr(layer, "down_proj_in_scale") else None),
                expert_idx_per_token,
                self.moe_quant_type,
                used_in_ep_low_latency,
            )
        else:
            ffn_out_without_down_proj_bias = fastdeploy.model_executor.ops.gpu.moe_expert_ffn(
                permute_input,
                token_nums_per_expert,
                getattr(layer, self.added_weight_attrs[0]),
                getattr(layer, self.added_weight_attrs[1]),
                dequant_scale,
                (layer.up_gate_proj_bias if hasattr(layer, "up_gate_proj_bias") else None),
                (layer.up_gate_proj_weight_scale if hasattr(layer, "up_gate_proj_weight_scale") else None),
                (layer.down_proj_weight_scale if hasattr(layer, "down_proj_weight_scale") else None),
                (layer.down_proj_in_scale if hasattr(layer, "down_proj_in_scale") else None),
                expert_idx_per_token,
                self.moe_quant_type,
                used_in_ep_low_latency,
                estimate_total_token_nums,
                getattr(layer.moe_quant_config, "hadamard_block_size", 128),
                layer.activation,
            )

        if layer.with_bias:
            down_proj_bias_expand = paddle.index_select(layer.down_proj_bias, expert_idx_per_token, axis=0)
            ffn_out_without_down_proj_bias = paddle.add(ffn_out_without_down_proj_bias, down_proj_bias_expand)
        return ffn_out_without_down_proj_bias

    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Apply the EP prefill method.
        """
        gate_out = gate(x.cast("float32"))
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
            logger.debug(f"token_all_num {token_all_num}")
            (
                permute_input,
                permute_indices_per_token,
                recv_num_tokens_per_expert_list_cumsum,
                dst_weights,
                dst_indices,
                cumsum_idx_gpu,
                expert_idx_per_token,
                dequant_scale,
            ) = fastdeploy.model_executor.ops.gpu.ep_moe_expert_dispatch(
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                (layer.up_gate_proj_in_scale if hasattr(layer, "up_gate_proj_in_scale") else None),
                recv_num_tokens_per_expert_list,
                token_all_num,
                self.moe_quant_type,
            )
            if not layer.with_bias and self.moe_quant_type != "w4a8" and self.moe_quant_type != "w4afp8":
                # only w4a8 and w4afp8 need expert_idx_per_token
                # Other need not this tensor, so we make it None.
                expert_idx_per_token = None
            else:
                expert_idx_per_token = expert_idx_per_token.cast("int64")

            if hasattr(layer, "up_gate_proj_in_scale"):
                dequant_scale = None

            ffn_out = self.compute_ffn(
                layer,
                permute_input,
                recv_num_tokens_per_expert_list_cumsum,
                expert_idx_per_token,
                False,
                -1,
                dequant_scale,
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
        tmp_ffn_out, event = self.ep_prefill_runner.combine(tmp_ffn_out, handle, recv_topk_weights)
        if self.ep_prefill_runner.ep_engine.async_finish:
            event.current_stream_wait()
        return tmp_ffn_out

    def apply_ep_decode(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Apply the EP decoder method.
        """
        gate_out = gate(x.cast("float32"))
        estimate_total_token_nums = gate_out.shape[0] * layer.top_k
        # 1. Select topk experts and weights
        topk_idx, topk_weights = self.ep_decoder_runner.moe_select(layer, gate_out)
        expertwise_scale = None
        if hasattr(layer, "up_gate_proj_in_scale_all_experts"):  # only use in w4a8
            expertwise_scale = getattr(layer, "up_gate_proj_in_scale_all_experts", None)
        use_fp8 = self.moe_quant_type == "w4afp8"
        quant_group_size = -1 if self.moe_quant_type == "w4afp8" else 128
        # 2. EP Dispatch
        permute_input, token_nums_per_expert, handle = self.ep_decoder_runner.dispatch(
            x,
            topk_idx,
            topk_weights,
            expertwise_scale=expertwise_scale,
            use_fp8=use_fp8,
            quant_group_size=quant_group_size,
        )
        dequant_scale = None
        if self.moe_quant_type == "w4afp8" and expertwise_scale is None:
            (permute_input, dequant_scale) = permute_input
        # 3. Compute ffn
        if self.moe_quant_type == "w4a8" or self.moe_quant_type == "w4afp8":
            num_local_experts, max_num, _ = permute_input.shape
            expert_idx_per_token = paddle.arange(num_local_experts)[:, None].tile([1, max_num])
        elif self.moe_quant_type in ["weight_only_int8", "weight_only_int4", "w16a16"]:
            expert_idx_per_token = None
        else:
            raise NotImplementedError

        ffn_out = self.compute_ffn(
            layer,
            permute_input,
            token_nums_per_expert.cast("int64"),
            expert_idx_per_token,
            True,
            estimate_total_token_nums,
            dequant_scale,
        )

        # 4. EP combine
        return self.ep_decoder_runner.combine(ffn_out, topk_idx, topk_weights, handle)

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Paddle Cutlass compute Fused MoE.
        """
        gate_out = gate(x.cast("float32"))
        if layer.topk_method == "noaux_tc":
            gate_out, topk_weights, topk_idx = get_moe_scores(
                gate_out,
                layer.n_group,
                layer.topk_group,
                layer.top_k,
                layer.routed_scaling_factor,
                layer.gate_correction_bias,
                getattr(layer, "renormalize", True),
            )
            if current_platform.is_iluvatar():
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
                    self.moe_quant_type,
                    topk_only_mode=True,
                )
                dequant_scale = None
            else:
                (
                    permute_input,
                    token_nums_per_expert,
                    permute_indices_per_token,
                    topk_weights,
                    topk_idx,
                    expert_idx_per_token,
                    dequant_scale,
                ) = moe_expert_dispatch(
                    x,
                    gate_out,
                    None,  # Use layer.gate_correction_bias in get_moe_scores.
                    (
                        layer.up_gate_proj_in_scale if hasattr(layer, "up_gate_proj_in_scale") else None
                    ),  # if set, permute_input will be int8_t
                    layer.top_k,
                    False,
                    self.moe_quant_type,
                    topk_only_mode=True,
                )
        else:
            if current_platform.is_iluvatar():
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
                    (layer.up_gate_proj_in_scale if hasattr(layer, "up_gate_proj_in_scale") else None),
                    layer.top_k,
                    False,
                    self.moe_quant_type,
                    topk_only_mode=False,
                )
                dequant_scale = None
            else:
                (
                    permute_input,
                    token_nums_per_expert,
                    permute_indices_per_token,
                    topk_weights,
                    topk_idx,
                    expert_idx_per_token,
                    dequant_scale,
                ) = moe_expert_dispatch(
                    x,
                    gate_out,
                    layer.gate_correction_bias,
                    (layer.up_gate_proj_in_scale if hasattr(layer, "up_gate_proj_in_scale") else None),
                    layer.top_k,
                    False,
                    self.moe_quant_type,
                    topk_only_mode=False,
                )

        if hasattr(layer, "up_gate_proj_in_scale"):
            dequant_scale = None

        if not layer.with_bias and self.moe_quant_type != "w4a8" and self.moe_quant_type != "w4afp8":
            # only w4a8 need expert_idx_per_token
            # Other need not this tensor, so we make it None.
            expert_idx_per_token = None
        else:
            expert_idx_per_token = expert_idx_per_token.cast("int64")

        ffn_out = self.compute_ffn(
            layer, permute_input, token_nums_per_expert, expert_idx_per_token, False, -1, dequant_scale
        )

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

    def process_prequanted_weights(self, layer: nn.Layer, state_dict, is_rearrange: bool = False):
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
                is_rearrange,
            )
        )

        up_gate_proj_weight_scale = []
        down_proj_weight_scale = []
        up_gate_proj_in_scale_all_experts = []
        up_gate_proj_in_scale = []
        down_proj_in_scale = []

        if isinstance(state_dict, list):
            state_dict = dict(state_dict)

        if layer.ep_size > 1:
            for expert_idx in ep_rank_to_expert_id_list:
                scale_tensor = get_tensor(
                    (
                        state_dict[up_gate_proj_expert_in_scale_key.format(expert_idx)]
                        if up_gate_proj_expert_in_scale_key.format(expert_idx) in state_dict
                        else up_gate_proj_expert_in_scale_key.format(expert_idx)
                    ),
                    layer.fd_config.model_config.model,
                )
                up_gate_proj_in_scale_all_experts.append(scale_tensor)

        for expert_idx in logical_expert_ids:
            up_gate_proj_weight_scale.append(
                get_tensor(
                    (
                        state_dict.pop(up_gate_proj_expert_weight_scale_key.format(expert_idx))
                        if up_gate_proj_expert_weight_scale_key.format(expert_idx) in state_dict
                        else up_gate_proj_expert_weight_scale_key.format(expert_idx)
                    ),
                    layer.fd_config.model_config.model,
                )
            )
            down_proj_weight_scale.append(
                get_tensor(
                    (
                        state_dict.pop(down_proj_expert_weight_scale_key.format(expert_idx))
                        if down_proj_expert_weight_scale_key.format(expert_idx) in state_dict
                        else down_proj_expert_weight_scale_key.format(expert_idx)
                    ),
                    layer.fd_config.model_config.model,
                )
            )
            up_gate_proj_in_scale.append(
                get_tensor(
                    (
                        state_dict.pop(up_gate_proj_expert_in_scale_key.format(expert_idx))
                        if up_gate_proj_expert_in_scale_key.format(expert_idx) in state_dict
                        else up_gate_proj_expert_in_scale_key.format(expert_idx)
                    ),
                    layer.fd_config.model_config.model,
                )
            )
            down_proj_in_scale.append(
                get_tensor(
                    (
                        state_dict.pop(down_proj_expert_in_scale_key.format(expert_idx))
                        if down_proj_expert_in_scale_key.format(expert_idx) in state_dict
                        else down_proj_expert_in_scale_key.format(expert_idx)
                    ),
                    layer.fd_config.model_config.model,
                )
            )

        up_gate_proj_weight = paddle.stack(up_gate_proj_weights, axis=0)
        down_proj_weight = paddle.stack(down_proj_weights, axis=0)
        up_gate_proj_weight_scale = paddle.stack(up_gate_proj_weight_scale, axis=0).cast(paddle.get_default_dtype())
        down_proj_weight_scale = paddle.stack(down_proj_weight_scale, axis=0).cast(paddle.get_default_dtype())
        up_gate_proj_in_scale_all_experts = paddle.stack(up_gate_proj_in_scale_all_experts, axis=0).squeeze()
        up_gate_proj_in_scale = paddle.stack(up_gate_proj_in_scale, axis=0).squeeze()
        down_proj_in_scale = paddle.stack(down_proj_in_scale, axis=0).squeeze()

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
            getattr(layer, name).set_value(tensor)

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        Paddle cutlass create weight process.
        """
        self.weight_dtype = "int8"
        self.up_gate_proj_weight_shape = [
            layer.num_local_experts,
            layer.hidden_size // 2,
            layer.moe_intermediate_size * 2,
        ]
        self.down_proj_weight_shape = [
            layer.num_local_experts,
            layer.moe_intermediate_size // 2,
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

        self.create_w4a8_scale_weights(layer, layer.weight_key_map)

        if layer.with_bias:
            layer.up_gate_proj_bias = layer.create_parameter(
                shape=[layer.num_experts, layer.moe_intermediate_size * 2],
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            layer.down_proj_bias = layer.create_parameter(
                shape=[layer.num_experts, layer.hidden_size],
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            set_weight_attrs(
                layer.up_gate_proj_bias,
                extra_weight_attrs,
            )
            set_weight_attrs(
                layer.down_proj_bias,
                extra_weight_attrs,
            )

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass load weight process.
        """
        up_gate_proj_weights, down_proj_weights, logical_expert_ids, ep_rank_to_expert_id_list = (
            layer.extract_moe_ffn_weights(state_dict)
        )
        self.check(layer, up_gate_proj_weights, down_proj_weights)
        for idx, weight_tensor in enumerate([up_gate_proj_weights, down_proj_weights]):
            weight_name = self.added_weight_attrs[idx]
            weight_list = []
            for i in range(layer.num_local_experts):
                quant_weight, scale = weight_quantize(weight_tensor[i], algo=self.moe_quant_type, arch=80)
                weight_list.append(quant_weight)
            quanted_weight = paddle.stack(weight_list, axis=0)
            getattr(layer, weight_name).set_value(quanted_weight)

        self.load_w4a8_scale_weights(
            layer, layer.weight_key_map, state_dict, logical_expert_ids, ep_rank_to_expert_id_list
        )

    def create_w4a8_scale_weights(self, layer: nn.Layer, weight_key_map: dict):
        """
        Get w4a8 weights from state dict and process them.
        Args:
            layer (nn.Layer): The layer to add parameters to.
            weight_key_map (dict): The weight key map.
        """
        self.default_dtype = layer._helper.get_default_dtype()
        if layer.ep_size > 1:
            setattr(
                layer,
                "up_gate_proj_in_scale_all_experts",
                layer.create_parameter(
                    shape=[layer.num_experts],
                    dtype="float32",
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )

        # in_scales
        for in_scale_name in ["up_gate_proj_in_scale", "down_proj_in_scale"]:
            setattr(
                layer,
                in_scale_name,
                layer.create_parameter(
                    shape=[layer.num_local_experts],
                    dtype="float32",
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

    def load_w4a8_scale_weights(
        self,
        layer: nn.Layer,
        weight_key_map: dict,
        state_dict: dict,
        logical_expert_ids: paddle.Tensor,
        ep_rank_to_expert_id_list: list,
    ):
        """
        Get w4a8 weights from state dict and process them.
        Args:
            layer (nn.Layer): The layer to add parameters to.
            weight_key_map (dict): The weight key map.
            state_dict (dict): The state dict.
        """

        def _extract_scale_tensor(layer: nn.Layer, state_dict, key_template, expert_idx):
            return get_tensor(
                (
                    state_dict.pop(key_template.format(expert_idx))
                    if key_template.format(expert_idx) in state_dict
                    else key_template.format(expert_idx)
                ),
                layer.fd_config.model_config.model,
            )

        def _process_in_scale(name: str, in_scales: list[paddle.Tensor]):
            processed_in_scale = 1 / paddle.concat(in_scales)
            getattr(layer, name).set_value(processed_in_scale)
            return processed_in_scale

        def _process_weight_scale(
            name: str,
            weight_scales: list[paddle.Tensor],
            processed_in_scale: paddle.Tensor,
        ):
            processed_weight_scale = (
                paddle.stack(weight_scales, axis=0) / (127 * 112) / processed_in_scale[:, None]
            ).cast(paddle.get_default_dtype())
            getattr(layer, name).set_value(processed_weight_scale)

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
            for expert_idx in ep_rank_to_expert_id_list:
                scale_tensor = get_tensor(
                    (
                        state_dict[scale_key_map["up_gate_proj_in_scale"].format(expert_idx)]
                        if scale_key_map["up_gate_proj_in_scale"].format(expert_idx) in state_dict
                        else scale_key_map["up_gate_proj_in_scale"].format(expert_idx)
                    ),
                    layer.fd_config.model_config.model,
                )
                up_gate_proj_in_scales_all_experts.append(1 / scale_tensor)
            getattr(layer, "up_gate_proj_in_scale_all_experts").set_value(
                paddle.concat(up_gate_proj_in_scales_all_experts)
            )

        for expert_idx in logical_expert_ids:
            for name, scale_key_template in scale_key_map.items():
                scale_tensor = _extract_scale_tensor(layer, state_dict, scale_key_template, expert_idx)
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


class CutlassW4AFP8MoEMethod(CutlassMoEMethod):
    """
    w4a8 MoE Method
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.quant_config = quant_config
        self.moe_quant_type = "w4afp8"
        self.pack_num = 2

    def process_prequanted_weights(self, layer: nn.Layer, state_dict, is_rearrange: bool = False):
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
                is_rearrange,
            )
        )

        up_gate_proj_weight_scale = []
        down_proj_weight_scale = []
        up_gate_proj_in_scale_all_experts = []
        up_gate_proj_in_scale = []
        down_proj_in_scale = []

        if isinstance(state_dict, list):
            state_dict = dict(state_dict)

        if layer.ep_size > 1:
            for expert_idx in ep_rank_to_expert_id_list:
                scale_tensor = get_tensor(
                    (
                        state_dict[up_gate_proj_expert_in_scale_key.format(expert_idx)]
                        if up_gate_proj_expert_in_scale_key.format(expert_idx) in state_dict
                        else up_gate_proj_expert_in_scale_key.format(expert_idx)
                    ),
                    layer.fd_config.model_config.model,
                )
                up_gate_proj_in_scale_all_experts.append(scale_tensor)

        for expert_idx in logical_expert_ids:
            up_gate_proj_weight_scale.append(
                get_tensor(
                    (
                        state_dict.pop(up_gate_proj_expert_weight_scale_key.format(expert_idx))
                        if up_gate_proj_expert_weight_scale_key.format(expert_idx) in state_dict
                        else up_gate_proj_expert_weight_scale_key.format(expert_idx)
                    ),
                    layer.fd_config.model_config.model,
                )
            )
            down_proj_weight_scale.append(
                get_tensor(
                    (
                        state_dict.pop(down_proj_expert_weight_scale_key.format(expert_idx))
                        if down_proj_expert_weight_scale_key.format(expert_idx) in state_dict
                        else down_proj_expert_weight_scale_key.format(expert_idx)
                    ),
                    layer.fd_config.model_config.model,
                )
            )
            up_gate_proj_in_scale.append(
                get_tensor(
                    (
                        state_dict.pop(up_gate_proj_expert_in_scale_key.format(expert_idx))
                        if up_gate_proj_expert_in_scale_key.format(expert_idx) in state_dict
                        else up_gate_proj_expert_in_scale_key.format(expert_idx)
                    ),
                    layer.fd_config.model_config.model,
                )
            )
            down_proj_in_scale.append(
                get_tensor(
                    (
                        state_dict.pop(down_proj_expert_in_scale_key.format(expert_idx))
                        if down_proj_expert_in_scale_key.format(expert_idx) in state_dict
                        else down_proj_expert_in_scale_key.format(expert_idx)
                    ),
                    layer.fd_config.model_config.model,
                )
            )

        up_gate_proj_weight = paddle.stack(up_gate_proj_weights, axis=0)
        down_proj_weight = paddle.stack(down_proj_weights, axis=0)
        up_gate_proj_weight_scale = paddle.stack(up_gate_proj_weight_scale, axis=0)
        down_proj_weight_scale = paddle.stack(down_proj_weight_scale, axis=0)
        up_gate_proj_in_scale_all_experts = paddle.stack(up_gate_proj_in_scale_all_experts, axis=0).squeeze()
        up_gate_proj_in_scale = paddle.stack(up_gate_proj_in_scale, axis=0).squeeze()
        down_proj_in_scale = paddle.stack(down_proj_in_scale, axis=0).squeeze()

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
            getattr(layer, name).set_value(tensor)

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        Paddle cutlass create weight process.
        """
        self.weight_dtype = "int8"
        self.ffn1_weight_shape = [
            layer.num_local_experts,
            layer.hidden_size // 2,
            layer.moe_intermediate_size * 2,
        ]
        self.ffn2_weight_shape = [
            layer.num_local_experts,
            layer.moe_intermediate_size // 2,
            layer.hidden_size,
        ]
        setattr(
            layer,
            self.added_weight_attrs[0],
            layer.create_parameter(
                shape=self.ffn1_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            self.added_weight_attrs[1],
            layer.create_parameter(
                shape=self.ffn2_weight_shape,
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )

        self.create_w4afp8_scale_weights(layer, layer.weight_key_map)

        if layer.with_bias:
            layer.up_gate_proj_bias = layer.create_parameter(
                shape=[layer.num_experts, layer.moe_intermediate_size * 2],
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            layer.down_proj_bias = layer.create_parameter(
                shape=[layer.num_experts, layer.hidden_size],
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            set_weight_attrs(
                layer.up_gate_proj_bias,
                extra_weight_attrs,
            )
            set_weight_attrs(
                layer.down_proj_bias,
                extra_weight_attrs,
            )

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass load weight process.
        """
        up_gate_proj_weights, down_proj_weights, logical_expert_ids, ep_rank_to_expert_id_list = (
            layer.extract_moe_ffn_weights(state_dict)
        )
        self.check(layer, up_gate_proj_weights, down_proj_weights)
        for idx, weight_tensor in enumerate([up_gate_proj_weights, down_proj_weights]):
            weight_name = self.added_weight_attrs[idx]
            weight_list = []
            for i in range(layer.num_local_experts):
                quant_weight = w4afp8_gemm_weight_convert(weight_tensor[i])
                weight_list.append(quant_weight)
            quanted_weight = paddle.stack(weight_list, axis=0)
            getattr(layer, weight_name).set_value(quanted_weight)

        self.load_w4afp8_scale_weights(
            layer, layer.weight_key_map, state_dict, logical_expert_ids, ep_rank_to_expert_id_list
        )

    def create_w4afp8_scale_weights(self, layer: nn.Layer, weight_key_map: dict):
        """
        Get w4afp8 weights from state dict and process them.
        Args:
            layer (nn.Layer): The layer to add parameters to.
            weight_key_map (dict): The weight key map.
        """

        self.default_dtype = layer._helper.get_default_dtype()
        if layer.ep_size > 1 and not layer.moe_quant_config.moe_dynamic_quant:
            setattr(
                layer,
                "up_gate_proj_in_scale_all_experts",
                layer.create_parameter(
                    shape=[layer.num_experts],
                    dtype="float32",
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )

        # in_scales
        if not layer.moe_quant_config.moe_dynamic_quant:
            for in_scale_name in ["up_gate_proj_in_scale", "down_proj_in_scale"]:
                setattr(
                    layer,
                    in_scale_name,
                    layer.create_parameter(
                        shape=[layer.num_local_experts],
                        dtype="float32",
                        default_initializer=paddle.nn.initializer.Constant(0),
                    ),
                )

        # weight_scales
        setattr(
            layer,
            "up_gate_proj_weight_scale",
            layer.create_parameter(
                shape=[layer.num_local_experts, layer.moe_intermediate_size * 2],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            "down_proj_weight_scale",
            layer.create_parameter(
                shape=[layer.num_local_experts, layer.hidden_size],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )

    def load_w4afp8_scale_weights(
        self,
        layer: nn.Layer,
        weight_key_map: dict,
        state_dict: dict,
        logical_expert_ids: paddle.Tensor,
        ep_rank_to_expert_id_list: list,
    ):
        """
        Get w4afp8 weights from state dict and process them.
        Args:
            layer (nn.Layer): The layer to add parameters to.
            weight_key_map (dict): The weight key map.
            state_dict (dict): The state dict.
        """

        def _extract_scale_tensor(layer: nn.Layer, state_dict, key_template, expert_idx):
            return get_tensor(
                (
                    state_dict.pop(key_template.format(expert_idx))
                    if key_template.format(expert_idx) in state_dict
                    else key_template.format(expert_idx)
                ),
                layer.fd_config.model_config.model,
            )

        def _process_in_scale(name: str, in_scales: list[paddle.Tensor]):
            processed_in_scale = 1 / paddle.concat(in_scales)
            getattr(layer, name).set_value(processed_in_scale)
            return processed_in_scale

        def _permute_weight_scale(weight_scale: paddle.Tensor):
            weight_scale = w4afp8_gemm_scale_permute(weight_scale)
            return weight_scale

        def _process_weight_scale(name: str, weight_scales: list[paddle.Tensor], processed_in_scale: paddle.Tensor):
            if processed_in_scale is not None:
                processed_weight_scale = paddle.stack(weight_scales, axis=0) / (448 * 7 * 2 ** (-9))
                if len(processed_weight_scale.shape) == 3:
                    processed_weight_scale = (
                        processed_weight_scale.transpose([0, 2, 1]) / processed_in_scale[:, None, None]
                    )
                else:
                    processed_weight_scale = processed_weight_scale / processed_in_scale[:, None]
            else:
                processed_weight_scale = paddle.stack(weight_scales, axis=0) / (440 * 7 * 2 ** (-9))

            if len(processed_weight_scale.shape) == 3:
                if name == "up_gate_proj_weight_scale" and processed_weight_scale.shape[-1] * 128 != layer.hidden_size:
                    assert (
                        layer.hidden_size // 128 % processed_weight_scale.shape[-1] == 0
                    ), "weight_scale_group_size must be a multiple of 128"
                    # If it is a multiple of 128, repeat to 128
                    processed_weight_scale = processed_weight_scale.repeat_interleave(
                        layer.hidden_size // 128 // processed_weight_scale.shape[-1], axis=-1
                    )
                elif (
                    name == "down_proj_weight_scale"
                    and processed_weight_scale.shape[-1] * 128 != layer.moe_intermediate_size
                ):
                    assert (
                        layer.moe_intermediate_size // 128 % processed_weight_scale.shape[-1] == 0
                    ), "weight_scale_group_size must be a multiple of 128"
                    # If it is a multiple of 128, repeat to 128
                    processed_weight_scale = processed_weight_scale.repeat_interleave(
                        layer.moe_intermediate_size // 128 // processed_weight_scale.shape[-1], axis=-1
                    )

                origin_shape = processed_weight_scale.shape
                processed_weight_scale = processed_weight_scale.transpose([0, 2, 1])
                processed_weight_scale = processed_weight_scale.reshape([-1, processed_weight_scale.shape[-1]])
                processed_weight_scale = _permute_weight_scale(processed_weight_scale)
                processed_weight_scale = processed_weight_scale.reshape(
                    [origin_shape[0], origin_shape[2], origin_shape[1] // 128, 128]
                )
                processed_weight_scale = processed_weight_scale.transpose([0, 2, 1, 3])
                setattr(
                    layer,
                    name,
                    layer.create_parameter(
                        shape=processed_weight_scale.shape,
                        dtype="float32",
                        default_initializer=paddle.nn.initializer.Constant(0),
                    ),
                )
            else:
                processed_weight_scale = _permute_weight_scale(processed_weight_scale)
            getattr(layer, name).set_value(processed_weight_scale)

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
            if hasattr(layer, name) and value is None:
                raise ValueError(f"scale {name} should not be none in w4a8 mode.")

        # 2. Extract scale tensor from state dict
        if layer.ep_size > 1 and not layer.moe_quant_config.moe_dynamic_quant:
            for expert_idx in ep_rank_to_expert_id_list:
                scale_tensor = get_tensor(
                    (
                        state_dict[scale_key_map["up_gate_proj_in_scale"].format(expert_idx)]
                        if scale_key_map["up_gate_proj_in_scale"].format(expert_idx) in state_dict
                        else scale_key_map["up_gate_proj_in_scale"].format(expert_idx)
                    ),
                    layer.fd_config.model_config.model,
                )
                up_gate_proj_in_scales_all_experts.append(1 / scale_tensor)
            getattr(layer, "up_gate_proj_in_scale_all_experts").set_value(
                paddle.concat(up_gate_proj_in_scales_all_experts)
            )

        for expert_idx in logical_expert_ids:
            for name, scale_key_template in scale_key_map.items():
                if hasattr(layer, name):
                    scale_tensor = _extract_scale_tensor(layer, state_dict, scale_key_template, expert_idx)
                    scale_weight_map[name].append(scale_tensor)

        for i, weight_scale_name in enumerate(["up_gate_proj_weight_scale", "down_proj_weight_scale"]):
            in_scale_name = weight_scale_name.replace("_weight_scale", "_in_scale")
            in_scale = None
            if hasattr(layer, in_scale_name) and in_scale_name in scale_weight_map.keys():
                in_scale = _process_in_scale(in_scale_name, scale_weight_map[in_scale_name])
            _process_weight_scale(
                weight_scale_name,
                scale_weight_map[weight_scale_name],
                in_scale,
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

    def process_prequanted_weights(self, layer: nn.Layer, state_dict, is_rearrange: bool = False):
        """
        Paddle cutlass process prequanted weights.
        """
        up_gate_proj_expert_weight_key = layer.weight_key_map.get("up_gate_proj_expert_weight_key", None)
        down_proj_expert_weight_key = layer.weight_key_map.get("down_proj_expert_weight_key", None)
        up_gate_proj_expert_weight_scale_key = layer.weight_key_map.get("up_gate_proj_expert_weight_scale_key", None)
        down_proj_expert_weight_scale_key = layer.weight_key_map.get("down_proj_expert_weight_scale_key", None)

        up_gate_proj_weights, down_proj_weights, logical_expert_ids, _ = layer.load_experts_weight(
            state_dict, up_gate_proj_expert_weight_key, down_proj_expert_weight_key, is_rearrange
        )
        # self.check(layer, up_gate_proj_weights, down_proj_weights)
        up_gate_proj_weight_scale = []
        down_proj_weight_scale = []

        if isinstance(state_dict, list):
            state_dict = dict(state_dict)

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
            getattr(layer, name).set_value(tensor)

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        Paddle cutlass create weight process.
        """
        self.default_dtype = layer._helper.get_default_dtype()
        if self.moe_quant_type == "weight_only_int4":
            self.up_gate_proj_weight_shape = [
                layer.num_local_experts,
                layer.moe_intermediate_size,
                layer.hidden_size,
            ]
        else:
            self.up_gate_proj_weight_shape = [
                layer.num_local_experts,
                layer.moe_intermediate_size * 2,
                layer.hidden_size,
            ]
        if self.moe_quant_type == "weight_only_int4":
            self.down_proj_weight_shape = [
                layer.num_local_experts,
                layer.hidden_size // 2,
                layer.moe_intermediate_size,
            ]
        else:
            self.down_proj_weight_shape = [
                layer.num_local_experts,
                layer.hidden_size,
                layer.moe_intermediate_size,
            ]
        self.up_gate_proj_scale_shape = [layer.num_local_experts, layer.moe_intermediate_size * 2]
        self.down_proj_scale_shape = [layer.num_local_experts, layer.hidden_size]
        self.model_format = extra_weight_attrs.get("model_format")
        # TODO(bukejiyu): remove v1 loader check when v0 loader is removed
        if self.quant_config.is_checkpoint_bf16 and layer.fd_config.load_config.load_choices == "default_v1":
            if self.model_format != "torch":
                up_gate_proj_weight_shape = [
                    layer.num_local_experts,
                    layer.hidden_size,
                    layer.moe_intermediate_size * 2,
                ]
                down_proj_weight_shape = [layer.num_local_experts, layer.moe_intermediate_size, layer.hidden_size]
                up_gate_proj_attrs = {
                    **extra_weight_attrs,
                    "tensor_track": TensorTracker(shape=up_gate_proj_weight_shape, output_dim=True),
                }
                down_proj_attrs = {
                    **extra_weight_attrs,
                    "tensor_track": TensorTracker(shape=down_proj_weight_shape, output_dim=False),
                }
            else:
                up_gate_proj_weight_shape = [
                    layer.num_local_experts,
                    layer.moe_intermediate_size * 2,
                    layer.hidden_size,
                ]
                down_proj_weight_shape = [layer.num_local_experts, layer.hidden_size, layer.moe_intermediate_size]
                up_gate_proj_attrs = {
                    **extra_weight_attrs,
                    "tensor_track": TensorTracker(shape=up_gate_proj_weight_shape, output_dim=False),
                    "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0},
                }
                down_proj_attrs = {
                    **extra_weight_attrs,
                    "tensor_track": TensorTracker(shape=down_proj_weight_shape, output_dim=True),
                    "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0},
                }

            layer.up_gate_proj_weight = layer.create_parameter(
                shape=up_gate_proj_weight_shape,
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            layer.down_proj_weight = layer.create_parameter(
                shape=down_proj_weight_shape,
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )
            set_weight_attrs(
                layer.up_gate_proj_weight,
                up_gate_proj_attrs,
            )
            set_weight_attrs(
                layer.down_proj_weight,
                down_proj_attrs,
            )
        else:
            self.weight_dtype = "int8"

            up_gate_proj_weight_name = self.added_weight_attrs[0]
            down_proj_weight_name = self.added_weight_attrs[1]
            up_gate_proj_scale_name = self.added_scale_attrs[0]
            down_proj_scale_name = self.added_scale_attrs[1]

            setattr(
                layer,
                up_gate_proj_weight_name,
                layer.create_parameter(
                    shape=self.up_gate_proj_weight_shape,
                    dtype=self.weight_dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            setattr(
                layer,
                down_proj_weight_name,
                layer.create_parameter(
                    shape=self.down_proj_weight_shape,
                    dtype=self.weight_dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            # weight_scale
            setattr(
                layer,
                up_gate_proj_scale_name,
                layer.create_parameter(
                    shape=self.up_gate_proj_scale_shape,
                    dtype=self.default_dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            setattr(
                layer,
                down_proj_scale_name,
                layer.create_parameter(
                    shape=self.down_proj_scale_shape,
                    dtype=self.default_dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            # The v1 loader currently does not support loading offline quantized weight-only weights.
            moe_extra_weight_attrs = {**extra_weight_attrs, "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0}}
            set_weight_attrs(layer.up_gate_proj_weight, moe_extra_weight_attrs)
            set_weight_attrs(layer.down_proj_weight, moe_extra_weight_attrs)
            scale_extra_weight_attrs = {
                **extra_weight_attrs,
                "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "up": 0, "down": None},
            }
            set_weight_attrs(layer.up_gate_proj_weight_scale, scale_extra_weight_attrs)
            set_weight_attrs(layer.down_proj_weight_scale, scale_extra_weight_attrs)

        if layer.with_bias:
            layer.up_gate_proj_bias = layer.create_parameter(
                shape=[layer.num_experts, layer.moe_intermediate_size * 2],
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            layer.down_proj_bias = layer.create_parameter(
                shape=[layer.num_experts, layer.hidden_size],
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            set_weight_attrs(
                layer.up_gate_proj_bias,
                extra_weight_attrs,
            )
            set_weight_attrs(
                layer.down_proj_bias,
                extra_weight_attrs,
            )

    def process_weights_after_loading(self, layer):
        def _process_quantize(weight_idx):
            # 1.init shape and type
            # quantized_weight_name
            weight_name = self.added_weight_attrs[weight_idx]
            unquantized_weight_name = weight_name.replace("quant_weight", "weight")
            weight_shape = self.up_gate_proj_weight_shape if weight_type == "gate_up" else self.down_proj_weight_shape
            weight_dtype = "int8"
            # scale
            scale_name = self.added_scale_attrs[weight_idx]
            scale_shape = self.up_gate_proj_scale_shape if weight_type == "gate_up" else self.down_proj_scale_shape
            scale_dtype = self.default_dtype

            # 2.crate tmp tensor

            weight = paddle.empty(weight_shape, dtype=weight_dtype)
            scale = paddle.empty(scale_shape, dtype=scale_dtype)

            # 3.quantize weight

            for expert_id in range(layer.num_local_experts):
                weight[expert_id], scale[expert_id] = weight_quantize(
                    getattr(layer, unquantized_weight_name)[expert_id], algo=self.moe_quant_type
                )

            free_tensor(getattr(layer, unquantized_weight_name))

            # create weight
            setattr(
                layer,
                weight_name,
                layer.create_parameter(
                    shape=weight_shape,
                    dtype=weight_dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            # create scale
            setattr(
                layer,
                scale_name,
                layer.create_parameter(
                    shape=scale_shape,
                    dtype=scale_dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            getattr(layer, weight_name).copy_(weight, False)
            getattr(layer, scale_name).copy_(scale, False)

        if self.quant_config.is_checkpoint_bf16:
            weight_id_map = {"gate_up": 0, "down": 1}
            if weight_fully_copied(layer.up_gate_proj_weight):
                weight_type = "gate_up"
            else:
                weight_type = "down"

            if self.model_format == "torch":
                unquantized_weight_name = self.added_weight_attrs[weight_id_map[weight_type]].replace(
                    "quant_weight", "weight"
                )
                process_weight_transpose(layer, unquantized_weight_name)
            _process_quantize(weight_id_map[weight_type])
        else:
            return

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass load weight process.
        """
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)
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
            getattr(layer, weight_name).set_value(quanted_weight)

            quanted_weight_scale = paddle.stack(weight_scale_list, axis=0)
            getattr(layer, scale_name).set_value(quanted_weight_scale)
