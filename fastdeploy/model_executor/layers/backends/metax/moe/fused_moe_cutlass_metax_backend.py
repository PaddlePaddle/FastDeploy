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

import os

import paddle
from paddle import nn
from paddle.nn.quant import weight_quantize

from fastdeploy.model_executor.layers.moe.fused_moe_backend_base import (
    MoEMethodBase,
    UnquantizedFusedMoEMethod,
)
from fastdeploy.model_executor.layers.moe.moe import get_moe_scores
from fastdeploy.model_executor.layers.quantization.weight_only import WeightOnlyConfig
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.ops.gpu import (
    fused_expert_moe,
    moe_expert_dispatch,
    moe_expert_ffn,
    moe_expert_reduce,
)
from fastdeploy.model_executor.utils import (
    TensorTracker,
    free_tensor,
    process_weight_transpose,
    set_weight_attrs,
    weight_fully_copied,
)


class MetaxCutlassUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
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
    ):
        """
        Paddle Cutlass compute Fused MoE.
        """
        raise NotImplementedError

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

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Paddle Cutlass compute Fused MoE.
        """
        """
        Paddle Cutlass compute Fused MoE.
        """
        if layer.topk_method == "noaux_tc":
            gate_out = gate(x.cast("float32"))

            gate_out, topk_weights, topk_idx = get_moe_scores(
                gate_out,
                layer.n_group,
                layer.topk_group,
                layer.top_k,
                layer.routed_scaling_factor,
                layer.gate_correction_bias,
                getattr(layer, "renormalize", True),
            )

            (
                permute_input,
                token_nums_per_expert,
                permute_indices_per_token,
                topk_weights,
                topk_idx,
            ) = moe_expert_dispatch(
                x,
                gate_out,
                layer.top_k,
                False,
                True,
            )

            ffn_out = self.compute_ffn(layer, permute_input, token_nums_per_expert, None)

            fused_moe_out = moe_expert_reduce(
                ffn_out,
                topk_weights,
                permute_indices_per_token,
                topk_idx,
                None,
                False,
                1.0,
            )
        else:
            raise NotImplementedError

            fused_moe_out = fused_expert_moe(
                x,
                gate.weight,
                getattr(layer, self.added_weight_attrs[0]),
                getattr(layer, self.added_weight_attrs[1]),
                None,
                (layer.up_gate_proj_weight_scale if hasattr(layer, "up_gate_proj_weight_scale") else None),
                None,
                (layer.down_proj_weight_scale if hasattr(layer, "down_proj_weight_scale") else None),
                "weight_only_int8",
                layer.top_k,
                True,
                False,
            )

        return fused_moe_out


class MetaxCutlassMoEMethod(MoEMethodBase):
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

    def compute_ffn(
        self,
        layer: nn.Layer,
        permute_input: paddle.Tensor,
        token_nums_per_expert: paddle.Tensor,
        expert_idx_per_token: paddle.Tensor,
        used_in_ep_low_latency: bool = False,
        estimate_total_token_nums: int = -1,
    ):
        """
        Paddle Cutlass compute Fused MoE.
        """
        return moe_expert_ffn(
            permute_input,
            token_nums_per_expert,
            getattr(layer, self.added_weight_attrs[0]),
            getattr(layer, self.added_weight_attrs[1]),
            None,
            (layer.up_gate_proj_weight_scale if hasattr(layer, "up_gate_proj_weight_scale") else None),
            (layer.down_proj_weight_scale if hasattr(layer, "down_proj_weight_scale") else None),
            "weight_only_int8",
        )

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

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Paddle Cutlass compute Fused MoE.
        """
        if layer.topk_method == "noaux_tc":
            gate_out = gate(x.cast("float32"))

            gate_out, topk_weights, topk_idx = get_moe_scores(
                gate_out,
                layer.n_group,
                layer.topk_group,
                layer.top_k,
                layer.routed_scaling_factor,
                layer.gate_correction_bias,
                getattr(layer, "renormalize", True),
            )

            (
                permute_input,
                token_nums_per_expert,
                permute_indices_per_token,
                topk_weights,
                topk_idx,
            ) = moe_expert_dispatch(
                x,
                gate_out,
                layer.top_k,
                False,
                True,
            )

            ffn_out = self.compute_ffn(layer, permute_input, token_nums_per_expert, None)

            fused_moe_out = moe_expert_reduce(
                ffn_out,
                topk_weights,
                permute_indices_per_token,
                topk_idx,
                None,
                False,
                1.0,
            )
        else:
            fused_moe_out = fused_expert_moe(
                x,
                gate.weight,
                getattr(layer, self.added_weight_attrs[0]),
                getattr(layer, self.added_weight_attrs[1]),
                None,
                (layer.up_gate_proj_weight_scale if hasattr(layer, "up_gate_proj_weight_scale") else None),
                None,
                (layer.down_proj_weight_scale if hasattr(layer, "down_proj_weight_scale") else None),
                "weight_only_int8",
                layer.top_k,
                True,
                False,
            )

        return fused_moe_out


class MetaxCutlassWeightOnlyMoEMethod(MetaxCutlassMoEMethod):
    """
    weight only for moe
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)
        if quant_config is None:
            self.quant_config = WeightOnlyConfig(algo="weight_only_int8", is_checkpoint_bf16=True)
        else:
            self.quant_config = quant_config
        self.moe_quant_type = self.quant_config.algo
        self.pack_num = 1
        self.weight_only_linear_arch = os.getenv("FLAGS_weight_only_linear_arch")
        if self.weight_only_linear_arch is not None:
            self.weight_only_linear_arch = int(self.weight_only_linear_arch)

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
            # extra_weight_attrs["weight_need_transpose"] = extra_weight_attrs.get("model_format") == "torch"
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
            # extra_weight_attrs["weight_need_transpose"] = not extra_weight_attrs.get("model_format") == "torch"
            moe_extra_weight_attrs = {**extra_weight_attrs, "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0}}
            set_weight_attrs(layer.up_gate_proj_weight, moe_extra_weight_attrs)
            set_weight_attrs(layer.down_proj_weight, moe_extra_weight_attrs)
            scale_extra_weight_attrs = {
                **extra_weight_attrs,
                "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "up": 0, "down": None},
            }
            set_weight_attrs(layer.up_gate_proj_weight_scale, scale_extra_weight_attrs)
            set_weight_attrs(layer.down_proj_weight_scale, scale_extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        def _process_quantize(weight_idx):
            # 1.init shape and type
            weight_name = self.added_weight_attrs[weight_idx]
            unquantized_weight_name = weight_name.replace("quant_weight", "weight")
            weight_shape = self.up_gate_proj_weight_shape if weight_type == "gate_up" else self.down_proj_weight_shape
            transposed_weight_shape = [weight_shape[0], weight_shape[2], weight_shape[1]]
            weight_dtype = "int8"
            # scale
            scale_name = self.added_scale_attrs[weight_idx]
            scale_shape = self.up_gate_proj_scale_shape if weight_type == "gate_up" else self.down_proj_scale_shape
            scale_dtype = self.default_dtype

            # 2.crate tmp tensor

            weight = paddle.empty(transposed_weight_shape, dtype=weight_dtype)
            scale = paddle.empty(scale_shape, dtype=scale_dtype)

            # 3.quantize weight

            for expert_id in range(layer.num_local_experts):
                weight[expert_id], scale[expert_id] = weight_quantize(
                    getattr(layer, unquantized_weight_name)[expert_id],
                    algo=self.moe_quant_type,
                    arch=self.weight_only_linear_arch,
                )

            free_tensor(getattr(layer, unquantized_weight_name))

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
            getattr(layer, weight_name).copy_(weight.transpose([0, 2, 1]), False)
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
                quant_weight, scale = weight_quantize(
                    weight_tensor[i], algo=self.moe_quant_type, arch=self.weight_only_linear_arch
                )
                quant_weight = paddle.transpose(quant_weight, [1, 0])
                weight_list.append(quant_weight)
                weight_scale_list.append(scale)
            quanted_weight = paddle.stack(weight_list, axis=0)
            getattr(layer, weight_name).set_value(quanted_weight)

            quanted_weight_scale = paddle.stack(weight_scale_list, axis=0)
            getattr(layer, scale_name).set_value(quanted_weight_scale)
