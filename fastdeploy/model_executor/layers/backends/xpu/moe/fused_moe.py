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

from fastdeploy.model_executor.layers.moe.fused_moe_backend_base import MoEMethodBase
from fastdeploy.model_executor.layers.quantization.weight_only import WeightOnlyConfig
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.ops.xpu import (
    ep_moe_expert_combine,
    ep_moe_expert_dispatch,
    moe_expert_ffn,
    moe_topk_select,
    weight_quantize_xpu,
    xpu_moe_layer,
)
from fastdeploy.model_executor.utils import (
    TensorTracker,
    default_weight_loader,
    free_tensor,
    set_weight_attrs,
)


class XPUMoEMethod(MoEMethodBase):
    """
    XPU MOE
    """

    def __init__(
        self,
        quant_config: WeightOnlyConfig,
    ) -> None:
        super().__init__(quant_config)

        if self.moe_quant_type in ["w16a16"]:
            self.weight_dtype = "bfloat16"
        elif self.moe_quant_type in ["weight_only_int8", "w8a8", "weight_only_int4", "w4a8"]:
            self.weight_dtype = "int8"
        else:
            raise ValueError(f"Unsupported moe quant type: {self.moe_quant_type}")
        self.scale_dtype = "float32"
        self.bias_dtype = "float32"

    def import_backend_ep_runner(self) -> None:
        from .ep import XPUEPDecoderRunner, XPUEPPrefillRunner

        self.EPPrefillRunner = XPUEPPrefillRunner
        self.EPDecoderRunner = XPUEPDecoderRunner

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        create weight process.
        """
        if layer.fd_config.load_config.load_choices == "default_v1" and self.moe_quant_type in [
            "w16a16",
            "weight_only_int8",
            "weight_only_int4",
        ]:
            self.up_gate_proj_weight_shape = [
                layer.num_local_experts,
                layer.moe_intermediate_size * 2,
                layer.hidden_size,
            ]
            self.down_proj_weight_shape = [layer.num_local_experts, layer.hidden_size, layer.moe_intermediate_size]
            layer.up_gate_proj_weight = layer.create_parameter(
                shape=self.up_gate_proj_weight_shape,
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            layer.down_proj_weight = layer.create_parameter(
                shape=self.down_proj_weight_shape,
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            set_weight_attrs(
                layer.up_gate_proj_weight,
                {
                    "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0},
                    "weight_loader": extra_weight_attrs.get("weight_loader", default_weight_loader(layer.fd_config)),
                    "weight_need_transpose": not extra_weight_attrs.get("model_format") == "torch",
                    "tensor_track": TensorTracker(shape=layer.up_gate_proj_weight.shape, output_dim=False),
                },
            )
            set_weight_attrs(
                layer.down_proj_weight,
                {
                    "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0},
                    "weight_loader": extra_weight_attrs.get("weight_loader", default_weight_loader(layer.fd_config)),
                    "weight_need_transpose": not extra_weight_attrs.get("model_format") == "torch",
                    "tensor_track": TensorTracker(shape=layer.down_proj_weight.shape, output_dim=True),
                },
            )
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
                    {
                        "weight_loader": extra_weight_attrs.get(
                            "weight_loader", default_weight_loader(layer.fd_config)
                        ),
                    },
                )
                set_weight_attrs(
                    layer.down_proj_bias,
                    {
                        "weight_loader": extra_weight_attrs.get(
                            "weight_loader", default_weight_loader(layer.fd_config)
                        ),
                    },
                )
            if self.moe_quant_type in ["weight_only_int8", "weight_only_int4"]:
                self.up_gate_proj_scale_shape = [
                    layer.num_local_experts,
                    layer.moe_intermediate_size * 2,
                ]
                self.down_proj_scale_shape = [
                    layer.num_local_experts,
                    layer.hidden_size,
                ]

        else:
            self.up_gate_proj_weight_shape = [
                layer.num_local_experts,
                layer.moe_intermediate_size * 2,
                layer.hidden_size,
            ]
            self.down_proj_weight_shape = [
                layer.num_local_experts,
                layer.hidden_size,
                layer.moe_intermediate_size,
            ]
            if self.moe_quant_type in ["weight_only_int4", "w4a8"]:
                self.up_gate_proj_weight_shape[-1] //= 2
                self.down_proj_weight_shape[-1] //= 2

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

            if self.moe_quant_type in ["weight_only_int8", "w8a8", "weight_only_int4", "w4a8"]:
                self.up_gate_proj_scale_shape = [
                    layer.num_local_experts,
                    layer.moe_intermediate_size * 2,
                ]
                self.down_proj_scale_shape = [
                    layer.num_local_experts,
                    layer.hidden_size,
                ]
                setattr(
                    layer,
                    self.added_scale_attrs[0],
                    layer.create_parameter(
                        shape=self.up_gate_proj_scale_shape,
                        dtype=self.scale_dtype,
                        default_initializer=paddle.nn.initializer.Constant(0),
                    ),
                )
                setattr(
                    layer,
                    self.added_scale_attrs[1],
                    layer.create_parameter(
                        shape=self.down_proj_scale_shape,
                        dtype=self.scale_dtype,
                        default_initializer=paddle.nn.initializer.Constant(0),
                    ),
                )

            if self.moe_quant_type in ["w8a8", "w4a8"]:
                for in_scale_name in self.added_in_scale_attrs:
                    setattr(
                        layer,
                        in_scale_name,
                        layer.create_parameter(
                            shape=[layer.num_local_experts],
                            dtype=self.scale_dtype,
                            default_initializer=paddle.nn.initializer.Constant(0),
                        ),
                    )

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)
        for weights in [up_gate_proj_weights, down_proj_weights]:
            for idx, weight in enumerate(weights):
                weights[idx] = weight.transpose([1, 0])
        stacked_up_gate_proj_weights = paddle.stack(up_gate_proj_weights, axis=0)
        stacked_down_proj_weights = paddle.stack(down_proj_weights, axis=0)

        layer.up_gate_proj_weight.set_value(stacked_up_gate_proj_weights)
        layer.down_proj_weight.set_value(stacked_down_proj_weights)

    def apply_tp_fused_op(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Apply TP Fused Op.
        """
        fused_moe_out = xpu_moe_layer(
            x,
            gate.weight.transpose([1, 0]),
            layer.gate_correction_bias,
            layer.up_gate_proj_weight,
            layer.down_proj_weight,
            None,  # up_gate_proj bias
            None,  # down_proj bias
            getattr(layer, self.added_scale_attrs[0], None),
            getattr(layer, self.added_scale_attrs[1], None),
            getattr(layer, self.added_in_scale_attrs[0], None),
            self.moe_quant_type,
            layer.top_k,
            False,  # moe group, used in deepseek
        )

        return fused_moe_out

    def apply_tp_scatter_op(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Apply TP Scatter Op.
        """
        gate_out = gate(x.cast("float32"))
        topk_idx, topk_weights = moe_topk_select(
            gate_out,
            layer.gate_correction_bias,
            layer.top_k,
            True,
        )
        token_nums_per_expert_list = list(range(64))  # placeholder, not use
        (
            permute_input,
            permute_indices_per_token,
            token_num_lod,
            dst_weights,
            ffn1_act_scale_per_token,
        ) = ep_moe_expert_dispatch(
            x,
            topk_idx,
            topk_weights,
            getattr(layer, self.added_in_scale_attrs[0], None),
            token_nums_per_expert_list,
            x.shape[0] * layer.top_k,
            self.moe_quant_type,
        )

        if not hasattr(layer, self.added_in_scale_attrs[0]):
            ffn1_act_scale_per_token = None
        ffn_out = self.compute_ffn(
            layer,
            permute_input,
            token_num_lod,
            x.shape[0] * layer.top_k,
            ffn1_act_scale_per_token,
        )

        topk_weights_bf16 = topk_weights.astype("bfloat16")
        tmp_ffn_out = ep_moe_expert_combine(
            ffn_out,
            permute_indices_per_token,
            topk_weights_bf16,
            permute_indices_per_token.shape[0],
            ffn_out.shape[0],
            ffn_out.shape[1],
            permute_indices_per_token.shape[1],
        )

        return tmp_ffn_out

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        apply tp
        """
        if self.moe_quant_type in ["w16a16"]:
            fused_moe_out = self.apply_tp_fused_op(layer, x, gate)
        else:
            fused_moe_out = self.apply_tp_scatter_op(layer, x, gate)

        return fused_moe_out

    def compute_ffn(
        self,
        layer: nn.Layer,
        permute_input,
        token_num_lod,
        valid_token_num,
        ffn1_act_scale_per_token=None,
    ):
        """
        Calculate moe
        """
        if self.moe_quant_type in ["w4a8"]:
            hadamard_block_size = getattr(layer.moe_quant_config, "hadamard_block_size", 128)
        else:
            hadamard_block_size = -1
        ffn_out = moe_expert_ffn(
            permute_input,
            token_num_lod,
            getattr(layer, self.added_weight_attrs[0]),
            getattr(layer, self.added_weight_attrs[1]),
            None,
            None,
            ffn1_act_scale_per_token,
            getattr(layer, self.added_in_scale_attrs[1], None),
            getattr(layer, self.added_scale_attrs[0], None),
            getattr(layer, self.added_scale_attrs[1], None),
            None,
            None,
            self.moe_quant_type,
            hadamard_block_size,
            valid_token_num,
        )
        return ffn_out

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
        # 2. Dynamic compute blockwise quantization scales
        # x, x_scale_tensor = fastdeploy.model_executor.ops.xpu.per_token_quant(x)
        x_scale_tensor = None
        # 3. EP Dispatch
        (
            recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            recv_num_tokens_per_expert_list,
            _,
        ) = self.ep_prefill_runner.dispatch(
            x,
            topk_idx,
            topk_weights,
            x_scale_tensor=x_scale_tensor,
        )

        token_num_per_expert = recv_num_tokens_per_expert_list.numpy().tolist()
        token_all_num = sum(token_num_per_expert)

        # 4. Compute ffn
        moe_dispatch_scale = None
        (
            permute_input,
            permute_indices_per_token,
            token_num_lod,
            dst_weights,
            ffn1_act_scale_per_token,
        ) = ep_moe_expert_dispatch(
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            moe_dispatch_scale,
            token_num_per_expert,
            token_all_num,
            self.moe_quant_type,
        )

        ffn_out = self.compute_ffn(
            layer,
            permute_input,
            token_num_lod,
            token_all_num,
        )

        # prmt back per rank
        recv_topk_weights_bf16 = recv_topk_weights.astype("bfloat16")
        tmp_ffn_out = ep_moe_expert_combine(
            ffn_out,
            permute_indices_per_token,
            recv_topk_weights_bf16,
            permute_indices_per_token.shape[0],
            ffn_out.shape[0],
            ffn_out.shape[1],
            permute_indices_per_token.shape[1],
        )

        # 5. EP combine
        handle = None
        return self.ep_prefill_runner.combine(tmp_ffn_out, handle, recv_topk_weights)

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

        # 1. Select topk experts and weights
        topk_idx, topk_weights = self.ep_decoder_runner.moe_select(layer, gate_out)

        # 2. EP Dispatch
        expertwise_scale = None
        use_fp8 = False
        (
            permute_input,
            token_nums_per_expert,
            handle,
            valid_token_num,
        ) = self.ep_decoder_runner.dispatch(
            x,
            topk_idx,
            topk_weights,
            expertwise_scale=expertwise_scale,
            use_fp8=use_fp8,
        )

        # 3. Compute ffn
        ffn_out = self.compute_ffn(
            layer,
            permute_input,
            token_nums_per_expert,
            valid_token_num,
        )

        # 4. EP combine
        return self.ep_decoder_runner.combine(
            ffn_out,
            topk_idx,
            topk_weights,
            handle,
        )

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        compute Fused MoE.
        """
        if layer.ep_size > 1:
            if layer.fd_config.model_config.moe_phase.phase == "prefill":
                return self.apply_ep_prefill(layer, x, gate)
            elif layer.fd_config.model_config.moe_phase.phase == "decode":
                return self.apply_ep_decode(layer, x, gate)
            else:
                raise ValueError(f"Unsupported phase: {layer.fd_config.model_config.moe_phase.phase}")
        else:
            return self.apply_tp(layer, x, gate)


class XPUWeightOnlyMoEMethod(XPUMoEMethod):
    """
    XPU Fused MoE Method.
    """

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle xpu load weight process.
        """
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)
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

        for idx, weight_tensor in enumerate([up_gate_proj_weights, down_proj_weights]):
            weight_name = self.added_weight_attrs[idx]
            scale_name = self.added_scale_attrs[idx]

            weight_list = []
            weight_scale_list = []
            for i in range(layer.num_local_experts):
                quant_weight, scale = weight_quantize_xpu(
                    weight_tensor[i], self.moe_quant_type, -1, -1
                )  # quant_weight is [k,n]
                # transpose quant_weight to [n,k]
                weight_list.append(quant_weight.transpose([1, 0]))
                weight_scale_list.append(scale)

            quanted_weight = paddle.stack(weight_list, axis=0)
            getattr(layer, weight_name).set_value(quanted_weight)
            quanted_weight_scale = paddle.stack(weight_scale_list, axis=0)
            getattr(layer, scale_name).set_value(quanted_weight_scale)

    def process_weights_after_loading(self, layer):
        """ """
        if not self.quant_config.is_checkpoint_bf16:
            return
        weight_id_map = {"gate_up": 0, "down": 1}
        if (
            hasattr(layer.up_gate_proj_weight, "tensor_track")
            and layer.up_gate_proj_weight.tensor_track is not None
            and layer.up_gate_proj_weight.tensor_track.is_fully_copied()
        ):
            weight_type = "gate_up"
        else:
            weight_type = "down"

        # 1.init shape and type
        # weight
        weight_name = self.added_weight_attrs[weight_id_map[weight_type]]
        unquantized_weight_name = weight_name.replace("quant_weight", "weight")
        if weight_type == "gate_up":
            weight_shape = [
                layer.num_local_experts,
                layer.moe_intermediate_size * 2,
                layer.hidden_size,
            ]
        else:
            weight_shape = [
                layer.num_local_experts,
                layer.hidden_size,
                layer.moe_intermediate_size,
            ]
        weight_dtype = "int8"
        # scale
        scale_name = self.added_scale_attrs[weight_id_map[weight_type]]
        scale_shape = self.up_gate_proj_scale_shape if weight_type == "gate_up" else self.down_proj_scale_shape
        if self.moe_quant_type in ["weight_only_int4"]:
            weight_shape[-1] //= 2
        scale_dtype = "float32"

        # 2.crate tmp tensor

        # weight = paddle.empty(weight_shape, dtype=weight_dtype)
        # scale = paddle.empty(scale_shape, dtype=scale_dtype)

        # 3.quantize weight
        weight_list = []
        weight_scale_list = []
        for expert_id in range(layer.num_local_experts):
            quant_weight, scale = weight_quantize_xpu(
                getattr(layer, unquantized_weight_name)[expert_id].transpose([1, 0]), self.moe_quant_type, -1, -1
            )
            weight_list.append(quant_weight.transpose([1, 0]))
            weight_scale_list.append(scale)
        quanted_weight = paddle.stack(weight_list, axis=0)
        quanted_weight_scale = paddle.stack(weight_scale_list, axis=0)

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

        getattr(layer, weight_name).set_value(quanted_weight)
        getattr(layer, scale_name).set_value(quanted_weight_scale)


class XPUW4A8MoEMethod(XPUMoEMethod):
    """
    XPU w4a8 MoE Method
    """

    def paddle_swap_int4_pack_int4_0123_to_int8_1032in_int8(self, weight_tensor: paddle.Tensor) -> paddle.Tensor:
        """
        Pack the last dimension of a tensor into int8 format by combining adjacent int4 values.
        """
        mask = paddle.full_like(weight_tensor, 0x0F, dtype="int8")
        high_4bit = (weight_tensor >> 4) & mask
        low_4bit = weight_tensor & mask
        swapped = (low_4bit << 4) | high_4bit
        return swapped

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        load weight and process.
        """
        (
            up_gate_proj_weights,
            down_proj_weights,
            logical_expert_ids,
            ep_rank_to_expert_id_list,
        ) = layer.extract_moe_ffn_weights(state_dict)
        assert len(up_gate_proj_weights) == layer.num_local_experts
        assert len(down_proj_weights) == layer.num_local_experts
        assert up_gate_proj_weights[0].shape == [
            layer.hidden_size // 2,
            layer.moe_intermediate_size * 2,
        ]
        assert down_proj_weights[0].shape == [
            layer.moe_intermediate_size // 2,
            layer.hidden_size,
        ]

        for idx, weight_tensor in enumerate([up_gate_proj_weights, down_proj_weights]):
            weight_name = self.added_weight_attrs[idx]
            weight_list = []
            for i in range(layer.num_local_experts):
                weight_list.append(weight_tensor[i].transpose([1, 0]))  # transpone to [n, k]
            quanted_weight = paddle.stack(weight_list, axis=0)
            getattr(layer, weight_name).set_value(quanted_weight)

        self.load_w4a8_scale_weights(
            layer,
            layer.weight_key_map,
            state_dict,
            logical_expert_ids,
        )

    def load_w4a8_scale_weights(
        self,
        layer: nn.Layer,
        weight_key_map: dict,
        state_dict: dict,
        logical_expert_ids: paddle.Tensor,
    ):
        """
        Get w4a8 weights from state dict and process them.
        Args:
            layer (nn.Layer): The layer to add parameters to.
            weight_key_map (dict): The weight key map.
            state_dict (dict): The state dict.
        """

        def _extract_scale_tensor(
            layer: nn.Layer,
            state_dict,
            key_template,
            expert_idx,
        ):
            return get_tensor(
                (
                    state_dict.pop(key_template.format(expert_idx))
                    if key_template.format(expert_idx) in state_dict
                    else key_template.format(expert_idx)
                ),
                layer.fd_config.model_config.model,
            )

        # 1. Init scale containers and maps
        up_gate_proj_weight_scales = []
        down_proj_weight_scales = []
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

        for expert_idx in logical_expert_ids:
            for name, scale_key_template in scale_key_map.items():
                scale_tensor = _extract_scale_tensor(
                    layer,
                    state_dict,
                    scale_key_template,
                    expert_idx,
                )
                scale_weight_map[name].append(scale_tensor)

        # 2. Process scale tensor and set to layer
        for in_scale_name in self.added_in_scale_attrs:
            getattr(layer, in_scale_name).set_value(paddle.concat(scale_weight_map[in_scale_name]))

        for weight_scale_name in self.added_scale_attrs:
            getattr(layer, weight_scale_name).set_value(paddle.stack(scale_weight_map[weight_scale_name], axis=0))
