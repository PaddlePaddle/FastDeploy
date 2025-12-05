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

import paddle
from paddle import nn

import fastdeploy
from fastdeploy.model_executor.layers.moe.moe import get_moe_scores
from fastdeploy.model_executor.layers.quantization.quant_base import QuantMethodBase
from fastdeploy.model_executor.ops.gpu import tritonmoe_preprocess
from fastdeploy.model_executor.utils import TensorTracker, set_weight_attrs
from fastdeploy.utils import ceil_div

from .triton_moe_kernels import fused_moe_kernel_paddle


class MetaxTritonWeightOnlyMoEMethod(QuantMethodBase):
    """
    Use Triton Group Gemm to compute Fused MoE.
    """

    def __init__(self, quant_config=None):
        """
        Triton Group Gemm to compute Fused MoE.
        """
        self.quant_config = quant_config
        self.added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
        self.added_scale_attrs = [
            "up_gate_proj_weight_scale",
            "down_proj_weight_scale",
        ]

    def process_prequanted_weights(self, layer: nn.Layer, state_dict, is_rearrange: bool = False) -> None:
        """process_prequanted_weights"""
        pass

    @paddle.no_grad()
    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        Triton MoE create weight process.
        """
        self.weight_dtype = "int8" if self.quant_config is not None else "bfloat16"
        self.default_dtype = layer._helper.get_default_dtype()
        up_gate_proj_weight_name = self.added_weight_attrs[0]
        down_proj_weight_name = self.added_weight_attrs[1]
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
        # TODO(bukejiyu): remove v1 loader check when v0 loader is removed
        is_checkpoint_bf16 = self.quant_config.is_checkpoint_bf16 if self.quant_config is not None else True
        if is_checkpoint_bf16 and layer.fd_config.load_config.load_choices == "default_v1":
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
            extra_weight_attrs["weight_need_transpose"] = extra_weight_attrs.get("model_format") == "torch"

            set_weight_attrs(
                layer.up_gate_proj_weight,
                {
                    **extra_weight_attrs,
                    "tensor_track": TensorTracker(shape=layer.up_gate_proj_weight.shape, output_dim=True),
                },
            )
            set_weight_attrs(
                layer.down_proj_weight,
                {
                    **extra_weight_attrs,
                    "tensor_track": TensorTracker(shape=layer.down_proj_weight.shape, output_dim=False),
                },
            )
        else:
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
                self.added_scale_attrs[0],
                layer.create_parameter(
                    shape=[layer.num_local_experts, layer.moe_intermediate_size * 2],
                    dtype=self.default_dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            setattr(
                layer,
                self.added_scale_attrs[1],
                layer.create_parameter(
                    shape=[layer.num_local_experts, layer.hidden_size],
                    dtype=self.default_dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            # support cache feature in future

    @paddle.no_grad()
    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        Triton MoE load weight process.
        """
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)
        assert len(up_gate_proj_weights) == layer.num_local_experts
        assert len(down_proj_weights) == layer.num_local_experts

        if self.quant_config is not None:
            algo = layer.quant_method.quant_config.name()
            assert algo == "wint8"
            max_bound = 127

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

        for idx, weight_tensor in enumerate([up_gate_proj_tensor, down_proj_tensor]):
            weight_name = self.added_weight_attrs[idx]
            scale_name = self.added_scale_attrs[idx]

            quanted_weight_scale = weight_tensor.abs().max(axis=1)

            if self.quant_config is not None:
                quanted_weight = weight_tensor / quanted_weight_scale[:, None, :] * max_bound
                quanted_weight = paddle.round(quanted_weight).astype("int8")
                quanted_weight_scale = quanted_weight_scale / max_bound

                getattr(layer, weight_name).set_value(quanted_weight)
            else:
                getattr(layer, weight_name).set_value(weight_tensor)

            getattr(layer, scale_name).set_value(quanted_weight_scale)

    @paddle.no_grad()
    def process_weights_after_loading(self, layer):
        """ """
        is_checkpoint_bf16 = self.quant_config.is_checkpoint_bf16 if self.quant_config is not None else True
        if not is_checkpoint_bf16:
            return

        if self.quant_config is not None:
            algo = layer.quant_method.quant_config.name()
            assert algo == "wint8"
            max_bound = 127
        weight_id_map = {"gate_up": 0, "down": 1}
        if (
            hasattr(layer.up_gate_proj_weight, "tensor_track")
            and layer.up_gate_proj_weight.tensor_track is not None
            and layer.up_gate_proj_weight.tensor_track.is_fully_copied()
        ):
            weight_type = "gate_up"
            layer.up_gate_proj_weight.tensor_track = None
        else:
            weight_type = "down"
            layer.down_proj_weight.tensor_track = None

        # weight
        weight_name = self.added_weight_attrs[weight_id_map[weight_type]]
        # scale
        scale_name = self.added_scale_attrs[weight_id_map[weight_type]]

        weight_tensor = getattr(layer, weight_name)
        quanted_weight_scale = weight_tensor.abs().max(axis=1)
        if self.quant_config is not None:
            quanted_weight = weight_tensor / quanted_weight_scale[:, None, :] * max_bound
            quanted_weight = paddle.round(quanted_weight).astype("int8")
            quanted_weight_scale = quanted_weight_scale / max_bound

            getattr(layer, weight_name).value().get_tensor()._clear()
            # create weight
            setattr(
                layer,
                weight_name,
                layer.create_parameter(
                    shape=weight_tensor.shape,
                    dtype=quanted_weight.dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            getattr(layer, weight_name).copy_(quanted_weight, False)

        # create scale
        setattr(
            layer,
            scale_name,
            layer.create_parameter(
                shape=quanted_weight_scale.shape,
                dtype=quanted_weight_scale.dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        getattr(layer, scale_name).copy_(quanted_weight_scale, False)

    @paddle.no_grad()
    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Triton compute Fused MoE.
        """
        gate_out = gate(x.cast("float32"))
        token_num = x.shape[0]
        top_k = layer.top_k
        num_local_experts = layer.num_local_experts
        moe_intermediate_size = layer.moe_intermediate_size
        hidden_size = layer.hidden_size

        if layer.topk_method == "noaux_tc":
            gate_out, topk_weights, topk_ids = get_moe_scores(
                gate_out,
                layer.n_group,
                layer.topk_group,
                layer.top_k,
                layer.routed_scaling_factor,
                layer.gate_correction_bias,
                getattr(layer, "renormalize", True),
            )
        else:
            topk_ids, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
                gate_out,
                layer.gate_correction_bias,
                layer.top_k,
                True,  # apply_norm_weight
                False,
            )
        up_gate_proj_out = paddle.empty(
            [token_num * top_k, moe_intermediate_size * 2],
            dtype=x.dtype,
        )

        config = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 4,
        }
        sorted_token_ids, expert_ids, num_tokens_post_padded = tritonmoe_preprocess(
            topk_ids, num_local_experts, config["BLOCK_SIZE_M"]
        )
        max_possible_num_post_padded = sorted_token_ids.shape[0]
        grid = (
            ceil_div(max_possible_num_post_padded, config["BLOCK_SIZE_M"])
            * ceil_div(moe_intermediate_size * 2, config["BLOCK_SIZE_N"]),
        )

        fused_moe_kernel_paddle[grid](
            x,
            layer.up_gate_proj_weight,
            up_gate_proj_out,
            None,
            layer.up_gate_proj_weight_scale,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            max_possible_num_post_padded,
            token_num * top_k,
            N=moe_intermediate_size * 2,
            K=hidden_size,
            stride_am=x.strides[0],
            stride_ak=x.strides[1],
            stride_be=layer.up_gate_proj_weight.strides[0],
            stride_bk=layer.up_gate_proj_weight.strides[1],
            stride_bn=layer.up_gate_proj_weight.strides[2],
            stride_cm=up_gate_proj_out.strides[0],
            stride_cn=up_gate_proj_out.strides[1],
            #
            stride_asm=-1,
            stride_ask=-1,
            stride_bse=layer.up_gate_proj_weight_scale.strides[0],
            stride_bsk=-1,
            stride_bsn=layer.up_gate_proj_weight_scale.strides[1],
            group_n=-1,
            group_k=-1,
            # Meta-parameters
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            MUL_ROUTED_WEIGHT=False,
            top_k=top_k,
            compute_type_enum=1,
            use_fp8_w8a8=False,
            use_int8_w8a16=True if self.quant_config is not None else False,
            per_channel_quant=False,
            even_Ks=hidden_size % config["BLOCK_SIZE_K"] == 0,
        )

        down_proj_input = paddle.incubate.nn.functional.swiglu(up_gate_proj_out)

        down_proj_out = paddle.empty(
            (token_num * top_k, hidden_size),
            dtype=x.dtype,
        )

        grid = (
            ceil_div(max_possible_num_post_padded, config["BLOCK_SIZE_M"])
            * ceil_div(hidden_size, config["BLOCK_SIZE_N"]),
        )
        fused_moe_kernel_paddle[grid](
            down_proj_input,
            layer.down_proj_weight,
            down_proj_out,
            None,
            layer.down_proj_weight_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            max_possible_num_post_padded,
            token_num * top_k,
            N=hidden_size,
            K=moe_intermediate_size,
            stride_am=down_proj_input.strides[0],
            stride_ak=down_proj_input.strides[1],
            stride_be=layer.down_proj_weight.strides[0],
            stride_bk=layer.down_proj_weight.strides[1],
            stride_bn=layer.down_proj_weight.strides[2],
            stride_cm=down_proj_out.strides[0],
            stride_cn=down_proj_out.strides[1],
            stride_asm=-1,
            stride_ask=-1,
            stride_bse=layer.down_proj_weight_scale.strides[0],
            stride_bsk=-1,
            stride_bsn=layer.down_proj_weight_scale.strides[1],
            group_n=-1,
            group_k=-1,
            # Meta-parameters
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            MUL_ROUTED_WEIGHT=True,
            top_k=1,
            compute_type_enum=1,
            use_fp8_w8a8=False,
            use_int8_w8a16=True if self.quant_config is not None else False,
            per_channel_quant=False,
            even_Ks=moe_intermediate_size % config["BLOCK_SIZE_K"] == 0,
        )

        down_proj_out.reshape_([token_num, top_k, hidden_size])
        out = down_proj_out.sum(axis=1)
        return out
