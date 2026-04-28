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

from typing import Callable

import paddle
from paddle import nn

import fastdeploy
from fastdeploy import envs
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.utils import (
    TensorTracker,
    free_tensor,
    process_weight_transpose,
    set_weight_attrs,
    weight_fully_copied,
)
from fastdeploy.platforms import current_platform
from fastdeploy.utils import ceil_div, register_custom_python_op

from ..quantization.quant_base import QuantMethodBase

try:
    from fastdeploy.model_executor.ops.gpu import tritonmoe_preprocess_func

    from .triton_moe_kernels import fused_moe_kernel_paddle
except ImportError:
    pass
from fastdeploy.model_executor.layers.moe.moe import get_moe_scores
from fastdeploy.model_executor.layers.quantization.fp8_utils import (
    fused_stack_transpose_quant,
    quant_weight_ue8m0,
    transform_scale_ue8m0,
)
from fastdeploy.model_executor.layers.quantization.ops import scaled_fp8_quant


class TritonWeightOnlyMoEMethod(QuantMethodBase):
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

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        Triton MoE create weight process.
        """
        self.default_dtype = layer._helper.get_default_dtype()
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
                    shape=[layer.num_local_experts, layer.moe_intermediate_size * 2],
                    dtype=self.default_dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            setattr(
                layer,
                down_proj_scale_name,
                layer.create_parameter(
                    shape=[layer.num_local_experts, layer.hidden_size],
                    dtype=self.default_dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            # support cache feature in future

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        Triton MoE load weight process.
        """
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)
        assert len(up_gate_proj_weights) == layer.num_local_experts
        assert len(down_proj_weights) == layer.num_local_experts

        algo = layer.quant_method.quant_config.name()

        assert algo == "wint8"

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

        if algo == "wint8":
            max_bound = 127
        elif algo == "wint4":
            max_bound = 7

        for idx, weight_tensor in enumerate([up_gate_proj_tensor, down_proj_tensor]):
            weight_name = self.added_weight_attrs[idx]
            scale_name = self.added_scale_attrs[idx]

            quanted_weight_scale = weight_tensor.abs().max(axis=1)
            quanted_weight = weight_tensor / quanted_weight_scale[:, None, :] * max_bound
            quanted_weight = paddle.round(quanted_weight).astype("int8")
            quanted_weight_scale = quanted_weight_scale / max_bound

            getattr(layer, weight_name).set_value(quanted_weight)
            getattr(layer, scale_name).set_value(quanted_weight_scale)

    def process_weights_after_loading(self, layer):
        """ """

        def _process_quantize(weight_idx):
            algo = layer.quant_method.quant_config.name()
            assert algo == "wint8"
            max_bound = 127
            # weight
            weight_name = self.added_weight_attrs[weight_id_map[weight_type]]
            # scale
            scale_name = self.added_scale_attrs[weight_id_map[weight_type]]

            weight_tensor = getattr(layer, weight_name)
            quanted_weight_scale = weight_tensor.abs().max(axis=1)
            quanted_weight = weight_tensor / quanted_weight_scale[:, None, :] * max_bound
            quanted_weight = paddle.round(quanted_weight).astype("int8")
            quanted_weight_scale = quanted_weight_scale / max_bound

            free_tensor(getattr(layer, weight_name))

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
            getattr(layer, weight_name).copy_(quanted_weight, False)
            getattr(layer, scale_name).copy_(quanted_weight_scale, False)

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

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
        fc1_latent_proj: nn.Layer = None,
        fc2_latent_proj: nn.Layer = None,
    ) -> paddle.Tensor:
        """
        Triton compute Fused MoE.
        """
        token_num = x.shape[0]
        if token_num == 0:
            return paddle.zeros([token_num, layer.hidden_size], dtype=x.dtype)
        gate_out = gate(x)
        top_k = layer.top_k
        num_local_experts = layer.num_local_experts
        top_k = layer.top_k
        moe_intermediate_size = layer.moe_intermediate_size
        hidden_size = layer.hidden_size

        if layer.topk_method == "noaux_tc":
            use_fused = not fastdeploy.envs.FD_ENABLE_RL and current_platform.is_cuda()
            if not use_fused:
                gate_out = gate_out.cast("float32")
            gate_out, topk_weights, topk_ids = get_moe_scores(
                gate_out,
                layer.n_group,
                layer.topk_group,
                layer.top_k,
                layer.routed_scaling_factor,
                layer.gate_correction_bias,
                getattr(layer, "renormalize", True),
                use_fused_cast=use_fused,
            )
        else:
            gate_out = gate_out.cast("float32")
            topk_ids, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
                gate_out,
                layer.gate_correction_bias,
                top_k,
                True,  # apply_norm_weight,
                False,
            )

        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids=topk_ids)

        up_gate_proj_out = paddle.empty(
            [token_num * top_k, moe_intermediate_size * 2],
            dtype=x.dtype,
        )

        config = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
        }
        sorted_token_ids, expert_ids, num_tokens_post_padded = tritonmoe_preprocess_func(
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
            use_int8_w8a16=True,
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
            use_int8_w8a16=True,
            per_channel_quant=False,
            even_Ks=moe_intermediate_size % config["BLOCK_SIZE_K"] == 0,
        )

        down_proj_out.reshape_([token_num, top_k, hidden_size])
        out = down_proj_out.sum(axis=1)

        return out


class Wfp8Afp8MoEMethod(QuantMethodBase):
    """
    Use Triton Group Gemm to compute Fused wfp8afp8 Quant MoE.
    """

    def __init__(self, quant_config):
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

        raise NotImplementedError

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        Triton MoE create weight process.
        """
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
        self.up_gate_proj_scale_shape = [
            layer.num_local_experts,
            layer.moe_intermediate_size * 2,
            1,
        ]
        self.down_proj_scale_shape = [
            layer.num_local_experts,
            layer.hidden_size,
            1,
        ]
        self.model_format = extra_weight_attrs.get("model_format")
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
            self.weight_dtype = paddle.float8_e4m3fn
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
                    dtype="float32",
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            setattr(
                layer,
                down_proj_scale_name,
                layer.create_parameter(
                    shape=self.down_proj_scale_shape,
                    dtype="float32",
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )

    def process_weights_after_loading(self, layer):
        """ """

        def _process_quantize(weight_idx):
            # weight
            weight_name = self.added_weight_attrs[weight_idx]
            weight_shape = self.up_gate_proj_weight_shape if weight_type == "gate_up" else self.down_proj_weight_shape
            weight_dtype = paddle.float8_e4m3fn
            # scale
            scale_name = self.added_scale_attrs[weight_idx]
            scale_shape = self.up_gate_proj_scale_shape if weight_type == "gate_up" else self.down_proj_scale_shape
            scale_dtype = "float32"

            # 2.crate tmp tensor

            weight = paddle.empty(shape=weight_shape, dtype=weight_dtype)
            scale = paddle.empty(shape=scale_shape, dtype=scale_dtype)

            # 3.quantize weight
            from fastdeploy.model_executor.layers.utils import per_token_cast_to_fp8

            for expert_id in range(layer.num_experts):
                weight_quant, scale[expert_id] = per_token_cast_to_fp8(
                    getattr(layer, weight_name)[expert_id].transpose([1, 0]).contiguous(),
                )
                weight[expert_id].copy_(weight_quant, False)

            free_tensor(getattr(layer, weight_name))

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
            # dynamic quantize
            weight_id_map = {"gate_up": 0, "down": 1}
            if weight_fully_copied(layer.up_gate_proj_weight):
                weight_type = "gate_up"
            else:
                weight_type = "down"
            if self.model_format == "torch":
                # pt model
                process_weight_transpose(layer, self.added_weight_attrs[weight_id_map[weight_type]])

            _process_quantize(weight_id_map[weight_type])
        else:
            return

    def check(self, layer: nn.Layer, up_gate_proj_weights, down_proj_weights):
        """
        check layer is valid for this method
        """
        assert up_gate_proj_weights[0].shape == [
            layer.moe_intermediate_size * 2,
            layer.hidden_size,
        ]
        assert down_proj_weights[0].shape == [
            layer.hidden_size,
            layer.moe_intermediate_size,
        ]

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
        fc1_latent_proj: nn.Layer = None,
        fc2_latent_proj: nn.Layer = None,
    ) -> paddle.Tensor:
        """
        Triton compute Fused MoE.
        """
        token_num = x.shape[0]
        if token_num == 0:
            return paddle.zeros([token_num, layer.hidden_size], dtype=x.dtype)
        gate_out = gate(x)
        gate_out = gate_out.cast("float32")
        top_k = layer.top_k
        num_local_experts = layer.num_local_experts
        moe_intermediate_size = layer.moe_intermediate_size
        hidden_size = layer.hidden_size
        E, N1, _ = getattr(layer, self.added_weight_attrs[0]).shape

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

        config = {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
            "num_warps": 8,
            "num_stages": 4,
        }
        if token_num <= E:
            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 4,
            }

        sorted_token_ids, expert_ids, num_tokens_post_padded = tritonmoe_preprocess_func(
            topk_ids, num_local_experts, config["BLOCK_SIZE_M"]
        )
        max_possible_num_post_padded = sorted_token_ids.shape[0]
        grid = (
            ceil_div(max_possible_num_post_padded, config["BLOCK_SIZE_M"])
            * ceil_div(moe_intermediate_size * 2, config["BLOCK_SIZE_N"]),
        )

        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids=topk_ids)

        up_gate_proj_out = paddle.empty(
            [token_num * top_k, moe_intermediate_size * 2],
            dtype=x.dtype,
        )

        from .triton_moe_kernels import fused_moe_kernel_paddle

        x_q, x_scale = scaled_fp8_quant(x, use_per_token_if_dynamic=True)

        fused_moe_kernel_paddle[grid](
            x_q,
            layer.up_gate_proj_weight,
            up_gate_proj_out,
            x_scale,
            layer.up_gate_proj_weight_scale,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            max_possible_num_post_padded,
            token_num * top_k,
            N=moe_intermediate_size * 2,
            K=hidden_size,
            stride_am=x_q.strides[0],
            stride_ak=x_q.strides[1],
            stride_be=layer.up_gate_proj_weight.strides[0],
            stride_bk=layer.up_gate_proj_weight.strides[2],
            stride_bn=layer.up_gate_proj_weight.strides[1],
            stride_cm=up_gate_proj_out.strides[0],
            stride_cn=up_gate_proj_out.strides[1],
            #
            stride_asm=x_scale.strides[0],
            stride_ask=x_scale.strides[1],
            stride_bse=layer.up_gate_proj_weight_scale.strides[0],
            stride_bsk=layer.up_gate_proj_weight_scale.strides[2],
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
            use_fp8_w8a8=True,
            use_int8_w8a16=False,
            per_channel_quant=True,
            even_Ks=hidden_size % config["BLOCK_SIZE_K"] == 0,
            num_warps=config.get("num_warps", 4),
            num_stages=config.get("num_stages", 4),
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

        x_q, x_scale = scaled_fp8_quant(down_proj_input, use_per_token_if_dynamic=True)

        fused_moe_kernel_paddle[grid](
            x_q,
            layer.down_proj_weight,
            down_proj_out,
            x_scale,
            layer.down_proj_weight_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            max_possible_num_post_padded,
            token_num * top_k,
            N=hidden_size,
            K=moe_intermediate_size,
            stride_am=x_q.strides[0],
            stride_ak=x_q.strides[1],
            stride_be=layer.down_proj_weight.strides[0],
            stride_bk=layer.down_proj_weight.strides[2],
            stride_bn=layer.down_proj_weight.strides[1],
            stride_cm=down_proj_out.strides[0],
            stride_cn=down_proj_out.strides[1],
            stride_asm=x_scale.strides[0],
            stride_ask=x_scale.strides[1],
            stride_bse=layer.down_proj_weight_scale.strides[0],
            stride_bsk=layer.down_proj_weight_scale.strides[2],
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
            use_fp8_w8a8=True,
            use_int8_w8a16=False,
            per_channel_quant=True,
            even_Ks=moe_intermediate_size % config["BLOCK_SIZE_K"] == 0,
            num_warps=config.get("num_warps", 4),
            num_stages=config.get("num_stages", 4),
        )

        down_proj_out.reshape_([token_num, top_k, hidden_size])
        out = down_proj_out.sum(axis=1)

        return out


class TensorWiseFP8MoEMethod(QuantMethodBase):
    """
    Use Triton Group Gemm to compute Fused MoE.
    """

    def __init__(self, quant_method=None):
        """
        Triton Group Gemm to compute Fused MoE.
        """
        self.quant_method = quant_method
        self.added_wfp8afp8_attrs = [
            "up_gate_proj_weight",
            "down_proj_weight",
            "up_gate_proj_weight_scale",
            "down_proj_weight_scale",
            "up_gate_proj_in_scale",
            "down_proj_in_scale",
        ]

    def process_prequanted_weights(self, layer: nn.Layer, state_dict, is_rearrange: bool = False) -> None:
        """process_prequanted_weights"""

        up_gate_proj_tensor, down_proj_tensor = layer.extract_moe_ffn_weights(state_dict)
        assert up_gate_proj_tensor[0].shape == [
            layer.hidden_size,
            layer.moe_intermediate_size * 2,
        ]
        assert down_proj_tensor[0].shape == [
            layer.moe_intermediate_size,
            layer.hidden_size,
        ]

        up_gate_proj_tensor = paddle.stack(up_gate_proj_tensor, axis=0).view(paddle.float8_e4m3fn)
        down_proj_tensor = paddle.stack(down_proj_tensor, axis=0).view(paddle.float8_e4m3fn)

        def _extract_scale_tensor(key_template):
            result = []
            for i in range(layer.num_experts):
                result.append(get_tensor(state_dict.pop(key_template.format(i))))
            return paddle.concat(result).cast("float32")

        weight_key_map = layer.weight_key_map
        up_gate_proj_weight_scale = _extract_scale_tensor(weight_key_map["up_gate_proj_expert_weight_scale_key"])
        down_proj_weight_scale = _extract_scale_tensor(weight_key_map["down_proj_expert_weight_scale_key"])
        up_gate_proj_in_scale = _extract_scale_tensor(weight_key_map["up_gate_proj_expert_in_scale_key"])
        down_proj_in_scale = _extract_scale_tensor(weight_key_map["down_proj_expert_in_scale_key"])

        for idx, weight_tensor in enumerate(
            [
                up_gate_proj_tensor,
                down_proj_tensor,
                up_gate_proj_weight_scale,
                down_proj_weight_scale,
                up_gate_proj_in_scale,
                down_proj_in_scale,
            ]
        ):
            name = self.added_wfp8afp8_attrs[idx]
            if weight_tensor.dtype == paddle.float8_e4m3fn:
                getattr(layer, name).copy_(weight_tensor, False)
            else:
                getattr(layer, name).set_value(weight_tensor)

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        Triton MoE create weight process.
        """
        self.weight_dtype = paddle.float8_e4m3fn
        self.default_dtype = layer._helper.get_default_dtype()
        up_gate_proj_weight_name = self.added_wfp8afp8_attrs[0]
        down_proj_weight_name = self.added_wfp8afp8_attrs[1]
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
        for idx in range(2, len(self.added_wfp8afp8_attrs)):
            setattr(
                layer,
                self.added_wfp8afp8_attrs[idx],
                layer.create_parameter(
                    shape=[layer.num_local_experts],
                    dtype="float32",
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
        fc1_latent_proj: nn.Layer = None,
        fc2_latent_proj: nn.Layer = None,
    ) -> paddle.Tensor:
        """
        Triton compute Fused MoE.
        """
        token_num = x.shape[0]
        if token_num == 0:
            return paddle.zeros([token_num, layer.hidden_size], dtype=x.dtype)
        gate_out = gate(x)
        gate_out = gate_out.cast("float32")
        top_k = layer.top_k
        num_local_experts = layer.num_local_experts
        moe_intermediate_size = layer.moe_intermediate_size
        hidden_size = layer.hidden_size

        topk_ids, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
            gate_out,
            layer.gate_correction_bias,
            top_k,
            True,  # apply_norm_weight,
            False,
        )

        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids=topk_ids)

        up_gate_proj_out = paddle.empty(
            [token_num * top_k, moe_intermediate_size * 2],
            dtype=x.dtype,
        )

        config_up_gate_proj = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 1,
        }

        sorted_token_ids, expert_ids, num_tokens_post_padded = tritonmoe_preprocess_func(
            topk_ids, num_local_experts, config_up_gate_proj["BLOCK_SIZE_M"]
        )
        max_possible_num_post_padded = sorted_token_ids.shape[0]
        grid = (
            ceil_div(
                max_possible_num_post_padded,
                config_up_gate_proj["BLOCK_SIZE_M"],
            )
            * ceil_div(moe_intermediate_size * 2, config_up_gate_proj["BLOCK_SIZE_N"]),
        )

        permute_x = fastdeploy.model_executor.ops.gpu.moe_fused_hadamard_quant_fp8(
            x,
            scale=layer.up_gate_proj_in_scale,
            topk_ids=topk_ids,
            top_k=top_k,
            intermediate_size=hidden_size,
            tiled=False,
        )

        fused_moe_kernel_paddle[grid](
            permute_x,
            layer.up_gate_proj_weight,
            up_gate_proj_out,
            layer.up_gate_proj_in_scale,
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
            stride_asm=-1,  # only used in blockwise fp8
            stride_ask=-1,  # only used in blockwise fp8
            stride_bse=-1,
            stride_bsk=-1,
            stride_bsn=-1,
            group_n=-1,
            group_k=-1,
            # Meta-parameters
            BLOCK_SIZE_M=config_up_gate_proj["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config_up_gate_proj["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config_up_gate_proj["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config_up_gate_proj["GROUP_SIZE_M"],
            MUL_ROUTED_WEIGHT=False,
            top_k=1,
            compute_type_enum=1,
            use_fp8_w8a8=True,
            use_int8_w8a16=False,
            per_channel_quant=False,
            even_Ks=hidden_size % config_up_gate_proj["BLOCK_SIZE_K"] == 0,
        )

        down_proj_input = paddle.incubate.nn.functional.swiglu(up_gate_proj_out)

        down_proj_input = fastdeploy.model_executor.ops.gpu.moe_fused_hadamard_quant_fp8(
            down_proj_input,
            scale=layer.down_proj_in_scale,
            topk_ids=topk_ids,
            top_k=top_k,
            intermediate_size=moe_intermediate_size,
            tiled=True,
        )

        config_down_proj = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }

        down_proj_out = paddle.empty(
            (token_num * top_k, hidden_size),
            dtype=x.dtype,
        )

        grid = (
            ceil_div(max_possible_num_post_padded, config_down_proj["BLOCK_SIZE_M"])
            * ceil_div(hidden_size, config_down_proj["BLOCK_SIZE_N"]),
        )

        fused_moe_kernel_paddle[grid](
            down_proj_input,
            layer.down_proj_weight,
            down_proj_out,
            layer.down_proj_in_scale,
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
            stride_bse=-1,
            stride_bsk=-1,
            stride_bsn=-1,
            group_n=-1,
            group_k=-1,
            # Meta-parameters
            BLOCK_SIZE_M=config_down_proj["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config_down_proj["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config_down_proj["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config_down_proj["GROUP_SIZE_M"],
            MUL_ROUTED_WEIGHT=True,
            top_k=1,
            compute_type_enum=1,
            use_fp8_w8a8=True,
            use_int8_w8a16=False,
            per_channel_quant=False,
            even_Ks=moe_intermediate_size % config_down_proj["BLOCK_SIZE_K"] == 0,
        )

        down_proj_out.reshape_([token_num, top_k, hidden_size])
        out = down_proj_out.sum(axis=1)

        return out


def python_op_fused_moe_kernel_paddle_infer_meta(
    x,
    layer_added_weight_attrs_0,
    layer_added_scale_attrs_0,
    layer_added_weight_attrs1,
    layer_added_scale_attrs1,
    gate_out,
    gate_correction_bias,
    top_k: int,
    N1: int,
    N2: int,
    num_local_experts: int,
    moe_intermediate_size: int,
    hidden_size: int,
    config: dict,
    quant_config,
    topk_ids_hookfunc,
    layer,
    fc1_latent_proj,
    fc2_latent_proj,
):
    token_num = x.shape[0]
    return paddle.static.MetaTensor(shape=[token_num, hidden_size], dtype=x.dtype)


@register_custom_python_op(
    name="python_op_fused_moe_kernel_paddle",
    infer_meta=python_op_fused_moe_kernel_paddle_infer_meta,
    input_names=[
        "x",
        "layer_added_weight_attrs_0",
        "layer_added_scale_attrs_0",
        "layer_added_weight_attrs1",
        "layer_added_scale_attrs1",
        "gate_out",
        "gate_correction_bias",
    ],
    output_names=["out"],
    inplace_map={},
)
def python_op_fused_moe_kernel_paddle(
    x: paddle.Tensor,
    layer_added_weight_attrs_0: paddle.Tensor,
    layer_added_scale_attrs_0: paddle.Tensor,
    layer_added_weight_attrs1: paddle.Tensor,
    layer_added_scale_attrs1: paddle.Tensor,
    gate_out: paddle.Tensor,
    gate_correction_bias: paddle.Tensor,
    top_k: int,
    N1: int,
    N2: int,
    num_local_experts: int,
    moe_intermediate_size: int,
    hidden_size: int,
    config: dict,
    quant_config,
    topk_ids_hookfunc,
    layer,
    fc1_latent_proj,
    fc2_latent_proj,
):

    token_num = x.shape[0]
    if x.shape[0] == 0:
        return paddle.zeros([token_num, hidden_size], dtype=x.dtype)

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
            gate_correction_bias,
            top_k,
            True,  # apply_norm_weight
            False,
        )

    if topk_ids_hookfunc is not None:
        topk_ids_hookfunc(topk_ids=topk_ids)

    from fastdeploy.model_executor.ops.gpu import tritonmoe_preprocess_func

    sorted_token_ids, expert_ids, num_tokens_post_padded = tritonmoe_preprocess_func(
        topk_ids, num_local_experts, config["BLOCK_SIZE_M"]
    )
    # cache13 = create_empty_tensor(tuple([token_num * top_k * max(N1, N2)]), x.dtype)
    cache13 = paddle.empty([token_num * top_k * max(N1, N2)], dtype=x.dtype)
    intermediate_cache1 = cache13[: token_num * top_k * N1].view([token_num * top_k, N1])
    max_num_tokens_padded = sorted_token_ids.shape[0]

    grid = (
        ceil_div(max_num_tokens_padded, config["BLOCK_SIZE_M"])
        * ceil_div(moe_intermediate_size * 2, config["BLOCK_SIZE_N"]),
    )

    from .triton_moe_kernels import fused_moe_kernel_paddle

    if fc1_latent_proj is not None:
        x = fc1_latent_proj(x)

    if not fastdeploy.envs.FD_USE_PHI_FP8_QUANT:
        x_q, x_scale = fastdeploy.model_executor.ops.gpu.per_token_quant(x, quant_config.weight_block_size[0], False)
    else:
        x_q, x_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            x, using_pow2_scale=fastdeploy.envs.FD_FP8_QUANT_WITH_POW2SCALE, output_scale_transpose=False
        )
        x_scale = x_scale[: x.shape[0]]

    fused_moe_kernel_paddle[grid](
        x_q,
        layer_added_weight_attrs_0,
        intermediate_cache1,
        x_scale,
        layer_added_scale_attrs_0,
        None,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        max_num_tokens_padded,
        token_num * top_k,
        N=moe_intermediate_size * 2,
        K=hidden_size,
        stride_am=x_q.strides[0],
        stride_ak=x_q.strides[1],
        stride_be=layer_added_weight_attrs_0.strides[0],
        stride_bk=layer_added_weight_attrs_0.strides[2],
        stride_bn=layer_added_weight_attrs_0.strides[1],
        stride_cm=intermediate_cache1.strides[0],
        stride_cn=intermediate_cache1.strides[1],
        #
        stride_asm=x_scale.strides[0],  # only used in blockwise fp8
        stride_ask=x_scale.strides[1],  # only used in blockwise fp8
        stride_bse=layer_added_scale_attrs_0.strides[0],
        stride_bsk=layer_added_scale_attrs_0.strides[2],
        stride_bsn=layer_added_scale_attrs_0.strides[1],
        group_n=quant_config.weight_block_size[1],
        group_k=quant_config.weight_block_size[0],
        # Meta-parameters
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=False,
        top_k=top_k,
        compute_type_enum=1,
        use_fp8_w8a8=True,
        use_int8_w8a16=False,
        per_channel_quant=False,
        even_Ks=hidden_size % config["BLOCK_SIZE_K"] == 0,
    )

    intermediate_cache2 = paddle.incubate.nn.functional.swiglu(intermediate_cache1)

    intermediate_cache3 = cache13[: token_num * top_k * N2].view([token_num * top_k, N2])

    grid = (ceil_div(max_num_tokens_padded, config["BLOCK_SIZE_M"]) * ceil_div(hidden_size, config["BLOCK_SIZE_N"]),)
    if not fastdeploy.envs.FD_USE_PHI_FP8_QUANT:
        x_q, x_scale = fastdeploy.model_executor.ops.gpu.per_token_quant(
            intermediate_cache2, quant_config.weight_block_size[0], False
        )
    else:
        x_q, x_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            intermediate_cache2,
            using_pow2_scale=fastdeploy.envs.FD_FP8_QUANT_WITH_POW2SCALE,
            output_scale_transpose=False,
        )
        x_scale = x_scale[: x_q.shape[0]]

    fused_moe_kernel_paddle[grid](
        x_q,
        layer_added_weight_attrs1,
        intermediate_cache3,
        x_scale,
        layer_added_scale_attrs1,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        max_num_tokens_padded,
        token_num * top_k,
        N=hidden_size,
        K=moe_intermediate_size,
        stride_am=x_q.strides[0],
        stride_ak=x_q.strides[1],
        stride_be=layer_added_weight_attrs1.strides[0],
        stride_bk=layer_added_weight_attrs1.strides[2],
        stride_bn=layer_added_weight_attrs1.strides[1],
        stride_cm=intermediate_cache3.strides[0],
        stride_cn=intermediate_cache3.strides[1],
        stride_asm=x_scale.strides[0],  # only used in blockwise fp8
        stride_ask=x_scale.strides[1],  # only used in blockwise fp8
        stride_bse=layer_added_scale_attrs1.strides[0],
        stride_bsk=layer_added_scale_attrs1.strides[2],
        stride_bsn=layer_added_scale_attrs1.strides[1],
        group_n=quant_config.weight_block_size[1],
        group_k=quant_config.weight_block_size[0],
        # Meta-parameters
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=True,
        top_k=1,
        compute_type_enum=1,
        use_fp8_w8a8=True,
        use_int8_w8a16=False,
        per_channel_quant=False,
        even_Ks=moe_intermediate_size % config["BLOCK_SIZE_K"] == 0,
    )

    intermediate_cache3.reshape_([token_num, top_k, hidden_size])
    out = intermediate_cache3.sum(axis=1)

    if fc2_latent_proj is not None:
        out = fc2_latent_proj(out)

    return out


class BlockWiseFP8MoEMethod(QuantMethodBase):
    """
    Use Triton Group Gemm to compute Fused BlockWise FP8 Quant MoE.
    """

    def __init__(self, quant_config):
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

        raise NotImplementedError

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        Triton MoE create weight process.
        """
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
        if not self.quant_config.moe_blockwise_gemm_scale_ue8m0:
            self.up_gate_proj_scale_shape = [
                layer.num_local_experts,
                ceil_div(layer.moe_intermediate_size * 2, self.quant_config.weight_block_size[0]),
                ceil_div(layer.hidden_size, self.quant_config.weight_block_size[1]),
            ]
            self.down_proj_scale_shape = [
                layer.num_local_experts,
                ceil_div(layer.hidden_size, self.quant_config.weight_block_size[0]),
                ceil_div(layer.moe_intermediate_size, self.quant_config.weight_block_size[1]),
            ]
        else:
            up_num_scales = ceil_div(
                layer.hidden_size,
                self.quant_config.weight_block_size[1],
            )
            up_num_scale_packs = (up_num_scales + 3) // 4
            self.up_gate_proj_scale_shape = [
                layer.num_local_experts,
                layer.moe_intermediate_size * 2,
                up_num_scale_packs,
            ]
            down_num_scales = ceil_div(
                layer.moe_intermediate_size,
                self.quant_config.weight_block_size[1],
            )
            down_num_scale_packs = (down_num_scales + 3) // 4
            self.down_proj_scale_shape = [
                layer.num_local_experts,
                layer.hidden_size,
                down_num_scale_packs,
            ]
        # TODO(bukejiyu): remove v1 loader check when v0 loader is removed
        self.model_format = extra_weight_attrs.get("model_format")

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
            # offline quant
            # 1.init shape
            extra_weight_attrs = {**extra_weight_attrs}
            if layer.fd_config.load_config.load_choices == "default_v1":
                if self.model_format != "torch":
                    # transpose [0,2,1]
                    up_gate_proj_weight_shape = (
                        self.up_gate_proj_weight_shape[:1] + self.up_gate_proj_weight_shape[1:][::-1]
                    )
                    up_gate_proj_scale_shape = (
                        self.up_gate_proj_scale_shape[:1] + self.up_gate_proj_scale_shape[1:][::-1]
                    )
                    down_proj_weight_shape = self.down_proj_weight_shape[:1] + self.down_proj_weight_shape[1:][::-1]
                    down_proj_scale_shape = self.down_proj_scale_shape[:1] + self.down_proj_scale_shape[1:][::-1]
                    up_gate_proj_attrs = {
                        **extra_weight_attrs,
                    }
                    down_proj_attrs = {
                        **extra_weight_attrs,
                    }
                else:
                    up_gate_proj_weight_shape = self.up_gate_proj_weight_shape
                    up_gate_proj_scale_shape = self.up_gate_proj_scale_shape
                    down_proj_weight_shape = self.down_proj_weight_shape
                    down_proj_scale_shape = self.down_proj_scale_shape
                    up_gate_proj_attrs = {
                        **extra_weight_attrs,
                        "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0},
                    }
                    down_proj_attrs = {
                        **extra_weight_attrs,
                        "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0},
                    }
            else:
                # v0 loader
                up_gate_proj_weight_shape = self.up_gate_proj_weight_shape
                up_gate_proj_scale_shape = self.up_gate_proj_scale_shape
                down_proj_weight_shape = self.down_proj_weight_shape
                down_proj_scale_shape = self.down_proj_scale_shape
                up_gate_proj_attrs = {}
                down_proj_attrs = {}

            self.weight_dtype = paddle.float8_e4m3fn
            self.added_scale_attrs = ["up_gate_proj_weight_scale_inv", "down_proj_weight_scale_inv"]
            up_gate_proj_weight_name = self.added_weight_attrs[0]
            down_proj_weight_name = self.added_weight_attrs[1]
            up_gate_proj_scale_name = self.added_scale_attrs[0]
            down_proj_scale_name = self.added_scale_attrs[1]

            setattr(
                layer,
                up_gate_proj_weight_name,
                layer.create_parameter(
                    shape=up_gate_proj_weight_shape,
                    dtype=self.weight_dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            setattr(
                layer,
                down_proj_weight_name,
                layer.create_parameter(
                    shape=down_proj_weight_shape,
                    dtype=self.weight_dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            # weight_scale
            if not self.quant_config.moe_blockwise_gemm_scale_ue8m0:
                setattr(
                    layer,
                    up_gate_proj_scale_name,
                    layer.create_parameter(
                        shape=up_gate_proj_scale_shape,
                        dtype="float32",
                        default_initializer=paddle.nn.initializer.Constant(0),
                    ),
                )
                setattr(
                    layer,
                    down_proj_scale_name,
                    layer.create_parameter(
                        shape=down_proj_scale_shape,
                        dtype="float32",
                        default_initializer=paddle.nn.initializer.Constant(0),
                    ),
                )
            else:
                setattr(
                    layer,
                    up_gate_proj_scale_name,
                    layer.create_parameter(
                        shape=up_gate_proj_scale_shape,
                        dtype="int32",
                        default_initializer=paddle.nn.initializer.Constant(0),
                    ),
                )
                setattr(
                    layer,
                    down_proj_scale_name,
                    layer.create_parameter(
                        shape=down_proj_scale_shape,
                        dtype="int32",
                        default_initializer=paddle.nn.initializer.Constant(0),
                    ),
                )
            set_weight_attrs(
                getattr(layer, up_gate_proj_weight_name),
                up_gate_proj_attrs,
            )
            set_weight_attrs(
                getattr(layer, up_gate_proj_scale_name),
                up_gate_proj_attrs,
            )

            set_weight_attrs(
                getattr(layer, down_proj_weight_name),
                down_proj_attrs,
            )
            set_weight_attrs(
                getattr(layer, down_proj_scale_name),
                down_proj_attrs,
            )

    def process_weights_after_loading(self, layer):

        def _process_quantize(weight_idx):
            # 1.init shape and type
            self.added_scale_attrs = ["up_gate_proj_weight_scale_inv", "down_proj_weight_scale_inv"]
            # weight
            weight_name = self.added_weight_attrs[weight_idx]
            unquantized_weight_name = weight_name.replace("quant_weight", "weight")
            weight_shape = self.up_gate_proj_weight_shape if weight_type == "gate_up" else self.down_proj_weight_shape
            weight_dtype = paddle.float8_e4m3fn
            # scale
            scale_name = self.added_scale_attrs[weight_idx]
            scale_shape = self.up_gate_proj_scale_shape if weight_type == "gate_up" else self.down_proj_scale_shape

            # 2.crate tmp tensor and 3.quantize weight
            if not self.quant_config.moe_blockwise_gemm_scale_ue8m0:
                scale_dtype = "float32"
                weight = paddle.empty(shape=[weight_shape[0], weight_shape[2], weight_shape[1]], dtype=weight_dtype)
                scale = paddle.empty(shape=[scale_shape[0], scale_shape[2], scale_shape[1]], dtype=scale_dtype)

                from fastdeploy.model_executor.layers.utils import per_block_cast_to_fp8

                for expert_id in range(layer.num_local_experts):
                    weight_quant, scale[expert_id] = per_block_cast_to_fp8(
                        getattr(layer, unquantized_weight_name)[expert_id], self.quant_config.weight_block_size
                    )
                    weight[expert_id].copy_(weight_quant, False)
            else:
                if fastdeploy.envs.FD_USE_PHI_FP8_QUANT:
                    num_expert = layer.num_local_experts
                    expert_weight_list = [getattr(layer, unquantized_weight_name)[i] for i in range(num_expert)]
                    weight = paddle.empty(shape=weight_shape, dtype=weight_dtype)
                    scale_list = []
                    chunk_size = 64

                    for start_idx in range(0, num_expert, chunk_size):
                        end_idx = min(start_idx + chunk_size, num_expert)
                        local_chunk_size = end_idx - start_idx
                        chunk_experts = [w.contiguous() for w in expert_weight_list[start_idx:end_idx]]

                        w1_t_quant, w1_t_scale = fused_stack_transpose_quant(
                            chunk_experts, use_ue8m0=self.quant_config.moe_blockwise_gemm_scale_ue8m0
                        )
                        w1_t_quant = w1_t_quant.reshape([local_chunk_size, -1, w1_t_quant.shape[-1]])
                        w1_t_scale = w1_t_scale.reshape([local_chunk_size, -1, w1_t_scale.shape[-1]])

                        weight[start_idx:end_idx].copy_(w1_t_quant, False)
                        scale_list.append(w1_t_scale)

                    scale = paddle.concat(scale_list, axis=0)
                else:
                    weight = paddle.empty(shape=weight_shape, dtype=weight_dtype)
                    scale_list = []

                    for expert_id in range(layer.num_local_experts):
                        w_q, s_fp32 = quant_weight_ue8m0(
                            weight_dequant=getattr(layer, unquantized_weight_name)[expert_id]
                            .transpose([1, 0])
                            .contiguous(),
                            weight_block_size=self.quant_config.weight_block_size,
                        )
                        s_ue8m0 = transform_scale_ue8m0(
                            s_fp32, mn=w_q.shape[-2], weight_block_size=self.quant_config.weight_block_size
                        )
                        weight[expert_id].copy_(w_q, False)
                        scale_list.append(s_ue8m0)
                    scale = paddle.to_tensor(scale_list)
                scale = scale.transpose([0, 2, 1]).contiguous().transpose([0, 2, 1])

            free_tensor(getattr(layer, unquantized_weight_name))
            free_tensor(getattr(layer, weight_name))
            setattr(
                layer,
                weight_name,
                layer.create_parameter(
                    shape=weight.shape,
                    dtype=weight.dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            setattr(
                layer,
                scale_name,
                layer.create_parameter(
                    shape=scale.shape,
                    dtype=scale.dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )

            if not self.quant_config.moe_blockwise_gemm_scale_ue8m0:
                getattr(layer, weight_name).copy_(weight.transpose([0, 2, 1]).contiguous(), False)
                getattr(layer, scale_name).copy_(scale.transpose([0, 2, 1]).contiguous(), False)
            else:
                getattr(layer, weight_name).copy_(weight, False)
                scale_param = getattr(layer, scale_name)
                scale_param.data = scale
            if bool(envs.FD_USE_BLACKWELL_GEMM):
                import blackwell_ops

                scale_bw = blackwell_ops.unpack_and_convert_scale(scale, None)
                scale_bw_name = scale_name + "_bw"
                setattr(
                    layer,
                    scale_bw_name,
                    scale_bw,
                )
                if layer.fd_config.scheduler_config.splitwise_role != "mixed":
                    setattr(
                        layer,
                        scale_name,
                        None,
                    )

        if self.quant_config.is_checkpoint_bf16:
            # dynamic quantize
            weight_id_map = {"gate_up": 0, "down": 1}
            if weight_fully_copied(layer.up_gate_proj_weight):
                weight_type = "gate_up"
            else:
                weight_type = "down"
            if self.model_format == "torch":
                # pt model
                unquantized_weight_name = self.added_weight_attrs[weight_id_map[weight_type]].replace(
                    "quant_weight", "weight"
                )
                process_weight_transpose(layer, unquantized_weight_name)
            _process_quantize(weight_id_map[weight_type])
        else:
            if self.model_format != "torch":
                up_gate_proj_weight_name = self.added_weight_attrs[0]
                down_proj_weight_name = self.added_weight_attrs[1]
                up_gate_proj_scale_name = self.added_scale_attrs[0]
                down_proj_scale_name = self.added_scale_attrs[1]
                process_weight_transpose(layer, up_gate_proj_weight_name)
                process_weight_transpose(layer, down_proj_weight_name)
                process_weight_transpose(layer, up_gate_proj_scale_name)
                process_weight_transpose(layer, down_proj_scale_name)
            if self.quant_config.moe_blockwise_gemm_scale_ue8m0:
                up_gate_proj_scale = getattr(layer, self.added_scale_attrs[0])
                new_up_gate_proj_scale = paddle.empty(
                    up_gate_proj_scale.shape[:1] + up_gate_proj_scale.shape[1:][::-1], dtype=up_gate_proj_scale.dtype
                )
                new_up_gate_proj_scale = new_up_gate_proj_scale.transpose([0, 2, 1])
                getattr(layer, self.added_scale_attrs[0]).data = new_up_gate_proj_scale
                down_proj_scale = getattr(layer, self.added_scale_attrs[1])
                new_down_proj_scale = paddle.empty(
                    down_proj_scale.shape[:1] + down_proj_scale.shape[1:][::-1], dtype=down_proj_scale.dtype
                )
                new_down_proj_scale = new_down_proj_scale.transpose([0, 2, 1])
                getattr(layer, self.added_scale_attrs[1]).data = new_down_proj_scale

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        Triton MoE create weight process.
        """
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)

        self.check(layer, up_gate_proj_weights, down_proj_weights)
        for idx, weight_tensor in enumerate([up_gate_proj_weights, down_proj_weights]):
            weight_name = self.added_weight_attrs[idx]
            scale_name = self.added_scale_attrs[idx]

            weight_list = []
            weight_scale_list = []
            for i in range(layer.num_local_experts):
                from fastdeploy.model_executor.layers.utils import per_block_cast_to_fp8

                quant_weight, scale = per_block_cast_to_fp8(weight_tensor[i], self.quant_config.weight_block_size)

                weight_list.append(quant_weight)
                weight_scale_list.append(scale)
            quanted_weight = paddle.stack(weight_list, axis=0)
            quanted_weight = quanted_weight.transpose([0, 2, 1]).contiguous().view(paddle.float8_e4m3fn)
            getattr(layer, weight_name).copy_(quanted_weight, False)

            quanted_weight_scale = paddle.stack(weight_scale_list, axis=0)
            quanted_weight_scale = quanted_weight_scale.transpose([0, 2, 1]).contiguous()
            getattr(layer, scale_name).set_value(quanted_weight_scale)

    def check(self, layer: nn.Layer, up_gate_proj_weights, down_proj_weights):
        """
        check layer is valid for this method
        """
        assert up_gate_proj_weights[0].shape == [
            layer.hidden_size,
            layer.moe_intermediate_size * 2,
        ]
        assert down_proj_weights[0].shape == [
            layer.moe_intermediate_size,
            layer.hidden_size,
        ]

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
        fc1_latent_proj: nn.Layer = None,
        fc2_latent_proj: nn.Layer = None,
    ) -> paddle.Tensor:
        """
        Triton compute Fused MoE.
        """

        gate_out = gate(x)
        gate_out = gate_out.cast("float32")
        top_k = layer.top_k
        num_local_experts = layer.num_local_experts
        moe_intermediate_size = layer.moe_intermediate_size
        hidden_size = layer.hidden_size
        E, N1, _ = getattr(layer, self.added_weight_attrs[0]).shape
        N2 = getattr(layer, self.added_weight_attrs[1]).shape[1]

        gate_correction_bias = layer.gate_correction_bias
        # for triton op input
        layer_added_weight_attrs_0 = getattr(layer, self.added_weight_attrs[0])
        layer_added_scale_attrs_0 = getattr(layer, self.added_scale_attrs[0])
        layer_added_weight_attrs1 = getattr(layer, self.added_weight_attrs[1])
        layer_added_scale_attrs1 = getattr(layer, self.added_scale_attrs[1])

        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": self.quant_config.weight_block_size[1],
            "BLOCK_SIZE_K": self.quant_config.weight_block_size[0],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }

        return python_op_fused_moe_kernel_paddle(
            x,
            layer_added_weight_attrs_0,
            layer_added_scale_attrs_0,
            layer_added_weight_attrs1,
            layer_added_scale_attrs1,
            gate_out,
            gate_correction_bias,
            top_k,
            N1,
            N2,
            num_local_experts,
            moe_intermediate_size,
            hidden_size,
            config,
            self.quant_config,
            topk_ids_hookfunc,
            layer,
            fc1_latent_proj,
            fc2_latent_proj,
        )
