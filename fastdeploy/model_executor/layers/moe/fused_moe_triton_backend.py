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
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.utils import (
    TensorTracker,
    free_tensor,
    process_weight_transpose,
    set_weight_attrs,
    weight_fully_copied,
)
from fastdeploy.utils import ceil_div

from ..quantization.quant_base import QuantMethodBase

try:
    from fastdeploy.model_executor.ops.gpu import tritonmoe_preprocess_func

    from .triton_moe_kernels import fused_moe_kernel_paddle
except ImportError:
    pass
from fastdeploy.model_executor.layers.moe.moe import get_moe_scores
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
    ) -> paddle.Tensor:
        """
        Triton compute Fused MoE.
        """
        token_num = x.shape[0]
        if token_num == 0:
            return paddle.zeros([token_num, layer.hidden_size], dtype=x.dtype)
        gate_out = gate(x.cast("float32"))
        top_k = layer.top_k
        num_local_experts = layer.num_local_experts
        top_k = layer.top_k
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
                top_k,
                True,  # apply_norm_weight,
                False,
            )
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
    ) -> paddle.Tensor:
        """
        Triton compute Fused MoE.
        """
        token_num = x.shape[0]
        if token_num == 0:
            return paddle.zeros([token_num, layer.hidden_size], dtype=x.dtype)
        gate_out = gate(x.cast("float32"))
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
    ) -> paddle.Tensor:
        """
        Triton compute Fused MoE.
        """
        token_num = x.shape[0]
        if token_num == 0:
            return paddle.zeros([token_num, layer.hidden_size], dtype=x.dtype)
        gate_out = gate(x.cast("float32"))
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
                        "tensor_track": TensorTracker(
                            shape=up_gate_proj_weight_shape,
                            output_dim=False,
                        ),
                    }
                    down_proj_attrs = {
                        **extra_weight_attrs,
                        "tensor_track": TensorTracker(
                            shape=down_proj_weight_shape,
                            output_dim=False,
                        ),
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
            scale_dtype = "float32"

            # 2.crate tmp tensor

            weight = paddle.empty(shape=[weight_shape[0], weight_shape[2], weight_shape[1]], dtype=weight_dtype)
            scale = paddle.empty(shape=[scale_shape[0], scale_shape[2], scale_shape[1]], dtype=scale_dtype)

            # 3.quantize weight
            from fastdeploy.model_executor.layers.utils import per_block_cast_to_fp8

            for expert_id in range(layer.num_local_experts):
                weight_quant, scale[expert_id] = per_block_cast_to_fp8(
                    getattr(layer, unquantized_weight_name)[expert_id], self.quant_config.weight_block_size
                )
                weight[expert_id].copy_(weight_quant, False)

            free_tensor(getattr(layer, unquantized_weight_name))

            # create weight
            setattr(
                layer,
                weight_name,
                layer.create_parameter(
                    shape=weight.shape,
                    dtype=weight_dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            # create scale
            setattr(
                layer,
                scale_name,
                layer.create_parameter(
                    shape=scale.shape,
                    dtype=scale_dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            getattr(layer, weight_name).copy_(weight.transpose([0, 2, 1]).contiguous(), False)
            getattr(layer, scale_name).copy_(scale.transpose([0, 2, 1]).contiguous(), False)

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
            else:
                return

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
    ) -> paddle.Tensor:
        """
        Triton compute Fused MoE.
        """
        token_num = x.shape[0]
        if token_num == 0:
            return paddle.zeros([token_num, layer.hidden_size], dtype=x.dtype)
        gate_out = gate(x.cast("float32"))
        top_k = layer.top_k
        num_local_experts = layer.num_local_experts
        moe_intermediate_size = layer.moe_intermediate_size
        hidden_size = layer.hidden_size
        E, N1, _ = getattr(layer, self.added_weight_attrs[0]).shape
        N2 = getattr(layer, self.added_weight_attrs[1]).shape[1]

        topk_ids, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
            gate_out,
            layer.gate_correction_bias,
            layer.top_k,
            True,  # apply_norm_weight
            False,
        )

        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": self.quant_config.weight_block_size[1],
            "BLOCK_SIZE_K": self.quant_config.weight_block_size[0],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }
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

        x_q, x_scale = fastdeploy.model_executor.ops.gpu.per_token_quant(x, self.quant_config.weight_block_size[0])

        fused_moe_kernel_paddle[grid](
            x_q,
            getattr(layer, self.added_weight_attrs[0]),
            intermediate_cache1,
            x_scale,
            getattr(layer, self.added_scale_attrs[0]),
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
            stride_be=getattr(layer, self.added_weight_attrs[0]).strides[0],
            stride_bk=getattr(layer, self.added_weight_attrs[0]).strides[2],
            stride_bn=getattr(layer, self.added_weight_attrs[0]).strides[1],
            stride_cm=intermediate_cache1.strides[0],
            stride_cn=intermediate_cache1.strides[1],
            #
            stride_asm=x_scale.strides[0],  # only used in blockwise fp8
            stride_ask=x_scale.strides[1],  # only used in blockwise fp8
            stride_bse=getattr(layer, self.added_scale_attrs[0]).strides[0],
            stride_bsk=getattr(layer, self.added_scale_attrs[0]).strides[2],
            stride_bsn=getattr(layer, self.added_scale_attrs[0]).strides[1],
            group_n=self.quant_config.weight_block_size[1],
            group_k=self.quant_config.weight_block_size[0],
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

        grid = (
            ceil_div(max_num_tokens_padded, config["BLOCK_SIZE_M"]) * ceil_div(hidden_size, config["BLOCK_SIZE_N"]),
        )

        x_q, x_scale = fastdeploy.model_executor.ops.gpu.per_token_quant(
            intermediate_cache2, self.quant_config.weight_block_size[0]
        )

        fused_moe_kernel_paddle[grid](
            x_q,
            getattr(layer, self.added_weight_attrs[1]),
            intermediate_cache3,
            x_scale,
            getattr(layer, self.added_scale_attrs[1]),
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
            stride_be=getattr(layer, self.added_weight_attrs[1]).strides[0],
            stride_bk=getattr(layer, self.added_weight_attrs[1]).strides[2],
            stride_bn=getattr(layer, self.added_weight_attrs[1]).strides[1],
            stride_cm=intermediate_cache3.strides[0],
            stride_cn=intermediate_cache3.strides[1],
            stride_asm=x_scale.strides[0],  # only used in blockwise fp8
            stride_ask=x_scale.strides[1],  # only used in blockwise fp8
            stride_bse=getattr(layer, self.added_scale_attrs[1]).strides[0],
            stride_bsk=getattr(layer, self.added_scale_attrs[1]).strides[2],
            stride_bsn=getattr(layer, self.added_scale_attrs[1]).strides[1],
            group_n=self.quant_config.weight_block_size[1],
            group_k=self.quant_config.weight_block_size[0],
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

        return out
