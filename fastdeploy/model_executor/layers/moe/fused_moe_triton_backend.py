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
from fastdeploy.distributed.communication_op import tensor_model_parallel_all_reduce
from fastdeploy.model_executor.layers.utils import create_and_set_parameter, get_tensor
from fastdeploy.utils import ceil_div

from ..quantization.quant_base import QuantMethodBase

try:
    from fastdeploy.model_executor.ops.gpu import tritonmoe_preprocess_func

    from .triton_moe_kernels import fused_moe_kernel_paddle
except ImportError:
    pass


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

    def process_prequanted_weights(self, layer: nn.Layer, state_dict) -> None:
        """process_prequanted_weights"""
        pass

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Triton MoE create weight process.
        """
        up_gate_proj_weights, down_proj_weights = layer.extract_moe_ffn_weights(state_dict)
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

            setattr(
                layer,
                weight_name,
                layer.create_parameter(
                    shape=quanted_weight.shape,
                    dtype=quanted_weight.dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            getattr(layer, weight_name).set_value(quanted_weight)

            setattr(
                layer,
                scale_name,
                layer.create_parameter(
                    shape=quanted_weight_scale.shape,
                    dtype=quanted_weight_scale.dtype,
                ),
            )
            getattr(layer, scale_name).set_value(quanted_weight_scale)

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Triton compute Fused MoE.
        """
        token_num = x.shape[0]
        top_k = layer.top_k
        num_local_experts = layer.num_local_experts
        top_k = layer.top_k
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

    def process_prequanted_weights(self, layer: nn.Layer, state_dict) -> None:
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

        added_wfp8afp8_attrs = [
            "up_gate_proj_weight",
            "down_proj_weight",
            "up_gate_proj_weight_scale",
            "down_proj_weight_scale",
            "up_gate_proj_in_scale",
            "down_proj_in_scale",
        ]

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
            name = added_wfp8afp8_attrs[idx]
            setattr(
                layer,
                name,
                layer.create_parameter(
                    shape=weight_tensor.shape,
                    dtype=weight_tensor.dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            if weight_tensor.dtype == paddle.float8_e4m3fn:
                getattr(layer, name).copy_(weight_tensor, False)
            else:
                getattr(layer, name).set_value(weight_tensor)

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Triton MoE create weight process.
        """
        pass

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Triton compute Fused MoE.
        """

        token_num = x.shape[0]
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
            even_Ks=moe_intermediate_size % config_down_proj["BLOCK_SIZE_K"] == 0,
        )

        down_proj_out.reshape_([token_num, top_k, hidden_size])
        out = down_proj_out.sum(axis=1)

        if layer.tp_size > 1:
            tensor_model_parallel_all_reduce(out)

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

    def process_prequanted_weights(self, layer: nn.Layer, state_dict) -> None:
        """process_prequanted_weights"""

        raise NotImplementedError

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Triton MoE create weight process.
        """
        up_gate_proj_weights, down_proj_weights = layer.extract_moe_ffn_weights(state_dict)

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
            quanted_weight = quanted_weight.transpose([0, 2, 1]).contiguous()
            create_and_set_parameter(layer, weight_name, quanted_weight)

            quanted_weight_scale = paddle.stack(weight_scale_list, axis=0)
            quanted_weight_scale = quanted_weight_scale.transpose([0, 2, 1]).contiguous()
            create_and_set_parameter(layer, scale_name, quanted_weight_scale)

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
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Triton compute Fused MoE.
        """

        token_num = x.shape[0]
        top_k = layer.top_k
        num_local_experts = layer.num_local_experts
        moe_intermediate_size = layer.moe_intermediate_size
        hidden_size = layer.hidden_size
        E, N1, _ = layer.up_gate_proj_weight.shape
        N2 = layer.down_proj_weight.shape[1]

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
        from fastdeploy.model_executor.ops.gpu import tritonmoe_preprocess

        sorted_token_ids, expert_ids, num_tokens_post_padded = tritonmoe_preprocess(
            topk_ids, num_local_experts, config["BLOCK_SIZE_M"]
        )
        max_num_tokens_padded = sorted_token_ids.shape[0]

        grid = (
            ceil_div(max_num_tokens_padded, config["BLOCK_SIZE_M"])
            * ceil_div(moe_intermediate_size * 2, config["BLOCK_SIZE_N"]),
        )

        from .triton_moe_kernels import fused_moe_kernel_paddle

        x_q, x_scale = fastdeploy.model_executor.ops.gpu.per_token_quant(x, self.quant_config.weight_block_size[0])

        cache13 = paddle.empty([token_num * top_k * max(N1, N2)], dtype=x.dtype)
        intermediate_cache1 = cache13[: token_num * top_k * N1].view([token_num * top_k, N1])
        intermediate_cache3 = cache13[: token_num * top_k * N2].view([token_num * top_k, N2])

        fused_moe_kernel_paddle[grid](
            x_q,
            layer.up_gate_proj_weight.view(paddle.float8_e4m3fn),
            intermediate_cache1,
            x_scale,
            layer.up_gate_proj_weight_scale,
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
            stride_be=layer.up_gate_proj_weight.strides[0],
            stride_bk=layer.up_gate_proj_weight.strides[2],
            stride_bn=layer.up_gate_proj_weight.strides[1],
            stride_cm=intermediate_cache1.strides[0],
            stride_cn=intermediate_cache1.strides[1],
            #
            stride_asm=x_scale.strides[0],  # only used in blockwise fp8
            stride_ask=x_scale.strides[1],  # only used in blockwise fp8
            stride_bse=layer.up_gate_proj_weight_scale.strides[0],
            stride_bsk=layer.up_gate_proj_weight_scale.strides[2],
            stride_bsn=layer.up_gate_proj_weight_scale.strides[1],
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
            even_Ks=hidden_size % config["BLOCK_SIZE_K"] == 0,
        )

        intermediate_cache2 = paddle.incubate.nn.functional.swiglu(intermediate_cache1)

        grid = (
            ceil_div(max_num_tokens_padded, config["BLOCK_SIZE_M"]) * ceil_div(hidden_size, config["BLOCK_SIZE_N"]),
        )

        x_q, x_scale = fastdeploy.model_executor.ops.gpu.per_token_quant(
            intermediate_cache2, self.quant_config.weight_block_size[0]
        )

        fused_moe_kernel_paddle[grid](
            x_q,
            layer.down_proj_weight.view(paddle.float8_e4m3fn),
            intermediate_cache3,
            x_scale,
            layer.down_proj_weight_scale,
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
            stride_be=layer.down_proj_weight.strides[0],
            stride_bk=layer.down_proj_weight.strides[2],
            stride_bn=layer.down_proj_weight.strides[1],
            stride_cm=intermediate_cache3.strides[0],
            stride_cn=intermediate_cache3.strides[1],
            stride_asm=x_scale.strides[0],  # only used in blockwise fp8
            stride_ask=x_scale.strides[1],  # only used in blockwise fp8
            stride_bse=layer.down_proj_weight_scale.strides[0],
            stride_bsk=layer.down_proj_weight_scale.strides[2],
            stride_bsn=layer.down_proj_weight_scale.strides[1],
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
            even_Ks=moe_intermediate_size % config["BLOCK_SIZE_K"] == 0,
        )

        intermediate_cache3.reshape_([token_num, top_k, hidden_size])
        out = intermediate_cache3.sum(axis=1)

        if layer.tp_size > 1:
            tensor_model_parallel_all_reduce(out)

        return out
