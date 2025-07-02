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
from fastdeploy.distributed.communication_op import \
    tensor_model_parallel_all_reduce
from fastdeploy.model_executor.layers.utils import (create_hadamard_matrix_map,
                                                    get_tensor)
from fastdeploy.utils import ceil_div

from ..quantization.quant_base import QuantMethodBase

try:
    from fastdeploy.model_executor.ops.gpu import tritonmoe_preprocess_func

    from .triton_moe_kernels import fused_moe_kernel_paddle
except:
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
        self.added_weight_attrs = ["moe_ffn1_weight", "moe_ffn2_weight"]
        self.added_scale_attrs = [
            "moe_ffn1_weight_scale", "moe_ffn2_weight_scale"
        ]

    def process_prequanted_weights(self, layer: nn.Layer, state_dict) -> None:
        """process_prequanted_weights"""
        pass

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Triton MoE create weight process.
        """
        ffn1_weights, ffn2_weights = layer.extract_moe_ffn_weights(state_dict)
        assert len(ffn1_weights) == layer.num_local_experts
        assert len(ffn2_weights) == layer.num_local_experts

        algo = layer.quant_method.quant_config.name()

        assert algo == "wint8"

        assert ffn1_weights[0].shape == [
            layer.hidden_size, layer.moe_intermediate_size * 2
        ]
        assert ffn2_weights[0].shape == [
            layer.moe_intermediate_size, layer.hidden_size
        ]

        ffn1_tensor = paddle.stack(ffn1_weights, axis=0)
        ffn2_tensor = paddle.stack(ffn2_weights, axis=0)

        if algo == "wint8":
            max_bound = 127
        elif algo == "wint4":
            max_bound = 7

        for idx, weight_tensor in enumerate([ffn1_tensor, ffn2_tensor]):
            weight_name = self.added_weight_attrs[idx]
            scale_name = self.added_scale_attrs[idx]

            quanted_weight_scale = weight_tensor.abs().max(axis=1)
            quanted_weight = weight_tensor / quanted_weight_scale[:,
                                                                  None, :] * max_bound
            quanted_weight = paddle.round(quanted_weight).astype("int8")
            quanted_weight_scale = quanted_weight_scale / max_bound

            setattr(
                layer, weight_name,
                layer.create_parameter(
                    shape=quanted_weight.shape,
                    dtype=quanted_weight.dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ))
            getattr(layer, weight_name).set_value(quanted_weight)

            setattr(
                layer, scale_name,
                layer.create_parameter(
                    shape=quanted_weight_scale.shape,
                    dtype=quanted_weight_scale.dtype,
                ))
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
        intermediate_cache1 = paddle.empty(
            [token_num * top_k, moe_intermediate_size * 2],
            dtype=x.dtype,
        )
        intermediate_cache2 = paddle.empty(
            (token_num * top_k, moe_intermediate_size),
            dtype=x.dtype,
        )
        intermediate_cache3 = paddle.empty(
            (token_num * top_k, hidden_size),
            dtype=x.dtype,
        )

        config = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
        }
        sorted_token_ids, expert_ids, num_tokens_post_padded = tritonmoe_preprocess_func(
            topk_ids, num_local_experts, config["BLOCK_SIZE_M"])
        max_possible_num_post_padded = sorted_token_ids.shape[0]
        grid = (
            ceil_div(max_possible_num_post_padded, config["BLOCK_SIZE_M"]) *
            ceil_div(moe_intermediate_size * 2, config["BLOCK_SIZE_N"]), )

        fused_moe_kernel_paddle[grid](
            x,
            layer.moe_ffn1_weight,
            intermediate_cache1,
            None,
            layer.moe_ffn1_weight_scale,
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
            stride_be=layer.moe_ffn1_weight.strides[0],
            stride_bk=layer.moe_ffn1_weight.strides[1],
            stride_bn=layer.moe_ffn1_weight.strides[2],
            stride_cm=intermediate_cache1.strides[0],
            stride_cn=intermediate_cache1.strides[1],
            #
            stride_asm=-1,
            stride_ask=-1,
            stride_bse=layer.moe_ffn1_weight_scale.strides[0],
            stride_bsk=-1,
            stride_bsn=layer.moe_ffn1_weight_scale.strides[1],
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

        intermediate_cache2 = paddle.incubate.nn.functional.swiglu(
            intermediate_cache1)

        grid = (
            ceil_div(max_possible_num_post_padded, config["BLOCK_SIZE_M"]) *
            ceil_div(hidden_size, config["BLOCK_SIZE_N"]), )
        fused_moe_kernel_paddle[grid](
            intermediate_cache2,
            layer.moe_ffn2_weight,
            intermediate_cache3,
            None,
            layer.moe_ffn2_weight_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            max_possible_num_post_padded,
            token_num * top_k,
            N=hidden_size,
            K=moe_intermediate_size,
            stride_am=intermediate_cache2.strides[0],
            stride_ak=intermediate_cache2.strides[1],
            stride_be=layer.moe_ffn2_weight.strides[0],
            stride_bk=layer.moe_ffn2_weight.strides[1],
            stride_bn=layer.moe_ffn2_weight.strides[2],
            stride_cm=intermediate_cache3.strides[0],
            stride_cn=intermediate_cache3.strides[1],
            stride_asm=-1,
            stride_ask=-1,
            stride_bse=layer.moe_ffn2_weight_scale.strides[0],
            stride_bsk=-1,
            stride_bsn=layer.moe_ffn2_weight_scale.strides[1],
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

        intermediate_cache3.reshape_([token_num, top_k, hidden_size])
        out = intermediate_cache3.sum(axis=1)
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

        ffn1_tensor, ffn2_tensor = layer.extract_moe_ffn_weights(state_dict)
        assert ffn1_tensor[0].shape == [
            layer.hidden_size, layer.moe_intermediate_size * 2
        ]
        assert ffn2_tensor[0].shape == [
            layer.moe_intermediate_size, layer.hidden_size
        ]

        ffn1_tensor = paddle.stack(ffn1_tensor, axis=0)
        ffn2_tensor = paddle.stack(ffn2_tensor, axis=0)

        added_wfp8afp8_attrs = [
            "moe_ffn1_weight", "moe_ffn2_weight", "moe_ffn1_weight_scale",
            "moe_ffn2_weight_scale", "moe_ffn1_in_scale", "moe_ffn2_in_scale"
        ]

        def _extract_scale_tensor(key_template):
            result = []
            for i in range(layer.num_experts):
                result.append(
                    get_tensor(state_dict.pop(key_template.format(i))))
            return paddle.concat(result).cast("float32")

        weight_key_map = layer.weight_key_map
        moe_ffn1_weight_scale = _extract_scale_tensor(
            weight_key_map["ffn1_expert_weight_scale_key"])
        moe_ffn2_weight_scale = _extract_scale_tensor(
            weight_key_map["ffn2_expert_weight_scale_key"])
        moe_ffn1_in_scale = _extract_scale_tensor(
            weight_key_map["ffn1_expert_in_scale_key"])
        moe_ffn2_in_scale = _extract_scale_tensor(
            weight_key_map["ffn2_expert_in_scale_key"])

        for idx, weight_tensor in enumerate([
                ffn1_tensor, ffn2_tensor, moe_ffn1_weight_scale,
                moe_ffn2_weight_scale, moe_ffn1_in_scale, moe_ffn2_in_scale
        ]):
            name = added_wfp8afp8_attrs[idx]
            setattr(
                layer, name,
                layer.create_parameter(
                    shape=weight_tensor.shape,
                    dtype=weight_tensor.dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ))
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

        scores = paddle.nn.functional.softmax(gate_out, axis=-1)

        topk_weights, topk_ids = paddle.topk(scores,
                                             k=top_k,
                                             axis=-1,
                                             sorted=False)
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdim=True)

        intermediate_cache1 = paddle.empty(
            [token_num * top_k, moe_intermediate_size * 2],
            dtype=x.dtype,
        )
        intermediate_cache2 = paddle.empty(
            (token_num * top_k, moe_intermediate_size),
            dtype=x.dtype,
        )
        intermediate_cache3 = paddle.empty(
            (token_num * top_k, hidden_size),
            dtype=x.dtype,
        )

        config = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
        }

        sorted_token_ids, expert_ids, num_tokens_post_padded = tritonmoe_preprocess_func(
            topk_ids, num_local_experts, config["BLOCK_SIZE_M"])
        max_possible_num_post_padded = sorted_token_ids.shape[0]
        grid = (
            ceil_div(max_possible_num_post_padded, config["BLOCK_SIZE_M"]) *
            ceil_div(moe_intermediate_size * 2, config["BLOCK_SIZE_N"]), )

        adamard_matrix = create_hadamard_matrix_map[hidden_size]
        x = paddle.matmul(x.cast("float32"), adamard_matrix)

        permute_x = x[:, None, :].tile([1, top_k, 1])
        permute_x = permute_x.reshape([-1, hidden_size])

        quant_activation_scale = layer.moe_ffn1_in_scale[topk_ids].reshape(
            [-1, 1])
        permute_x = permute_x / quant_activation_scale
        permute_x = permute_x.astype("float8_e4m3fn")

        fused_moe_kernel_paddle[grid](
            permute_x,
            layer.moe_ffn1_weight.view(paddle.float8_e4m3fn),
            intermediate_cache1,
            layer.moe_ffn1_in_scale,
            layer.moe_ffn1_weight_scale,
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
            stride_be=layer.moe_ffn1_weight.strides[0],
            stride_bk=layer.moe_ffn1_weight.strides[1],
            stride_bn=layer.moe_ffn1_weight.strides[2],
            stride_cm=intermediate_cache1.strides[0],
            stride_cn=intermediate_cache1.strides[1],
            #
            stride_asm=-1,  # only used in blockwise fp8
            stride_ask=-1,  # only used in blockwise fp8
            stride_bse=-1,
            stride_bsk=-1,
            stride_bsn=-1,
            group_n=-1,
            group_k=-1,
            # Meta-parameters
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            MUL_ROUTED_WEIGHT=False,
            top_k=1,
            compute_type_enum=1,
            use_fp8_w8a8=True,
            use_int8_w8a16=False,
            even_Ks=hidden_size % config["BLOCK_SIZE_K"] == 0,
        )

        intermediate_cache2 = paddle.incubate.nn.functional.swiglu(
            intermediate_cache1)

        hadamard_matrix = create_hadamard_matrix_map[moe_intermediate_size]
        intermediate_cache2 = paddle.matmul(
            intermediate_cache2.cast("float32"), hadamard_matrix)
        quant_activation_scale = layer.moe_ffn2_in_scale[topk_ids].reshape(
            [-1, 1])
        intermediate_cache2 = intermediate_cache2 / quant_activation_scale
        intermediate_cache2 = intermediate_cache2.astype("float8_e4m3fn")

        grid = (
            ceil_div(max_possible_num_post_padded, config["BLOCK_SIZE_M"]) *
            ceil_div(hidden_size, config["BLOCK_SIZE_N"]), )

        fused_moe_kernel_paddle[grid](
            intermediate_cache2,
            layer.moe_ffn2_weight.view(paddle.float8_e4m3fn),
            intermediate_cache3,
            layer.moe_ffn2_in_scale,
            layer.moe_ffn2_weight_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            max_possible_num_post_padded,
            token_num * top_k,
            N=hidden_size,
            K=moe_intermediate_size,
            stride_am=intermediate_cache2.strides[0],
            stride_ak=intermediate_cache2.strides[1],
            stride_be=layer.moe_ffn2_weight.strides[0],
            stride_bk=layer.moe_ffn2_weight.strides[1],
            stride_bn=layer.moe_ffn2_weight.strides[2],
            stride_cm=intermediate_cache3.strides[0],
            stride_cn=intermediate_cache3.strides[1],
            stride_asm=-1,
            stride_ask=-1,
            stride_bse=-1,
            stride_bsk=-1,
            stride_bsn=-1,
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
            even_Ks=moe_intermediate_size % config["BLOCK_SIZE_K"] == 0,
        )

        intermediate_cache3.reshape_([token_num, top_k, hidden_size])
        out = intermediate_cache3.sum(axis=1)

        if layer.tp_size > 1:
            tensor_model_parallel_all_reduce(out)

        return out
