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

from fastdeploy.model_executor.layers.quantization.quant_base import QuantMethodBase
from fastdeploy.utils import ceil_div


class DCUTritonWeightOnlyMoEMethod(QuantMethodBase):
    """
    Use Triton Group Gemm to compute Fused MoE.
    """

    def __init__(self, quant_method=None):
        """
        Triton Group Gemm to compute Fused MoE.
        """
        self.quant_method = quant_method
        self.added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
        self.added_scale_attrs = [
            "up_gate_proj_weight_scale",
            "down_proj_weight_scale",
        ]

    def process_prequanted_weights(self, layer: nn.Layer, state_dict, is_rearrange: bool = False) -> None:
        """process_prequanted_weights"""
        pass

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Triton MoE create weight process.
        """
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)
        assert len(up_gate_proj_weights) == layer.num_local_experts
        assert len(down_proj_weights) == layer.num_local_experts
        assert self.quant_method.name() == "wint8"
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

        if self.quant_method.name() == "wint8":
            max_bound = 127
        elif self.quant_method.name() == "wint4":
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
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Triton compute Fused MoE.
        """
        gate_out = gate(x.cast("float32"))
        token_num = x.shape[0]
        top_k = layer.top_k
        num_local_experts = layer.num_local_experts
        top_k = layer.top_k
        moe_intermediate_size = layer.moe_intermediate_size
        hidden_size = layer.hidden_size

        scores = paddle.nn.functional.softmax(gate_out, axis=-1)
        scores += layer.gate_correction_bias
        topk_weights, topk_ids = paddle.topk(scores, k=top_k, axis=-1, sorted=False)
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
        from fastdeploy.model_executor.ops.gpu import tritonmoe_preprocess

        from .triton_moe_kernels import fused_moe_kernel_paddle

        sorted_token_ids, expert_ids, num_tokens_post_padded = tritonmoe_preprocess(
            topk_ids, num_local_experts, config["BLOCK_SIZE_M"]
        )
        max_num_tokens_padded = sorted_token_ids.shape[0]
        grid = (
            ceil_div(max_num_tokens_padded, config["BLOCK_SIZE_M"])
            * ceil_div(moe_intermediate_size * 2, config["BLOCK_SIZE_N"]),
        )

        fused_moe_kernel_paddle[grid](
            x,
            layer.up_gate_proj_weight,
            intermediate_cache1,
            None,
            layer.up_gate_proj_weight_scale,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            moe_intermediate_size * 2,
            hidden_size,
            max_num_tokens_padded,
            token_num * top_k,
            stride_am=x.strides[0],
            stride_ak=x.strides[1],
            stride_be=layer.up_gate_proj_weight.strides[0],
            stride_bk=layer.up_gate_proj_weight.strides[1],
            stride_bn=layer.up_gate_proj_weight.strides[2],
            stride_cm=intermediate_cache1.strides[0],
            stride_cn=intermediate_cache1.strides[1],
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

        intermediate_cache2 = paddle.incubate.nn.functional.swiglu(intermediate_cache1)

        grid = (
            ceil_div(max_num_tokens_padded, config["BLOCK_SIZE_M"]) * ceil_div(hidden_size, config["BLOCK_SIZE_N"]),
        )
        fused_moe_kernel_paddle[grid](
            intermediate_cache2,
            layer.down_proj_weight,
            intermediate_cache3,
            None,
            layer.down_proj_weight_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            hidden_size,
            moe_intermediate_size,
            max_num_tokens_padded,
            token_num * top_k,
            stride_am=intermediate_cache2.strides[0],
            stride_ak=intermediate_cache2.strides[1],
            stride_be=layer.down_proj_weight.strides[0],
            stride_bk=layer.down_proj_weight.strides[1],
            stride_bn=layer.down_proj_weight.strides[2],
            stride_cm=intermediate_cache3.strides[0],
            stride_cn=intermediate_cache3.strides[1],
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

        intermediate_cache3.reshape_([token_num, top_k, hidden_size])
        out = intermediate_cache3.sum(axis=1)
        return out
