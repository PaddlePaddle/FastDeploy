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
from fastdeploy.model_executor.ops.gpu import (
    MoeWna16MarlinGemmApi,
    tritonmoe_preprocess_func,
)

from ..quantization.quant_base import QuantMethodBase


def gptq_marlin_moe_repack(
    b_q_weight: paddle.Tensor,
    perm: paddle.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> paddle.Tensor:
    """
    Util function.
    """
    from fastdeploy.model_executor.ops.gpu import gptq_marlin_repack

    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = paddle.empty(
        [num_experts, size_k // 16, size_n * (num_bits // 2)],
        dtype=b_q_weight.dtype,
    )
    for e in range(num_experts):
        output[e] = gptq_marlin_repack(b_q_weight[e], perm[e], size_k, size_n, num_bits)
    return output


def get_scale_perms():
    """
    Util function.
    """
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(s: paddle.Tensor, size_k: int, size_n: int, group_size: int) -> paddle.Tensor:
    """
    Util function.
    """
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape([-1, len(scale_perm)])[:, scale_perm]
    else:
        s = s.reshape([-1, len(scale_perm_single)])[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


def marlin_moe_permute_scales(
    s: paddle.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
):
    """
    Util function.
    """
    num_experts = s.shape[0]
    output = paddle.empty(
        [num_experts, s.shape[1], s.shape[2]],
        dtype=s.dtype,
    )

    for e in range(num_experts):
        output[e] = marlin_permute_scales(s[e], size_k, size_n, group_size)
    return output


class MarlinWeightOnlyMoEMethod(QuantMethodBase):
    """
    Use Marlin Group Gemm to compute Fused MoE.
    """

    def __init__(self, quant_method=None):
        """
        Marlin Group Gemm to compute Fused MoE.
        """
        self.quant_method = quant_method
        self.added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
        self.added_scale_attrs = [
            "up_gate_proj_weight_scale",
            "down_proj_weight_scale",
        ]
        self.added_zeros_attrs = ["zeros0", "zeros1"]

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        self.default_dtype = layer._helper.get_default_dtype()
        self.weight_dtype = "int32"

        up_gate_proj_weight_name = self.added_weight_attrs[0]
        down_proj_weight_name = self.added_weight_attrs[1]
        self.up_gate_proj_weight_shape = [
            layer.num_local_experts,
            layer.hidden_size // 16,
            layer.moe_intermediate_size * 4,
        ]
        self.down_proj_weight_shape = [
            layer.num_local_experts,
            layer.moe_intermediate_size // 16,
            layer.hidden_size * 2,
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
        # weight_scale
        setattr(
            layer,
            self.added_scale_attrs[0],
            layer.create_parameter(
                shape=[layer.num_local_experts, 1, layer.moe_intermediate_size * 2],
                dtype=self.default_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            self.added_scale_attrs[1],
            layer.create_parameter(
                shape=[layer.num_local_experts, 1, layer.hidden_size],
                dtype=self.default_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        Marlin MoE load weight process.
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

        up_gate_proj_tensor = paddle.stack(up_gate_proj_weights, axis=0)
        down_proj_tensor = paddle.stack(down_proj_weights, axis=0)

        max_bound = 7

        for idx, weight_tensor in enumerate([up_gate_proj_tensor, down_proj_tensor]):
            weight_name = self.added_weight_attrs[idx]
            scale_name = self.added_scale_attrs[idx]

            weight_scale = weight_tensor.abs().max(axis=1)
            quanted_weight = weight_tensor / weight_scale[:, None, :] * max_bound
            quanted_weight = paddle.round(quanted_weight).astype("int32")

            quanted_weight[quanted_weight > 7] = 7
            quanted_weight[quanted_weight < -7] = -7
            quanted_weight += 8

            E, K, N = quanted_weight.shape
            quanted_weight = quanted_weight.reshape([0, K // 8, 8, N])
            res = paddle.zeros([E, K // 8, N], dtype="int32")
            for j in range(8):
                tmp = quanted_weight[:, :, j, :]
                res = res | (tmp << (j * 4))
            quanted_weight = paddle.assign(res)
            weight_scale = weight_scale / max_bound
            weight_scale = weight_scale[:, None, :]

            group_size = -1  # means per_channel

            g_idx_sort_indices = paddle.empty([E, 0], dtype="int32")
            quanted_weight = gptq_marlin_moe_repack(
                quanted_weight,
                g_idx_sort_indices,
                K,
                N,
                4,
            )

            weight_scale = marlin_moe_permute_scales(
                weight_scale,
                size_k=layer.moe_intermediate_size,  # useless
                size_n=N,
                group_size=group_size,
            )

            for name, tensor in [
                (weight_name, quanted_weight),
                (scale_name, weight_scale),
            ]:
                getattr(layer, name).set_value(tensor)

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Marlin compute Fused MoE.
        """
        gate_out = gate(x.cast("float32"))
        token_num = x.shape[0]
        top_k = layer.top_k
        top_k = layer.top_k
        moe_intermediate_size = layer.moe_intermediate_size
        hidden_size = layer.hidden_size
        num_experts = layer.num_experts
        topk_method = layer.topk_method

        if topk_method == "noaux_tc":
            from fastdeploy.model_executor.layers.moe.moe import get_moe_scores

            _, topk_weights, topk_ids = get_moe_scores(
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

        block_size_m = 64

        for m in [8, 16, 32, 48, 64]:
            if token_num * top_k / num_experts / m < 0.9:
                block_size_m = m
                break

        topk = top_k

        # for H100 132 sms
        workspace = paddle.empty([528], dtype="int32")

        sorted_token_ids, expert_ids, num_tokens_post_padded = tritonmoe_preprocess_func(
            topk_ids, num_experts, block_size_m
        )

        ffn_out = MoeWna16MarlinGemmApi(
            x,
            c_or_none=None,
            b_q_weight=layer.up_gate_proj_weight,
            b_scales=layer.up_gate_proj_weight_scale,
            global_scale_or_none=None,
            b_zeros_or_none=None,
            g_idx_or_none=None,
            perm_or_none=None,
            workspace=workspace,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            topk_weights=topk_weights,
            moe_block_size=block_size_m,
            top_k=topk,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type_str="uint4b8",
            size_m=token_num,
            size_n=moe_intermediate_size * 2,
            size_k=hidden_size,
            is_k_full=True,
            use_atomic_add=True,
            use_fp32_reduce=True,
            is_zp_float=False,
        )[0]

        swiglu_out = paddle.incubate.nn.functional.swiglu(ffn_out)

        ffn_out = MoeWna16MarlinGemmApi(
            swiglu_out,
            c_or_none=None,
            b_q_weight=layer.down_proj_weight,
            b_scales=layer.down_proj_weight_scale,
            global_scale_or_none=None,
            b_zeros_or_none=None,
            g_idx_or_none=None,
            perm_or_none=None,
            workspace=workspace,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            topk_weights=topk_weights,
            moe_block_size=block_size_m,
            top_k=1,
            mul_topk_weights=True,
            is_ep=False,
            b_q_type_str="uint4b8",
            size_m=token_num * topk,
            size_n=hidden_size,
            size_k=moe_intermediate_size,
            is_k_full=True,
            use_atomic_add=True,
            use_fp32_reduce=True,
            is_zp_float=False,
        )[0]

        ffn_out.reshape_([token_num, -1, hidden_size])
        ffn_out = ffn_out.sum(axis=1)

        return ffn_out
