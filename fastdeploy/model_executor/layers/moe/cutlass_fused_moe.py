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
from paddle.distributed import fleet
from paddle.framework import in_dynamic_or_pir_mode
from paddle.nn.quant import weight_quantize

from fastdeploy.model_executor.ops.gpu import (moe_expert_dispatch,
                                               moe_expert_ffn,
                                               moe_expert_reduce)

from .fused_moe_method_base import FusedMoEMethodBase


class CutlassFusedMoeMethod(FusedMoEMethodBase):
    """
    Use Cutlass Group Gemm to compute Fused MoE.
    This method is the oldest way to compute MoE in Paddle.
    """

    def create_weights(
            self,
            layer: nn.Layer,
            moe_compute_params,
            ffn1_tensor,
            ffn2_tensor,
            ffn1_bias=None,
            ffn2_bias=None,
            # belows only used in w4a8.
            moe_ffn1_weight_scale=None,
            moe_ffn2_weight_scale=None,
            moe_ffn1_in_scale=None,
            moe_ffn2_in_scale=None):
        """
        Paddle cutlass create weight process.
        """

        num_local_experts = moe_compute_params.num_local_experts
        moe_quant_type = moe_compute_params.moe_quant_type

        assert len(ffn1_tensor) == num_local_experts
        assert len(ffn2_tensor) == num_local_experts
        assert ffn1_tensor[0].shape == [
            moe_compute_params.hidden_size,
            moe_compute_params.moe_intermediate_size * 2
        ]
        assert ffn2_tensor[0].shape == [
            moe_compute_params.moe_intermediate_size,
            moe_compute_params.hidden_size
        ]

        added_weight_attrs = ["moe_ffn1_weight", "moe_ffn2_weight"]
        added_scale_attrs = ["moe_ffn1_weight_scale", "moe_ffn2_weight_scale"]

        if moe_quant_type == "w4a8":
            moe_ffn1_in_scale = paddle.concat(moe_ffn1_in_scale)
            moe_ffn2_in_scale = paddle.concat(moe_ffn2_in_scale)
            moe_ffn1_in_scale = 1 / moe_ffn1_in_scale
            moe_ffn2_in_scale = 1 / moe_ffn2_in_scale
            moe_ffn1_weight_scale = paddle.stack(moe_ffn1_weight_scale, axis=0)
            moe_ffn2_weight_scale = paddle.stack(moe_ffn2_weight_scale, axis=0)

            moe_ffn1_weight_scale = moe_ffn1_weight_scale / (127 * 112)
            moe_ffn2_weight_scale = moe_ffn2_weight_scale / (127 * 112)
            moe_ffn1_weight_scale = moe_ffn1_weight_scale / moe_ffn1_in_scale[:,
                                                                              None]
            moe_ffn2_weight_scale = moe_ffn2_weight_scale / moe_ffn2_in_scale[:,
                                                                              None]
            moe_ffn1_weight_scale = moe_ffn1_weight_scale.cast(
                paddle.get_default_dtype())
            moe_ffn2_weight_scale = moe_ffn2_weight_scale.cast(
                paddle.get_default_dtype())

        if moe_quant_type in ["weight_only_int4", "weight_only_int8", "w4a8"]:

            for idx, weight_tensor in enumerate([ffn1_tensor, ffn2_tensor]):
                weight_name = added_weight_attrs[idx]
                scale_name = added_scale_attrs[idx]

                weight_list = []
                weight_scale_list = []
                for i in range(num_local_experts):
                    quant_weight, scale = weight_quantize(weight_tensor[i],
                                                          algo=moe_quant_type,
                                                          arch=80)
                    weight_list.append(quant_weight)
                    if moe_quant_type != "w4a8":
                        # scale holds no memoty in w4a8, don't touch it!
                        weight_scale_list.append(scale)
                quanted_weight = paddle.stack(weight_list, axis=0)
                setattr(
                    layer, weight_name,
                    layer.create_parameter(
                        shape=quanted_weight.shape,
                        dtype=quanted_weight.dtype,
                        default_initializer=paddle.nn.initializer.Constant(0),
                    ))
                getattr(layer, weight_name).set_value(quanted_weight)

                # this scale only useful for wint8/4.
                if moe_quant_type != "w4a8":
                    quanted_weight_scale = paddle.stack(weight_scale_list,
                                                        axis=0)
                    setattr(
                        layer, scale_name,
                        layer.create_parameter(
                            shape=quanted_weight_scale.shape,
                            dtype=quanted_weight_scale.dtype,
                        ))
                    getattr(layer, scale_name).set_value(quanted_weight_scale)

        if moe_quant_type == "w4a8":
            assert moe_ffn1_weight_scale is not None
            assert moe_ffn2_weight_scale is not None
            assert moe_ffn1_in_scale is not None
            assert moe_ffn2_in_scale is not None
            added_w4a8_attrs = [
                "moe_ffn1_weight_scale", "moe_ffn2_weight_scale",
                "moe_ffn1_in_scale", "moe_ffn2_in_scale"
            ]
            for idx, weight_tensor in enumerate([
                    moe_ffn1_weight_scale, moe_ffn2_weight_scale,
                    moe_ffn1_in_scale, moe_ffn2_in_scale
            ]):
                name = added_w4a8_attrs[idx]
                setattr(
                    layer, name,
                    layer.create_parameter(
                        shape=weight_tensor.shape,
                        dtype=weight_tensor.dtype,
                        default_initializer=paddle.nn.initializer.Constant(0),
                    ))
                getattr(layer, name).set_value(weight_tensor)

    def apply(
        self,
        layer: nn.Layer,
        moe_compute_params,
        x: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Paddle Cutlass compute Fused MoE.
        """

        gate_out = paddle.matmul(x.cast("float32"), layer.gate_weight)

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
            (layer.moe_ffn1_in_scale if hasattr(layer, "moe_ffn1_in_scale")
             else None),  # if set, permute_input will be int8_t
            moe_compute_params.top_k,
            False,
            topk_only_mode=False,
        )

        if moe_compute_params.moe_quant_type != "w4a8":
            # only w4a8 need expert_idx_per_token
            # Other need not this tensor, so we make it None.
            expert_idx_per_token = None
        else:
            expert_idx_per_token = expert_idx_per_token.cast("int64")

        ffn_out = moe_expert_ffn(
            permute_input,
            token_nums_per_expert,
            layer.moe_ffn1_weight,
            layer.moe_ffn2_weight,
            None,
            (layer.moe_ffn1_weight_scale
             if hasattr(layer, "moe_ffn1_weight_scale") else None),
            (layer.moe_ffn2_weight_scale
             if hasattr(layer, "moe_ffn2_weight_scale") else None),
            (layer.moe_ffn2_in_scale
             if hasattr(layer, "moe_ffn2_in_scale") else None),
            expert_idx_per_token,
            moe_compute_params.moe_quant_type,
            False,  # used_in_ep_low_latency
        )

        if False:
            if in_dynamic_or_pir_mode():
                hcg = fleet.get_hybrid_communicate_group()
                mp_group = hcg.get_model_parallel_group()
                paddle.distributed.all_reduce(ffn_out, group=mp_group)
            else:
                paddle.distributed.all_reduce(ffn_out, group=mp_group)

        # reduce 中会做 topk 个 weight 的 norm 和 routed_scaling_factor
        fused_moe_out = moe_expert_reduce(
            ffn_out,
            topk_weights,
            permute_indices_per_token,
            topk_idx,
            None,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )
        return fused_moe_out
