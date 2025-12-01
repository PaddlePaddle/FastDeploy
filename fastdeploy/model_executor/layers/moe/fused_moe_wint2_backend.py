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
from fastdeploy.model_executor.ops.gpu import moe_expert_dispatch, moe_expert_reduce
from fastdeploy.utils import ceil_div

from ..quantization.quant_base import QuantMethodBase
from ..utils import get_tensor


class Wint2MoeMethod(QuantMethodBase):
    """
    Use  compute Fused MoE.
    """

    def __init__(self, quant_config):
        super().__init__()
        self.moe_quant_type = quant_config.moe_quant_type
        self.added_weight_attrs = ["up_gate_proj_weight", "down_proj_weight"]
        self.added_scale_attrs = [
            "up_gate_proj_weight_scale",
            "down_proj_weight_scale",
        ]

    def process_loaded_weights(self, layer, weights) -> None:
        """
        process_loaded_weights
        """
        pass

    def check(self, layer: nn.Layer, up_gate_proj_weights, down_proj_weights):
        """
        check layer is valid for this method
        """
        assert (
            len(up_gate_proj_weights) == layer.num_local_experts
        ), "up_gate_proj_weights length should be equal to num_local_experts."
        assert (
            len(down_proj_weights) == layer.num_local_experts
        ), "down_proj_weights length should be equal to num_local_experts."

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        Paddle cutlass create weight process.
        """
        self.weight_dtype = "uint8"
        self.default_dtype = layer._helper.get_default_dtype()
        setattr(
            layer,
            "up_gate_proj_weight",
            layer.create_parameter(
                shape=[layer.num_local_experts, layer.hidden_size // 4, layer.moe_intermediate_size * 2],
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            "down_proj_weight",
            layer.create_parameter(
                shape=[layer.num_local_experts, layer.moe_intermediate_size // 4, layer.hidden_size],
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            "up_gate_proj_weight_scale",
            layer.create_parameter(
                shape=[layer.num_local_experts, layer.hidden_size // 128, layer.moe_intermediate_size * 2],
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            "down_proj_weight_scale",
            layer.create_parameter(
                shape=[layer.num_local_experts, layer.moe_intermediate_size // 128, layer.hidden_size],
                dtype=self.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            "up_gate_proj_super_scales",
            layer.create_parameter(
                shape=[layer.num_local_experts, layer.moe_intermediate_size * 2],
                dtype=self.default_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            "down_proj_super_scales",
            layer.create_parameter(
                shape=[layer.num_local_experts, layer.hidden_size],
                dtype=self.default_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            "up_gate_proj_code_scale",
            layer.create_parameter(
                shape=[layer.num_local_experts, layer.moe_intermediate_size * 2],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            "down_proj_code_scale",
            layer.create_parameter(
                shape=[layer.num_local_experts, layer.hidden_size],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            "up_gate_proj_code_zp",
            layer.create_parameter(
                shape=[layer.num_local_experts, layer.moe_intermediate_size * 2],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )
        setattr(
            layer,
            "down_proj_code_zp",
            layer.create_parameter(
                shape=[layer.num_local_experts, layer.hidden_size],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0),
            ),
        )


class CutlassWint2FusedMoeMethod(Wint2MoeMethod):
    """
    Use Triton Group Gemm to compute Fused MoE.
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)

    def process_loaded_weights(self, layer, weights) -> None:
        """
        process_loaded_weights
        """
        pass

    def process_prequanted_weights(self, layer: nn.Layer, state_dict, is_rearrange: bool = False):
        """
        Paddle cutlass process prequanted weights.
        """
        up_gate_proj_expert_weight_key = layer.weight_key_map.get("up_gate_proj_expert_weight_key", None)
        down_proj_expert_weight_key = layer.weight_key_map.get("down_proj_expert_weight_key", None)
        up_gate_proj_expert_weight_scale_key = layer.weight_key_map.get("up_gate_proj_expert_weight_scale_key", None)
        down_proj_expert_weight_scale_key = layer.weight_key_map.get("down_proj_expert_weight_scale_key", None)
        up_gate_proj_expert_super_scales_key = layer.weight_key_map.get("up_gate_proj_expert_super_scales_key", None)
        down_proj_expert_super_scales_key = layer.weight_key_map.get("down_proj_expert_super_scales_key", None)
        up_gate_proj_expert_code_scale_key = layer.weight_key_map.get("up_gate_proj_expert_code_scale_key", None)
        down_proj_expert_code_scale_key = layer.weight_key_map.get("down_proj_expert_code_scale_key", None)
        up_gate_proj_expert_code_zp_key = layer.weight_key_map.get("up_gate_proj_expert_code_zp_key", None)
        down_proj_expert_code_zp_key = layer.weight_key_map.get("down_proj_expert_code_zp_key", None)

        up_gate_proj_weights, down_proj_weights, _, _ = layer.load_experts_weight(
            state_dict,
            up_gate_proj_expert_weight_key,
            down_proj_expert_weight_key,
        )
        # self.check(layer, up_gate_proj_weights, down_proj_weights)

        up_gate_proj_weight_scale = []
        down_proj_weight_scale = []
        up_gate_proj_super_scales = []
        down_proj_super_scales = []
        up_gate_proj_code_scale = []
        down_proj_code_scale = []
        up_gate_proj_code_zp = []
        down_proj_code_zp = []
        for i in range(layer.num_experts):
            expert_idx = layer.expert_id_offset + i
            up_gate_proj_weight_scale.append(
                get_tensor(state_dict.pop(up_gate_proj_expert_weight_scale_key.format(expert_idx)))
            )
            down_proj_weight_scale.append(
                get_tensor(state_dict.pop(down_proj_expert_weight_scale_key.format(expert_idx)))
            )
            up_gate_proj_super_scales.append(
                get_tensor(state_dict.pop(up_gate_proj_expert_super_scales_key.format(expert_idx)))
            )
            down_proj_super_scales.append(
                get_tensor(state_dict.pop(down_proj_expert_super_scales_key.format(expert_idx)))
            )
            up_gate_proj_code_scale.append(
                get_tensor(state_dict.pop(up_gate_proj_expert_code_scale_key.format(expert_idx)))
            )
            down_proj_code_scale.append(get_tensor(state_dict.pop(down_proj_expert_code_scale_key.format(expert_idx))))
            up_gate_proj_code_zp.append(get_tensor(state_dict.pop(up_gate_proj_expert_code_zp_key.format(expert_idx))))
            down_proj_code_zp.append(get_tensor(state_dict.pop(down_proj_expert_code_zp_key.format(expert_idx))))

        up_gate_proj_weight = paddle.stack(up_gate_proj_weights, axis=0)
        down_proj_weight = paddle.stack(down_proj_weights, axis=0)
        up_gate_proj_weight_scale = paddle.stack(up_gate_proj_weight_scale, axis=0)
        down_proj_weight_scale = paddle.stack(down_proj_weight_scale, axis=0)
        up_gate_proj_super_scales = paddle.stack(up_gate_proj_super_scales, axis=0)
        down_proj_super_scales = paddle.stack(down_proj_super_scales, axis=0)
        up_gate_proj_code_scale = paddle.stack(up_gate_proj_code_scale, axis=0)
        down_proj_code_scale = paddle.stack(down_proj_code_scale, axis=0)
        up_gate_proj_code_zp = paddle.stack(up_gate_proj_code_zp, axis=0)
        down_proj_code_zp = paddle.stack(down_proj_code_zp, axis=0)

        # Here we pre-arrange the n-dim weight matrix
        w1_shape = up_gate_proj_weight.shape
        up_gate_proj_weight = up_gate_proj_weight.reshape([w1_shape[0], w1_shape[1] // 16, 16, w1_shape[2] // 8, 8])
        up_gate_proj_weight = paddle.transpose(up_gate_proj_weight, perm=[0, 3, 1, 4, 2])
        up_gate_proj_weight = up_gate_proj_weight.reshape(w1_shape)

        w2_shape = down_proj_weight.shape
        down_proj_weight = down_proj_weight.reshape([w2_shape[0], w2_shape[1] // 16, 16, w2_shape[2] // 8, 8])
        down_proj_weight = paddle.transpose(down_proj_weight, perm=[0, 3, 1, 4, 2])
        down_proj_weight = down_proj_weight.reshape(w2_shape)

        name_tensor_map = {
            "up_gate_proj_weight": up_gate_proj_weight,
            "down_proj_weight": down_proj_weight,
            "up_gate_proj_weight_scale": up_gate_proj_weight_scale,
            "down_proj_weight_scale": down_proj_weight_scale,
            "up_gate_proj_super_scales": up_gate_proj_super_scales,
            "down_proj_super_scales": down_proj_super_scales,
            "up_gate_proj_code_scale": up_gate_proj_code_scale,
            "down_proj_code_scale": down_proj_code_scale,
            "up_gate_proj_code_zp": up_gate_proj_code_zp,
            "down_proj_code_zp": down_proj_code_zp,
        }
        for name, tensor in name_tensor_map.items():
            getattr(layer, name).set_value(tensor)

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Use Wint2 Triton Fusedmoe compute Fused MoE.
        """
        gate_out = gate(x.cast("float32"))

        (
            permute_input,
            token_nums_per_expert,
            permute_indices_per_token,
            topk_weights,
            topk_idx,
            expert_idx_per_token,
            dequant_scale,
        ) = moe_expert_dispatch(
            x,
            gate_out,
            layer.gate_correction_bias,
            (
                layer.up_gate_proj_in_scale if hasattr(layer, "up_gate_proj_in_scale") else None
            ),  # if set, permute_input will be int8_t
            layer.top_k,
            False,
            self.moe_quant_type,
            topk_only_mode=False,
        )

        ffn_out = fastdeploy.model_executor.ops.gpu.moe_expert_ffn_wint2(
            permute_input,
            token_nums_per_expert,
            layer.up_gate_proj_weight,
            layer.down_proj_weight,
            None,
            layer.up_gate_proj_super_scales,
            layer.down_proj_super_scales,
            layer.up_gate_proj_weight_scale,
            layer.up_gate_proj_code_scale,
            layer.up_gate_proj_code_zp,
            layer.down_proj_weight_scale,
            layer.down_proj_code_scale,
            layer.down_proj_code_zp,
            False,
        )

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


class TritonWint2FusedMoeMethod(CutlassWint2FusedMoeMethod):
    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.moe_quant_type = quant_config.moe_quant_type

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Use Wint2 Triton Fusedmoe compute Fused MoE.
        """
        gate_out = gate(x.cast("float32"))
        from fastdeploy.model_executor.ops.triton_ops import moe_wint2_ffn_kernel

        topk_ids, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
            gate_out,
            layer.gate_correction_bias,
            layer.top_k,
            True,  # apply_norm_weight,
            False,
        )

        num_tokens, K = x.shape
        E, _, N = layer.up_gate_proj_weight.shape
        M = num_tokens

        top_k = topk_ids.shape[1]

        intermediate_cache1 = paddle.empty(
            [M, top_k, N],
            dtype=x.dtype,
        )
        intermediate_cache3 = paddle.empty(
            (M, top_k, K),
            dtype=x.dtype,
        )

        double_quant = True
        num_valid_tokens = topk_ids.shape[0] * topk_ids.shape[1]

        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 512,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 16,
        }
        from fastdeploy.model_executor.ops.gpu import tritonmoe_preprocess

        sorted_token_ids, expert_ids, num_tokens_post_padded = tritonmoe_preprocess(
            topk_ids, E, config["BLOCK_SIZE_M"]
        )

        max_possible_num_post_padded = sorted_token_ids.shape[0]
        grid = (ceil_div(max_possible_num_post_padded, config["BLOCK_SIZE_M"]) * ceil_div(N, config["BLOCK_SIZE_N"]),)

        moe_wint2_ffn_kernel[grid](
            x,
            layer.up_gate_proj_weight,
            intermediate_cache1,
            layer.up_gate_proj_weight_scale,
            layer.up_gate_proj_super_scales,
            layer.up_gate_proj_code_scale,
            layer.up_gate_proj_code_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            num_valid_tokens,
            max_possible_num_post_padded,
            # Matrix dimensions
            N=layer.up_gate_proj_weight.shape[-1],
            K=x.shape[-1],
            # The stride variables represent how much to increase the ptr by when
            # moving by 1 element in a particular dimension. E.g. `stride_am` is
            # how much to increase `a_ptr` by to get the element one row down
            # (A has M rows).
            stride_am=x.strides[0],
            stride_ak=x.strides[1],
            stride_be=layer.up_gate_proj_weight.strides[0],
            stride_bk=layer.up_gate_proj_weight.strides[1],
            stride_bn=1,
            stride_cm=intermediate_cache1.strides[-2],
            stride_cn=1,
            stride_bse=layer.up_gate_proj_weight_scale.strides[0],
            stride_bsk=layer.up_gate_proj_weight_scale.strides[1],
            stride_bsn=1,
            stride_bce=layer.up_gate_proj_code_scale.strides[0],
            stride_bck=1,
            stride_bcn=1,
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            MUL_ROUTED_WEIGHT=False,
            USE_DOUBLE_QUANT=double_quant,
            top_k=top_k,
        )

        intermediate_cache2 = paddle.incubate.nn.functional.swiglu(intermediate_cache1.reshape([-1, N]))

        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 2,
            "num_warps": 4,
            "num_stages": 8,
        }

        grid = (
            ceil_div(max_possible_num_post_padded, config["BLOCK_SIZE_M"])
            * ceil_div(layer.down_proj_weight.shape[-1], config["BLOCK_SIZE_N"]),
        )

        moe_wint2_ffn_kernel[grid](
            intermediate_cache2,
            layer.down_proj_weight,
            intermediate_cache3,
            layer.down_proj_weight_scale,
            layer.down_proj_super_scales,
            layer.down_proj_code_scale,
            layer.down_proj_code_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            num_valid_tokens,
            max_possible_num_post_padded,
            # Matrix dimensions
            N=layer.down_proj_weight.shape[-1],
            K=intermediate_cache2.shape[-1],
            # The stride variables represent how much to increase the ptr by when
            # moving by 1 element in a particular dimension. E.g. `stride_am` is
            # how much to increase `a_ptr` by to get the element one row down
            # (A has M rows).
            stride_am=intermediate_cache2.strides[0],
            stride_ak=1,
            stride_be=layer.down_proj_weight.strides[0],
            stride_bk=layer.down_proj_weight.strides[1],
            stride_bn=1,
            stride_cm=intermediate_cache3.strides[-2],
            stride_cn=1,
            stride_bse=layer.down_proj_weight_scale.strides[0],
            stride_bsk=layer.down_proj_weight_scale.strides[1],
            stride_bsn=1,
            stride_bce=layer.down_proj_code_scale.strides[0],
            stride_bck=1,
            stride_bcn=1,
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            MUL_ROUTED_WEIGHT=True,
            USE_DOUBLE_QUANT=double_quant,
            top_k=1,
        )

        fused_moe_out = paddle.sum(intermediate_cache3, axis=1)

        return fused_moe_out
