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
from paddleformers.utils.log import logger

import fastdeploy
from fastdeploy.distributed.communication import tensor_model_parallel_all_reduce
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.ops.gpu import count_tokens_per_expert_func, deep_gemm
from fastdeploy.model_executor.utils import TensorTracker, set_weight_attrs
from fastdeploy.utils import ceil_div

from .fused_moe_backend_base import MoEMethodBase


class DeepGemmFusedMoeMethod(MoEMethodBase):
    """
    DeepGemmFusedMoeMethod is a class that implements the MoEMethodBase interface for DeepGemm backend.
    """

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        deepgemm create weight process.
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
        if self.quant_config.is_checkpoint_bf16 and layer.fd_config.load_config.load_choices == "default_v1":
            layer.up_gate_proj_weight = layer.create_parameter(
                shape=[layer.num_local_experts, layer.hidden_size, layer.moe_intermediate_size * 2],
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            layer.down_proj_weight = layer.create_parameter(
                shape=[layer.num_local_experts, layer.moe_intermediate_size, layer.hidden_size],
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
            extra_weight_attrs["weight_need_transpose"] = not extra_weight_attrs.get("model_format") == "torch"
            extra_weight_attrs = {**extra_weight_attrs, "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0}}
            set_weight_attrs(
                getattr(layer, up_gate_proj_weight_name),
                extra_weight_attrs,
            )
            set_weight_attrs(
                getattr(layer, up_gate_proj_scale_name),
                extra_weight_attrs,
            )

            set_weight_attrs(
                getattr(layer, down_proj_weight_name),
                extra_weight_attrs,
            )
            set_weight_attrs(
                getattr(layer, down_proj_scale_name),
                extra_weight_attrs,
            )

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
            layer.up_gate_proj_weight.tensor_track = None
        else:
            weight_type = "down"
            layer.down_proj_weight.tensor_track = None

        # 1.init shape and type
        self.added_scale_attrs = ["up_gate_proj_weight_scale_inv", "down_proj_weight_scale_inv"]
        # weight
        weight_name = self.added_weight_attrs[weight_id_map[weight_type]]
        unquantized_weight_name = weight_name.replace("quant_weight", "weight")
        weight_shape = self.up_gate_proj_weight_shape if weight_type == "gate_up" else self.down_proj_weight_shape
        weight_dtype = paddle.float8_e4m3fn
        # scale
        scale_name = self.added_scale_attrs[weight_id_map[weight_type]]
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

        getattr(layer, unquantized_weight_name).value().get_tensor()._clear()

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

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        """
        deepgemm create weight process.
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
            quanted_weight = quanted_weight.transpose([0, 2, 1]).contiguous()
            getattr(layer, weight_name).copy_(quanted_weight, False)

            quanted_weight_scale = paddle.stack(weight_scale_list, axis=0)
            quanted_weight_scale = quanted_weight_scale.transpose([0, 2, 1]).contiguous()
            getattr(layer, scale_name).set_value(quanted_weight_scale)

    def process_prequanted_weights(self, layer: nn.Layer, state_dict, is_rearrange: bool = False):
        """
        Paddle cutlass process prequanted weights.
        """
        up_gate_proj_expert_weight_key = layer.weight_key_map.get("up_gate_proj_expert_weight_key", None)
        down_proj_expert_weight_key = layer.weight_key_map.get("down_proj_expert_weight_key", None)
        up_gate_proj_expert_weight_scale_key = layer.weight_key_map.get("up_gate_proj_expert_weight_scale_key", None)
        down_proj_expert_weight_scale_key = layer.weight_key_map.get("down_proj_expert_weight_scale_key", None)

        up_gate_proj_weights, down_proj_weights, logical_expert_ids, _ = layer.load_experts_weight(
            state_dict, up_gate_proj_expert_weight_key, down_proj_expert_weight_key, is_rearrange
        )
        # self.check(layer, up_gate_proj_weights, down_proj_weights)
        up_gate_proj_weight_scale = []
        down_proj_weight_scale = []
        for expert_idx in logical_expert_ids:
            up_gate_proj_expert_weight_scale_key_name = up_gate_proj_expert_weight_scale_key.format(expert_idx)
            down_proj_expert_weight_scale_key_name = down_proj_expert_weight_scale_key.format(expert_idx)

            up_gate_proj_weight_scale.append(
                get_tensor(
                    (
                        state_dict.pop(up_gate_proj_expert_weight_scale_key_name)
                        if up_gate_proj_expert_weight_scale_key_name in state_dict
                        else up_gate_proj_expert_weight_scale_key_name
                    ),
                    layer.fd_config.model_config.model,
                )
            )
            down_proj_weight_scale.append(
                get_tensor(
                    (
                        state_dict.pop(down_proj_expert_weight_scale_key_name)
                        if down_proj_expert_weight_scale_key_name in state_dict
                        else down_proj_expert_weight_scale_key_name
                    ),
                    layer.fd_config.model_config.model,
                )
            )

        up_gate_proj_weight = (
            paddle.stack(up_gate_proj_weights, axis=0).transpose([0, 2, 1]).contiguous().view("float8_e4m3fn")
        )
        down_proj_weight = (
            paddle.stack(down_proj_weights, axis=0).transpose([0, 2, 1]).contiguous().view("float8_e4m3fn")
        )
        up_gate_proj_weight_scale = paddle.stack(up_gate_proj_weight_scale, axis=0).transpose([0, 2, 1]).contiguous()
        down_proj_weight_scale = paddle.stack(down_proj_weight_scale, axis=0).transpose([0, 2, 1]).contiguous()

        name_tensor_map = {
            "up_gate_proj_weight": up_gate_proj_weight,
            "down_proj_weight": down_proj_weight,
            "up_gate_proj_weight_scale_inv": up_gate_proj_weight_scale,
            "down_proj_weight_scale_inv": down_proj_weight_scale,
        }
        for name, tensor in name_tensor_map.items():
            getattr(layer, name).set_value(tensor)

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
        x, x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant(
            x, self.quant_config.weight_block_size[0]
        )
        # 3. EP Dispatch
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_num_tokens_per_expert_list,
            handle,
            _,
        ) = self.ep_prefill_runner.dispatch(
            x, topk_idx, topk_weights, x_scale_tensor=x_scale_tensor, expert_alignment=128
        )

        token_all_num = sum(recv_num_tokens_per_expert_list)

        # 4. Compute ffn
        if token_all_num > 0:
            logger.debug(f"token_all_num {token_all_num}")
            (recv_x, recv_x_scale) = recv_x

            token_nums_this_rank = count_tokens_per_expert_func(recv_topk_idx, layer.num_local_experts)

            (
                permute_input,
                permute_scale,
                permute_indices_per_token,
                recv_num_tokens_per_expert_list_cumsum,
                recv_num_tokens_per_expert_list_padded_cumsum,
                dst_weights,
                dst_indices,
                cumsum_idx_gpu,
                m_indices,
            ) = fastdeploy.model_executor.ops.gpu.ep_moe_expert_dispatch_fp8(
                recv_x,
                recv_x_scale,
                recv_topk_idx,
                recv_topk_weights,
                token_nums_this_rank[0],
                token_nums_this_rank[1],
                True,  # use_in_ep
                token_all_num,
            )

            permute_scale = permute_scale.transpose([1, 0]).contiguous()
            permute_scale = permute_scale.transpose([1, 0])

            # up_gate_proj
            ffn_out = paddle.empty(
                (permute_input.shape[0], getattr(layer, self.added_weight_attrs[0]).shape[1]),
                dtype=paddle.bfloat16,
            )
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                (permute_input, permute_scale),
                (getattr(layer, self.added_weight_attrs[0]), getattr(layer, self.added_scale_attrs[0])),
                ffn_out,
                m_indices,
            )
            # swiglu
            ffn_out = paddle.incubate.nn.functional.swiglu(ffn_out, None)

            # down_proj
            ffn_in_x, ffn_in_x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant(
                ffn_out, self.quant_config.weight_block_size[0]
            )
            ffn_in_x_scale_tensor = ffn_in_x_scale_tensor.transpose([1, 0]).contiguous()
            ffn_in_x_scale_tensor = ffn_in_x_scale_tensor.transpose([1, 0])

            ffn_out = paddle.empty(
                (ffn_out.shape[0], getattr(layer, self.added_weight_attrs[1]).shape[1]),
                dtype=paddle.bfloat16,
            )
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                (ffn_in_x, ffn_in_x_scale_tensor),
                (getattr(layer, self.added_weight_attrs[1]), getattr(layer, self.added_scale_attrs[1])),
                ffn_out,
                m_indices,
            )
            # prmt back per rank
            tmp_ffn_out = fastdeploy.model_executor.ops.gpu.ep_moe_expert_combine(
                ffn_out,
                dst_weights,
                permute_indices_per_token,
                dst_indices,
                None,  # down_proj_bias
                False,  # norm_topk_prob
                1.0,
            )[0]

        else:
            tmp_ffn_out = paddle.cast(recv_x[0], paddle.bfloat16)

        # 5. EP combine
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
        permute_input, token_nums_per_expert, handle = self.ep_decoder_runner.dispatch(
            x, topk_idx, topk_weights, use_fp8=True
        )

        # 3. Compute ffn
        assert isinstance(permute_input, tuple)
        up_gate_proj_out = paddle.empty(
            [
                layer.num_local_experts,
                layer.ep_size * layer.fd_config.model_config.num_max_dispatch_tokens_per_rank,
                layer.moe_intermediate_size * 2,
            ],
            dtype=paddle.bfloat16,
        )

        ffn_out = paddle.empty(
            [
                layer.num_local_experts,
                layer.ep_size * layer.fd_config.model_config.num_max_dispatch_tokens_per_rank,
                layer.hidden_size,
            ],
            dtype=paddle.bfloat16,
        )

        expected_m = 128
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            permute_input,
            (
                getattr(layer, self.added_weight_attrs[0]),
                getattr(layer, self.added_scale_attrs[0]),
            ),
            up_gate_proj_out,
            token_nums_per_expert,
            expected_m,
        )

        act_out = fastdeploy.model_executor.ops.gpu.group_swiglu_with_masked(up_gate_proj_out, token_nums_per_expert)

        act_out_fp8, scale = fastdeploy.model_executor.ops.gpu.masked_per_token_quant(
            act_out,
            token_nums_per_expert,
            self.quant_config.weight_block_size[0],
        )

        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            (act_out_fp8, scale),
            (
                getattr(layer, self.added_weight_attrs[1]),
                getattr(layer, self.added_scale_attrs[1]),
            ),
            ffn_out,
            token_nums_per_expert,
            expected_m,
        )

        # 4. EP combine
        return self.ep_decoder_runner.combine(ffn_out, topk_idx, topk_weights, handle)

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Paddle Use DeepGemm compute Fused MoE.
        below is TP compute method.
        """
        gate_out = gate(x.cast("float32"))

        if layer.topk_method == "noaux_tc":
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
                layer.top_k,
                True,  # apply_norm_weight
                False,
            )

        tmp = count_tokens_per_expert_func(topk_ids, layer.num_experts)

        recv_x, recv_x_scale = fastdeploy.model_executor.ops.gpu.per_token_quant(x, 128)

        (
            permute_input,
            permute_scale,
            permute_indices_per_token,
            recv_num_tokens_per_expert_list_cumsum,
            recv_num_tokens_per_expert_list_padded_cumsum,
            dst_weights,
            dst_indices,
            cumsum_idx_gpu,
            m_indices,
        ) = fastdeploy.model_executor.ops.gpu.ep_moe_expert_dispatch_fp8(
            recv_x,
            recv_x_scale,
            topk_ids,
            topk_weights,
            tmp[0],
            tmp[1],
            False,  # use_in_ep
            -1,
        )

        permute_scale = permute_scale.transpose([1, 0]).contiguous()
        permute_scale = permute_scale.transpose([1, 0])

        # up_gate_proj
        ffn_out = paddle.empty(
            (permute_input.shape[0], getattr(layer, self.added_weight_attrs[0]).shape[1]),
            dtype=paddle.bfloat16,
        )
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (permute_input, permute_scale),
            (getattr(layer, self.added_weight_attrs[0]), getattr(layer, self.added_scale_attrs[0])),
            ffn_out,
            m_indices,
        )
        # swiglu
        ffn_out = paddle.incubate.nn.functional.swiglu(ffn_out)

        # down_proj
        ffn_in_x, ffn_in_x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant(
            ffn_out, self.quant_config.weight_block_size[0]
        )

        ffn_in_x_scale_tensor = ffn_in_x_scale_tensor.transpose([1, 0]).contiguous()
        ffn_in_x_scale_tensor = ffn_in_x_scale_tensor.transpose([1, 0])

        ffn_out = paddle.empty(
            (ffn_out.shape[0], getattr(layer, self.added_weight_attrs[1]).shape[1]),
            dtype=paddle.bfloat16,
        )
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (ffn_in_x, ffn_in_x_scale_tensor),
            (getattr(layer, self.added_weight_attrs[1]), getattr(layer, self.added_scale_attrs[1])),
            ffn_out,
            m_indices,
        )
        # prmt back per rank
        tmp_ffn_out = fastdeploy.model_executor.ops.gpu.ep_moe_expert_combine(
            ffn_out,
            dst_weights,
            permute_indices_per_token,
            dst_indices,
            None,
            False,  # norm_topk_prob
            1.0,
        )[0]
        if layer.tp_size > 1:
            tmp_ffn_out = tensor_model_parallel_all_reduce(tmp_ffn_out)

        return tmp_ffn_out
