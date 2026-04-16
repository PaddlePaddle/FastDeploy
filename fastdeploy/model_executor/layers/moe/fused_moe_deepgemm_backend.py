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

import os
import threading
from typing import Callable

import paddle
from paddle import nn
from paddleformers.utils.log import logger

import fastdeploy
from fastdeploy.model_executor.layers.moe.ep import deep_ep
from fastdeploy.model_executor.layers.quantization.fp8_utils import (
    deep_gemm,
    paddlefleet_ops,
)
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.ops.gpu import (
    count_tokens_per_expert_func,
    depermute_prefill_combine,
    prefill_permute_to_masked_gemm,
)
from fastdeploy.platforms import current_platform
from fastdeploy.utils import register_custom_python_op
from fastdeploy.worker.tbo import let_another_thread_run

from .fused_moe_backend_base import MoEMethodBase
from .fused_moe_triton_backend import BlockWiseFP8MoEMethod

if current_platform.is_cuda():
    try:
        m_grouped_fp8_gemm_nt_contiguous = deep_gemm.m_grouped_fp8_gemm_nt_contiguous
        m_grouped_fp8_gemm_nt_masked = deep_gemm.m_grouped_fp8_gemm_nt_masked
    except:
        m_grouped_fp8_gemm_nt_contiguous = deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous
        m_grouped_fp8_gemm_nt_masked = deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked
else:
    m_grouped_fp8_gemm_nt_contiguous = None
    m_grouped_fp8_gemm_nt_masked = None

global_values = {}


def call_prefill_permute_to_masked_gemm(
    x: paddle.Tensor,
    scale: paddle.Tensor,
    topk_ids: paddle.Tensor,
    num_local_experts: int,
    max_token_num: int,
):
    """
    Permute input tokens and scales from token-major to expert-major layout
    for MoE masked GEMM operations.

    Args:
        x: Input hidden states [num_tokens, hidden].
        scale: Input scales [num_tokens, hidden_scale].
        topk_ids: Expert routing indices [num_tokens, topk] (int64 or int32).
        num_local_experts: Number of local experts on this device.
        max_token_num: Maximum tokens per expert buffer.

    Returns:
        tuple: (permute_x, permute_scale, permuted_indice_map, token_nums_per_expert)
    """
    if topk_ids.dtype != paddle.int64:
        topk_ids = topk_ids.cast(paddle.int64)

    results = prefill_permute_to_masked_gemm(x, scale, topk_ids, num_local_experts, max_token_num)

    return results[0], results[1], results[2], results[3]


def call_depermute_prefill_combine(
    x: paddle.Tensor,
    indice_map: paddle.Tensor,
    topk_weights: paddle.Tensor,
    num_worst_tokens: int,
):
    """
    Depermute and combine expert outputs back to token-major layout.

    Args:
        x: Expert outputs [num_local_experts, max_tokens_per_expert, hidden].
        indice_map: Flat index tensor [num_worst_tokens, topk] (int32).
        topk_weights: Combination weights [num_worst_tokens, topk] (float32).
        num_worst_tokens: Number of output tokens to produce.

    Returns:
        depermuted_x: Combined output [num_worst_tokens, hidden].
    """
    results = depermute_prefill_combine(x, indice_map, topk_weights, num_worst_tokens)

    return results


def m_grouped_fp8_gemm_nt_contiguous_custom_python_op_infermeta(
    permute_input: "paddle.static.MetaTensor",
    permute_scale: "paddle.static.MetaTensor",
    layer_added_weight_attrs_0: "paddle.static.MetaTensor",
    layer_added_scale_attrs_0: "paddle.static.MetaTensor",
    m_indices: "paddle.static.MetaTensor",
    layer_added_weight_attrs_1: "paddle.static.MetaTensor",
    layer_added_scale_attrs_1: "paddle.static.MetaTensor",
    quant_config_weight_block_size_0: int,
):
    return paddle.static.MetaTensor(
        shape=[permute_input.shape[0], layer_added_weight_attrs_1.shape[1]], dtype=paddle.bfloat16
    )


@register_custom_python_op(
    name="m_grouped_fp8_gemm_nt_contiguous_custom",
    infer_meta=m_grouped_fp8_gemm_nt_contiguous_custom_python_op_infermeta,
    input_names=[
        "permute_input",
        "permute_scale",
        "layer_added_weight_attrs_0",
        "layer_added_scale_attrs_0",
        "m_indices",
        "layer_added_weight_attrs_1",
        "layer_added_scale_attrs_1",
    ],
    output_names=["ffn_new_out"],
    inplace_map={},
)
def m_grouped_fp8_gemm_nt_contiguous_custom_python_op(
    permute_input: paddle.Tensor,
    permute_scale: paddle.Tensor,
    layer_added_weight_attrs_0: paddle.Tensor,  # getattr(layer, self.added_weight_attrs[0])
    layer_added_scale_attrs_0: paddle.Tensor,  # getattr(layer, self.added_scale_attrs[0])
    m_indices: paddle.Tensor,
    layer_added_weight_attrs_1: paddle.Tensor,  # getattr(layer, self.added_weight_attrs[1])
    layer_added_scale_attrs_1: paddle.Tensor,  # getattr(layer, self.added_scale_attrs[1])
    quant_config_weight_block_size_0: int,  # self.quant_config.weight_block_size[0]
    disable_ue8m0_cast: bool,
    dst_weights: paddle.Tensor,
):

    # up_gate_proj
    ffn_out = paddle.empty(
        (permute_input.shape[0], layer_added_weight_attrs_0.shape[1]),
        dtype=paddle.bfloat16,
    )
    # if disable_ue8m0_cast:
    if permute_scale.strides[0] != 1:
        permute_scale = permute_scale.transpose([1, 0]).contiguous()
        permute_scale = permute_scale.transpose([1, 0])
    # disable_ue8m0_cast is False for SM100
    m_grouped_fp8_gemm_nt_contiguous(
        (permute_input, permute_scale),
        (layer_added_weight_attrs_0, layer_added_scale_attrs_0),
        ffn_out,
        m_indices,
    )

    # swiglu
    if fastdeploy.envs.FD_MOE_PROB_IN_ADVANCE:
        ffn_in_x, ffn_in_x_scale_tensor = paddlefleet_ops.fuse_weighted_swiglu_fp8_quant(
            ffn_out, dst_weights, using_pow2_scaling=True, use_ue8m0=not disable_ue8m0_cast
        )

        ffn_in_x_scale_tensor = paddle.transpose(paddle.transpose(ffn_in_x_scale_tensor, [1, 0]).contiguous(), [1, 0])
    else:
        ffn_out = paddle.incubate.nn.functional.swiglu(ffn_out)

        # down_proj
        if not fastdeploy.envs.FD_USE_PHI_FP8_QUANT:
            ffn_in_x, ffn_in_x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant(
                ffn_out, quant_config_weight_block_size_0, not disable_ue8m0_cast
            )

            ffn_in_x_scale_tensor = ffn_in_x_scale_tensor.transpose([1, 0]).contiguous()
            ffn_in_x_scale_tensor = ffn_in_x_scale_tensor.transpose([1, 0])
        else:
            ffn_in_x, ffn_in_x_scale_tensor = paddle.incubate.nn.functional.fp8_quant_blockwise(
                ffn_out,
                using_pow2_scale=not disable_ue8m0_cast or fastdeploy.envs.FD_FP8_QUANT_WITH_POW2SCALE,
                using_ue8m0_scale=not disable_ue8m0_cast,
            )
            ffn_in_x_scale_tensor = ffn_in_x_scale_tensor.T[: ffn_in_x.shape[0]]

    ffn_out = paddle.empty(
        (permute_input.shape[0], layer_added_weight_attrs_1.shape[1]),
        dtype=paddle.bfloat16,
    )
    # disable_ue8m0_cast is False for SM100
    m_grouped_fp8_gemm_nt_contiguous(
        (ffn_in_x, ffn_in_x_scale_tensor),
        (layer_added_weight_attrs_1, layer_added_scale_attrs_1),
        ffn_out,
        m_indices,
    )
    return ffn_out


class DeepGemmFusedMoeMethod(MoEMethodBase):
    """
    DeepGemmFusedMoeMethod is a class that implements the MoEMethodBase interface for DeepGemm backend.
    """

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        deepgemm create weight process.
        """
        BlockWiseFP8MoEMethod.create_weights(self, layer, **extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        """ """
        BlockWiseFP8MoEMethod.process_weights_after_loading(self, layer)

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

        if isinstance(state_dict, list):
            state_dict = dict(state_dict)

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

        if not self.quant_config.deepgemm_scale_ue8m0:
            up_gate_proj_weight = (
                paddle.stack(up_gate_proj_weights, axis=0).transpose([0, 2, 1]).contiguous().view("float8_e4m3fn")
            )
            down_proj_weight = (
                paddle.stack(down_proj_weights, axis=0).transpose([0, 2, 1]).contiguous().view("float8_e4m3fn")
            )
            up_gate_proj_weight_scale = (
                paddle.stack(up_gate_proj_weight_scale, axis=0).transpose([0, 2, 1]).contiguous()
            )
            down_proj_weight_scale = paddle.stack(down_proj_weight_scale, axis=0).transpose([0, 2, 1]).contiguous()
        else:
            up_gate_proj_weight = (
                paddle.stack(up_gate_proj_weights, axis=0).transpose([0, 2, 1]).contiguous().view("float8_e4m3fn")
            )
            down_proj_weight = (
                paddle.stack(down_proj_weights, axis=0).transpose([0, 2, 1]).contiguous().view("float8_e4m3fn")
            )
            up_gate_proj_weight_scale = paddle.stack(up_gate_proj_weight_scale, axis=0).transpose([0, 2, 1])
            down_proj_weight_scale = paddle.stack(down_proj_weight_scale, axis=0).transpose([0, 2, 1])

        name_tensor_map = {
            "up_gate_proj_weight": up_gate_proj_weight,
            "down_proj_weight": down_proj_weight,
            "up_gate_proj_weight_scale_inv": up_gate_proj_weight_scale,
            "down_proj_weight_scale_inv": down_proj_weight_scale,
        }
        for name, tensor in name_tensor_map.items():
            getattr(layer, name).data = tensor

    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
    ) -> paddle.Tensor:
        """
        Apply the EP prefill method.
        """
        gate_out = gate(x)
        gate_out = gate_out.cast("float32")

        hidden_size = x.shape[1]

        # 1. Select topk experts and weights
        topk_idx, topk_weights = self.ep_prefill_runner.moe_select(layer, gate_out)

        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids=topk_idx)

        # 2. Dynamic compute blockwise quantization scales
        if not fastdeploy.envs.FD_USE_PHI_FP8_QUANT:
            x_fp8, x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant(
                x, self.quant_config.weight_block_size[0], self.quant_config.deepgemm_scale_ue8m0
            )
        else:
            x_fp8, x_scale_tensor = paddle.incubate.nn.functional.fp8_quant_blockwise(
                x,
                using_pow2_scale=self.quant_config.deepgemm_scale_ue8m0 or fastdeploy.envs.FD_FP8_QUANT_WITH_POW2SCALE,
                output_scale_transpose=self.quant_config.deepgemm_scale_ue8m0,
                using_ue8m0_scale=self.quant_config.deepgemm_scale_ue8m0,
            )
            x_scale_tensor = (
                x_scale_tensor[: x.shape[0]]
                if not self.quant_config.deepgemm_scale_ue8m0
                else x_scale_tensor.T[: x.shape[0]]
            )

        event = deep_ep.Buffer.capture()

        if self.ep_prefill_runner.num_worst_tokens <= 0:
            let_another_thread_run()
        # 3. EP Dispatch
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_num_tokens_per_expert_list,
            handle,
            event,
        ) = self.ep_prefill_runner.dispatch(
            x_fp8, topk_idx, topk_weights, x_scale_tensor=x_scale_tensor, expert_alignment=128, previous_event=event
        )

        if self.ep_prefill_runner.num_worst_tokens > 0:
            let_another_thread_run()

        thread_name = threading.current_thread().name

        if self.ep_prefill_runner.ep_engine.async_finish:
            event.current_stream_wait()

        global global_values

        if thread_name not in global_values:
            global_values[thread_name] = {}

        (recv_x_value, recv_x_scale) = recv_x
        (recv_x_value, recv_x_scale) = recv_x

        global_values[thread_name]["x"] = x
        global_values[thread_name]["topk_idx"] = topk_idx
        global_values[thread_name]["topk_weights"] = topk_weights
        global_values[thread_name]["x_scale_tensor"] = x_scale_tensor

        global_values[thread_name]["recv_x_value"] = recv_x_value
        global_values[thread_name]["recv_x_scale"] = recv_x_scale
        global_values[thread_name]["recv_topk_idx"] = recv_topk_idx
        global_values[thread_name]["recv_topk_weights"] = recv_topk_weights
        global_values[thread_name]["handle"] = handle
        global_values[thread_name]["recv_num_tokens_per_expert_list"] = recv_num_tokens_per_expert_list

        token_all_num = sum(recv_num_tokens_per_expert_list)

        # Note(ZKK):
        # below code have many del, so ugly!
        # but considering MoE Prefill will reach peak GPU memory,
        # so here we manually del a var as soon as it's not used.

        # 4. Compute ffn
        if self.ep_prefill_runner.num_worst_tokens > 0:
            token_split_factor = 2 if int(os.getenv("USE_TBO", "0")) == 1 else 1
            max_tokens_per_rank = (
                layer.fd_config.scheduler_config.max_num_batched_tokens
                // layer.fd_config.parallel_config.tensor_parallel_size
                // token_split_factor
            )
            expected_m = max_tokens_per_rank

            logger.debug(f"max_tokens_per_rank {max_tokens_per_rank}")

            permute_input, permute_scale, permuted_indice_map, token_nums_per_expert = (
                call_prefill_permute_to_masked_gemm(
                    x=recv_x_value,
                    scale=recv_x_scale,
                    topk_ids=recv_topk_idx,
                    num_local_experts=layer.num_local_experts,
                    max_token_num=layer.ep_size * max_tokens_per_rank,
                )
            )

            up_gate_proj_out = paddle.empty(
                [
                    layer.num_local_experts,
                    layer.ep_size * max_tokens_per_rank,
                    layer.moe_intermediate_size * 2,
                ],
                dtype=paddle.bfloat16,
            )

            m_grouped_fp8_gemm_nt_masked(
                (permute_input, permute_scale),
                (
                    getattr(layer, self.added_weight_attrs[0]),
                    getattr(layer, self.added_scale_attrs[0]),
                ),
                up_gate_proj_out,
                token_nums_per_expert,
                expected_m,
                disable_ue8m0_cast=not self.quant_config.deepgemm_scale_ue8m0,
            )

            act_out_fp8, scale = fastdeploy.model_executor.ops.gpu.fused_mask_swiglu_fp8_quant(
                up_gate_proj_out,
                token_nums_per_expert,
                self.quant_config.weight_block_size[0],
                use_ue8m0=self.quant_config.deepgemm_scale_ue8m0,
            )

            if layer.hidden_size == layer.moe_intermediate_size * 2:
                ffn_out = up_gate_proj_out
            else:
                ffn_out = paddle.empty(
                    [
                        layer.num_local_experts,
                        layer.ep_size * max_tokens_per_rank,
                        layer.hidden_size,
                    ],
                    dtype=paddle.bfloat16,
                )

            m_grouped_fp8_gemm_nt_masked(
                (act_out_fp8, scale),
                (
                    getattr(layer, self.added_weight_attrs[1]),
                    getattr(layer, self.added_scale_attrs[1]),
                ),
                ffn_out,
                token_nums_per_expert,
                expected_m,
                disable_ue8m0_cast=not self.quant_config.deepgemm_scale_ue8m0,
            )

            tmp_ffn_out = call_depermute_prefill_combine(
                x=ffn_out,
                indice_map=permuted_indice_map,
                topk_weights=recv_topk_weights,
                num_worst_tokens=recv_x_value.shape[0],
            )

        elif token_all_num > 0:
            logger.debug(f"token_all_num {token_all_num}")

            if fastdeploy.envs.FD_USE_PHI_MOE_PERMUTE:
                recv_topk_idx = recv_topk_idx.astype(paddle.int32)
                (
                    permute_input,
                    permute_indices_per_token,  # == zipped_expertwise_rowmap
                    dst_weights,
                    permute_scale,
                    m_indices,
                ) = paddle.nn.functional.moe_permute(
                    hidden_states=recv_x_value,
                    scale=recv_x_scale,
                    expert_routemap_topk=recv_topk_idx,
                    expert_prob_topk=recv_topk_weights,
                    num_experts=layer.num_local_experts,
                    tokens_per_expert=[],
                    padding_alignment=128,
                    return_expert_indices=True,
                    override_buffer_size=token_all_num,
                    using_ue8m0_scale=self.quant_config.deepgemm_scale_ue8m0,
                )
            else:
                token_nums_this_rank = count_tokens_per_expert_func(recv_topk_idx, layer.num_local_experts, False)
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
                    recv_x_value,
                    recv_x_scale,
                    recv_topk_idx,
                    recv_topk_weights,
                    token_nums_this_rank[0],
                    token_nums_this_rank[1],
                    True,  # use_in_ep
                    token_all_num,
                )

            assert permute_input.shape[0] == token_all_num

            if permute_scale.strides[0] != 1:
                permute_scale = permute_scale.transpose([1, 0]).contiguous().transpose([1, 0])

            # up_gate_proj
            ffn_out = paddle.empty(
                (token_all_num, getattr(layer, self.added_weight_attrs[0]).shape[1]),
                dtype=paddle.bfloat16,
            )
            m_grouped_fp8_gemm_nt_contiguous(
                (permute_input, permute_scale),
                (getattr(layer, self.added_weight_attrs[0]), getattr(layer, self.added_scale_attrs[0])),
                ffn_out,
                m_indices,
            )

            if fastdeploy.envs.FD_MOE_PROB_IN_ADVANCE:
                ffn_in_x, ffn_in_x_scale_tensor = paddlefleet_ops.fuse_weighted_swiglu_fp8_quant(
                    ffn_out, dst_weights, using_pow2_scaling=True, use_ue8m0=self.quant_config.deepgemm_scale_ue8m0
                )

                ffn_in_x_scale_tensor = paddle.transpose(
                    paddle.transpose(ffn_in_x_scale_tensor, [1, 0]).contiguous(), [1, 0]
                )
            else:
                # swiglu
                ffn_out = paddle.incubate.nn.functional.swiglu(ffn_out, None)

                # down_proj
                if not fastdeploy.envs.FD_USE_PHI_FP8_QUANT:
                    ffn_in_x, ffn_in_x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant(
                        ffn_out, self.quant_config.weight_block_size[0], self.quant_config.deepgemm_scale_ue8m0
                    )
                    ffn_in_x_scale_tensor = ffn_in_x_scale_tensor.transpose([1, 0]).contiguous().transpose([1, 0])
                else:
                    ffn_in_x, ffn_in_x_scale_tensor = paddle.incubate.nn.functional.fp8_quant_blockwise(
                        ffn_out,
                        using_pow2_scale=self.quant_config.deepgemm_scale_ue8m0
                        or fastdeploy.envs.FD_FP8_QUANT_WITH_POW2SCALE,
                        using_ue8m0_scale=self.quant_config.deepgemm_scale_ue8m0,
                    )
                    ffn_in_x_scale_tensor = ffn_in_x_scale_tensor.T[: ffn_in_x.shape[0]]

            ffn_out = paddle.empty(
                (token_all_num, getattr(layer, self.added_weight_attrs[1]).shape[1]),
                dtype=paddle.bfloat16,
            )
            m_grouped_fp8_gemm_nt_contiguous(
                (ffn_in_x, ffn_in_x_scale_tensor),
                (getattr(layer, self.added_weight_attrs[1]), getattr(layer, self.added_scale_attrs[1])),
                ffn_out,
                m_indices,
            )
            if fastdeploy.envs.FD_USE_PHI_MOE_PERMUTE:
                tmp_ffn_out, out_probs = paddle.nn.functional.moe_unpermute(
                    hidden_states_unzipped=ffn_out,
                    zipped_expertwise_rowmap=permute_indices_per_token,
                    expert_routemap_topk=recv_topk_idx,
                    token_prob_unzipped=dst_weights,
                    total_zipped_tokens=recv_x_value.shape[0],
                    num_experts=layer.num_local_experts,
                    using_weighted_combine=not fastdeploy.envs.FD_MOE_PROB_IN_ADVANCE,
                )

            else:
                # prmt back per rank
                tmp_ffn_out = fastdeploy.model_executor.ops.gpu.ep_moe_expert_combine(
                    ffn_out,
                    dst_weights,
                    permute_indices_per_token,
                    dst_indices,
                    None,  # down_proj_bias
                    False,  # norm_topk_prob
                    1.0,
                )
        else:
            tmp_ffn_out = paddle.empty([0, hidden_size], paddle.bfloat16)

        if shared_experts is not None:
            s_x = shared_experts(x)

        # 5. EP combine
        event = deep_ep.Buffer.capture()
        if self.ep_prefill_runner.num_worst_tokens <= 0:
            let_another_thread_run()

        global_values[thread_name]["combine_in"] = tmp_ffn_out
        tmp_ffn_out, event = self.ep_prefill_runner.combine(tmp_ffn_out, handle, recv_topk_weights, event)

        if self.ep_prefill_runner.num_worst_tokens > 0:
            let_another_thread_run()

        if self.ep_prefill_runner.ep_engine.async_finish:
            event.current_stream_wait()

        global_values[thread_name]["combine_out"] = tmp_ffn_out
        if shared_experts is not None:
            tmp_ffn_out += s_x

        return tmp_ffn_out

    def apply_ep_decode(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        shared_experts: nn.Layer = None,
    ) -> paddle.Tensor:
        """
        Apply the EP decoder method.
        """
        gate_out = gate(x)
        gate_out = gate_out.cast("float32")
        # 1. Select topk experts and weights
        topk_idx, topk_weights = self.ep_decoder_runner.moe_select(layer, gate_out)

        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids=topk_idx)

        # 2. EP Dispatch
        permute_input, token_nums_per_expert, handle = self.ep_decoder_runner.dispatch(
            x, topk_idx, topk_weights, use_fp8=True, use_ue8m0=self.quant_config.deepgemm_scale_ue8m0
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
        # disable_ue8m0_cast is False for SM100
        m_grouped_fp8_gemm_nt_masked(
            permute_input,
            (
                getattr(layer, self.added_weight_attrs[0]),
                getattr(layer, self.added_scale_attrs[0]),
            ),
            up_gate_proj_out,
            token_nums_per_expert,
            expected_m,
        )

        act_out_fp8, scale = fastdeploy.model_executor.ops.gpu.fused_mask_swiglu_fp8_quant(
            up_gate_proj_out,
            token_nums_per_expert,
            self.quant_config.weight_block_size[0],
            use_ue8m0=self.quant_config.deepgemm_scale_ue8m0,
        )

        # disable_ue8m0_cast is False for SM100
        m_grouped_fp8_gemm_nt_masked(
            (act_out_fp8, scale),
            (
                getattr(layer, self.added_weight_attrs[1]),
                getattr(layer, self.added_scale_attrs[1]),
            ),
            ffn_out,
            token_nums_per_expert,
            expected_m,
        )

        if shared_experts is not None:
            s_x = shared_experts(x)

        # 4. EP combine
        out = self.ep_decoder_runner.combine(ffn_out, topk_idx, topk_weights, handle)

        if shared_experts is not None:
            out += s_x
        return out

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
    ) -> paddle.Tensor:
        """
        Paddle Use DeepGemm compute Fused MoE.
        below is TP compute method.
        """
        gate_out = gate(x)
        gate_out = gate_out.cast("float32")

        if layer.topk_method == "noaux_tc":
            _, topk_weights, topk_ids = fastdeploy.model_executor.layers.moe.moe.get_moe_scores(
                gate_out,
                layer.n_group,
                layer.topk_group,
                layer.top_k,
                layer.routed_scaling_factor,
                layer.gate_correction_bias,
                getattr(layer, "renormalize", True),
                topk_reduce_func=getattr(layer, "topk_reduce_func", None),
            )
        else:
            topk_ids, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
                gate_out,
                layer.gate_correction_bias,
                layer.top_k,
                True,  # apply_norm_weight
                False,
            )

        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids=topk_ids)

        if not fastdeploy.envs.FD_USE_PHI_FP8_QUANT:
            recv_x, recv_x_scale = fastdeploy.model_executor.ops.gpu.per_token_quant(
                x, 128, self.quant_config.deepgemm_scale_ue8m0
            )
        else:
            recv_x, recv_x_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                x,
                using_pow2_scale=self.quant_config.deepgemm_scale_ue8m0 or fastdeploy.envs.FD_FP8_QUANT_WITH_POW2SCALE,
                output_scale_transpose=self.quant_config.deepgemm_scale_ue8m0,
                using_ue8m0_scale=self.quant_config.deepgemm_scale_ue8m0,
            )
            recv_x_scale = (
                recv_x_scale[: recv_x.shape[0]]
                if not self.quant_config.deepgemm_scale_ue8m0
                else recv_x_scale.T[: recv_x.shape[0]]
            )

        if fastdeploy.envs.FD_USE_PHI_MOE_PERMUTE:
            topk_ids = topk_ids.astype(paddle.int32)
            override_buffer_size = recv_x.shape[0] * layer.top_k + layer.num_experts * (128 - 1)
            (
                permute_input,
                permute_indices_per_token,  # == zipped_expertwise_rowmap
                dst_weights,
                permute_scale,
                m_indices,
            ) = paddle.nn.functional.moe_permute(
                hidden_states=recv_x,
                scale=recv_x_scale,
                expert_routemap_topk=topk_ids,
                expert_prob_topk=topk_weights,
                num_experts=layer.num_experts,
                tokens_per_expert=[],
                padding_alignment=128,
                return_expert_indices=True,
                override_buffer_size=override_buffer_size,
                using_ue8m0_scale=self.quant_config.deepgemm_scale_ue8m0,
            )
        else:
            tmp = count_tokens_per_expert_func(topk_ids, layer.num_experts, False)
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

        ffn_out = m_grouped_fp8_gemm_nt_contiguous_custom_python_op(
            permute_input,
            permute_scale,
            getattr(layer, self.added_weight_attrs[0]),
            getattr(layer, self.added_scale_attrs[0]),
            m_indices,
            getattr(layer, self.added_weight_attrs[1]),
            getattr(layer, self.added_scale_attrs[1]),
            self.quant_config.weight_block_size[0],
            disable_ue8m0_cast=not self.quant_config.deepgemm_scale_ue8m0,
            dst_weights=dst_weights if fastdeploy.envs.FD_MOE_PROB_IN_ADVANCE else None,
        )

        # prmt back per rank
        if fastdeploy.envs.FD_USE_PHI_MOE_PERMUTE:
            tmp_ffn_out, out_probs = paddle.nn.functional.moe_unpermute(
                hidden_states_unzipped=ffn_out,
                zipped_expertwise_rowmap=permute_indices_per_token,
                expert_routemap_topk=topk_ids,
                token_prob_unzipped=dst_weights,
                total_zipped_tokens=recv_x.shape[0],
                num_experts=layer.num_experts,
                using_weighted_combine=not fastdeploy.envs.FD_MOE_PROB_IN_ADVANCE,
            )
        else:
            tmp_ffn_out = fastdeploy.model_executor.ops.gpu.ep_moe_expert_combine(
                ffn_out,
                dst_weights,
                permute_indices_per_token,
                dst_indices,
                None,
                False,  # norm_topk_prob
                1.0,
            )
        return tmp_ffn_out
