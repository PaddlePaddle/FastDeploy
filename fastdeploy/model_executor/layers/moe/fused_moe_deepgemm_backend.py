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
import paddle.nn.functional as F
from paddle import nn
from paddleformers.utils.log import logger
import fastdeploy
from fastdeploy.model_executor.layers.moe.ep import deep_ep, EPRunner, FakeEPRunner
from fastdeploy.model_executor.layers.quantization.fp8_utils import (
    deep_gemm,
    paddlefleet_ops,
    _interleave_weights,
    _transpose_sf_for_utccp,
)
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.ops.gpu import (
    count_tokens_per_expert_func,
    depermute_prefill_combine,
    prefill_permute_to_masked_gemm,
    mega_moe_pre_dispatch,
)
from fastdeploy.platforms import current_platform
from fastdeploy.utils import ceil_div, register_custom_python_op, singleton
from fastdeploy.worker.tbo import let_another_thread_run
from fastdeploy.model_executor.utils import (
    TensorTracker,
    set_weight_attrs,
    free_tensor,
    weight_fully_copied,
    get_sm_version,
)
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

    results = prefill_permute_to_masked_gemm(x, scale, topk_ids, num_local_experts, max_token_num, False)

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
        fc1_latent_proj: nn.Layer = None,
        fc2_latent_proj: nn.Layer = None,
    ) -> paddle.Tensor:
        """
        Apply the EP prefill method.
        """
        gate_out = gate(x)
        gate_out = gate_out.cast("float32")

        hidden_size = layer.hidden_size

        # 1. Select topk experts and weights
        topk_idx, topk_weights = EPRunner.moe_select(layer, gate_out)

        if layer.routed_scaling_factor_learnable:
            safe_topk_indices = paddle.clip(topk_idx, min=0)
            gathered_scales = F.embedding(safe_topk_indices, layer.per_expert_scale.unsqueeze(1)).squeeze(-1)
            topk_weights = topk_weights * gathered_scales

        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids=topk_idx)

        if fc1_latent_proj:
            x = fc1_latent_proj(x)

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

        if fc2_latent_proj:
            tmp_ffn_out = fc2_latent_proj(tmp_ffn_out)

        return tmp_ffn_out

    def apply_ep_decode(
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
        Apply the EP decoder method.
        """
        gate_out = gate(x)
        gate_out = gate_out.cast("float32")
        # 1. Select topk experts and weights
        topk_idx, topk_weights = EPRunner.moe_select(layer, gate_out)

        if layer.routed_scaling_factor_learnable:
            safe_topk_indices = paddle.clip(topk_idx, min=0)
            gathered_scales = F.embedding(safe_topk_indices, layer.per_expert_scale.unsqueeze(1)).squeeze(-1)
            topk_weights = topk_weights * gathered_scales

        if topk_ids_hookfunc is not None:
            topk_ids_hookfunc(topk_ids=topk_idx)

        # 2. EP Dispatch
        if fc1_latent_proj:
            x = fc1_latent_proj(x)

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

        if fc2_latent_proj:
            out = fc2_latent_proj(out)

        return out

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
        topk_ids_hookfunc: Callable = None,
        fc1_latent_proj: nn.Layer = None,
        fc2_latent_proj: nn.Layer = None,
    ) -> paddle.Tensor:
        """
        Paddle Use DeepGemm compute Fused MoE.
        below is TP compute method.
        """
        gate_out = gate(x)

        if layer.topk_method == "noaux_tc":
            use_fused = (
                layer.fd_config.scheduler_config.enable_moe_scores_elementwise_fuse and current_platform.is_cuda()
            )
            if not use_fused:
                gate_out = gate_out.cast("float32")
            _, topk_weights, topk_ids = fastdeploy.model_executor.layers.moe.moe.get_moe_scores(
                gate_out,
                layer.n_group,
                layer.topk_group,
                layer.top_k,
                layer.routed_scaling_factor,
                layer.gate_correction_bias,
                getattr(layer, "renormalize", True),
                topk_reduce_func=getattr(layer, "topk_reduce_func", None),
                use_fused_cast=use_fused,
            )
        else:
            gate_out = gate_out.cast("float32")
            topk_ids, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
                gate_out,
                layer.gate_correction_bias,
                layer.top_k,
                True,  # apply_norm_weight
                False,
            )

        if layer.routed_scaling_factor_learnable:
            safe_topk_indices = paddle.clip(topk_ids, min=0)
            gathered_scales = F.embedding(safe_topk_indices, layer.per_expert_scale.unsqueeze(1)).squeeze(-1)
            topk_weights = topk_weights * gathered_scales

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


@singleton
class MegaMoEBuffer:
    """
    A wrapper class for DeepEP engine.
    Manages buffer lifecycle based on role and phase.
    """

    def __init__(
        self,
        ep_group,
        num_experts: int,
        num_max_tokens_per_rank: int,
        top_k: int,
        hidden_size: int,
        moe_intermediate_size: int,
    ):
        self.buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            ep_group,
            num_experts,
            num_max_tokens_per_rank,
            top_k,
            hidden_size,
            moe_intermediate_size,
        )


class DeepGemmMegaMoEMethod(DeepGemmFusedMoeMethod):
    def __init__(self, quant_config):
        if not get_sm_version() >= 100:
            raise ValueError("MegaMoE now only support sm100+ devices.")
        super().__init__(quant_config)
        self.added_scale_attrs = ["up_gate_proj_weight_scale", "down_proj_weight_scale"]
        self.quant_config.deepgemm_scale_ue8m0 = True
        self.gran_k = 32

    def create_weights(self, layer: nn.Layer, **extra_weight_attrs):
        """
        Triton MoE create weight process.
        """
        logger.info("mega create_weights")
        self.model_format = extra_weight_attrs.get("model_format")
        self.up_gate_proj_quant_weight_shape = [
            layer.num_local_experts,
            layer.moe_intermediate_size * 2,
            layer.hidden_size,
        ]
        self.down_proj_quant_weight_shape = [
            layer.num_local_experts,
            layer.hidden_size,
            layer.moe_intermediate_size,
        ]
        self.up_gate_proj_pretranspose_weight_shape = [
            layer.num_local_experts,
            layer.hidden_size,
            layer.moe_intermediate_size * 2,
        ]
        self.down_proj_pretranspose_weight_shape = [
            layer.num_local_experts,
            layer.moe_intermediate_size,
            layer.hidden_size,
        ]
        if self.model_format != "torch":
            self.up_gate_proj_bf16_weight_shape = self.up_gate_proj_pretranspose_weight_shape
            self.down_proj_bf16_weight_shape = self.down_proj_pretranspose_weight_shape
        else:
            self.up_gate_proj_bf16_weight_shape = self.up_gate_proj_quant_weight_shape
            self.down_proj_bf16_weight_shape = self.down_proj_quant_weight_shape
        self.up_gate_proj_packed_weight_shape = [
            layer.num_local_experts,
            layer.moe_intermediate_size * 2,
            layer.hidden_size // 2, # 4-bit packing
        ]
        self.down_proj_packed_weight_shape = [
            layer.num_local_experts,
            layer.hidden_size,
            layer.moe_intermediate_size // 2, # 4-bit packing
        ]
        up_num_scales = ceil_div(layer.hidden_size, self.gran_k)
        down_num_scales = ceil_div(layer.moe_intermediate_size, self.gran_k)
        self.up_gate_proj_scale_shape = [
            layer.num_local_experts,
            layer.moe_intermediate_size * 2,
            (up_num_scales + 3) // 4,
        ]
        self.down_proj_scale_shape = [
            layer.num_local_experts,
            layer.hidden_size,
            (down_num_scales + 3) // 4,
        ]
        self.up_gate_proj_weight_shape = self.up_gate_proj_quant_weight_shape
        self.down_proj_weight_shape = self.down_proj_quant_weight_shape

        logger.info(
            "MegaMoE create_weights: "
            f"is_checkpoint_bf16={self.quant_config.is_checkpoint_bf16}, "
            f"load_choices={layer.fd_config.load_config.load_choices}, "
            f"model_format={self.model_format}, "
            f"up_gate_bf16_shape={self.up_gate_proj_bf16_weight_shape}, "
            f"down_bf16_shape={self.down_proj_bf16_weight_shape}, "
            f"up_gate_quant_shape={self.up_gate_proj_quant_weight_shape}, "
            f"down_quant_shape={self.down_proj_quant_weight_shape}, "
            f"up_gate_packed_shape={self.up_gate_proj_packed_weight_shape}, "
            f"down_packed_shape={self.down_proj_packed_weight_shape}, "
            f"up_gate_scale_shape={self.up_gate_proj_scale_shape}, "
            f"down_scale_shape={self.down_proj_scale_shape}"
        )

        if self.quant_config.is_checkpoint_bf16 and layer.fd_config.load_config.load_choices == "default_v1":
            if self.model_format != "torch":
                up_gate_proj_attrs = {
                    **extra_weight_attrs,
                    "tensor_track": TensorTracker(shape=self.up_gate_proj_bf16_weight_shape, output_dim=True),
                    "SHARD_ID_TO_SHARDED_DIM": {"gate": 1, "down": 0, "up": 1},
                }
                down_proj_attrs = {
                    **extra_weight_attrs,
                    "tensor_track": TensorTracker(shape=self.down_proj_bf16_weight_shape, output_dim=False),
                    "SHARD_ID_TO_SHARDED_DIM": {"gate": 1, "down": 0, "up": 1},
                }
            else:
                up_gate_proj_attrs = {
                    **extra_weight_attrs,
                    "tensor_track": TensorTracker(shape=self.up_gate_proj_bf16_weight_shape, output_dim=False),
                    "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0},
                }
                down_proj_attrs = {
                    **extra_weight_attrs,
                    "tensor_track": TensorTracker(shape=self.down_proj_bf16_weight_shape, output_dim=True),
                    "SHARD_ID_TO_SHARDED_DIM": {"gate": 0, "down": 1, "up": 0},
                }
            layer.up_gate_proj_weight = layer.create_parameter(
                shape=self.up_gate_proj_bf16_weight_shape,
                dtype=layer.weight_dtype,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

            layer.down_proj_weight = layer.create_parameter(
                shape=self.down_proj_bf16_weight_shape,
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
            self.up_gate_proj_weight_shape = self.up_gate_proj_packed_weight_shape
            self.down_proj_weight_shape = self.down_proj_packed_weight_shape
            up_gate_proj_attrs = {}
            down_proj_attrs = {}

            self.weight_dtype = paddle.int8
            up_gate_proj_weight_name = self.added_weight_attrs[0]
            down_proj_weight_name = self.added_weight_attrs[1]
            up_gate_proj_scale_name = self.added_scale_attrs[0]
            down_proj_scale_name = self.added_scale_attrs[1]

            setattr(
                layer,
                up_gate_proj_weight_name,
                layer.create_parameter(
                    shape=self.up_gate_proj_packed_weight_shape,
                    dtype=self.weight_dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            setattr(
                layer,
                down_proj_weight_name,
                layer.create_parameter(
                    shape=self.down_proj_packed_weight_shape,
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
                    dtype="int32",
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            setattr(
                layer,
                down_proj_scale_name,
                layer.create_parameter(
                    shape=self.down_proj_scale_shape,
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

    def init_ep(self, layer: nn.Layer) -> None:
        logger.info("USE MegaMoE backend")
        if layer.ep_size <= 1:
            return

        config = layer.fd_config
        splitwise_role = config.scheduler_config.splitwise_role

        if splitwise_role == "mixed" or splitwise_role == "prefill":
            self.num_max_tokens_per_rank = config.scheduler_config.max_num_batched_tokens
        elif splitwise_role == "decode":
            num_spec_tokens = config.speculative_config.num_speculative_tokens
            self.num_max_tokens_per_rank = config.scheduler_config.max_num_seqs * (num_spec_tokens + 1)
        else:
            raise ValueError(f"Unsupported splitwise role: {splitwise_role}")

        self.mega_moe_buffer = MegaMoEBuffer(
            layer.fd_config.parallel_config.ep_group,
            layer.num_experts,
            self.num_max_tokens_per_rank,
            layer.top_k,
            layer.hidden_size,
            layer.moe_intermediate_size,
        ).buffer
        self.num_max_tokens_per_rank = self.mega_moe_buffer.num_max_tokens_per_rank
        self.cumulative_local_expert_recv_stats = paddle.zeros(
            (layer.num_local_experts,), dtype=paddle.int32
        )

        self.ep_prefill_runner = FakeEPRunner()
        self.ep_decoder_runner = FakeEPRunner()
    
    def process_weights_after_loading(self, layer):
        def cast_grouped_weights_to_fp4(bf16_weights: paddle.Tensor):
            num_groups, n, k = bf16_weights.shape
            w = paddle.empty((num_groups, n, k // 2), dtype=paddle.int8)
            w_sf = paddle.empty((num_groups, n, k // self.gran_k), dtype=paddle.float32)
            for i in range(num_groups):
                w[i], w_sf[i] = deep_gemm.per_token_cast_to_fp4(
                    bf16_weights[i], use_ue8m0=True, gran_k=self.gran_k
                )
            w = w.contiguous()
            w_sf = w_sf.contiguous()
            w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, self.gran_k), num_groups)
            return w, w_sf

        def _process_quantize_mega_moe(weight_type):
            weight_idx = 0 if weight_type == "gate_up" else 1
            weight_name = self.added_weight_attrs[weight_idx]
            scale_name = self.added_scale_attrs[weight_idx]
            weight = getattr(layer, weight_name)
            if not hasattr(weight, "tensor_track") or weight.tensor_track is None:
                return

            if self.model_format != "torch":
                weight = weight.transpose([0, 2, 1]).contiguous()

            expected_weight_shape = (
                self.up_gate_proj_quant_weight_shape if weight_type == "gate_up" else self.down_proj_quant_weight_shape
            )
            expected_packed_shape = (
                self.up_gate_proj_packed_weight_shape if weight_type == "gate_up" else self.down_proj_packed_weight_shape
            )
            expected_scale_shape = self.up_gate_proj_scale_shape if weight_type == "gate_up" else self.down_proj_scale_shape

            if list(weight.shape) != list(expected_weight_shape):
                raise ValueError(
                    f"MegaMoE {weight_type} BF16 weight shape mismatch for {weight_name}: "
                    f"got {list(weight.shape)}, expected {list(expected_weight_shape)}"
                )
            if weight.dtype != paddle.bfloat16:
                weight = weight.astype(paddle.bfloat16)
            weight = weight.contiguous()

            logger.info(
                f"MegaMoE dynamic quantize {weight_type}: input shape={list(weight.shape)}, dtype={weight.dtype}"
            )
            weight_quantized, scale = cast_grouped_weights_to_fp4(weight)
            if weight_type == "gate_up":
                weight_quantized, scale = _interleave_weights((weight_quantized, scale))
            scale = _transpose_sf_for_utccp(scale)

            if list(weight_quantized.shape) != list(expected_packed_shape):
                raise ValueError(
                    f"MegaMoE {weight_type} packed weight shape mismatch: "
                    f"got {list(weight_quantized.shape)}, expected {list(expected_packed_shape)}"
                )
            if list(scale.shape) != list(expected_scale_shape):
                raise ValueError(
                    f"MegaMoE {weight_type} scale shape mismatch: "
                    f"got {list(scale.shape)}, expected {list(expected_scale_shape)}"
                )
            if weight_quantized.dtype != paddle.int8:
                raise ValueError(f"MegaMoE {weight_type} packed weight dtype mismatch: got {weight_quantized.dtype}")
            if scale.dtype != paddle.int32:
                raise ValueError(f"MegaMoE {weight_type} scale dtype mismatch: got {scale.dtype}")

            free_tensor(getattr(layer, weight_name))
            setattr(
                layer,
                weight_name,
                layer.create_parameter(
                    shape=weight_quantized.shape,
                    dtype=paddle.int8,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ),
            )
            setattr(
                layer,
                scale_name,
                layer.create_parameter(
                    shape=scale.shape,
                    dtype=paddle.int32,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ).as_strided(scale.shape, scale.stride()),
            )
            getattr(layer, weight_name).copy_(weight_quantized, False)
            getattr(layer, scale_name).copy_(scale, False)
            logger.info(
                f"MegaMoE dynamic quantize {weight_type}: packed weight shape={list(weight_quantized.shape)}, "
                f"scale shape={list(scale.shape)}"
            )

        if not self.quant_config.is_checkpoint_bf16:
            return
        if hasattr(layer, "up_gate_proj_weight") and weight_fully_copied(layer.up_gate_proj_weight):
            _process_quantize_mega_moe("gate_up")
        if hasattr(layer, "down_proj_weight") and weight_fully_copied(layer.down_proj_weight):
            _process_quantize_mega_moe("down")

    def process_prequanted_weights(self, layer: nn.Layer, state_dict, is_rearrange: bool = False):
        """
        Paddle cutlass process prequanted weights.
        """
        logger.info(f"start process_prequanted_weights in megamoe")
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

            up_gate_weight_scale = get_tensor(
                (
                    state_dict.pop(up_gate_proj_expert_weight_scale_key_name)
                    if up_gate_proj_expert_weight_scale_key_name in state_dict
                    else up_gate_proj_expert_weight_scale_key_name
                ),
                layer.fd_config.model_config.model,
            )
            down_weight_scale = get_tensor(
                (
                    state_dict.pop(down_proj_expert_weight_scale_key_name)
                    if down_proj_expert_weight_scale_key_name in state_dict
                    else down_proj_expert_weight_scale_key_name
                ),
                layer.fd_config.model_config.model,
            )

            up_gate_proj_weight_scale.append(
                up_gate_weight_scale
            )
            down_proj_weight_scale.append(
                down_weight_scale
            )


        up_gate_proj_weight = (
            paddle.stack(up_gate_proj_weights, axis=0)
        )
        down_proj_weight = (
            paddle.stack(down_proj_weights, axis=0)
        )
        up_gate_proj_weight_scale = paddle.stack(up_gate_proj_weight_scale, axis=0).transpose([0, 2, 1])
        down_proj_weight_scale = paddle.stack(down_proj_weight_scale, axis=0).transpose([0, 2, 1])

        name_tensor_map = {
            self.added_weight_attrs[0]: up_gate_proj_weight,
            self.added_weight_attrs[1]: down_proj_weight,
            self.added_scale_attrs[0]: up_gate_proj_weight_scale,
            self.added_scale_attrs[1]: down_proj_weight_scale,
        }
        for name, tensor in name_tensor_map.items():
            getattr(layer, name).data = tensor

    def apply_ep_prefill(self, layer, x, gate, topk_ids_hookfunc, shared_experts, fc1_latent_proj, fc2_latent_proj):
        return self.apply_mage_moe(
            layer, x, gate, topk_ids_hookfunc, shared_experts, fc1_latent_proj, fc2_latent_proj
        )

    def apply_ep_decode(self, layer, x, gate, topk_ids_hookfunc, shared_experts, fc1_latent_proj, fc2_latent_proj):
        return self.apply_mage_moe(
            layer, x, gate, topk_ids_hookfunc, shared_experts, fc1_latent_proj, fc2_latent_proj
        )

    def apply_mage_moe(self, layer, x, gate, topk_ids_hookfunc, shared_experts, fc1_latent_proj, fc2_latent_proj):
        hidden_size = layer.hidden_size
        num_tokens = x.shape[0]

        gate_out = gate(x).cast("float32")

        # 1. Select topk experts and weights.
        topk_idx, topk_weights = EPRunner.moe_select(layer, gate_out)

        buffer_capacity = self.mega_moe_buffer.x.shape[0]
        if num_tokens > buffer_capacity:
            raise ValueError(
                f"MegaMoE buffer capacity exceeded: num_tokens={num_tokens}, capacity={buffer_capacity}"
            )

        # copy x, topk_idx, topk_weights to mega_moe_buffer and quantization.
        mega_moe_pre_dispatch(
            x,
            topk_idx,
            topk_weights,
            self.mega_moe_buffer.x,
            self.mega_moe_buffer.x_sf,
            self.mega_moe_buffer.topk_idx,
            self.mega_moe_buffer.topk_weights,
            self.num_max_tokens_per_rank,
            32, # group_size
        )

        l1_weight = getattr(layer, self.added_weight_attrs[0])
        l1_scale = getattr(layer, self.added_scale_attrs[0])
        l2_weight = getattr(layer, self.added_weight_attrs[1])
        l2_scale = getattr(layer, self.added_scale_attrs[1])
        y = paddle.empty((num_tokens, hidden_size), dtype=paddle.bfloat16)

        swiglu_limit = getattr(layer.fd_config.model_config, "swiglu_limit", 10)
        deep_gemm.fp8_fp4_mega_moe(
            y,
            (l1_weight, l1_scale),
            (l2_weight, l2_scale),
            self.mega_moe_buffer,
            cumulative_local_expert_recv_stats=self.cumulative_local_expert_recv_stats,
            recipe=(1, 1, self.gran_k),
            activation="swiglu",
            activation_clamp=swiglu_limit,
            fast_math=True,
        )

        return y
