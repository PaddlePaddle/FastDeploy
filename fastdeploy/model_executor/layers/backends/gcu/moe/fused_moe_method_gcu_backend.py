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

import multiprocessing
import os

import numpy as np
import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.model_executor.layers.moe.fused_moe_backend_base import (
    UnquantizedFusedMoEMethod,
)
from fastdeploy.model_executor.layers.utils import (
    CpuGuard,
    create_and_set_parameter,
    get_tensor,
)
from fastdeploy.model_executor.ops.gcu import (
    invoke_fused_moe_kernel,
    moe_align_block_size,
    topk_softmax,
    weight_quantize_custom_rtn,
    weight_quantize_rtn,
)


class GCUFusedMoeMethod(UnquantizedFusedMoEMethod):
    """
    Use GCU to compute Fused MoE.
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.group_size = -1

    def process_loaded_weights(self, layer: nn.Layer, state_dict):
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)
        stacked_up_gate_proj_weights = paddle.stack(up_gate_proj_weights, axis=0)
        stacked_down_proj_weights = paddle.stack(down_proj_weights, axis=0)
        layer.up_gate_proj_weight.set_value(paddle.transpose(stacked_up_gate_proj_weights, [0, 2, 1]))
        layer.down_proj_weight.set_value(paddle.transpose(stacked_down_proj_weights, [0, 2, 1]))

    @paddle.no_grad()
    def compute_ffn(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
        enable_quant=False,
    ) -> paddle.Tensor:
        """
        Paddle gcu compute Fused MoE.
        """
        token_num, hidden_size = x.shape
        top_k = layer.top_k
        moe_intermediate_size = layer.moe_intermediate_size
        num_experts = layer.num_local_experts

        topk_weights = paddle.empty([token_num, top_k], dtype=gate_out.dtype)
        topk_indices = paddle.empty([token_num, top_k], dtype="int32")
        token_expert_indices = paddle.empty(
            [token_num, top_k],
            dtype="int32",
        )
        topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gate_out,
            norm_topk_prob=True,
        )

        config = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
        }

        block_size = config["BLOCK_SIZE_M"]
        max_num_tokens_padded = np.prod(topk_indices.shape) + num_experts * (block_size - 1)
        max_num_m_blocks = max_num_tokens_padded // block_size
        sorted_token_ids = paddle.empty([max_num_tokens_padded], dtype="int32")
        expert_ids = paddle.zeros(shape=[max_num_m_blocks], dtype="int32")
        num_tokens_post_pad = paddle.empty([1], dtype="int32")

        sorted_token_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            topk_indices,
            num_experts,
            block_size,
        )

        intermediate_cache1 = paddle.empty(
            [token_num, top_k, moe_intermediate_size * 2],
            dtype=x.dtype,
        )

        up_gate_proj_B_scale = layer.up_gate_proj_weight_scale if enable_quant else None
        up_gate_proj_B_zeros = layer.up_gate_proj_weight_zeros if enable_quant else None

        invoke_fused_moe_kernel(
            x,  # input
            layer.up_gate_proj_weight,  # weight
            intermediate_cache1,  # output
            None,  # A_scale
            up_gate_proj_B_scale,  # B_scale
            up_gate_proj_B_zeros,  # B_zp
            topk_weights,
            topk_indices,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            False,  # mul_routed_weight
            top_k,
            config,
            enable_quant,  # use_int4_w4a16
            [0, self.group_size],  # block_shape
        )

        intermediate_cache2 = paddle.empty(
            (token_num, top_k, moe_intermediate_size),
            dtype=x.dtype,
        )

        intermediate_cache2 = paddle.incubate.nn.functional.swiglu(intermediate_cache1)

        intermediate_cache2 = intermediate_cache2.reshape([-1, moe_intermediate_size])

        intermediate_cache3 = paddle.empty(
            (token_num, top_k, hidden_size),
            dtype=x.dtype,
        )

        down_proj_B_scale = layer.down_proj_weight_scale if enable_quant else None
        down_proj_B_zeros = layer.down_proj_weight_zeros if enable_quant else None

        invoke_fused_moe_kernel(
            intermediate_cache2,  # input
            layer.down_proj_weight,  # weight
            intermediate_cache3,  # output
            None,  # A_scale
            down_proj_B_scale,  # B_scale
            down_proj_B_zeros,  # B_zp
            topk_weights,
            topk_indices,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            True,  # mul_routed_weight
            1,
            config,
            enable_quant,  # use_int4_w4a16
            [0, self.group_size],  # block_shape
        )

        intermediate_cache3.reshape_([token_num, top_k, hidden_size])
        fused_moe_out = intermediate_cache3.sum(axis=1)
        fused_moe_out = fused_moe_out.reshape_([token_num, hidden_size])

        return fused_moe_out

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Paddle gcu compute Fused MoE.
        """
        gate_out = gate(x.cast("float32"))
        return self.compute_ffn(layer, x, gate_out, enable_quant=False)

    def apply_ep_prefill(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Apply the EP prefill method.
        """
        raise NotImplementedError

    def apply_ep_decode(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Apply the EP decoder method.
        """
        raise NotImplementedError

    def apply_tp(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Paddle Cutlass compute Fused MoE.
        """
        raise NotImplementedError


class GCUWeightOnlyMoEMethod(GCUFusedMoeMethod):
    """
    weight only for moe
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.quant_config = quant_config
        self.moe_quant_type = self.quant_config.algo
        self.pack_num = 1

        assert (
            self.quant_config.algo == "weight_only_int4"
        ), "GCUWeightOnlyMoEMethod only support weight_only_int4, but got:{self.quant_config.algo}"

        self.added_qzeros_attrs = [
            "up_gate_proj_weight_zeros",
            "down_proj_weight_zeros",
        ]
        self.group_size = 64

        self.quant_multi_process_group_size = int(os.getenv("FD_MOE_QUANT_MULTI_PROCESS_GROUP_SIZE", 8))
        logger.info(f"GCUWeightOnlyMoEMethod quant_multi_process_group_size: {self.quant_multi_process_group_size}")

    def process_prequanted_weights(self, layer: nn.Layer, state_dict, is_rearrange: bool = False):
        """
        Paddle gcu process prequanted weights.
        """
        up_gate_proj_expert_weight_key = layer.weight_key_map.get("up_gate_proj_expert_weight_key", None)
        down_proj_expert_weight_key = layer.weight_key_map.get("down_proj_expert_weight_key", None)
        up_gate_proj_expert_weight_scale_key = layer.weight_key_map.get("up_gate_proj_expert_weight_scale_key", None)
        down_proj_expert_weight_scale_key = layer.weight_key_map.get("down_proj_expert_weight_scale_key", None)

        up_gate_proj_weights, down_proj_weights, _, _ = layer.load_experts_weight(
            state_dict,
            up_gate_proj_expert_weight_key,
            down_proj_expert_weight_key,
        )
        # self.check(layer, up_gate_proj_weights, down_proj_weights)
        up_gate_proj_weight_scale = []
        down_proj_weight_scale = []
        for i in range(layer.num_experts):
            expert_idx = layer.expert_id_offset + i
            up_gate_proj_weight_scale.append(
                get_tensor(state_dict.pop(up_gate_proj_expert_weight_scale_key.format(expert_idx)))
            )
            down_proj_weight_scale.append(
                get_tensor(state_dict.pop(down_proj_expert_weight_scale_key.format(expert_idx)))
            )

        up_gate_proj_weight = paddle.stack(up_gate_proj_weights, axis=0)
        down_proj_weight = paddle.stack(down_proj_weights, axis=0)
        up_gate_proj_weight_scale = paddle.stack(up_gate_proj_weight_scale, axis=0)
        down_proj_weight_scale = paddle.stack(down_proj_weight_scale, axis=0)

        name_tensor_map = {
            "up_gate_proj_weight": up_gate_proj_weight,
            "down_proj_weight": down_proj_weight,
            "up_gate_proj_weight_scale": up_gate_proj_weight_scale,
            "down_proj_weight_scale": down_proj_weight_scale,
        }
        for name, tensor in name_tensor_map.items():
            create_and_set_parameter(layer, name, tensor)

    @paddle.no_grad()
    def create_weights(self, layer: nn.Layer, state_dict):
        """
        Paddle cutlass create weight process.
        """
        up_gate_proj_weights, down_proj_weights, _, _ = layer.extract_moe_ffn_weights(state_dict)
        self.check(layer, up_gate_proj_weights, down_proj_weights)

        def quant_worker(p_group_idx, shared_dict, weights, moe_quant_type, group_size):
            with CpuGuard():
                p_group_size = len(weights)
                for group_j in range(p_group_size):
                    # weight shape [K, N] -> [N/2, K] -> [N, K/2]
                    quant_weight, scale = weight_quantize_custom_rtn(
                        weights[group_j],
                        moe_quant_type,
                        group_size,  # group_size
                    )
                    shared_dict[p_group_size * p_group_idx + group_j] = (
                        quant_weight,
                        scale,
                    )

        for idx, weight_tensor in enumerate([up_gate_proj_weights, down_proj_weights]):
            weight_name = self.added_weight_attrs[idx]
            scale_name = self.added_scale_attrs[idx]
            zeros_name = self.added_qzeros_attrs[idx]

            if self.quant_multi_process_group_size > 0:
                process_group_size = self.quant_multi_process_group_size
                process_group_num = layer.num_local_experts // process_group_size
                grouped_weights_num = process_group_num * process_group_size
                remain_weights_start_idx = grouped_weights_num

                weight_list = [None] * grouped_weights_num
                weight_scale_list = [None] * grouped_weights_num

                with multiprocessing.Manager() as manager:
                    shared_dict = manager.dict({})
                    processes = []

                    for i in range(process_group_num):
                        w = []
                        for j in range(process_group_size):
                            w.append(weight_tensor[process_group_size * i + j].to("cpu"))

                        p = multiprocessing.Process(
                            target=quant_worker,
                            args=(
                                i,
                                shared_dict,
                                w,
                                self.moe_quant_type,
                                self.group_size,
                            ),
                        )
                        p.start()
                        processes.append(p)

                    for p in processes:
                        p.join()

                    dict_ = dict(shared_dict)

                    for k, v in dict_.items():
                        weight_list[k] = v[0].to(up_gate_proj_weights[0].place)
                        weight_scale_list[k] = v[1].to(up_gate_proj_weights[0].place)
            else:
                remain_weights_start_idx = 0

            if remain_weights_start_idx < layer.num_local_experts:
                for i in range(remain_weights_start_idx, layer.num_local_experts):
                    # weight shape [K, N] -> [N/2, K] -> [N, K/2]
                    quant_weight, scale = weight_quantize_rtn(
                        weight_tensor[i],
                        self.moe_quant_type,
                        self.group_size,  # group_size
                    )
                    weight_list.append(quant_weight)
                    weight_scale_list.append(scale)
            quanted_weight = paddle.stack(weight_list, axis=0)
            create_and_set_parameter(layer, weight_name, quanted_weight)

            quanted_weight_scale = paddle.stack(weight_scale_list, axis=0)
            create_and_set_parameter(layer, scale_name, quanted_weight_scale)

            quanted_weight_zeros = quanted_weight_scale * 8
            create_and_set_parameter(layer, zeros_name, quanted_weight_zeros)

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate: nn.Layer,
    ) -> paddle.Tensor:
        """
        Paddle gcu compute Fused MoE.
        """
        gate_out = gate(x.cast("float32"))
        return self.compute_ffn(layer, x, gate_out, enable_quant=True)
