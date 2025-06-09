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
import paddle
import fastdeploy
import fastdeploy.model_executor.ops.gpu.deep_gemm as deep_gemm
from fastdeploy.model_executor.layers.moe.moe import MoELayer


class MoeTPDecoerDeepDeepGEMMLayer(MoELayer):
    """
    MoeTPDecoerDeepDeepGEMMLayer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, **kwargs):
        """
        forward
        """
        gate_out = paddle.matmul(x.cast("float32"), self.gate_weight)
        if os.getenv("EP_DECODER_PERF_TEST", "False") == "True":
            gate_out = paddle.rand(shape=gate_out.shape, dtype=gate_out.dtype)
        ffn1_out = paddle.empty(
            [
                self.num_local_experts,
                self.max_batch_size,
                self.moe_intermediate_size * 2,
            ],
            dtype=self._dtype,
        )

        ffn_out = paddle.empty(
            [
                self.num_local_experts,
                self.max_batch_size,
                self.embed_dim,
            ],
            dtype=self._dtype,
        )

        topk_idx, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
            gate_out,
            (
                self.gate_correction_bias
                if self.moe_config.moe_use_gate_correction_bias
                else None
            ),
            self.top_k,
            True,  # apply_norm_weight
            False,
        )
        permute_input, token_nums_per_expert, permute_indices_per_token = (
            fastdeploy.model_executor.ops.gpu.moe_deepgemm_permute(
                x, topk_idx, self.num_local_experts, self.max_batch_size
            )
        )

        expected_m = 128

        permute_input_fp8, scale = fastdeploy.model_executor.ops.gpu.masked_per_token_quant(
            permute_input, token_nums_per_expert, 128
        )
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            (permute_input_fp8, scale),
            (
                self.moe_ffn1_weight,
                self.moe_ffn1_weight_scale,
            ),
            ffn1_out,
            token_nums_per_expert,
            expected_m,
        )

        act_out = fastdeploy.model_executor.ops.gpu.group_swiglu_with_masked(
            ffn1_out, token_nums_per_expert
        )

        act_out_fp8, scale = fastdeploy.model_executor.ops.gpu.masked_per_token_quant(
            act_out, token_nums_per_expert, 128
        )

        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            (act_out_fp8, scale),
            (
                self.moe_ffn2_weight,
                self.moe_ffn2_weight_scale,
            ),
            ffn_out,
            token_nums_per_expert,
            expected_m,
        )

        fused_moe_out = fastdeploy.model_executor.ops.gpu.moe_deepgemm_depermute(
            ffn_out, permute_indices_per_token, topk_idx, topk_weights
        )[0]

        return fused_moe_out


class MoeTPPrefillDeepDeepGEMMLayer(MoELayer):
    """
    MoeTPPrefillDeepDeepGEMMLayer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, **kwargs):
        """
        forward
        """
        raise NotImplementedError("Prefill is comming soon...")
