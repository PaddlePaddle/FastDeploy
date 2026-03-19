# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import unittest

import paddle

paddle.compat.enable_torch_proxy(scope={"flashinfer"})

from fastdeploy.model_executor.layers.moe.flashinfer_cutedsl_moe import (
    flashinfer_cutedsl_moe_masked,
)


class TestFlashinferCuteDslMoeMasked(unittest.TestCase):
    """Unit tests for flashinfer_cutedsl_moe_masked."""

    def test_flashinfer_cutedsl_moe_masked_runs_with_bf16_inputs(self):
        """
        Verify that flashinfer_cutedsl_moe_masked can run end-to-end with
        standard (bf16) inputs using real FlashInfer kernels.
        This directly exercises the path where hidden_states[1] is None.
        """

        num_experts = 2
        m = 3
        k = 64
        n = 32

        # Standard (non-prequantized) path: bf16 [num_experts, m, k], hidden_states[1] is None.
        a_bf16 = paddle.zeros([num_experts, m, k], dtype=paddle.bfloat16)
        hidden_states = (a_bf16, None)

        input_global_scale = paddle.ones([num_experts], dtype=paddle.float32)
        masked_m = paddle.full([num_experts], m, dtype=paddle.int32)

        w1 = paddle.zeros([num_experts, 2 * n, k // 2], dtype=paddle.uint8)
        # blockscale tensors must use float8_e4m3fn to satisfy runtime dtype checks
        w1_blockscale = paddle.zeros([1], dtype=paddle.float8_e4m3fn)
        w1_alpha = paddle.ones([num_experts], dtype=paddle.float32)

        w2 = paddle.zeros([num_experts, k, n // 2], dtype=paddle.uint8)
        a2_global_scale = paddle.ones([num_experts], dtype=paddle.float32)
        w2_blockscale = paddle.zeros([1], dtype=paddle.float8_e4m3fn)
        w2_alpha = paddle.ones([num_experts], dtype=paddle.float32)

        out = flashinfer_cutedsl_moe_masked(
            hidden_states=hidden_states,
            input_global_scale=input_global_scale,
            w1=w1,
            w1_blockscale=w1_blockscale,
            w1_alpha=w1_alpha,
            w2=w2,
            a2_global_scale=a2_global_scale,
            w2_blockscale=w2_blockscale,
            w2_alpha=w2_alpha,
            masked_m=masked_m,
        )
        self.assertEqual(list(out.shape), [num_experts, m, k])
        self.assertEqual(out.dtype, paddle.bfloat16)


if __name__ == "__main__":
    unittest.main()