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

import importlib
import sys
import types
import unittest

import paddle


def _install_fake_flashinfer_for_cutedsl(
    scaled_fp4_grouped_quantize=None,
    silu_and_mul_scaled_nvfp4_experts_quantize=None,
    grouped_gemm_nt_masked=None,
):
    """
    Install a fake flashinfer module (and cute_dsl.blockscaled_gemm submodule)
    so that importing flashinfer_cutedsl_moe does not require the real
    flashinfer (and thus does not import torch).
    """
    prev_flashinfer = sys.modules.get("flashinfer")
    prev_cute_dsl = sys.modules.get("flashinfer.cute_dsl")
    prev_blockscaled = sys.modules.get("flashinfer.cute_dsl.blockscaled_gemm")

    fake_flashinfer = types.ModuleType("flashinfer")
    if scaled_fp4_grouped_quantize is not None:
        fake_flashinfer.scaled_fp4_grouped_quantize = scaled_fp4_grouped_quantize
    if silu_and_mul_scaled_nvfp4_experts_quantize is not None:
        fake_flashinfer.silu_and_mul_scaled_nvfp4_experts_quantize = silu_and_mul_scaled_nvfp4_experts_quantize

    fake_blockscaled = types.ModuleType("flashinfer.cute_dsl.blockscaled_gemm")
    if grouped_gemm_nt_masked is not None:
        fake_blockscaled.grouped_gemm_nt_masked = grouped_gemm_nt_masked
    fake_cute_dsl = types.ModuleType("flashinfer.cute_dsl")
    fake_cute_dsl.blockscaled_gemm = fake_blockscaled

    sys.modules["flashinfer"] = fake_flashinfer
    sys.modules["flashinfer.cute_dsl"] = fake_cute_dsl
    sys.modules["flashinfer.cute_dsl.blockscaled_gemm"] = fake_blockscaled

    return prev_flashinfer, prev_cute_dsl, prev_blockscaled


class TestFlashinferCuteDslMoeMasked(unittest.TestCase):
    """Unit tests for flashinfer_cutedsl_moe_masked."""

    def test_flashinfer_cutedsl_moe_masked_runs_with_bf16_inputs(self):
        """
        Verify that flashinfer_cutedsl_moe_masked can run end-to-end with
        standard (bf16) inputs when FlashInfer kernels are mocked.
        This directly exercises the path where hidden_states[1] is None.
        """

        num_experts = 2
        m = 3
        k = 32
        n = 8

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

        def fake_scaled_fp4_grouped_quantize(x, masked_m, input_global_scale):
            # x: [num_experts, m, k] -> produce pre-quantized tensors with valid shapes.
            num_experts, m, k = x.shape
            a_q = paddle.zeros([m, k // 2, num_experts], dtype=paddle.uint8)
            a_q_sf = paddle.zeros([m, k // 16, num_experts], dtype=paddle.float8_e4m3fn)
            return a_q, a_q_sf

        def fake_grouped_gemm_nt_masked(a, b, out, masked_m, **kwargs):
            # Simply zero out the output tensor while preserving shape and dtype.
            out.set_value(paddle.zeros_like(out))

        def fake_silu_and_mul_scaled_nvfp4_experts_quantize(x, masked_m, a2_global_scale):
            # Return dummy FP4-packed activations; grouped_gemm_nt_masked ignores the contents.
            num_experts, m, k2 = x.shape  # k2 = 2 * n
            n = k2 // 2
            diq = paddle.zeros([m, n // 2, num_experts], dtype=paddle.uint8)
            diq_sf = paddle.zeros([m, n // 8, num_experts], dtype=paddle.float8_e4m3fn)
            return diq, diq_sf

        # Install fake flashinfer BEFORE importing the target module,
        # so that no real flashinfer/torch is loaded.
        prev_flashinfer, prev_cute_dsl, prev_blockscaled = _install_fake_flashinfer_for_cutedsl(
            scaled_fp4_grouped_quantize=fake_scaled_fp4_grouped_quantize,
            silu_and_mul_scaled_nvfp4_experts_quantize=fake_silu_and_mul_scaled_nvfp4_experts_quantize,
            grouped_gemm_nt_masked=fake_grouped_gemm_nt_masked,
        )

        cutedsl_moe_module = importlib.import_module("fastdeploy.model_executor.layers.moe.flashinfer_cutedsl_moe")
        out = cutedsl_moe_module.flashinfer_cutedsl_moe_masked(
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
