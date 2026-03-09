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

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import cutlass_fp8_fp8_half_block_gemm_fused

E4M3_MAX_POS = 448.0
BLOCK_SIZE = 128

paddle.seed(2025)
np.random.seed(2025)


def _expand_x_scale(x_scale, m, k):
    """Expand x block-scale from [ceil(K/128), M] to [M, K]."""
    expanded = paddle.repeat_interleave(x_scale, repeats=BLOCK_SIZE, axis=0)[:k, :]
    return expanded.transpose([1, 0])


def _expand_y_scale(y_scale, n, k):
    """Expand y block-scale from [ceil(N/128), ceil(K/128)] to [N, K]."""
    expanded = paddle.repeat_interleave(y_scale, repeats=BLOCK_SIZE, axis=0)[:n, :]
    return paddle.repeat_interleave(expanded, repeats=BLOCK_SIZE, axis=1)[:, :k]


def _reference_block_gemm(
    x_fp8, y_fp8, x_scale_exp, y_scale_exp, bias=None, output_dtype="bfloat16", activation_type=""
):
    """Dequantize FP8, matmul in fp32, optional bias/activation, cast to output dtype."""
    x_deq = x_fp8.astype("float32") * x_scale_exp.astype("float32")
    y_deq = y_fp8.astype("float32") * y_scale_exp.astype("float32")
    ref = paddle.matmul(x_deq, y_deq, transpose_y=True)
    if bias is not None:
        ref = ref + bias.astype("float32")
    if activation_type == "leaky_relu":
        ref = paddle.where(ref >= 0, ref, ref * 0.01)
    out_dtype = paddle.bfloat16 if output_dtype == "bfloat16" else paddle.float16
    return ref.astype(out_dtype)


class TestCutlassFp8Fp8HalfBlockGemmFused(unittest.TestCase):
    """Tests for cutlass_fp8_fp8_half_block_gemm_fused (FP8 block-scaled GEMM)."""

    def setUp(self):
        paddle.set_device("gpu")
        self.prop = paddle.device.cuda.get_device_properties()
        self.sm_version = self.prop.major * 10 + self.prop.minor

    def _skip_if_not_sm90(self):
        if self.sm_version < 90:
            self.skipTest(f"Requires SM90+ (current: SM{self.sm_version})")

    def _check_output(self, m, n, k, output_dtype="bfloat16", use_bias=True, activation_type="", rtol=5e-2, atol=5e-2):
        """Run block GEMM and verify against reference."""
        scale_k = (k + BLOCK_SIZE - 1) // BLOCK_SIZE
        scale_n = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

        x_bf16 = paddle.rand([m, k], dtype="float32").astype("bfloat16")
        y_bf16 = paddle.rand([n, k], dtype="float32").astype("bfloat16")
        x_scale = paddle.rand([scale_k, m], dtype="float32") * 0.9 + 0.1
        y_scale = paddle.rand([scale_n, scale_k], dtype="float32") * 0.9 + 0.1

        x_scale_exp = _expand_x_scale(x_scale, m, k)
        y_scale_exp = _expand_y_scale(y_scale, n, k)

        scaled = x_bf16.astype("float32") / x_scale_exp.astype("float32")
        x_fp8 = scaled.clip(min=-E4M3_MAX_POS, max=E4M3_MAX_POS).astype("float8_e4m3fn")
        scaled = y_bf16.astype("float32") / y_scale_exp.astype("float32")
        y_fp8 = scaled.clip(min=-E4M3_MAX_POS, max=E4M3_MAX_POS).astype("float8_e4m3fn")

        bias = None
        if use_bias:
            cast_dtype = "bfloat16" if output_dtype == "bfloat16" else "float16"
            bias = paddle.rand([n], dtype="float32").astype(cast_dtype)

        ref_out = _reference_block_gemm(
            x_fp8,
            y_fp8,
            x_scale_exp,
            y_scale_exp,
            bias=bias,
            output_dtype=output_dtype,
            activation_type=activation_type,
        )

        result = cutlass_fp8_fp8_half_block_gemm_fused(
            x_fp8,
            y_fp8,
            x_scale,
            y_scale,
            bias,
            transpose_x=False,
            transpose_y=True,
            output_dtype=output_dtype,
            act=activation_type,
        )

        expected_dtype = paddle.bfloat16 if output_dtype == "bfloat16" else paddle.float16
        self.assertEqual(result.shape, [m, n])
        self.assertEqual(result.dtype, expected_dtype)
        np.testing.assert_allclose(
            ref_out.astype("float32").numpy(),
            result.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
        )

    def test_bfloat16_various_shapes(self):
        """BF16 output correctness with multiple M/N/K configs."""
        self._skip_if_not_sm90()
        for m, n, k in [(16, 2048, 2048), (32, 4096, 4096), (64, 5120, 5120)]:
            with self.subTest(m=m, n=n, k=k):
                self._check_output(m, n, k, output_dtype="bfloat16")

    def test_float16_output(self):
        """FP16 output correctness."""
        self._skip_if_not_sm90()
        for m, n, k in [(16, 2048, 2048), (64, 4096, 4096)]:
            with self.subTest(m=m, n=n, k=k):
                self._check_output(m, n, k, output_dtype="float16")

    def test_non_aligned_dimensions(self):
        """N and K not aligned to block size 128."""
        self._skip_if_not_sm90()
        for m, n, k in [(16, 2048, 5504), (32, 6144, 2048), (16, 5120, 13824)]:
            with self.subTest(m=m, n=n, k=k):
                self._check_output(m, n, k)

    def test_bias_and_activation_variants(self):
        """Without bias and with leaky_relu activation."""
        self._skip_if_not_sm90()
        self._check_output(32, 2048, 2048, use_bias=False)
        self._check_output(16, 2048, 2048, activation_type="leaky_relu")


if __name__ == "__main__":
    unittest.main()
