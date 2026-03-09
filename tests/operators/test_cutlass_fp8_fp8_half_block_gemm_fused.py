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


def _expand_x_scale(x_scale, m, k):
    """Expand x block-scale from [ceil(K/128), M] to [M, K].

    Each column of x_scale corresponds to one row of x, and each row
    covers a 128-element block along the K dimension.
    """
    # [ceil(K/128), M] -> repeat rows -> [K_padded, M] -> truncate -> [K, M]
    expanded = paddle.repeat_interleave(x_scale, repeats=BLOCK_SIZE, axis=0)
    expanded = expanded[:k, :]
    # Transpose to match x layout: [M, K]
    return expanded.transpose([1, 0])


def _expand_y_scale(y_scale, n, k):
    """Expand y block-scale from [ceil(N/128), ceil(K/128)] to [N, K].

    y_scale is a 2D block-scale tensor where each element covers a
    128x128 block of the y matrix.
    """
    # Expand along N: [ceil(N/128), ceil(K/128)] -> [N, ceil(K/128)]
    expanded = paddle.repeat_interleave(y_scale, repeats=BLOCK_SIZE, axis=0)
    expanded = expanded[:n, :]
    # Expand along K: [N, ceil(K/128)] -> [N, K]
    expanded = paddle.repeat_interleave(expanded, repeats=BLOCK_SIZE, axis=1)
    return expanded[:, :k]


def _quantize_to_fp8(tensor_bf16, scale_expanded):
    """Quantize bf16 tensor to FP8 using element-wise scales.

    Follows the reviewer-recommended approach (PR #4096 review by @ckl117):
    start from bf16, divide by scale, clip to E4M3 range, cast to FP8.
    This ensures both kernel and reference see identical FP8 values.
    """
    scaled = tensor_bf16.astype("float32") / scale_expanded.astype("float32")
    return scaled.clip(min=-E4M3_MAX_POS, max=E4M3_MAX_POS).astype("float8_e4m3fn")


def _reference_block_gemm(
    x_fp8,
    y_fp8,
    x_scale_exp,
    y_scale_exp,
    bias=None,
    output_dtype="bfloat16",
    activation_type="",
):
    """Compute reference output for block-scaled FP8 GEMM.

    Dequantizes FP8 inputs using element-wise scales, computes matmul
    in float32 for maximum accuracy, then applies optional bias and
    activation before casting to the target output dtype.

    The CUTLASS kernel computes:
        out = (dequant(x) * x_scale) @ (dequant(y) * y_scale)^T [+ bias] [act]
    """
    # Dequantize in float32 for accurate reference
    x_deq = x_fp8.astype("float32") * x_scale_exp.astype("float32")
    y_deq = y_fp8.astype("float32") * y_scale_exp.astype("float32")

    ref = paddle.matmul(x_deq, y_deq, transpose_y=True)

    if bias is not None:
        ref = ref + bias.astype("float32")

    if activation_type == "leaky_relu":
        ref = paddle.where(ref >= 0, ref, ref * 0.01)

    out_paddle_dtype = paddle.bfloat16 if output_dtype == "bfloat16" else paddle.float16
    return ref.astype(out_paddle_dtype)


class TestCutlassFp8Fp8HalfBlockGemmFused(unittest.TestCase):
    """Unit tests for cutlass_fp8_fp8_half_block_gemm_fused.

    The operator performs FP8 x FP8 -> FP16/BF16 block-scaled GEMM using
    CUTLASS 3.x on SM90+ (Hopper) GPUs. Block scales are applied at
    128-element granularity.

    Scale tensor shapes (for x=[M,K], y=[N,K], transpose_y=True):
        x_scale: [ceil(K/128), M]  — per-row, per-K-block
        y_scale: [ceil(N/128), ceil(K/128)] — per 128x128 block
    """

    def setUp(self):
        paddle.set_device("gpu")
        paddle.seed(2025)
        np.random.seed(2025)
        self.prop = paddle.device.cuda.get_device_properties()
        self.sm_version = self.prop.major * 10 + self.prop.minor

    def _skip_if_not_sm90(self):
        if self.sm_version < 90:
            self.skipTest(f"cutlass_fp8_fp8_half_block_gemm_fused requires SM90+ " f"(current: SM{self.sm_version})")

    def _run_block_gemm(
        self,
        m,
        n,
        k,
        output_dtype="bfloat16",
        use_bias=True,
        activation_type="",
        rtol=5e-2,
        atol=5e-2,
    ):
        """Run one block GEMM test case and verify correctness.

        Creates bf16 data, quantizes to FP8, runs both the CUTLASS kernel
        and a Python reference, and compares.
        """
        scale_k = (k + BLOCK_SIZE - 1) // BLOCK_SIZE
        scale_n = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Random bf16 inputs
        x_bf16 = paddle.rand([m, k], dtype="float32").astype("bfloat16")
        y_bf16 = paddle.rand([n, k], dtype="float32").astype("bfloat16")

        # Block scales — positive, bounded away from zero
        x_scale = paddle.rand([scale_k, m], dtype="float32") * 0.9 + 0.1
        y_scale = paddle.rand([scale_n, scale_k], dtype="float32") * 0.9 + 0.1

        # Expand to element-wise for quantization and reference
        x_scale_exp = _expand_x_scale(x_scale, m, k)
        y_scale_exp = _expand_y_scale(y_scale, n, k)

        # Quantize bf16 -> FP8
        x_fp8 = _quantize_to_fp8(x_bf16, x_scale_exp)
        y_fp8 = _quantize_to_fp8(y_bf16, y_scale_exp)

        # Optional bias in output dtype
        bias = None
        if use_bias:
            cast_dtype = "bfloat16" if output_dtype == "bfloat16" else "float16"
            bias = paddle.rand([n], dtype="float32").astype(cast_dtype)

        # Reference
        ref_out = _reference_block_gemm(
            x_fp8,
            y_fp8,
            x_scale_exp,
            y_scale_exp,
            bias=bias,
            output_dtype=output_dtype,
            activation_type=activation_type,
        )

        # Run CUTLASS kernel — positional args for inputs to avoid
        # the "x_sacle" typo in the .Inputs() registration
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

        # Shape
        self.assertEqual(result.shape, [m, n])

        # Dtype
        expected_dtype = paddle.bfloat16 if output_dtype == "bfloat16" else paddle.float16
        self.assertEqual(result.dtype, expected_dtype)

        # Numerical correctness
        np.testing.assert_allclose(
            ref_out.astype("float32").numpy(),
            result.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
        )

    # ------------------------------------------------------------------
    # Category A: Numerical Correctness
    # ------------------------------------------------------------------

    def test_basic_bfloat16(self):
        """Block GEMM correctness with bfloat16 output, various M/N/K."""
        self._skip_if_not_sm90()
        nk_pairs = [
            [2048, 2048],
            [4096, 4096],
            [5120, 5120],
        ]
        for m in [16, 32, 64]:
            for n, k in nk_pairs:
                with self.subTest(m=m, n=n, k=k):
                    self._run_block_gemm(m, n, k, output_dtype="bfloat16")

    def test_basic_float16(self):
        """Block GEMM correctness with float16 output."""
        self._skip_if_not_sm90()
        for m in [16, 64]:
            for n, k in [[2048, 2048], [4096, 4096]]:
                with self.subTest(m=m, n=n, k=k):
                    self._run_block_gemm(m, n, k, output_dtype="float16")

    def test_without_bias(self):
        """Block GEMM without bias for both output dtypes."""
        self._skip_if_not_sm90()
        for m in [16, 32]:
            for out_dtype in ["bfloat16", "float16"]:
                with self.subTest(m=m, dtype=out_dtype):
                    self._run_block_gemm(
                        m,
                        2048,
                        2048,
                        output_dtype=out_dtype,
                        use_bias=False,
                    )

    def test_non_aligned_dimensions(self):
        """N and K not aligned to block size 128."""
        self._skip_if_not_sm90()
        nk_pairs = [
            [2048, 5504],
            [6144, 2048],
            [5120, 13824],
            [15360, 5120],
        ]
        for m in [16, 32]:
            for n, k in nk_pairs:
                with self.subTest(m=m, n=n, k=k):
                    self._run_block_gemm(m, n, k)

    def test_leaky_relu_activation(self):
        """Fused leaky_relu activation (alpha=0.01 hardcoded in kernel)."""
        self._skip_if_not_sm90()
        for m in [16, 64]:
            with self.subTest(m=m):
                self._run_block_gemm(m, 2048, 2048, activation_type="leaky_relu")

    # ------------------------------------------------------------------
    # Category B/C: Shape & Dtype Validation
    # ------------------------------------------------------------------

    def test_output_shape_and_dtype(self):
        """Verify output shape and dtype for various configurations."""
        self._skip_if_not_sm90()
        configs = [
            (16, 2048, 2048, "bfloat16"),
            (64, 4096, 5120, "bfloat16"),
            (32, 2048, 2048, "float16"),
        ]
        for m, n, k, out_dtype in configs:
            with self.subTest(m=m, n=n, k=k, dtype=out_dtype):
                scale_k = (k + BLOCK_SIZE - 1) // BLOCK_SIZE
                scale_n = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
                x_fp8 = (
                    paddle.rand([m, k], dtype="float32")
                    .clip(min=-E4M3_MAX_POS, max=E4M3_MAX_POS)
                    .astype("float8_e4m3fn")
                )
                y_fp8 = (
                    paddle.rand([n, k], dtype="float32")
                    .clip(min=-E4M3_MAX_POS, max=E4M3_MAX_POS)
                    .astype("float8_e4m3fn")
                )
                x_scale = paddle.ones([scale_k, m], dtype="float32")
                y_scale = paddle.ones([scale_n, scale_k], dtype="float32")

                result = cutlass_fp8_fp8_half_block_gemm_fused(
                    x_fp8,
                    y_fp8,
                    x_scale,
                    y_scale,
                    None,
                    transpose_x=False,
                    transpose_y=True,
                    output_dtype=out_dtype,
                    act="",
                )
                self.assertEqual(result.shape, [m, n])
                expected_dtype = paddle.bfloat16 if out_dtype == "bfloat16" else paddle.float16
                self.assertEqual(result.dtype, expected_dtype)

    # ------------------------------------------------------------------
    # Category E: Determinism
    # ------------------------------------------------------------------

    def test_determinism(self):
        """Same inputs produce bit-identical outputs across two calls."""
        self._skip_if_not_sm90()
        m, n, k = 16, 2048, 2048
        scale_k = (k + BLOCK_SIZE - 1) // BLOCK_SIZE
        scale_n = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

        paddle.seed(12345)
        x_fp8 = paddle.rand([m, k], dtype="float32").clip(min=-E4M3_MAX_POS, max=E4M3_MAX_POS).astype("float8_e4m3fn")
        y_fp8 = paddle.rand([n, k], dtype="float32").clip(min=-E4M3_MAX_POS, max=E4M3_MAX_POS).astype("float8_e4m3fn")
        x_scale = paddle.rand([scale_k, m], dtype="float32") * 0.9 + 0.1
        y_scale = paddle.rand([scale_n, scale_k], dtype="float32") * 0.9 + 0.1

        result1 = cutlass_fp8_fp8_half_block_gemm_fused(
            x_fp8,
            y_fp8,
            x_scale,
            y_scale,
            None,
            transpose_x=False,
            transpose_y=True,
            output_dtype="bfloat16",
            act="",
        )
        result2 = cutlass_fp8_fp8_half_block_gemm_fused(
            x_fp8,
            y_fp8,
            x_scale,
            y_scale,
            None,
            transpose_x=False,
            transpose_y=True,
            output_dtype="bfloat16",
            act="",
        )
        np.testing.assert_array_equal(
            result1.astype("float32").numpy(),
            result2.astype("float32").numpy(),
        )


if __name__ == "__main__":
    unittest.main()
