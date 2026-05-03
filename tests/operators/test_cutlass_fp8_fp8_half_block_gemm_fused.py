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

import os
import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import cutlass_fp8_fp8_half_block_gemm_fused

BLOCK_SIZE = 128

paddle.seed(2025)
np.random.seed(2025)


class TestCutlassFp8Fp8HalfBlockGemmFused(unittest.TestCase):
    """Tests for cutlass_fp8_fp8_half_block_gemm_fused (FP8 block-scaled GEMM)."""

    def setUp(self):
        paddle.set_device("gpu")
        self.prop = paddle.device.cuda.get_device_properties()
        self.sm_version = self.prop.major * 10 + self.prop.minor
        # Auto-tune mode lets the kernel find a valid config for each MNK.
        os.environ["FLAGS_use_cutlass_device_best_config_path"] = "tune"

    def tearDown(self):
        os.environ.pop("FLAGS_use_cutlass_device_best_config_path", None)

    def _skip_if_not_sm90(self):
        if self.sm_version < 90:
            self.skipTest(f"Requires SM90+ (current: SM{self.sm_version})")

    def _check_output(self, m, n, k, output_dtype="bfloat16"):
        """Run block GEMM and verify against dequant-matmul reference."""
        scale_k = (k + BLOCK_SIZE - 1) // BLOCK_SIZE
        scale_n = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

        x_fp8 = paddle.rand([m, k], dtype="bfloat16").astype("float8_e4m3fn")
        y_fp8 = paddle.rand([n, k], dtype="bfloat16").astype("float8_e4m3fn")
        x_scale = paddle.rand([scale_k, m], dtype="float32") * 0.9 + 0.1
        y_scale = paddle.rand([scale_n, scale_k], dtype="float32") * 0.9 + 0.1

        # Dequantize: expand block scales, then matmul in fp32
        x_s = paddle.repeat_interleave(x_scale, BLOCK_SIZE, axis=0)[:k, :].transpose([1, 0])
        y_s = paddle.repeat_interleave(y_scale, BLOCK_SIZE, axis=0)[:n, :]
        y_s = paddle.repeat_interleave(y_s, BLOCK_SIZE, axis=1)[:, :k]
        ref = paddle.matmul(
            x_fp8.astype("float32") * x_s.astype("float32"),
            y_fp8.astype("float32") * y_s.astype("float32"),
            transpose_y=True,
        )
        out_t = paddle.bfloat16 if output_dtype == "bfloat16" else paddle.float16
        ref = ref.astype(out_t)

        result = cutlass_fp8_fp8_half_block_gemm_fused(
            x_fp8,
            y_fp8,
            x_scale,
            y_scale,
            None,
            transpose_x=False,
            transpose_y=True,
            output_dtype=output_dtype,
            act="",
        )

        self.assertEqual(result.shape, [m, n])
        self.assertEqual(result.dtype, out_t)
        np.testing.assert_allclose(
            ref.astype("float32").numpy(),
            result.astype("float32").numpy(),
            rtol=5e-2,
            atol=5e-2,
        )

    def test_bfloat16_correctness(self):
        """BF16 output correctness with multiple shapes."""
        self._skip_if_not_sm90()
        for m, n, k in [(32, 2048, 2048), (64, 4096, 4096), (128, 5120, 5120)]:
            with self.subTest(m=m, n=n, k=k):
                self._check_output(m, n, k)

    def test_float16_output(self):
        """FP16 output correctness."""
        self._skip_if_not_sm90()
        self._check_output(64, 2048, 2048, output_dtype="float16")

    def test_non_aligned_dimensions(self):
        """N and K not aligned to block size 128."""
        self._skip_if_not_sm90()
        self._check_output(32, 2048, 5504)


if __name__ == "__main__":
    unittest.main()
