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

from fastdeploy.model_executor.ops.gpu import winx_unzip

paddle.seed(42)
np.random.seed(42)


def wint25_unzip_ref(zipped_weight_np, super_scale_np):
    """NumPy reference for weight_only_int2.5 decompression."""
    batch, k_zipped, n_cols = zipped_weight_np.shape
    num_groups = k_zipped // 10
    num_rows = num_groups * 64
    weight = np.zeros((batch, num_rows, n_cols), dtype=np.float32)
    shift_bits = np.array([13, 11, 9, 6, 4, 2, 0], dtype=np.int32)

    for b in range(batch):
        for g in range(num_groups):
            zipped_last = zipped_weight_np[b, g * 10 + 9, :].astype(np.int32)
            local_scale = (zipped_last & 0x1FFF).astype(np.float32)
            scale = local_scale * super_scale_np[b]
            for zr in range(9):
                zv = zipped_weight_np[b, g * 10 + zr, :].astype(np.int32)
                for si in range(7):
                    shifted = (zv >> shift_bits[si]) & 0x7
                    weight[b, g * 64 + zr * 7 + si, :] = (shifted.astype(np.float32) - 4) * scale
            val_last = (zipped_last >> 13) & 0x7
            weight[b, g * 64 + 63, :] = (val_last.astype(np.float32) - 4) * scale
    return weight


def wint2_unzip_ref(zipped_weight_np, local_scale_np, code_scale_np, code_zp_np, super_scale_np):
    """NumPy reference for weight_only_int2 decompression."""
    batch, k_packed, n_cols = zipped_weight_np.shape
    num_rows = k_packed * 4
    num_groups = num_rows // 64
    weight = np.zeros((batch, num_rows, n_cols), dtype=np.float32)
    shift_bits = np.array([9, 6, 3, 0], dtype=np.int32)

    for b in range(batch):
        for g in range(num_groups):
            block_start_row = b * num_rows + g * 64
            ls_row = g // 2
            local_scale_shift = ((block_start_row // 64 + 0 + 1) & 1) * 4
            ls_byte = local_scale_np[b, ls_row, :].astype(np.int32)
            ls_float = ((ls_byte >> local_scale_shift) & 0xF).astype(np.float32)
            scale = ls_float * super_scale_np[b] if super_scale_np is not None else ls_float
            for zr in range(16):
                zv = zipped_weight_np[b, g * 16 + zr, :].astype(np.float32)
                decode_val = np.floor(zv * code_scale_np[b] + code_zp_np[b] + 0.5).astype(np.int32)
                for si in range(4):
                    shifted = (decode_val >> shift_bits[si]) & 0x3F
                    weight[b, g * 64 + zr * 4 + si, :] = (shifted.astype(np.float32) - 32) * scale
    return weight


@unittest.skipUnless(paddle.is_compiled_with_cuda(), "GPU required for winx_unzip")
class TestWinxUnzip(unittest.TestCase):
    """Correctness tests for the winx_unzip custom op."""

    def setUp(self):
        paddle.set_device("gpu")

    def _check_wint25(self, batch, k_zipped, n, seed=42):
        """Run wint2.5 op and compare against NumPy reference."""
        np.random.seed(seed)
        zipped_np = np.random.randint(0, 65536, (batch, k_zipped, n)).astype(np.uint16)
        super_scale_np = np.random.rand(batch, n).astype(np.float32) * 0.1 + 0.01

        expected = wint25_unzip_ref(zipped_np, super_scale_np)
        zipped_pd = paddle.to_tensor(zipped_np.view(np.int16), dtype=paddle.int16)
        super_scale_pd = paddle.to_tensor(super_scale_np.astype(np.float16), dtype=paddle.float16)
        out = winx_unzip(zipped_pd, None, None, None, super_scale_pd, "weight_only_int2.5")
        self.assertEqual(list(out.shape), [batch, k_zipped // 10 * 64, n])
        self.assertEqual(out.dtype, paddle.float16)
        np.testing.assert_allclose(out.astype(paddle.float32).numpy(), expected, rtol=5e-3, atol=5e-3)

    def _check_wint2(self, batch, k_packed, n, seed=42):
        """Run wint2 op and compare against NumPy reference."""
        np.random.seed(seed)
        num_ls_rows = (k_packed * 4 // 64 + 1) // 2
        zipped_np = np.random.randint(0, 256, (batch, k_packed, n)).astype(np.uint8)
        local_scale_np = np.random.randint(0, 256, (batch, num_ls_rows, n)).astype(np.uint8)
        code_scale_np = np.full((batch, n), 128.0, dtype=np.float32)
        code_zp_np = np.zeros((batch, n), dtype=np.float32)
        super_scale_np = (np.random.rand(batch, n) * 0.1 + 0.01).astype(np.float32)

        expected = wint2_unzip_ref(zipped_np, local_scale_np, code_scale_np, code_zp_np, super_scale_np)
        out = winx_unzip(
            paddle.to_tensor(zipped_np, dtype=paddle.uint8),
            paddle.to_tensor(local_scale_np, dtype=paddle.uint8),
            paddle.to_tensor(code_scale_np, dtype=paddle.float32),
            paddle.to_tensor(code_zp_np, dtype=paddle.float32),
            paddle.to_tensor(super_scale_np.astype(np.float16), dtype=paddle.float16),
            "weight_only_int2",
        )
        self.assertEqual(list(out.shape), [batch, k_packed * 4, n])
        self.assertEqual(out.dtype, paddle.float16)
        np.testing.assert_allclose(out.astype(paddle.float32).numpy(), expected, rtol=5e-2, atol=5e-2)

    def test_wint25_correctness(self):
        """wint2.5 single and multi-group correctness."""
        for batch, k_zipped, n in [(1, 10, 64), (2, 20, 128)]:
            with self.subTest(batch=batch, k_zipped=k_zipped, n=n):
                self._check_wint25(batch, k_zipped, n)

    def test_wint2_correctness(self):
        """wint2 single and multi-group correctness."""
        for batch, k_packed, n in [(1, 16, 256), (2, 32, 256)]:
            with self.subTest(batch=batch, k_packed=k_packed, n=n):
                self._check_wint2(batch, k_packed, n)

    def test_wint2_with_code_zp(self):
        """wint2 correctness with non-zero code_zp."""
        np.random.seed(99)
        batch, k_packed, n = 2, 32, 256
        num_ls_rows = 1
        zipped_np = np.random.randint(0, 256, (batch, k_packed, n)).astype(np.uint8)
        local_scale_np = np.random.randint(0, 256, (batch, num_ls_rows, n)).astype(np.uint8)
        code_scale_np = np.full((batch, n), 128.0, dtype=np.float32)
        code_zp_np = np.full((batch, n), 0.5, dtype=np.float32)
        super_scale_np = (np.random.rand(batch, n) * 0.05 + 0.005).astype(np.float32)

        expected = wint2_unzip_ref(zipped_np, local_scale_np, code_scale_np, code_zp_np, super_scale_np)
        out = winx_unzip(
            paddle.to_tensor(zipped_np, dtype=paddle.uint8),
            paddle.to_tensor(local_scale_np, dtype=paddle.uint8),
            paddle.to_tensor(code_scale_np, dtype=paddle.float32),
            paddle.to_tensor(code_zp_np, dtype=paddle.float32),
            paddle.to_tensor(super_scale_np.astype(np.float16), dtype=paddle.float16),
            "weight_only_int2",
        )
        np.testing.assert_allclose(out.astype(paddle.float32).numpy(), expected, rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
    unittest.main()
