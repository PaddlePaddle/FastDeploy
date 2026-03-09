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

"""Unit tests for winx_unzip custom op.

Tests the CUTLASS-based weight decompression operator for wint2 and wint2.5
quantised weight formats. Validates numerical correctness against a pure-NumPy
reference implementation derived directly from the CUDA kernel logic in:
  - custom_ops/gpu_ops/moe/winx_unzip.cu
  - custom_ops/gpu_ops/cutlass_extensions/gemm/threadblock/wint2x_unzip.h

wint2.5 encoding (kWeightOnlyInt25):
  - 64 output values are packed into 10 × uint16 words per group.
  - First 9 words each carry 7 × 3-bit values extracted at bit-shifts
    [13, 11, 9, 6, 4, 2, 0]. The 10th word carries the 64th 3-bit value
    in bits [15:13] and a 13-bit local_scale in bits [12:0].
  - Dequant: value = (raw_3bit − 4) × local_scale × super_scale.

wint2 encoding (kWeightOnlyInt2):
  - Each uint8 is first linearly decoded:
    decode = round(byte × code_scale + code_zp)
  - Then 4 × 6-bit values are extracted at bit-shifts [9, 6, 3, 0].
  - Per-group uint4 local_scale is stored nibble-packed in a separate tensor.
  - Dequant: value = (raw_6bit − 32) × local_scale_nibble × super_scale.
"""

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import winx_unzip

# ---------------------------------------------------------------------------
#  NumPy reference implementations
# ---------------------------------------------------------------------------


def wint25_unzip_ref(zipped_weight_np, super_scale_np):
    """Pure-NumPy reference for weight_only_int2.5 decompression.

    Parameters
    ----------
    zipped_weight_np : ndarray, shape [batch, K_zipped, N], dtype uint16
        Packed 2.5-bit weights.  K_zipped / 10 * 64 = num_output_rows.
    super_scale_np : ndarray, shape [batch, N], dtype float32
        Channel-wise super scale (already cast to fp32 for reference compute).

    Returns
    -------
    weight_np : ndarray, shape [batch, num_rows, N], dtype float32
    """
    batch, k_zipped, n_cols = zipped_weight_np.shape
    num_groups = k_zipped // 10
    num_rows = num_groups * 64
    weight = np.zeros((batch, num_rows, n_cols), dtype=np.float32)

    shift_bits = np.array([13, 11, 9, 6, 4, 2, 0], dtype=np.int32)
    kBZP = 4

    for b in range(batch):
        for g in range(num_groups):
            # Last zipped word carries local_scale in lower 13 bits
            zipped_last = zipped_weight_np[b, g * 10 + 9, :].astype(np.int32)
            local_scale = (zipped_last & 0x1FFF).astype(np.float32)
            scale = local_scale * super_scale_np[b]  # [N]

            # First 9 zipped words → 63 output rows
            for zr in range(9):
                zv = zipped_weight_np[b, g * 10 + zr, :].astype(np.int32)
                for si in range(7):
                    shifted = (zv >> shift_bits[si]) & 0x7
                    row_idx = g * 64 + zr * 7 + si
                    weight[b, row_idx, :] = (shifted.astype(np.float32) - kBZP) * scale

            # 64th value from upper 3 bits of last word
            val_last = (zipped_last >> 13) & 0x7
            weight[b, g * 64 + 63, :] = (val_last.astype(np.float32) - kBZP) * scale

    return weight


def wint2_unzip_ref(zipped_weight_np, local_scale_np, code_scale_np, code_zp_np, super_scale_np):
    """Pure-NumPy reference for weight_only_int2 decompression.

    Parameters
    ----------
    zipped_weight_np : ndarray, shape [batch, K_packed, N], dtype uint8
        Packed 2-bit weights.  K_packed * 4 = num_output_rows.
    local_scale_np : ndarray, shape [batch, ?, N], dtype uint8
        Nibble-packed uint4 local scales.
    code_scale_np : ndarray, shape [batch, N], dtype float32
    code_zp_np : ndarray, shape [batch, N], dtype float32
    super_scale_np : ndarray, shape [batch, N], dtype float32  (or None)

    Returns
    -------
    weight_np : ndarray, shape [batch, num_rows, N], dtype float32
    """
    batch, k_packed, n_cols = zipped_weight_np.shape
    num_rows = k_packed * 4
    num_groups = num_rows // 64
    weight = np.zeros((batch, num_rows, n_cols), dtype=np.float32)

    shift_bits = np.array([9, 6, 3, 0], dtype=np.int32)
    kWeightMask = 0x3F
    kLocalScaleMask = 0xF
    kBZP = 32

    for b in range(batch):
        for g in range(num_groups):
            # The CUDA kernel computes:
            #   block_start_row = batch_idx * num_rows + tile_y * TileRows
            # For global group index g (within batch b), tile_y = g, so:
            #   block_start_row = b * num_rows + g * 64
            block_start_row = b * num_rows + g * 64

            # Determine which nibble of the local_scale byte to use
            ls_row = g // 2  # each byte covers 2 groups of 64 rows
            local_scale_shift = ((block_start_row // 64 + 0 + 1) & 1) * 4
            ls_byte = local_scale_np[b, ls_row, :].astype(np.int32)
            shifted_ls = (ls_byte >> local_scale_shift) & kLocalScaleMask
            ls_float = shifted_ls.astype(np.float32)

            if super_scale_np is not None:
                scale = ls_float * super_scale_np[b]  # [N]
            else:
                scale = ls_float  # [N]

            # 16 zipped bytes per group of 64 rows
            for zr in range(16):
                zipped_idx = g * 16 + zr
                zv = zipped_weight_np[b, zipped_idx, :].astype(np.float32)
                # Linear decode: round(byte × code_scale + code_zp)
                decode_val = np.floor(zv * code_scale_np[b] + code_zp_np[b] + 0.5).astype(np.int32)

                row_base = g * 64 + zr * 4
                for si in range(4):
                    shifted = (decode_val >> shift_bits[si]) & kWeightMask
                    weight[b, row_base + si, :] = (shifted.astype(np.float32) - kBZP) * scale

    return weight


# ---------------------------------------------------------------------------
#  Test class
# ---------------------------------------------------------------------------


@unittest.skipUnless(paddle.is_compiled_with_cuda(), "GPU required for winx_unzip")
class TestWinxUnzip(unittest.TestCase):
    """Numerical-correctness tests for the winx_unzip custom op."""

    def setUp(self):
        paddle.set_device("gpu")
        self.seed = 42

    # ------------------------------------------------------------------
    #  wint2.5 tests
    # ------------------------------------------------------------------

    def test_wint25_basic_shape_and_dtype(self):
        """Output shape [B, K_zipped/10*64, N] and dtype matches super_scale."""
        np.random.seed(self.seed)
        batch, k_zipped, n = 1, 10, 64  # 1 group → 64 output rows
        zipped = paddle.to_tensor(
            np.random.randint(0, 65536, (batch, k_zipped, n), dtype=np.int32).astype(np.int16),
            dtype=paddle.int16,
        )
        super_scale = paddle.ones([batch, n], dtype=paddle.float16)

        out = winx_unzip(zipped, None, None, None, super_scale, "weight_only_int2.5")
        self.assertEqual(list(out.shape), [batch, 64, n])
        self.assertEqual(out.dtype, paddle.float16)

    def test_wint25_correctness_single_group(self):
        """One group (10 uint16 → 64 rows).  Verify against NumPy reference."""
        np.random.seed(self.seed)
        batch, k_zipped, n = 1, 10, 64
        zipped_np = np.random.randint(0, 65536, (batch, k_zipped, n)).astype(np.uint16)
        super_scale_np = np.random.rand(batch, n).astype(np.float32) * 0.1 + 0.01

        expected = wint25_unzip_ref(zipped_np, super_scale_np)

        zipped_pd = paddle.to_tensor(zipped_np.view(np.int16), dtype=paddle.int16)
        super_scale_pd = paddle.to_tensor(super_scale_np.astype(np.float16), dtype=paddle.float16)
        out = winx_unzip(zipped_pd, None, None, None, super_scale_pd, "weight_only_int2.5")
        out_np = out.astype(paddle.float32).numpy()

        np.testing.assert_allclose(out_np, expected, rtol=5e-3, atol=5e-3)

    def test_wint25_correctness_multi_group(self):
        """Two groups (20 uint16 → 128 rows) with batch=2."""
        np.random.seed(self.seed + 1)
        batch, k_zipped, n = 2, 20, 128
        zipped_np = np.random.randint(0, 65536, (batch, k_zipped, n)).astype(np.uint16)
        super_scale_np = np.random.rand(batch, n).astype(np.float32) * 0.05 + 0.005

        expected = wint25_unzip_ref(zipped_np, super_scale_np)

        zipped_pd = paddle.to_tensor(zipped_np.view(np.int16), dtype=paddle.int16)
        super_scale_pd = paddle.to_tensor(super_scale_np.astype(np.float16), dtype=paddle.float16)
        out = winx_unzip(zipped_pd, None, None, None, super_scale_pd, "weight_only_int2.5")
        out_np = out.astype(paddle.float32).numpy()

        np.testing.assert_allclose(out_np, expected, rtol=5e-3, atol=5e-3)

    def test_wint25_zero_super_scale(self):
        """All-zeros super_scale should produce all-zeros output."""
        np.random.seed(self.seed + 2)
        batch, k_zipped, n = 1, 10, 64
        zipped_np = np.random.randint(0, 65536, (batch, k_zipped, n)).astype(np.uint16)
        zipped_pd = paddle.to_tensor(zipped_np.view(np.int16), dtype=paddle.int16)
        super_scale_pd = paddle.zeros([batch, n], dtype=paddle.float16)

        out = winx_unzip(zipped_pd, None, None, None, super_scale_pd, "weight_only_int2.5")
        out_np = out.astype(paddle.float32).numpy()
        np.testing.assert_allclose(out_np, 0.0, atol=1e-6)

    def test_wint25_determinism(self):
        """Two calls with the same input produce identical output."""
        np.random.seed(self.seed + 3)
        batch, k_zipped, n = 1, 10, 128
        zipped_np = np.random.randint(0, 65536, (batch, k_zipped, n)).astype(np.uint16)
        zipped_pd = paddle.to_tensor(zipped_np.view(np.int16), dtype=paddle.int16)
        super_scale_pd = paddle.ones([batch, n], dtype=paddle.float16) * 0.01

        out1 = winx_unzip(zipped_pd, None, None, None, super_scale_pd, "weight_only_int2.5")
        out2 = winx_unzip(zipped_pd, None, None, None, super_scale_pd, "weight_only_int2.5")
        np.testing.assert_array_equal(out1.numpy(), out2.numpy())

    # ------------------------------------------------------------------
    #  wint2 tests
    # ------------------------------------------------------------------

    def test_wint2_basic_shape_and_dtype(self):
        """Output shape [B, K_packed*4, N] and dtype matches scale tensor."""
        np.random.seed(self.seed + 10)
        batch, k_packed, n = 1, 16, 256  # 16 bytes → 64 output rows
        zipped = paddle.to_tensor(
            np.random.randint(0, 256, (batch, k_packed, n), dtype=np.uint8),
            dtype=paddle.uint8,
        )
        local_scale = paddle.to_tensor(
            np.random.randint(0, 256, (batch, 1, n), dtype=np.uint8),
            dtype=paddle.uint8,
        )
        code_scale = paddle.ones([batch, n], dtype=paddle.float32)
        code_zp = paddle.zeros([batch, n], dtype=paddle.float32)
        super_scale = paddle.ones([batch, n], dtype=paddle.float16)

        out = winx_unzip(zipped, local_scale, code_scale, code_zp, super_scale, "weight_only_int2")
        self.assertEqual(list(out.shape), [batch, 64, n])
        self.assertEqual(out.dtype, paddle.float16)

    def test_wint2_correctness_single_group(self):
        """One group (16 bytes → 64 rows).  Verify against NumPy reference."""
        np.random.seed(self.seed + 11)
        batch, n = 1, 256
        k_packed = 16  # 64 output rows
        num_ls_rows = 1  # ceil(64 / 128) = 1

        zipped_np = np.random.randint(0, 256, (batch, k_packed, n)).astype(np.uint8)
        local_scale_np = np.random.randint(0, 256, (batch, num_ls_rows, n)).astype(np.uint8)
        # Use moderate code_scale so decode_value stays reasonable
        code_scale_np = np.full((batch, n), 128.0, dtype=np.float32)
        code_zp_np = np.zeros((batch, n), dtype=np.float32)
        super_scale_np = (np.random.rand(batch, n) * 0.1 + 0.01).astype(np.float32)

        expected = wint2_unzip_ref(zipped_np, local_scale_np, code_scale_np, code_zp_np, super_scale_np)

        zipped_pd = paddle.to_tensor(zipped_np, dtype=paddle.uint8)
        local_scale_pd = paddle.to_tensor(local_scale_np, dtype=paddle.uint8)
        code_scale_pd = paddle.to_tensor(code_scale_np, dtype=paddle.float32)
        code_zp_pd = paddle.to_tensor(code_zp_np, dtype=paddle.float32)
        super_scale_pd = paddle.to_tensor(super_scale_np.astype(np.float16), dtype=paddle.float16)

        out = winx_unzip(zipped_pd, local_scale_pd, code_scale_pd, code_zp_pd, super_scale_pd, "weight_only_int2")
        out_np = out.astype(paddle.float32).numpy()

        np.testing.assert_allclose(out_np, expected, rtol=5e-2, atol=5e-2)

    def test_wint2_correctness_multi_group(self):
        """128 output rows (2 groups) with batch=2."""
        np.random.seed(self.seed + 12)
        batch, n = 2, 256
        k_packed = 32  # 128 output rows
        num_ls_rows = 1  # ceil(128 / 128) = 1

        zipped_np = np.random.randint(0, 256, (batch, k_packed, n)).astype(np.uint8)
        local_scale_np = np.random.randint(0, 256, (batch, num_ls_rows, n)).astype(np.uint8)
        code_scale_np = np.full((batch, n), 128.0, dtype=np.float32)
        code_zp_np = np.full((batch, n), 0.5, dtype=np.float32)
        super_scale_np = (np.random.rand(batch, n) * 0.05 + 0.005).astype(np.float32)

        expected = wint2_unzip_ref(zipped_np, local_scale_np, code_scale_np, code_zp_np, super_scale_np)

        zipped_pd = paddle.to_tensor(zipped_np, dtype=paddle.uint8)
        local_scale_pd = paddle.to_tensor(local_scale_np, dtype=paddle.uint8)
        code_scale_pd = paddle.to_tensor(code_scale_np, dtype=paddle.float32)
        code_zp_pd = paddle.to_tensor(code_zp_np, dtype=paddle.float32)
        super_scale_pd = paddle.to_tensor(super_scale_np.astype(np.float16), dtype=paddle.float16)

        out = winx_unzip(zipped_pd, local_scale_pd, code_scale_pd, code_zp_pd, super_scale_pd, "weight_only_int2")
        out_np = out.astype(paddle.float32).numpy()

        np.testing.assert_allclose(out_np, expected, rtol=5e-2, atol=5e-2)

    def test_wint2_zero_code_scale(self):
        """code_scale=0 means decode_value = round(code_zp), constant output."""
        np.random.seed(self.seed + 13)
        batch, k_packed, n = 1, 16, 256
        num_ls_rows = 1

        zipped_np = np.random.randint(0, 256, (batch, k_packed, n)).astype(np.uint8)
        local_scale_np = np.random.randint(0, 256, (batch, num_ls_rows, n)).astype(np.uint8)
        code_scale_np = np.zeros((batch, n), dtype=np.float32)
        code_zp_np = np.zeros((batch, n), dtype=np.float32)
        super_scale_np = (np.random.rand(batch, n) * 0.1 + 0.01).astype(np.float32)

        expected = wint2_unzip_ref(zipped_np, local_scale_np, code_scale_np, code_zp_np, super_scale_np)

        zipped_pd = paddle.to_tensor(zipped_np, dtype=paddle.uint8)
        local_scale_pd = paddle.to_tensor(local_scale_np, dtype=paddle.uint8)
        code_scale_pd = paddle.to_tensor(code_scale_np, dtype=paddle.float32)
        code_zp_pd = paddle.to_tensor(code_zp_np, dtype=paddle.float32)
        super_scale_pd = paddle.to_tensor(super_scale_np.astype(np.float16), dtype=paddle.float16)

        out = winx_unzip(zipped_pd, local_scale_pd, code_scale_pd, code_zp_pd, super_scale_pd, "weight_only_int2")
        out_np = out.astype(paddle.float32).numpy()

        np.testing.assert_allclose(out_np, expected, rtol=5e-2, atol=5e-2)

    def test_wint2_determinism(self):
        """Two calls with the same input produce identical output."""
        np.random.seed(self.seed + 14)
        batch, k_packed, n = 1, 16, 256
        num_ls_rows = 1

        zipped_pd = paddle.to_tensor(
            np.random.randint(0, 256, (batch, k_packed, n), dtype=np.uint8),
            dtype=paddle.uint8,
        )
        local_scale_pd = paddle.to_tensor(
            np.random.randint(0, 256, (batch, num_ls_rows, n), dtype=np.uint8),
            dtype=paddle.uint8,
        )
        code_scale_pd = paddle.full([batch, n], 128.0, dtype=paddle.float32)
        code_zp_pd = paddle.zeros([batch, n], dtype=paddle.float32)
        super_scale_pd = paddle.ones([batch, n], dtype=paddle.float16) * 0.01

        out1 = winx_unzip(zipped_pd, local_scale_pd, code_scale_pd, code_zp_pd, super_scale_pd, "weight_only_int2")
        out2 = winx_unzip(zipped_pd, local_scale_pd, code_scale_pd, code_zp_pd, super_scale_pd, "weight_only_int2")
        np.testing.assert_array_equal(out1.numpy(), out2.numpy())

    def test_wint2_bfloat16_dtype(self):
        """wint2 with bfloat16 super_scale produces bfloat16 output."""
        np.random.seed(self.seed + 15)
        batch, k_packed, n = 1, 16, 256
        num_ls_rows = 1

        zipped_pd = paddle.to_tensor(
            np.random.randint(0, 256, (batch, k_packed, n), dtype=np.uint8),
            dtype=paddle.uint8,
        )
        local_scale_pd = paddle.to_tensor(
            np.random.randint(0, 256, (batch, num_ls_rows, n), dtype=np.uint8),
            dtype=paddle.uint8,
        )
        code_scale_pd = paddle.full([batch, n], 128.0, dtype=paddle.float32)
        code_zp_pd = paddle.zeros([batch, n], dtype=paddle.float32)
        super_scale_pd = paddle.ones([batch, n], dtype=paddle.bfloat16) * 0.01

        out = winx_unzip(zipped_pd, local_scale_pd, code_scale_pd, code_zp_pd, super_scale_pd, "weight_only_int2")
        self.assertEqual(out.dtype, paddle.bfloat16)
        # Verify non-trivial output
        self.assertTrue(np.any(out.astype(paddle.float32).numpy() != 0))


if __name__ == "__main__":
    unittest.main()
