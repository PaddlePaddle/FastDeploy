# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
"""Tests for cutlass_scaled_mm: fp8/int8 per-tensor/per-token/blockwise scaling."""

import math
import subprocess
import unittest

import numpy as np
import paddle
import paddle.nn.functional as F

from fastdeploy.model_executor.layers.quantization.ops import (
    cutlass_scaled_mm,
    scaled_fp8_quant,
)

_FP8_MAX = 448.0  # float8_e4m3fn max value


def _fp8_blockwise_quant_a(a: paddle.Tensor, block_size: int = 128):
    """
    Quantize A [M, K] to fp8 with a_scales [M, ceil(K/128)].
    Each scale covers one row-block of 128 elements along K.
    """
    M, K = a.shape
    num_k = math.ceil(K / block_size)
    K_pad = num_k * block_size
    # Pad: outermost-first convention → [N_right_pad, K_right_pad]
    a_f32 = F.pad(a.cast(paddle.float32), [0, 0, 0, K_pad - K]) if K_pad > K else a.cast(paddle.float32)
    # [M, num_k, block_size]
    a_3d = a_f32.reshape([M, num_k, block_size])
    scale = a_3d.abs().max(axis=2) / _FP8_MAX  # [M, num_k]
    scale = paddle.where(scale == 0, paddle.ones_like(scale), scale)
    a_q = paddle.clip(a_3d / scale.unsqueeze(2), -_FP8_MAX, _FP8_MAX)
    a_q = a_q.reshape([M, K_pad])[:, :K].cast(paddle.float8_e4m3fn)
    return a_q, scale  # scale: [M, ceil(K/128)]


def _fp8_blockwise_quant_b(b: paddle.Tensor, block_size: int = 128):
    """
    Quantize B [N, K] to fp8 with b_scales [ceil(K/128), ceil(N/128)].
    Each scale covers a 128×128 block spanning both the N and K dimensions.
    """
    N, K = b.shape
    num_k = math.ceil(K / block_size)
    num_n = math.ceil(N / block_size)
    K_pad = num_k * block_size
    N_pad = num_n * block_size
    # Pad to [N_pad, K_pad]; paddle F.pad outermost-first: [N_right, K_right]
    b_f32 = b.cast(paddle.float32)
    if N_pad > N or K_pad > K:
        b_f32 = F.pad(b_f32, [0, N_pad - N, 0, K_pad - K])
    # [num_n, block_size, num_k, block_size] → [num_n, num_k, block, block]
    b_4d = b_f32.reshape([num_n, block_size, num_k, block_size]).transpose([0, 2, 1, 3])
    # per-block scale: [num_n, num_k]
    scale_nk = b_4d.abs().reshape([num_n, num_k, block_size * block_size]).max(axis=2) / _FP8_MAX
    scale_nk = paddle.where(scale_nk == 0, paddle.ones_like(scale_nk), scale_nk)
    # Quantize
    scale_exp = scale_nk.unsqueeze(2).unsqueeze(3).expand([num_n, num_k, block_size, block_size])
    b_q_4d = paddle.clip(b_4d / scale_exp, -_FP8_MAX, _FP8_MAX)
    b_q = b_q_4d.transpose([0, 2, 1, 3]).reshape([N_pad, K_pad])[:N, :K].cast(paddle.float8_e4m3fn)
    # kernel expects b_scales [ceil(K/128), ceil(N/128)]
    b_scales = scale_nk.T.contiguous()
    return b_q, b_scales


class Test(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        np.random.seed(42)
        self.prop = paddle.device.cuda.get_device_properties()
        self.sm_version = self.prop.major * 10 + self.prop.minor
        print(self.prop)
        print(paddle.__git_commit__)
        nvcc_output = subprocess.check_output(["nvcc", "--version"], universal_newlines=True)
        output = nvcc_output.split()
        release_idx = output.index("release") + 1
        self.nvcc_cuda_version = float(output[release_idx].split(",")[0])

    # ─────────────────────────────────────────────────────────────────────────
    # Existing tests (SM89+): per-token fp8 and per-channel int8
    # ─────────────────────────────────────────────────────────────────────────

    def test_cutlass_scaled_mm_fp8(self):
        """Per-token fp8: verify cutlass_scaled_mm matches paddle.matmul baseline."""
        if self.sm_version < 89:
            self.skipTest("cutlass_scaled_mm fp8 requires sm89+")
        M, N, K = 32, 1024, 1024
        a = paddle.rand([M, K], dtype=paddle.bfloat16)
        b = paddle.rand([N, K], dtype=paddle.bfloat16)
        b_q, b_scales = scaled_fp8_quant(b, use_per_token_if_dynamic=False)
        a_q, a_scales = scaled_fp8_quant(a, use_per_token_if_dynamic=True)

        bias = paddle.rand([N], dtype=paddle.bfloat16)
        baseline = paddle.matmul(a, b, transpose_y=True) + bias
        c = cutlass_scaled_mm(a_q, b_q, a_scales, b_scales, paddle.bfloat16, bias)
        self.assertTrue(
            np.allclose(baseline.numpy(), c.numpy(), rtol=1e-2, atol=1e-2),
            "fp8 per-token result mismatch",
        )

    def test_cutlass_scaled_mm_int8(self):
        """Per-channel int8: verify cutlass_scaled_mm matches paddle.matmul baseline."""
        if self.sm_version >= 100:
            self.skipTest("int8 is not supported on SM100+ (no int8 kernel compiled)")
        M, N, K = 32, 1024, 512
        a = paddle.rand([M, K], dtype=paddle.bfloat16)
        b = paddle.rand([N, K], dtype=paddle.bfloat16)
        a_scales = (a.cast(paddle.float32).abs().max(axis=-1) / 127)[:, None]
        a_q = paddle.clip(a / a_scales, -127, 127).cast(paddle.int8)
        b_scales = (b.cast(paddle.float32).abs().max(axis=-1) / 127)[:, None]
        b_q = paddle.clip(b / b_scales, -127, 127).cast(paddle.int8)

        bias = paddle.rand([N], dtype=paddle.bfloat16)
        baseline = paddle.matmul(a, b, transpose_y=True) + bias
        c = cutlass_scaled_mm(a_q, b_q, a_scales, b_scales, paddle.bfloat16, bias)
        self.assertTrue(
            np.allclose(baseline.numpy(), c.numpy(), rtol=1e-2, atol=1e-2),
            "int8 per-channel result mismatch",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SM90 (Hopper): per-token fp8, multiple M shapes, fp16 output
    # ─────────────────────────────────────────────────────────────────────────

    def _run_fp8_per_token(self, M, N, K, out_dtype, bias=False):
        """Helper: run per-token fp8 cutlass_scaled_mm and compare to baseline."""
        a = paddle.rand([M, K], dtype=paddle.bfloat16)
        b = paddle.rand([N, K], dtype=paddle.bfloat16)
        b_q, b_scales = scaled_fp8_quant(b, use_per_token_if_dynamic=False)
        a_q, a_scales = scaled_fp8_quant(a, use_per_token_if_dynamic=True)
        bias_t = paddle.rand([N], dtype=out_dtype) if bias else None
        baseline = paddle.matmul(a, b, transpose_y=True).cast(out_dtype)
        if bias_t is not None:
            baseline = baseline + bias_t
        c = cutlass_scaled_mm(a_q, b_q, a_scales, b_scales, out_dtype, bias_t)
        return np.allclose(baseline.numpy(), c.numpy(), rtol=1e-2, atol=1e-2)

    def test_sm90_fp8_small_M(self):
        """SM90: fp8 per-token, M=8 (small-M path, N=512 ≤ 1280)."""
        if self.sm_version != 90:
            self.skipTest("SM90-specific test")
        self.assertTrue(self._run_fp8_per_token(8, 512, 1024, paddle.bfloat16))

    def test_sm90_fp8_small_M_large_N(self):
        """SM90: fp8 per-token, M=8, N=2048 > 1280 — hits M16_N8192 config."""
        if self.sm_version != 90:
            self.skipTest("SM90-specific test")
        self.assertTrue(self._run_fp8_per_token(8, 2048, 4096, paddle.bfloat16))

    def test_sm90_fp8_M64(self):
        """SM90: fp8 per-token, M=32, N=512 — hits M64_N1280 config."""
        if self.sm_version != 90:
            self.skipTest("SM90-specific test")
        self.assertTrue(self._run_fp8_per_token(32, 512, 1024, paddle.bfloat16))

    def test_sm90_fp8_M64_large_N(self):
        """SM90: fp8 per-token, M=64, N=4096 — hits M64_N8192 config."""
        if self.sm_version != 90:
            self.skipTest("SM90-specific test")
        self.assertTrue(self._run_fp8_per_token(64, 4096, 4096, paddle.bfloat16))

    def test_sm90_fp8_M128(self):
        """SM90: fp8 per-token, M=128 — hits M128 config."""
        if self.sm_version != 90:
            self.skipTest("SM90-specific test")
        self.assertTrue(self._run_fp8_per_token(128, 1024, 2048, paddle.bfloat16))

    def test_sm90_fp8_large_M(self):
        """SM90: fp8 per-token, M=256, K=4096 — hits default config."""
        if self.sm_version != 90:
            self.skipTest("SM90-specific test")
        self.assertTrue(self._run_fp8_per_token(256, 4096, 4096, paddle.bfloat16))

    def test_sm90_fp8_large_MK_cooperative(self):
        """SM90: fp8, M=8192, K=6144 — hits M8192_K6144 Cooperative config."""
        if self.sm_version != 90:
            self.skipTest("SM90-specific test")
        self.assertTrue(self._run_fp8_per_token(8192, 4096, 6144, paddle.bfloat16))

    def test_sm90_fp8_fp16_output(self):
        """SM90: fp8 per-token with float16 output dtype."""
        if self.sm_version != 90:
            self.skipTest("SM90-specific test")
        self.assertTrue(self._run_fp8_per_token(64, 1024, 1024, paddle.float16))

    def test_sm90_fp8_bias(self):
        """SM90: fp8 per-token with bias."""
        if self.sm_version != 90:
            self.skipTest("SM90-specific test")
        self.assertTrue(self._run_fp8_per_token(32, 1024, 1024, paddle.bfloat16, bias=True))

    # ─────────────────────────────────────────────────────────────────────────
    # SM90 (Hopper): blockwise fp8 (DeepSeek-V3 style)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_blockwise_fp8(self, M, N, K, out_dtype):
        """
        Helper: blockwise fp8 cutlass_scaled_mm.

        Scale shapes: a_scales [M, ceil(K/128)], b_scales [ceil(K/128), ceil(N/128)].
        SM90 additionally requires M % 4 == 0.
        Baseline uses the original float tensors; rtol/atol are loose to account
        for fp8 quantisation error.
        """
        a = paddle.rand([M, K], dtype=paddle.bfloat16)
        b = paddle.rand([N, K], dtype=paddle.bfloat16)

        a_q, a_scales = _fp8_blockwise_quant_a(a)  # a_scales: [M, ceil(K/128)]
        b_q, b_scales = _fp8_blockwise_quant_b(b)  # b_scales: [ceil(K/128), ceil(N/128)]

        baseline = paddle.matmul(a, b, transpose_y=True).cast(out_dtype)
        c = cutlass_scaled_mm(a_q, b_q, a_scales, b_scales, out_dtype)
        return np.allclose(baseline.numpy(), c.numpy(), rtol=1e-1, atol=1e-1)

    def test_sm90_fp8_blockwise(self):
        """SM90: blockwise fp8, M=128, N=4096, K=4096."""
        if self.sm_version != 90:
            self.skipTest("SM90 blockwise fp8 test")
        # N and K must be multiples of 128 for blockwise
        self.assertTrue(self._run_blockwise_fp8(128, 4096, 4096, paddle.bfloat16))

    def test_sm90_fp8_blockwise_small_M(self):
        """SM90: blockwise fp8, M=4 (minimum m%4==0 alignment)."""
        if self.sm_version != 90:
            self.skipTest("SM90 blockwise fp8 test")
        self.assertTrue(self._run_blockwise_fp8(4, 1024, 2048, paddle.bfloat16))

    def test_sm90_fp8_blockwise_fp16_output(self):
        """SM90: blockwise fp8 with float16 output."""
        if self.sm_version != 90:
            self.skipTest("SM90 blockwise fp8 test")
        self.assertTrue(self._run_blockwise_fp8(64, 2048, 4096, paddle.float16))

    # ─────────────────────────────────────────────────────────────────────────
    # SM100 (Blackwell GB200): per-token fp8 + blockwise fp8
    # ─────────────────────────────────────────────────────────────────────────

    def test_sm100_fp8_per_token(self):
        """SM100: fp8 per-token, M=64, standard path."""
        if self.sm_version < 100 or self.sm_version >= 120:
            self.skipTest("SM100-specific test")
        if self.nvcc_cuda_version < 12.9:
            self.skipTest("SM100 fp8 requires CUDA 12.9+")
        self.assertTrue(self._run_fp8_per_token(64, 2048, 4096, paddle.bfloat16))

    def test_sm100_fp8_per_token_small_M(self):
        """SM100: fp8 per-token, M=16 — hits swap_ab path."""
        if self.sm_version < 100 or self.sm_version >= 120:
            self.skipTest("SM100-specific test")
        if self.nvcc_cuda_version < 12.9:
            self.skipTest("SM100 fp8 requires CUDA 12.9+")
        self.assertTrue(self._run_fp8_per_token(16, 4096, 4096, paddle.bfloat16))

    def test_sm100_fp8_per_token_large_M(self):
        """SM100: fp8 per-token, M=512 — hits default large config."""
        if self.sm_version < 100 or self.sm_version >= 120:
            self.skipTest("SM100-specific test")
        if self.nvcc_cuda_version < 12.9:
            self.skipTest("SM100 fp8 requires CUDA 12.9+")
        self.assertTrue(self._run_fp8_per_token(512, 4096, 4096, paddle.bfloat16))

    def test_sm100_fp8_per_token_bias(self):
        """SM100: fp8 per-token with bias."""
        if self.sm_version < 100 or self.sm_version >= 120:
            self.skipTest("SM100-specific test")
        if self.nvcc_cuda_version < 12.9:
            self.skipTest("SM100 fp8 requires CUDA 12.9+")
        self.assertTrue(self._run_fp8_per_token(128, 2048, 2048, paddle.bfloat16, bias=True))

    def test_sm100_fp8_blockwise(self):
        """SM100: blockwise fp8, M=128, N=4096, K=4096."""
        if self.sm_version < 100 or self.sm_version >= 120:
            self.skipTest("SM100-specific test")
        if self.nvcc_cuda_version < 12.9:
            self.skipTest("SM100 fp8 requires CUDA 12.9+")
        self.assertTrue(self._run_blockwise_fp8(128, 4096, 4096, paddle.bfloat16))

    def test_sm100_fp8_blockwise_small_M(self):
        """SM100: blockwise fp8, M=8 — triggers swap_ab (m<16)."""
        if self.sm_version < 100 or self.sm_version >= 120:
            self.skipTest("SM100-specific test")
        if self.nvcc_cuda_version < 12.9:
            self.skipTest("SM100 fp8 requires CUDA 12.9+")
        self.assertTrue(self._run_blockwise_fp8(8, 1024, 2048, paddle.bfloat16))

    def test_sm100_fp8_int8_unsupported(self):
        """SM100: int8 is not supported — should raise."""
        if self.sm_version < 100 or self.sm_version >= 120:
            self.skipTest("SM100-specific test")
        if self.nvcc_cuda_version < 12.9:
            self.skipTest("SM100 fp8 requires CUDA 12.9+")
        M, N, K = 32, 1024, 512
        a = paddle.rand([M, K], dtype=paddle.bfloat16)
        b = paddle.rand([N, K], dtype=paddle.bfloat16)
        a_scales = (a.cast(paddle.float32).abs().max(axis=-1) / 127)[:, None]
        a_q = paddle.clip(a / a_scales, -127, 127).cast(paddle.int8)
        b_scales = (b.cast(paddle.float32).abs().max(axis=-1) / 127)[:, None]
        b_q = paddle.clip(b / b_scales, -127, 127).cast(paddle.int8)
        with self.assertRaises(Exception):
            cutlass_scaled_mm(a_q, b_q, a_scales, b_scales, paddle.bfloat16)

    # ─────────────────────────────────────────────────────────────────────────
    # SM120 (Blackwell RTX 5090): per-token fp8 + blockwise fp8
    # ─────────────────────────────────────────────────────────────────────────

    def test_sm120_fp8_per_token_M16(self):
        """SM120: fp8 per-token, M=16 — hits M16 small-tile config."""
        if self.sm_version < 120:
            self.skipTest("SM120-specific test")
        self.assertTrue(self._run_fp8_per_token(16, 1024, 2048, paddle.bfloat16))

    def test_sm120_fp8_per_token_M32(self):
        """SM120: fp8 per-token, M=32 — hits M32 custom-tile config."""
        if self.sm_version < 120:
            self.skipTest("SM120-specific test")
        self.assertTrue(self._run_fp8_per_token(32, 1024, 2048, paddle.bfloat16))

    def test_sm120_fp8_per_token_M64(self):
        """SM120: fp8 per-token, M=64 — hits M64 Pingpong config."""
        if self.sm_version < 120:
            self.skipTest("SM120-specific test")
        self.assertTrue(self._run_fp8_per_token(64, 2048, 4096, paddle.bfloat16))

    def test_sm120_fp8_per_token_M256(self):
        """SM120: fp8 per-token, M=256 — hits M64 config (≤256 branch)."""
        if self.sm_version < 120:
            self.skipTest("SM120-specific test")
        self.assertTrue(self._run_fp8_per_token(256, 2048, 4096, paddle.bfloat16))

    def test_sm120_fp8_per_token_large_M(self):
        """SM120: fp8 per-token, M=512 — hits default config."""
        if self.sm_version < 120:
            self.skipTest("SM120-specific test")
        self.assertTrue(self._run_fp8_per_token(512, 4096, 4096, paddle.bfloat16))

    def test_sm120_fp8_per_token_fp16_output(self):
        """SM120: fp8 per-token with float16 output."""
        if self.sm_version < 120:
            self.skipTest("SM120-specific test")
        self.assertTrue(self._run_fp8_per_token(64, 1024, 2048, paddle.float16))

    def test_sm120_fp8_per_token_bias(self):
        """SM120: fp8 per-token with bias."""
        if self.sm_version < 120:
            self.skipTest("SM120-specific test")
        self.assertTrue(self._run_fp8_per_token(128, 2048, 2048, paddle.bfloat16, bias=True))

    def test_sm120_fp8_blockwise(self):
        """SM120: blockwise fp8, M=128, N=4096, K=4096."""
        if self.sm_version < 120:
            self.skipTest("SM120-specific test")
        self.assertTrue(self._run_blockwise_fp8(128, 4096, 4096, paddle.bfloat16))

    def test_sm120_fp8_blockwise_small_M(self):
        """SM120: blockwise fp8, M=64 — hits small-M 64x128 tile."""
        if self.sm_version < 120:
            self.skipTest("SM120-specific test")
        self.assertTrue(self._run_blockwise_fp8(64, 2048, 4096, paddle.bfloat16))

    def test_sm120_fp8_int8_unsupported(self):
        """SM120: int8 is not supported — should raise."""
        if self.sm_version < 120:
            self.skipTest("SM120-specific test")
        M, N, K = 32, 1024, 512
        a = paddle.rand([M, K], dtype=paddle.bfloat16)
        b = paddle.rand([N, K], dtype=paddle.bfloat16)
        a_scales = (a.cast(paddle.float32).abs().max(axis=-1) / 127)[:, None]
        a_q = paddle.clip(a / a_scales, -127, 127).cast(paddle.int8)
        b_scales = (b.cast(paddle.float32).abs().max(axis=-1) / 127)[:, None]
        b_q = paddle.clip(b / b_scales, -127, 127).cast(paddle.int8)
        with self.assertRaises(Exception):
            cutlass_scaled_mm(a_q, b_q, a_scales, b_scales, paddle.bfloat16)


if __name__ == "__main__":
    unittest.main()
