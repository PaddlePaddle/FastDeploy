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
Flash Attention V2 / V3 determinism tests.

Verify bitwise determinism of flash-backend SDPA when explicitly
selecting FA version via FLAGS_flash_attn_version (2 or 3).
"""

import unittest

import pytest

pytestmark = pytest.mark.gpu

import paddle
import paddle.nn.functional as F

# --------------- constants ---------------
BATCH_SIZE = 2
NUM_HEADS = 32
HEAD_DIM = 64
SEQ_LEN = 2048
NUM_RUNS = 5


# --------------- helpers ---------------
def _make_qkv(batch_size, num_heads, seq_len, head_dim, dtype="float16", seed=42):
    """Create deterministic q/k/v tensors."""
    paddle.seed(seed)
    shape = [batch_size, num_heads, seq_len, head_dim]
    return (
        paddle.randn(shape, dtype=dtype),
        paddle.randn(shape, dtype=dtype),
        paddle.randn(shape, dtype=dtype),
    )


def _assert_deterministic(test_case, func, num_runs=NUM_RUNS):
    """Run *func* multiple times and assert all results are bitwise equal."""
    results = [func().clone() for _ in range(num_runs)]
    for i in range(1, num_runs):
        test_case.assertTrue(
            paddle.equal(results[0], results[i]).all().item(),
            f"Run 0 vs Run {i} differ",
        )


# --------------- test class ---------------
class TestFlashAttentionVersionsDeterminism(unittest.TestCase):
    """Test determinism when switching between FA2 and FA3."""

    FA_VERSIONS = [2, 3]

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("Flash Attention requires CUDA")
        paddle.set_device("gpu")
        # Save/restore flag to avoid cross-test pollution
        self._saved_version = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])["FLAGS_flash_attn_version"]

    def tearDown(self):
        paddle.set_flags({"FLAGS_flash_attn_version": self._saved_version})

    def _skip_if_fa3_unsupported(self):
        prop = paddle.device.cuda.get_device_properties()
        sm = prop.major * 10 + prop.minor
        if sm < 89 or sm >= 100:
            self.skipTest(f"FA3 requires SM89-SM99, current SM{sm}")

    def _set_fa_version(self, version):
        if version == 3:
            self._skip_if_fa3_unsupported()
        paddle.set_flags({"FLAGS_flash_attn_version": version})

    def _flash_sdpa(self, q, k, v, **kwargs):
        """Thin wrapper: synchronize then call flash-backend SDPA."""
        paddle.device.synchronize()
        return F.scaled_dot_product_attention(q, k, v, backend="flash", **kwargs)

    # ==================== tests ====================

    def test_determinism(self):
        """Multi-run determinism for FA2/FA3, causal and non-causal."""
        for version in self.FA_VERSIONS:
            for is_causal in [False, True]:
                with self.subTest(version=version, is_causal=is_causal):
                    self._set_fa_version(version)
                    q, k, v = _make_qkv(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
                    _assert_deterministic(
                        self,
                        lambda: self._flash_sdpa(q, k, v, is_causal=is_causal, enable_gqa=False),
                    )

    def test_batch_invariance(self):
        """First-sample result should be identical across batch sizes."""
        for version in self.FA_VERSIONS:
            with self.subTest(version=version):
                self._set_fa_version(version)
                max_bs = 8
                q, k, v = _make_qkv(max_bs, NUM_HEADS, SEQ_LEN, HEAD_DIM)

                ref = self._flash_sdpa(q[:1], k[:1], v[:1], is_causal=False, enable_gqa=False)
                for bs in [2, 4, 8]:
                    result = self._flash_sdpa(q[:bs], k[:bs], v[:bs], is_causal=False, enable_gqa=False)
                    self.assertTrue(
                        paddle.equal(ref, result[0:1]).all().item(),
                        f"FA{version} batch invariance failed at bs={bs}",
                    )

    def test_seq_length_determinism(self):
        """Determinism across various sequence lengths (including boundaries)."""
        seq_lengths = [1, 2, 4, 8, 16, 64, 128, 256, 512, 1024, 2048, 4096]
        for version in self.FA_VERSIONS:
            for seq_len in seq_lengths:
                with self.subTest(version=version, seq_len=seq_len):
                    self._set_fa_version(version)
                    q, k, v = _make_qkv(BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM)
                    _assert_deterministic(
                        self,
                        lambda: self._flash_sdpa(q, k, v, is_causal=False, enable_gqa=False),
                        num_runs=2,
                    )

    def test_dtype_determinism(self):
        """Determinism across float16 and float32."""
        for version in self.FA_VERSIONS:
            for dtype in ["float16", "float32"]:
                with self.subTest(version=version, dtype=dtype):
                    self._set_fa_version(version)
                    q, k, v = _make_qkv(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype)
                    _assert_deterministic(
                        self,
                        lambda: self._flash_sdpa(q, k, v, is_causal=False, enable_gqa=False),
                        num_runs=3,
                    )

    def test_head_config_determinism(self):
        """Determinism across different head configurations."""
        for version in self.FA_VERSIONS:
            for num_heads, head_dim in [(1, 64), (7, 64), (32, 64)]:
                with self.subTest(version=version, num_heads=num_heads, head_dim=head_dim):
                    self._set_fa_version(version)
                    q, k, v = _make_qkv(BATCH_SIZE, num_heads, SEQ_LEN, head_dim)
                    _assert_deterministic(
                        self,
                        lambda: self._flash_sdpa(q, k, v, is_causal=False, enable_gqa=False),
                        num_runs=2,
                    )

    def test_gqa_determinism(self):
        """Determinism with GQA enabled."""
        for version in self.FA_VERSIONS:
            with self.subTest(version=version):
                self._set_fa_version(version)
                q, k, v = _make_qkv(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
                _assert_deterministic(
                    self,
                    lambda: self._flash_sdpa(q, k, v, is_causal=False, enable_gqa=True),
                    num_runs=3,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
