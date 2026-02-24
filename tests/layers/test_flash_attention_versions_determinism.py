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
Test determinism for Flash Attention V2 and V3

Check determinism behavior across different Flash Attention backends:
- FA2 (Flash Attention V2)
- FA3 (Flash Attention V3)
"""

import unittest

import pytest

pytestmark = pytest.mark.gpu

import paddle
import paddle.nn.functional as F

print("=" * 70)
print(" FLASH ATTENTION V2/V3 DETERMINISM TEST")
print("=" * 70)
print()

# Base configuration
BATCH_SIZE = 2
NUM_HEADS = 32
HEAD_DIM = 64
SEQ_LEN = 2048


class TestFlashAttentionDeterminism(unittest.TestCase):
    """Test Flash Attention determinism"""

    def setUp(self):
        """Set up test environment and save current flash attn version"""
        self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(self.device)

        # Save current flash attn version to restore later
        self._saved_flash_attn_version = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])[
            "FLAGS_flash_attn_version"
        ]

    def tearDown(self):
        """Restore original flash attn version after each test"""
        paddle.set_flags({"FLAGS_flash_attn_version": self._saved_flash_attn_version})

    def _check_fa3_support(self):
        """Check if GPU architecture supports FA3"""
        prop = paddle.device.cuda.get_device_properties()
        sm_version = prop.major * 10 + prop.minor
        if sm_version < 89 or sm_version >= 100:
            self.skipTest(f"Flash Attention V3 requires SM89+ but SM100-. Current: SM{sm_version}")

    def _check_cuda_support(self):
        """Check if CUDA is supported"""
        if not paddle.is_compiled_with_cuda():
            self.skipTest("Flash Attention requires CUDA")

    def _test_determinism(self, version, num_runs=5, dtype="float16", **attn_kwargs):
        """Generic method for testing determinism

        Args:
            version: Flash Attention version (2 or 3)
            num_runs: Number of runs
            dtype: Data type
            **attn_kwargs: Parameters passed to scaled_dot_product_attention
        """
        self._check_cuda_support()
        if version == 3:
            self._check_fa3_support()

        paddle.set_flags({"FLAGS_flash_attn_version": version})

        paddle.seed(42)
        query = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype=dtype)
        key = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype=dtype)
        value = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype=dtype)

        results = []
        for _ in range(num_runs):
            paddle.device.synchronize()
            result = F.scaled_dot_product_attention(query, key, value, backend="flash", **attn_kwargs)
            results.append(result.clone())

        all_equal = True
        for i in range(1, num_runs):
            is_equal = paddle.equal(results[0], results[i]).all().item()
            if not is_equal:
                all_equal = False

        version_name = f"FA{version}"
        mode = "causal" if attn_kwargs.get("is_causal") else "non-causal"
        if all_equal:
            print(f"  Result: {version_name} {mode} mode is deterministic")
        else:
            print(f"  Result: {version_name} {mode} mode is NOT deterministic")

        return all_equal

    def _test_batch_invariance(self, version, dtype="float16", **attn_kwargs):
        """Generic method for testing batch invariance

        Verify that the same request's q,k,v yields consistent results
        across different batch sizes.
        """
        self._check_cuda_support()
        if version == 3:
            self._check_fa3_support()

        paddle.set_flags({"FLAGS_flash_attn_version": version})

        MAX_BATCH_SIZE = 8
        paddle.seed(42)
        full_query = paddle.randn([MAX_BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype=dtype)
        full_key = paddle.randn([MAX_BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype=dtype)
        full_value = paddle.randn([MAX_BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype=dtype)

        reference_result = None
        batch_sizes = [1, 2, 4, 8]
        all_equal = True

        for bs in batch_sizes:
            query = full_query[:bs]
            key = full_key[:bs]
            value = full_value[:bs]

            result = F.scaled_dot_product_attention(query, key, value, backend="flash", **attn_kwargs)
            current_result = result[0:1]

            if reference_result is None:
                reference_result = current_result.clone()
                is_equal = True
            else:
                is_equal = paddle.equal(reference_result, current_result).all().item()

            if not is_equal:
                all_equal = False

        version_name = f"FA{version}"
        if all_equal:
            print(f"  Result: {version_name} is batch invariant")
        else:
            print(f"  Result: {version_name} is NOT batch invariant")

        return all_equal

    def _test_seq_length_determinism(self, version, seq_lengths, dtype="float16", **attn_kwargs):
        """Generic method for testing determinism across different sequence lengths"""
        self._check_cuda_support()
        if version == 3:
            self._check_fa3_support()

        paddle.set_flags({"FLAGS_flash_attn_version": version})

        all_deterministic = []
        for seq_len in seq_lengths:
            paddle.seed(42)
            query = paddle.randn([BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM], dtype=dtype)
            key = paddle.randn([BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM], dtype=dtype)
            value = paddle.randn([BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM], dtype=dtype)

            result1 = F.scaled_dot_product_attention(query, key, value, backend="flash", **attn_kwargs)
            result2 = F.scaled_dot_product_attention(query, key, value, backend="flash", **attn_kwargs)

            is_equal = paddle.equal(result1, result2).all().item()
            all_deterministic.append(is_equal)

        all_equal = all(all_deterministic)
        version_name = f"FA{version}"
        if all_equal:
            print(f"  Result: {version_name} is deterministic for all sequence lengths")
        else:
            print(f"  Result: {version_name} is NOT deterministic for some sequence lengths")

        return all_equal

    # ==================== Basic Determinism Tests ====================

    def test_fa2_determinism(self):
        """Test Flash Attention V2 determinism"""
        print("\n[1] Flash Attention V2 Determinism Test")
        print("-" * 70)
        all_equal = self._test_determinism(2, num_runs=5, is_causal=False, enable_gqa=False)
        self.assertTrue(all_equal, "FA2 results are not equal")

    def test_fa3_determinism(self):
        """Test Flash Attention V3 determinism"""
        print("\n[2] Flash Attention V3 Determinism Test")
        print("-" * 70)
        all_equal = self._test_determinism(3, num_runs=5, is_causal=False, enable_gqa=False)
        self.assertTrue(all_equal, "FA3 results are not equal")

    def test_fa2_causal_determinism(self):
        """Test FA2 determinism with causal mask"""
        print("\n[3] FA2 Causal Mask Test")
        print("-" * 70)
        all_equal = self._test_determinism(2, num_runs=5, is_causal=True, enable_gqa=False)
        self.assertTrue(all_equal, "FA2 causal mode not deterministic")

    def test_fa3_causal_determinism(self):
        """Test FA3 determinism with causal mask"""
        print("\n[4] FA3 Causal Mask Test")
        print("-" * 70)
        all_equal = self._test_determinism(3, num_runs=5, is_causal=True, enable_gqa=False)
        self.assertTrue(all_equal, "FA3 causal mode not deterministic")

    # ==================== Batch Invariance Tests ====================

    def test_fa2_batch_invariance(self):
        """Test FA2 determinism across different batch sizes"""
        print("\n[5] FA2 Batch Invariance Test")
        print("-" * 70)
        all_equal = self._test_batch_invariance(2, is_causal=False, enable_gqa=False)
        self.assertTrue(all_equal, "FA2 is not batch invariant")

    def test_fa3_batch_invariance(self):
        """Test FA3 determinism across different batch sizes"""
        print("\n[6] FA3 Batch Invariance Test")
        print("-" * 70)
        all_equal = self._test_batch_invariance(3, is_causal=False, enable_gqa=False)
        self.assertTrue(all_equal, "FA3 is not batch invariant")

    # ==================== Sequence Length Tests ====================

    def test_fa2_seq_length_determinism(self):
        """Test FA2 determinism across different sequence lengths"""
        print("\n[7] FA2 Different Sequence Lengths Test")
        print("-" * 70)
        seq_lengths = [16, 32, 64, 128, 256, 512]
        all_equal = self._test_seq_length_determinism(2, seq_lengths, is_causal=False, enable_gqa=False)
        self.assertTrue(all_equal, "FA2 is not deterministic for all sequence lengths")

    def test_fa3_seq_length_determinism(self):
        """Test FA3 determinism across different sequence lengths"""
        print("\n[8] FA3 Different Sequence Lengths Test")
        print("-" * 70)
        seq_lengths = [16, 32, 64, 128, 256, 512]
        all_equal = self._test_seq_length_determinism(3, seq_lengths, is_causal=False, enable_gqa=False)
        self.assertTrue(all_equal, "FA3 is not deterministic for all sequence lengths")

    # ==================== Boundary Sequence Length Tests ====================

    def test_fa2_boundary_seq_lengths(self):
        """Test FA2 determinism with boundary sequence lengths"""
        print("\n[9] FA2 Boundary Sequence Lengths Test")
        print("-" * 70)
        # Include extremely small and large values to cover different code paths
        seq_lengths = [1, 2, 4, 8, 64, 128, 1024, 2048, 4096]
        all_equal = self._test_seq_length_determinism(2, seq_lengths, is_causal=False, enable_gqa=False)
        self.assertTrue(all_equal, "FA2 is not deterministic for boundary sequence lengths")

    def test_fa3_boundary_seq_lengths(self):
        """Test FA3 determinism with boundary sequence lengths"""
        print("\n[10] FA3 Boundary Sequence Lengths Test")
        print("-" * 70)
        seq_lengths = [1, 2, 4, 8, 64, 128, 1024, 2048, 4096]
        all_equal = self._test_seq_length_determinism(3, seq_lengths, is_causal=False, enable_gqa=False)
        self.assertTrue(all_equal, "FA3 is not deterministic for boundary sequence lengths")

    # ==================== Data Type Tests ====================

    def test_fa2_float16_determinism(self):
        """Test FA2 determinism with float16"""
        print("\n[11] FA2 float16 Data Type Test")
        print("-" * 70)
        all_equal = self._test_determinism(2, num_runs=3, dtype="float16", is_causal=False, enable_gqa=False)
        self.assertTrue(all_equal, "FA2 float16 results are not equal")

    def test_fa3_float16_determinism(self):
        """Test FA3 determinism with float16"""
        print("\n[12] FA3 float16 Data Type Test")
        print("-" * 70)
        all_equal = self._test_determinism(3, num_runs=3, dtype="float16", is_causal=False, enable_gqa=False)
        self.assertTrue(all_equal, "FA3 float16 results are not equal")

    def test_fa2_float32_determinism(self):
        """Test FA2 determinism with float32"""
        print("\n[13] FA2 float32 Data Type Test")
        print("-" * 70)
        all_equal = self._test_determinism(2, num_runs=3, dtype="float32", is_causal=False, enable_gqa=False)
        self.assertTrue(all_equal, "FA2 float32 results are not equal")

    def test_fa3_float32_determinism(self):
        """Test FA3 determinism with float32"""
        print("\n[14] FA3 float32 Data Type Test")
        print("-" * 70)
        all_equal = self._test_determinism(3, num_runs=3, dtype="float32", is_causal=False, enable_gqa=False)
        self.assertTrue(all_equal, "FA3 float32 results are not equal")

    # ==================== Head Configuration Tests ====================

    def _test_head_config_determinism(self, version, num_heads, head_dim):
        """Helper method for testing determinism with different head configurations"""
        self._check_cuda_support()
        if version == 3:
            self._check_fa3_support()

        paddle.set_flags({"FLAGS_flash_attn_version": version})

        paddle.seed(42)
        query = paddle.randn([BATCH_SIZE, num_heads, SEQ_LEN, head_dim], dtype="float16")
        key = paddle.randn([BATCH_SIZE, num_heads, SEQ_LEN, head_dim], dtype="float16")
        value = paddle.randn([BATCH_SIZE, num_heads, SEQ_LEN, head_dim], dtype="float16")

        result1 = F.scaled_dot_product_attention(query, key, value, backend="flash", is_causal=False, enable_gqa=False)
        result2 = F.scaled_dot_product_attention(query, key, value, backend="flash", is_causal=False, enable_gqa=False)

        is_equal = paddle.equal(result1, result2).all().item()
        return is_equal

    def test_fa2_single_head(self):
        """Test FA2 determinism with single head"""
        print("\n[15] FA2 Single Head Test")
        print("-" * 70)
        is_equal = self._test_head_config_determinism(2, num_heads=1, head_dim=HEAD_DIM)
        self.assertTrue(is_equal, "FA2 single head results are not equal")

    def test_fa3_single_head(self):
        """Test FA3 determinism with single head"""
        print("\n[16] FA3 Single Head Test")
        print("-" * 70)
        is_equal = self._test_head_config_determinism(3, num_heads=1, head_dim=HEAD_DIM)
        self.assertTrue(is_equal, "FA3 single head results are not equal")

    def test_fa2_odd_num_heads(self):
        """Test FA2 determinism with odd number of heads"""
        print("\n[17] FA2 Odd Number of Heads Test")
        print("-" * 70)
        is_equal = self._test_head_config_determinism(2, num_heads=7, head_dim=HEAD_DIM)
        self.assertTrue(is_equal, "FA2 odd number of heads results are not equal")

    def test_fa3_odd_num_heads(self):
        """Test FA3 determinism with odd number of heads"""
        print("\n[18] FA3 Odd Number of Heads Test")
        print("-" * 70)
        is_equal = self._test_head_config_determinism(3, num_heads=7, head_dim=HEAD_DIM)
        self.assertTrue(is_equal, "FA3 odd number of heads results are not equal")

    # ==================== GQA/MQA Tests ====================

    def test_fa2_gqa_determinism(self):
        """Test FA2 determinism with GQA"""
        print("\n[19] FA2 GQA Test")
        print("-" * 70)
        all_equal = self._test_determinism(2, num_runs=3, is_causal=False, enable_gqa=True)
        self.assertTrue(all_equal, "FA2 GQA results are not equal")

    def test_fa3_gqa_determinism(self):
        """Test FA3 determinism with GQA"""
        print("\n[20] FA3 GQA Test")
        print("-" * 70)
        all_equal = self._test_determinism(3, num_runs=3, is_causal=False, enable_gqa=True)
        self.assertTrue(all_equal, "FA3 GQA results are not equal")


if __name__ == "__main__":
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFlashAttentionDeterminism)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    print(" Test Summary")
    print("=" * 70)
    print(f"  Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failed: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    print(f"  Total: {result.testsRun}")
    print()

    if result.wasSuccessful():
        print(" ✓ All tests passed!")
    else:
        print(" ✗ Some tests failed or had errors")

    print("=" * 70)
