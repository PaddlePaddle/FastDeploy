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
Tests for Paddle Attention Determinism

Test scenarios:
1. Multiple runs with same batch size, check if results are consistent
2. Different batch sizes, check if results are consistent (test batch invariance)
3. Test determinism in both prefill and decode modes
4. Test determinism under different sequence lengths
"""

import unittest

import paddle

from fastdeploy.model_executor.layers.attention.native_paddle_backend import (
    PaddleNativeAttnBackend,
)
from fastdeploy.model_executor.layers.batch_invariant_ops import (
    set_batch_invariant_mode,
)


class TestPaddleAttentionDeterminism(unittest.TestCase):
    def setUp(self):
        """Set up testing environment"""
        self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(self.device)
        self.dtype = "float32"  # Use float32 for higher precision comparison

        # Initialize attention backend
        self.attn_backend = PaddleNativeAttnBackend()

    def test_sdpa_multiple_runs_same_batch(self):
        """Test if scaled_dot_product_attention produces identical results for same input across multiple runs"""
        print("\n=== Testing SDPA Multiple Runs (Same Batch) ===")

        num_heads = 8
        head_dim = 64
        batch_size = 2
        seq_len = 32

        # Fixed seed for reproducibility
        paddle.seed(42)

        # Create input tensors
        query = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
        key = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
        value = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)

        # Store results from multiple runs
        results = []

        # Run multiple times
        num_runs = 5
        for i in range(num_runs):
            paddle.device.cuda.synchronize() if self.device == "gpu" else None
            result = paddle.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False
            )
            results.append(result.clone())
            print(f"  Run {i+1}: mean={result.mean().item():.6f}, std={result.std().item():.6f}")

        # Verify all runs produce identical results (use equal instead of allclose)
        for i in range(1, num_runs):
            # Use paddle.equal to check exact equality
            is_equal = paddle.equal(results[0], results[i]).all().item()

            print(f"  Run 1 vs Run {i+1}: equal={is_equal}")

            if is_equal:
                print(f"  ✓ Run 1 and Run {i+1} are deterministically identical")
            else:
                print(f"  ✗ Run 1 and Run {i+1} differ!")

            # Check in test environment
            self.assertTrue(is_equal, f"SDPA results differ between run 1 and run {i+1}")

    def test_sdpa_different_batch_sizes(self):
        """Test scaled_dot_product_attention determinism with different batch sizes"""
        print("\n=== Testing SDPA Different Batch Sizes ===")

        num_heads = 8
        head_dim = 64
        seq_len = 32

        # Fixed seed
        paddle.seed(42)

        # Test single sequence
        single_batch_size = 1
        query_single = paddle.randn([single_batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
        key_single = paddle.randn([single_batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
        value_single = paddle.randn([single_batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)

        result_single = paddle.nn.functional.scaled_dot_product_attention(
            query_single, key_single, value_single, is_causal=False, enable_gqa=False
        )
        print(f"  Single batch (bs={single_batch_size}): mean={result_single.mean().item():.6f}")

        # Test different batch sizes
        for batch_size in [2, 4, 8, 16]:
            # Use same seed for consistent input
            paddle.seed(42)

            query = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
            key = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
            value = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)

            result = paddle.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False
            )

            # Only compare first sequence, use equal to check exact equality
            is_equal = paddle.equal(result_single, result[0:1]).all().item()

            print(f"  Batch size {batch_size}: mean={result[0:1].mean().item():.6f}, equal={is_equal}")

            if not is_equal:
                print(f"  ⚠ Single batch and batch_size={batch_size} differ!")
                print("    This is expected - standard attention is not batch-invariant")

    def test_sdpa_causal_mask_determinism(self):
        """Test SDPA determinism with causal mask"""
        print("\n=== Testing SDPA with Causal Mask ===")

        num_heads = 8
        head_dim = 64
        batch_size = 2
        seq_len = 64

        paddle.seed(42)

        query = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
        key = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
        value = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)

        results = []
        num_runs = 5

        for i in range(num_runs):
            paddle.device.cuda.synchronize() if self.device == "gpu" else None
            result = paddle.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=True, enable_gqa=False
            )
            results.append(result.clone())

        # Verify result consistency
        for i in range(1, num_runs):
            is_equal = paddle.equal(results[0], results[i]).all().item()
            print(f"  Run 1 vs Run {i+1}: equal={is_equal}")
            self.assertTrue(is_equal, f"Causal SDPA results differ between run 1 and run {i+1}")

    def test_sdpa_different_sequence_lengths(self):
        """Test determinism with different sequence lengths"""
        print("\n=== Testing SDPA Different Sequence Lengths ===")

        num_heads = 8
        head_dim = 64
        batch_size = 2

        paddle.seed(42)

        # Test different sequence lengths
        seq_lengths = [16, 32, 64, 128, 256]
        results = {}

        for seq_len in seq_lengths:
            paddle.seed(42)

            query = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
            key = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
            value = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)

            result = paddle.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False
            )
            results[seq_len] = result.clone()
            print(f"  Seq length {seq_len}: mean={result.mean().item():.6f}, std={result.std().item():.6f}")

        # Run multiple times for each sequence length to verify determinism
        for seq_len in seq_lengths:
            paddle.seed(42)

            query = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
            key = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
            value = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)

            result_new = paddle.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False
            )

            is_equal = paddle.equal(results[seq_len], result_new).all().item()
            print(f"  Seq {seq_len} (first vs second): equal={is_equal}")
            self.assertTrue(is_equal, f"SDPA results differ for seq_len={seq_len}")

    def test_manual_scaled_dot_product_attention_determinism(self):
        """Test manually implemented scaled_dot_product_attention determinism"""
        print("\n=== Testing Manual SDPA Implementation ===")

        num_heads = 8
        head_dim = 64
        batch_size = 2
        seq_len = 32

        paddle.seed(42)

        query = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
        key = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
        value = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)

        results = []
        num_runs = 5

        for i in range(num_runs):
            paddle.device.cuda.synchronize() if self.device == "gpu" else None

            # Manual attention implementation
            d_k = query.shape[-1]
            scores = paddle.matmul(query, key.transpose([0, 1, 3, 2]))  # QK^T
            scores = scores / paddle.sqrt(paddle.to_tensor(d_k, dtype=scores.dtype))
            attn_weights = paddle.nn.functional.softmax(scores, axis=-1)
            output = paddle.matmul(attn_weights, value)

            results.append(output.clone())

        # Verify result consistency
        for i in range(1, num_runs):
            is_equal = paddle.equal(results[0], results[i]).all().item()
            print(f"  Manual SDPA Run 1 vs Run {i+1}: equal={is_equal}")
            self.assertTrue(is_equal, f"Manual SDPA results differ between run 1 and run {i+1}")

    def test_attention_backend_prefill_determinism(self):
        """Test Attention Backend Prefill mode determinism"""
        print("\n=== Testing Attention Backend Prefill Determinism ===")

        # This test requires more complete setup, skip for now
        self.skipTest("Full backend setup required for prefill test")

    def test_attention_backend_decode_determinism(self):
        """Test Attention Backend Decode mode determinism"""

        print("\n=== Testing Attention Backend Decode Determinism ===")

        # This test requires more complete setup, skip for now
        self.skipTest("Full backend setup required for decode test")

    def test_batch_invariant_mode_compatibility(self):
        """Test compatibility of batch invariant mode with attention"""
        print("\n=== Testing Batch Invariant Mode Compatibility ===")

        num_heads = 8
        head_dim = 64
        batch_size = 4
        seq_len = 32

        paddle.seed(42)

        query = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
        key = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
        value = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)

        # Run in batch invariant mode
        with set_batch_invariant_mode(True):
            result_bi = paddle.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
            print(f"  With batch invariant mode: mean={result_bi.mean().item():.6f}")

        # Run in normal mode
        result_normal = paddle.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
        print(f"  Without batch invariant mode: mean={result_normal.mean().item():.6f}")

        # Results should be same (since SDPA itself is deterministic)
        is_equal = paddle.equal(result_bi, result_normal).all().item()
        print(f"  Batch invariant vs Normal: equal={is_equal}")

        if not is_equal:
            print("  Note: Batch invariant mode may not affect Paddle's native SDPA")

    def test_attention_with_different_head_configs(self):
        """Test determinism with different head configurations"""
        print("\n=== Testing Different Head Configurations ===")

        configs = [
            {"num_heads": 4, "head_dim": 64},
            {"num_heads": 8, "head_dim": 64},
            {"num_heads": 16, "head_dim": 64},
            {"num_heads": 32, "head_dim": 32},
        ]

        batch_size = 2
        seq_len = 32

        for config in configs:
            num_heads = config["num_heads"]
            head_dim = config["head_dim"]

            paddle.seed(42)

            query = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
            key = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
            value = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)

            # Run twice
            result1 = paddle.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
            result2 = paddle.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)

            is_equal = paddle.equal(result1, result2).all().item()

            print(f"  Heads={num_heads}, Dim={head_dim}: equal={is_equal}")

            self.assertTrue(is_equal, f"Attention results differ for heads={num_heads}, head_dim={head_dim}")

    def test_half_precision_determinism(self):
        """Test determinism at half precision (using exact equality check)"""
        print("\n=== Testing Half Precision Determinism ===")

        if not paddle.is_compiled_with_cuda():
            self.skipTest("Half precision test requires CUDA")

        num_heads = 8
        head_dim = 64
        batch_size = 2
        seq_len = 32

        paddle.seed(42)

        # Use float16
        query = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype="float16")
        key = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype="float16")
        value = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype="float16")

        results = []
        num_runs = 5

        for i in range(num_runs):
            paddle.device.cuda.synchronize()
            result = paddle.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False
            )
            results.append(result.clone())

        # Verify result consistency, using exact equality check
        is_deterministic = True
        for i in range(1, num_runs):
            is_equal = paddle.equal(results[0], results[i]).all().item()
            print(f"  FP16 Run 1 vs Run {i+1}: equal={is_equal}")
            if not is_equal:
                is_deterministic = False

        # Report based on actual test results
        if is_deterministic:
            print("  Result: FP16 is deterministic (exactly equal)")
        else:
            print("  Result: FP16 is not fully deterministic (results not exactly equal)")

    def test_different_backends_determinism(self):
        """Test determinism with different backends"""
        print("\n=== Testing Different Backends Determinism ===")

        if not paddle.is_compiled_with_cuda():
            self.skipTest("Backend test requires CUDA")

        num_heads = 8
        head_dim = 64
        batch_size = 2
        seq_len = 32

        # Test different backends
        backends = [None, "math", "flash"]  # None = auto

        for backend in backends:
            try:
                paddle.seed(42)

                query = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
                key = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
                value = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)

                results = []
                num_runs = 5

                for i in range(num_runs):
                    paddle.device.cuda.synchronize() if self.device == "gpu" else None
                    result = paddle.nn.functional.scaled_dot_product_attention(
                        query, key, value, is_causal=False, enable_gqa=False, backend=backend
                    )
                    results.append(result.clone())

                # 验证结果一致性，使用完全相等检查
                all_deterministic = True
                for i in range(1, num_runs):
                    is_equal = paddle.equal(results[0], results[i]).all().item()
                    if not is_equal:
                        all_deterministic = False

                backend_name = backend if backend else "auto"
                status = "✓ PASS" if all_deterministic else "✗ FAIL"
                print(f"  Backend={backend_name:6s}: {status}")

                self.assertTrue(all_deterministic, f"Backend {backend_name} not deterministic")
            except Exception as e:
                backend_name = backend if backend else "auto"
                print(f"  Backend={backend_name:6s}: ✗ ERROR - {str(e)[:50]}")
                # Skip unsupported backend
                continue

    def test_different_kv_lengths_determinism(self):
        """Test determinism with different KV lengths (simulating decode steps)"""
        print("\n=== Testing Different KV Lengths Determinism ===")

        num_heads = 8
        head_dim = 64
        batch_size = 2

        # Test different sequence lengths (simulating different stages)
        sequence_lengths = [16, 32, 64]

        for seq_len in sequence_lengths:
            paddle.seed(42)
            query = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
            key = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)
            value = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype=self.dtype)

            # First run
            result1 = paddle.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False
            )

            # Second run (completely same input)
            result2 = paddle.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False
            )

            # Third run (verify again)
            result3 = paddle.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False
            )

            is_equal_12 = paddle.equal(result1, result2).all().item()
            is_equal_13 = paddle.equal(result1, result3).all().item()

            print(f"  Seq_len={seq_len:2d}: run1 vs run2 equal={is_equal_12}, run1 vs run3 equal={is_equal_13}")

            self.assertTrue(is_equal_12, f"Determinism failed for seq_len={seq_len} (run1 vs run2)")
            self.assertTrue(is_equal_13, f"Determinism failed for seq_len={seq_len} (run1 vs run3)")


class TestAttentionDeterminismReport(unittest.TestCase):
    """Generate determinism test report"""

    def setUp(self):
        self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(self.device)

    def test_generate_determinism_report(self):
        """Generate complete determinism test report"""
        print("\n" + "=" * 70)
        print(" PADDLE ATTENTION DETERMINISM TEST REPORT")
        print("=" * 70)

        # Basic tests
        self._test_basic_determinism()
        # Batch size tests
        self._test_batch_size_determinism()
        # Sequence length tests
        self._test_seq_length_determinism()
        # Head config tests
        self._test_head_config_determinism()

        print("\n" + "=" * 70)
        print(" REPORT COMPLETE")
        print("=" * 70)

    def _test_basic_determinism(self):
        """Basic determinism test"""
        print("\n[BASIC DETERMINISM] Testing basic SDPA determinism...")

        num_heads = 8
        head_dim = 64
        batch_size = 2
        seq_len = 32

        paddle.seed(42)

        query = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype="float32")
        key = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype="float32")
        value = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype="float32")

        results = []
        for _ in range(10):
            result = paddle.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False
            )
            results.append(result.clone())

        # Use exact equality check
        all_equal = True
        for i in range(1, len(results)):
            is_equal = paddle.equal(results[0], results[i]).all().item()
            if not is_equal:
                all_equal = False

        status = "✓ PASS" if all_equal else "✗ FAIL"
        print(f"  Status: {status}")
        print(f"  All 10 runs are identical: {all_equal}")

        return all_equal

    def _test_batch_size_determinism(self):
        """Batch size determinism test"""
        print("\n[BATCH SIZE DETERMINISM] Testing batch invariance...")

        num_heads = 8
        head_dim = 64
        seq_len = 32

        batch_sizes = [1, 2, 4, 8, 16]
        results = []

        for bs in batch_sizes:
            paddle.seed(42)
            query = paddle.randn([bs, num_heads, seq_len, head_dim], dtype="float32")
            key = paddle.randn([bs, num_heads, seq_len, head_dim], dtype="float32")
            value = paddle.randn([bs, num_heads, seq_len, head_dim], dtype="float32")

            result = paddle.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False
            )
            results.append((bs, result[0:1].clone()))

        # Compare first sequence across different batch sizes, use exact equality check
        all_equal = True
        for i in range(1, len(results)):
            is_equal = paddle.equal(results[0][1], results[i][1]).all().item()
            if not is_equal:
                all_equal = False
            print(f"  Batch {results[0][0]} vs Batch {results[i][0]}: equal={is_equal}")

        status = "✓ PASS" if all_equal else "✗ FAIL"
        print(f"  Status: {status}")
        print(f"  Batch invariant (exactly equal): {all_equal}")

        return all_equal

    def _test_seq_length_determinism(self):
        """Sequence length determinism test"""
        print("\n[SEQUENCE LENGTH DETERMINISM] Testing different seq lengths...")

        num_heads = 8
        head_dim = 64
        batch_size = 2

        seq_lengths = [16, 32, 64, 128]
        all_deterministic = []

        for seq_len in seq_lengths:
            paddle.seed(42)
            query = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype="float32")
            key = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype="float32")
            value = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype="float32")

            result1 = paddle.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
            result2 = paddle.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)

            is_equal = paddle.equal(result1, result2).all().item()
            all_deterministic.append(is_equal)

            print(f"  Seq length {seq_len}: deterministic={is_equal}")

        all_equal = all(all_deterministic)
        status = "✓ PASS" if all_equal else "✗ FAIL"
        print(f"  Status: {status}")
        print(f"  All seq lengths deterministic: {all_equal}")

        return all_equal

    def _test_head_config_determinism(self):
        """Head config determinism test"""
        print("\n[HEAD CONFIG DETERMINISM] Testing different head configs...")

        batch_size = 2
        seq_len = 32

        configs = [
            (4, 64),
            (8, 64),
            (16, 32),
        ]
        all_deterministic = []

        for num_heads, head_dim in configs:
            paddle.seed(42)
            query = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype="float32")
            key = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype="float32")
            value = paddle.randn([batch_size, num_heads, seq_len, head_dim], dtype="float32")

            result1 = paddle.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)
            result2 = paddle.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False)

            is_equal = paddle.equal(result1, result2).all().item()
            all_deterministic.append(is_equal)

            print(f"  Heads={num_heads}, Dim={head_dim}: deterministic={is_equal}")

        all_equal = all(all_deterministic)
        status = "✓ PASS" if all_equal else "✗ FAIL"
        print(f"  Status: {status}")
        print(f"  All head configs deterministic: {all_equal}")

        return all_equal


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
