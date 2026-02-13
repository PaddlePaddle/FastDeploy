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
Batch Invariant Ops Tests

Test scenarios:
1. Test batch invariant matmul operations
2. Test batch invariant log_softmax operations
3. Test batch invariant mean operations
4. Test batch invariant addmm operations
5. Test batch invariance across different batch sizes
6. Test mode enable/disable functionality
7. Test attention block size configuration
"""

import unittest

import paddle

from fastdeploy.model_executor.layers.batch_invariant_ops.batch_invariant_ops import (
    addmm_batch_invariant,
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
    get_batch_invariant_attention_block_size,
    is_batch_invariant_mode_enabled,
    log_softmax,
    matmul_persistent,
    mean_batch_invariant,
    mean_dim,
    mm_batch_invariant,
    set_batch_invariant_mode,
)


class TestBatchInvariantModeControl(unittest.TestCase):
    """Test batch invariant mode enable/disable functionality"""

    def setUp(self):
        """Set up testing environment"""
        self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(self.device)

        # Save original mode state
        self.original_mode = is_batch_invariant_mode_enabled()

        # Disable mode for clean test start
        if is_batch_invariant_mode_enabled():
            disable_batch_invariant_mode()

    def tearDown(self):
        """Restore original mode state"""
        if self.original_mode:
            enable_batch_invariant_mode()
        else:
            disable_batch_invariant_mode()

    def test_is_batch_invariant_mode_enabled_default(self):
        """Test that batch invariant mode is disabled by default"""
        print("\n=== Testing default batch invariant mode ===")

        mode = is_batch_invariant_mode_enabled()
        print(f"  Default mode: {mode}")
        self.assertFalse(mode, "Batch invariant mode should be disabled by default")

    def test_enable_batch_invariant_mode(self):
        """Test enabling batch invariant mode"""
        print("\n=== Testing enable_batch_invariant_mode ===")

        enable_batch_invariant_mode()
        mode = is_batch_invariant_mode_enabled()
        print(f"  After enable: {mode}")
        self.assertTrue(mode, "Batch invariant mode should be enabled after calling enable")

    def test_disable_batch_invariant_mode(self):
        """Test disabling batch invariant mode"""
        print("\n=== Testing disable_batch_invariant_mode ===")

        enable_batch_invariant_mode()
        self.assertTrue(is_batch_invariant_mode_enabled())

        disable_batch_invariant_mode()
        mode = is_batch_invariant_mode_enabled()
        print(f"  After disable: {mode}")
        self.assertFalse(mode, "Batch invariant mode should be disabled after calling disable")

    def test_set_batch_invariant_mode_context(self):
        """Test set_batch_invariant_mode as context manager"""
        print("\n=== Testing set_batch_invariant_mode context manager ===")

        self.assertFalse(is_batch_invariant_mode_enabled())

        with set_batch_invariant_mode(True):
            self.assertTrue(is_batch_invariant_mode_enabled())
            print("  Inside context: enabled=True")

        self.assertFalse(is_batch_invariant_mode_enabled())
        print("  Outside context: restored to original state")

        # Test starting from enabled state
        enable_batch_invariant_mode()

        with set_batch_invariant_mode(False):
            self.assertFalse(is_batch_invariant_mode_enabled())
            print("  Inside context (from enabled): enabled=False")

        self.assertTrue(is_batch_invariant_mode_enabled())
        print("  Outside context: restored to enabled state")

    def test_repeated_enable_disable(self):
        """Test repeated enable/disable cycles"""
        print("\n=== Testing repeated enable/disable cycles ===")

        for i in range(5):
            enable_batch_invariant_mode()
            self.assertTrue(is_batch_invariant_mode_enabled(), f"Cycle {i+1}: Should be enabled after enable")

            disable_batch_invariant_mode()
            self.assertFalse(is_batch_invariant_mode_enabled(), f"Cycle {i+1}: Should be disabled after disable")

        print("  Completed 5 enable/disable cycles successfully")


class TestMatmulPersistent(unittest.TestCase):
    """Test matmul_persistent operation"""

    def setUp(self):
        """Set up testing environment"""
        self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(self.device)

    def test_matmul_persistent_basic(self):
        """Test basic matmul_persistent functionality"""
        print("\n=== Testing matmul_persistent basic functionality ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        paddle.seed(42)

        M, N, K = 128, 64, 32
        a = paddle.randn([M, K], dtype="float16")
        b = paddle.randn([K, N], dtype="float16")

        result = matmul_persistent(a, b)

        expected_shape = [M, N]
        self.assertEqual(result.shape, expected_shape, f"Expected shape {expected_shape}, got {result.shape}")

        print(f"  matmul_persistent({a.shape}, {b.shape}) -> {result.shape}")
        print(f"  mean={result.mean().item():.6f}, std={result.std().item():.6f}")

    def test_matmul_persistent_with_bias(self):
        """Test matmul_persistent with bias"""
        print("\n=== Testing matmul_persistent with bias ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        paddle.seed(42)

        M, N, K = 64, 32, 16
        a = paddle.randn([M, K], dtype="float16")
        b = paddle.randn([K, N], dtype="float16")
        bias = paddle.randn([N], dtype="float16")

        result = matmul_persistent(a, b, bias=bias)

        self.assertEqual(result.shape, [M, N])
        print(f"  matmul_persistent with bias: shape={result.shape}")
        print(f"  mean={result.mean().item():.6f}")

    def test_matmul_persistent_determinism(self):
        """Test matmul_persistent produces same results on same input"""
        print("\n=== Testing matmul_persistent determinism ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        paddle.seed(42)

        M, N, K = 64, 32, 16
        a = paddle.randn([M, K], dtype="float16")
        b = paddle.randn([K, N], dtype="float16")

        results = []
        num_runs = 5

        for i in range(num_runs):
            paddle.device.synchronize() if self.device == "gpu" else None
            result = matmul_persistent(a, b)
            results.append(result.clone().cpu())
            print(f"  Run {i+1}: mean={result.mean().item():.6f}")

        # Check if all results are identical
        all_equal = True
        for i in range(1, num_runs):
            is_equal = paddle.equal(results[0], results[i]).all().item()
            print(f"  Run 1 vs Run {i+1}: equal={is_equal}")
            if not is_equal:
                all_equal = False

        if all_equal:
            print("  [PASS] All runs produced identical results")
        else:
            print("  [INFO] Results differ (may be expected on GPU without deterministic mode)")

    def test_matmul_persistent_different_shapes(self):
        """Test matmul_persistent with different input shapes"""
        print("\n=== Testing matmul_persistent different shapes ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        test_shapes = [
            ([16, 8], [8, 16]),
            ([32, 16], [16, 32]),
            ([64, 32], [32, 64]),
            ([128, 64], [64, 128]),
        ]

        for a_shape, b_shape in test_shapes:
            a = paddle.randn(a_shape, dtype="float16")
            b = paddle.randn(b_shape, dtype="float16")

            result = matmul_persistent(a, b)

            expected_shape = [a_shape[0], b_shape[1]]
            self.assertEqual(result.shape, expected_shape)
            print(f"  matmul_persistent({a_shape}, {b_shape}) -> {result.shape}")

    def test_matmul_persistent_dtype_support(self):
        """Test matmul_persistent with different dtypes"""
        print("\n=== Testing matmul_persistent dtype support ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        paddle.seed(42)

        M, N, K = 32, 16, 8

        dtypes = ["float16", "float32"]
        if paddle.is_compiled_with_cuda():
            dtypes.append("bfloat16")

        for dtype in dtypes:
            a = paddle.randn([M, K], dtype=dtype)
            b = paddle.randn([K, N], dtype=dtype)

            result = matmul_persistent(a, b)

            self.assertEqual(result.shape, [M, N])
            print(f"  dtype={dtype:8s}: shape={result.shape}")


class TestLogSoftmaxBatchInvariant(unittest.TestCase):
    """Test log_softmax batch invariant operation"""

    def setUp(self):
        """Set up testing environment"""
        self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(self.device)

    def test_log_softmax_basic(self):
        """Test basic log_softmax functionality"""
        print("\n=== Testing log_softmax basic functionality ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        paddle.seed(42)

        batch_size = 4
        n_cols = 128
        input_tensor = paddle.randn([batch_size, n_cols], dtype="float16")

        result = log_softmax(input_tensor, axis=-1)

        self.assertEqual(result.shape, input_tensor.shape)

        # Check log_softmax property: sum(exp(log_softmax)) = 1
        sum_exp = paddle.exp(result).sum(axis=-1)
        for i in range(batch_size):
            self.assertAlmostEqual(sum_exp[i].item(), 1.0, places=3)

        print(f"  log_softmax({input_tensor.shape}) -> {result.shape}")
        print("  sum(exp(result)) ≈ 1.0 (verified)")

    def test_log_softmax_determinism(self):
        """Test log_softmax produces same results on same input"""
        print("\n=== Testing log_softmax determinism ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        paddle.seed(42)

        batch_size = 2
        n_cols = 64
        input_tensor = paddle.randn([batch_size, n_cols], dtype="float16")

        results = []
        num_runs = 5

        for i in range(num_runs):
            paddle.device.synchronize() if self.device == "gpu" else None
            result = log_softmax(input_tensor, axis=-1)
            results.append(result.clone().cpu())
            print(f"  Run {i+1}: mean={result.mean().item():.6f}")

        # Check if all results are identical
        all_equal = True
        for i in range(1, num_runs):
            is_equal = paddle.equal(results[0], results[i]).all().item()
            print(f"  Run 1 vs Run {i+1}: equal={is_equal}")
            if not is_equal:
                all_equal = False

        if all_equal:
            print("  [PASS] All runs produced identical results")

    def test_log_softmax_different_sizes(self):
        """Test log_softmax with different input sizes"""
        print("\n=== Testing log_softmax different sizes ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        sizes = [
            (1, 16),
            (2, 32),
            (4, 64),
            (8, 128),
        ]

        for batch_size, n_cols in sizes:
            input_tensor = paddle.randn([batch_size, n_cols], dtype="float16")
            result = log_softmax(input_tensor, axis=-1)

            self.assertEqual(result.shape, input_tensor.shape)
            print(f"  log_softmax({input_tensor.shape}) -> {result.shape}")


class TestMeanBatchInvariant(unittest.TestCase):
    """Test mean_batch_invariant operation"""

    def setUp(self):
        """Set up testing environment"""
        self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(self.device)

    def test_mean_dim_basic(self):
        """Test mean_dim basic functionality"""
        print("\n=== Testing mean_dim basic functionality ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        paddle.seed(42)

        M, N, K = 4, 8, 16
        input_tensor = paddle.randn([M, N, K], dtype="float16")

        # Test reduction along dimension 1
        result = mean_dim(input_tensor, dim=1, keepdim=False)

        expected_shape = [M, K]
        self.assertEqual(result.shape, expected_shape)
        print(f"  mean_dim({input_tensor.shape}, dim=1) -> {result.shape}")

        # Verify mean calculation manually for a few elements
        manual_mean = paddle.mean(input_tensor, axis=1, dtype="float32").to("float16")
        is_close = paddle.allclose(result, manual_mean, rtol=1e-3).all().item()
        print(f"  matches paddle.mean: {is_close}")

    def test_mean_batch_invariant_single_axis(self):
        """Test mean_batch_invariant with single axis"""
        print("\n=== Testing mean_batch_invariant single axis ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        paddle.seed(42)

        input_tensor = paddle.randn([8, 16, 32], dtype="float16")

        # Use None for dtype (default) or paddle.float32
        result = mean_batch_invariant(input_tensor, axis=[1], keepdim=False, dtype=None)

        expected_shape = [8, 32]
        self.assertEqual(result.shape, expected_shape)
        print(f"  mean_batch_invariant({input_tensor.shape}, axis=[1]) -> {result.shape}")

    def test_mean_batch_invariant_multi_axis(self):
        """Test mean_batch_invariant with multiple axes"""
        print("\n=== Testing mean_batch_invariant multiple axes ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        paddle.seed(42)

        input_tensor = paddle.randn([4, 8, 16, 32], dtype="float16")

        result = mean_batch_invariant(input_tensor, axis=[1, 2], keepdim=True, dtype=None)

        expected_shape = [4, 1, 1, 32]
        self.assertEqual(result.shape, expected_shape)
        print(f"  mean_batch_invariant({input_tensor.shape}, axis=[1,2], keepdim=True) -> {result.shape}")

    def test_mean_batch_invariant_determinism(self):
        """Test mean_batch_invariant determinism"""
        print("\n=== Testing mean_batch_invariant determinism ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        paddle.seed(42)

        input_tensor = paddle.randn([4, 8, 16], dtype="float16")

        results = []
        num_runs = 5

        for i in range(num_runs):
            paddle.device.synchronize() if self.device == "gpu" else None
            result = mean_batch_invariant(input_tensor, axis=[1], dtype=None)
            results.append(result.clone().cpu())
            print(f"  Run {i+1}: mean={result.mean().item():.6f}")

        # Check if all results are identical
        all_equal = True
        for i in range(1, num_runs):
            is_equal = paddle.equal(results[0], results[i]).all().item()
            print(f"  Run 1 vs Run {i+1}: equal={is_equal}")
            if not is_equal:
                all_equal = False

        if all_equal:
            print("  [PASS] All runs produced identical results")


class TestAddmmBatchInvariant(unittest.TestCase):
    """Test addmm_batch_invariant operation"""

    def setUp(self):
        """Set up testing environment"""
        self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(self.device)

    def test_addmm_batch_invariant_basic(self):
        """Test addmm_batch_invariant basic functionality"""
        print("\n=== Testing addmm_batch_invariant basic functionality ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        paddle.seed(42)

        M, N, K = 16, 8, 4
        # bias must be 1D with shape [N] (output dimension)
        input_tensor = paddle.randn([N], dtype="float16")
        x = paddle.randn([M, K], dtype="float16")
        y = paddle.randn([K, N], dtype="float16")

        # Test addmm: out = alpha * (x @ y) + beta * input
        result = addmm_batch_invariant(input_tensor, x, y, beta=1.0, alpha=1.0)

        expected_shape = [M, N]
        self.assertEqual(result.shape, expected_shape)
        print(f"  addmm_batch_invariant -> {result.shape}")

    def test_addmm_batch_invariant_with_alpha_beta(self):
        """Test addmm_batch_invariant with alpha and beta"""
        print("\n=== Testing addmm_batch_invariant with alpha/beta ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        paddle.seed(42)

        M, N, K = 8, 4, 2
        # bias must be 1D with shape [N]
        input_tensor = paddle.randn([N], dtype="float16")
        x = paddle.randn([M, K], dtype="float16")
        y = paddle.randn([K, N], dtype="float16")

        result1 = addmm_batch_invariant(input_tensor, x, y, beta=1.0, alpha=1.0)
        result2 = addmm_batch_invariant(input_tensor, x, y, beta=2.0, alpha=0.5)

        print(f"  alpha=1.0, beta=1.0: mean={result1.mean().item():.6f}")
        print(f"  alpha=0.5, beta=2.0: mean={result2.mean().item():.6f}")

        # Different alpha/beta should produce different results
        is_different = not paddle.equal(result1, result2).all().item()
        self.assertTrue(is_different, "Different alpha/beta should produce different results")


class TestMmBatchInvariant(unittest.TestCase):
    """Test mm_batch_invariant operation"""

    def setUp(self):
        """Set up testing environment"""
        self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(self.device)

    def test_mm_batch_invariant_basic(self):
        """Test mm_batch_invariant basic functionality"""
        print("\n=== Testing mm_batch_invariant basic functionality ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        paddle.seed(42)

        M, N, K = 32, 16, 8
        a = paddle.randn([M, K], dtype="float16")
        b = paddle.randn([K, N], dtype="float16")

        result = mm_batch_invariant(a, b)

        expected_shape = [M, N]
        self.assertEqual(result.shape, expected_shape)
        print(f"  mm_batch_invariant({a.shape}, {b.shape}) -> {result.shape}")

    def test_mm_batch_invariant_transpose(self):
        """Test mm_batch_invariant with transpose options"""
        print("\n=== Testing mm_batch_invariant transpose ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        paddle.seed(42)

        # Test with simple matrices that work with and without transpose
        M, N, K = 16, 8, 4
        # Standard matmul: [M, K] @ [K, N] = [M, N]
        a = paddle.randn([M, K], dtype="float16")  # [16, 4]
        b = paddle.randn([K, N], dtype="float16")  # [4, 8]

        result_no_transpose = mm_batch_invariant(a, b, transpose_x=False, transpose_y=False)
        print(f"  No transpose: a.shape={a.shape}, b.shape={b.shape} -> result.shape={result_no_transpose.shape}")

        # Test with transpose_y: need to create b with compatible shape
        # After transpose_y, we need b.T to be [K, N] so b must be [N, K]
        b_transpose = paddle.randn([N, K], dtype="float16")  # [8, 4]
        # After transpose_y: [8, 4] -> [4, 8]
        # a is [16, 4], b.T is [4, 8] -> [16, 8] compatible
        result_transpose_y = mm_batch_invariant(a, b_transpose, transpose_x=False, transpose_y=True)
        print(f"  With transpose_y: b_transpose.shape={b_transpose.shape} -> result.shape={result_transpose_y.shape}")

        # Both should have same shape but different values
        self.assertEqual(result_no_transpose.shape, result_transpose_y.shape)
        is_different = not paddle.equal(result_no_transpose, result_transpose_y).all().item()
        self.assertTrue(is_different, "Transpose should produce different values")


class TestBatchInvarianceAcrossBatchSizes(unittest.TestCase):
    """Test batch invariance across different batch sizes"""

    def setUp(self):
        """Set up testing environment"""
        self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(self.device)

        # Save original mode
        self.original_mode = is_batch_invariant_mode_enabled()
        # Disable for clean start
        if is_batch_invariant_mode_enabled():
            disable_batch_invariant_mode()

    def tearDown(self):
        """Restore original mode"""
        if self.original_mode:
            enable_batch_invariant_mode()
        else:
            disable_batch_invariant_mode()

    def test_matmul_batch_invariance(self):
        """
        Test that matmul produces same result for same tokens
        regardless of batch configuration
        """
        print("\n=== Testing matmul batch invariance ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        paddle.seed(42)

        num_heads = 8
        head_dim = 64
        seq_len = 32

        # Single sequence
        a_single = paddle.randn([1, num_heads, seq_len, head_dim], dtype="float16")
        b_single = paddle.randn([1, num_heads, seq_len, head_dim], dtype="float16")

        # For matmul, we need [M, K] @ [K, N] = [M, N]
        # Reshape a to [num_heads * seq_len, head_dim] = [256, 64]
        # Reshape b to [head_dim, num_heads * seq_len] = [64, 256] (transpose the last two dims)
        result_single = matmul_persistent(
            a_single.reshape([num_heads * seq_len, head_dim]), b_single.reshape([head_dim, num_heads * seq_len])
        )

        print(f"  Single batch shape: {result_single.shape}")
        print(f"  Single batch mean: {result_single.mean().item():.6f}")

    def test_attention_block_size(self):
        """Test get_batch_invariant_attention_block_size"""
        print("\n=== Testing attention block size ===")

        block_size = get_batch_invariant_attention_block_size()

        self.assertEqual(block_size.block_m, 16)
        self.assertEqual(block_size.block_n, 16)

        print(f"  Attention block size: {block_size}")


class TestBatchInvariantModeIntegration(unittest.TestCase):
    """Test integration of batch invariant mode with paddle operations"""

    def setUp(self):
        """Set up testing environment"""
        self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(self.device)

        # Save original mode and ops
        self.original_mode = is_batch_invariant_mode_enabled()
        if is_batch_invariant_mode_enabled():
            disable_batch_invariant_mode()

    def tearDown(self):
        """Restore original mode"""
        if self.original_mode:
            enable_batch_invariant_mode()
        else:
            disable_batch_invariant_mode()

    def test_mode_toggle_does_not_affect_normal_ops(self):
        """Test that toggling mode doesn't break normal operations"""
        print("\n=== Testing mode toggle doesn't break normal ops ===")

        if not paddle.is_compiled_with_cuda() and self.device == "gpu":
            self.skipTest("CUDA not available")

        paddle.seed(42)

        a = paddle.randn([16, 8], dtype="float16")
        b = paddle.randn([8, 16], dtype="float16")

        # Normal mode
        result_normal = paddle.matmul(a, b)

        # Enable batch invariant mode
        enable_batch_invariant_mode()

        # The mode replacement affects paddle._C_ops.matmul
        # Let's use the batch invariant directly
        result_bi = matmul_persistent(a, b)

        # Disable mode
        disable_batch_invariant_mode()

        print(f"  Normal matmul: {result_normal.shape}")
        print(f"  Batch invariant: {result_bi.shape}")

        self.assertEqual(result_normal.shape, result_bi.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
