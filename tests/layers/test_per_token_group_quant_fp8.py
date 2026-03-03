"""
Unit test for per_token_group_quant_fp8 triton kernel.
"""

import unittest

import paddle


class TestPerTokenGroupQuantFP8(unittest.TestCase):
    """Test cases for per_token_group_quant_fp8 function."""

    def setUp(self):
        """Set up test fixtures."""
        paddle.set_device("gpu")

    def test_basic_functionality(self):
        """Test basic quantization functionality."""
        from fastdeploy.model_executor.layers.quantization.fp8_utils import (
            per_token_group_quant_fp8,
        )

        # Create test input: [batch_size, seq_len] = [4, 128]
        batch_size = 4
        seq_len = 128
        group_size = 128

        # Create random input tensor
        x = paddle.randn([batch_size, seq_len], dtype=paddle.float32)
        x = paddle.to_tensor(x, place=paddle.CUDAPlace(0))

        # Run quantization
        x_q, x_s = per_token_group_quant_fp8(x, group_size=group_size)

        # Check output shapes
        expected_q_shape = [batch_size, seq_len]
        expected_s_shape = [batch_size, seq_len // group_size]

        self.assertEqual(
            list(x_q.shape), expected_q_shape, f"Expected x_q shape {expected_q_shape}, got {list(x_q.shape)}"
        )
        self.assertEqual(
            list(x_s.shape), expected_s_shape, f"Expected x_s shape {expected_s_shape}, got {list(x_s.shape)}"
        )

        # Check dtype
        self.assertEqual(x_q.dtype, paddle.float8_e4m3fn, f"Expected dtype float8_e4m3fn, got {x_q.dtype}")
        self.assertEqual(x_s.dtype, paddle.float32, f"Expected scale dtype float32, got {x_s.dtype}")

        print("✓ Basic functionality test passed")
        print(f"  Input shape: {list(x.shape)}")
        print(f"  Output x_q shape: {list(x_q.shape)}, dtype: {x_q.dtype}")
        print(f"  Output x_s shape: {list(x_s.shape)}, dtype: {x_s.dtype}")

    def test_different_group_sizes(self):
        """Test with different group sizes."""
        from fastdeploy.model_executor.layers.quantization.fp8_utils import (
            per_token_group_quant_fp8,
        )

        batch_size = 2
        seq_len = 256

        x = paddle.randn([batch_size, seq_len], dtype=paddle.float32)
        x = paddle.to_tensor(x, place=paddle.CUDAPlace(0))

        # Test different group sizes
        for group_size in [64, 128, 256]:
            x_q, x_s = per_token_group_quant_fp8(x, group_size=group_size)

            expected_s_shape = [batch_size, seq_len // group_size]
            self.assertEqual(
                list(x_s.shape),
                expected_s_shape,
                f"group_size={group_size}: Expected x_s shape {expected_s_shape}, got {list(x_s.shape)}",
            )
            print(f"✓ group_size={group_size} test passed, scale shape: {list(x_s.shape)}")

    def test_larger_batch(self):
        """Test with larger batch and sequence length."""
        from fastdeploy.model_executor.layers.quantization.fp8_utils import (
            per_token_group_quant_fp8,
        )

        batch_size = 16
        seq_len = 512
        group_size = 128

        x = paddle.randn([batch_size, seq_len], dtype=paddle.float32)
        x = paddle.to_tensor(x, place=paddle.CUDAPlace(0))

        x_q, x_s = per_token_group_quant_fp8(x, group_size=group_size)

        expected_q_shape = [batch_size, seq_len]
        expected_s_shape = [batch_size, seq_len // group_size]

        self.assertEqual(list(x_q.shape), expected_q_shape)
        self.assertEqual(list(x_s.shape), expected_s_shape)

        print("✓ Larger batch test passed")
        print(f"  Input: [{batch_size}, {seq_len}]")
        print(f"  Output: x_q {list(x_q.shape)}, x_s {list(x_s.shape)}")

    def test_use_ue8m0_flag(self):
        """Test with use_ue8m0 flag."""
        from fastdeploy.model_executor.layers.quantization.fp8_utils import (
            per_token_group_quant_fp8,
        )

        batch_size = 4
        seq_len = 128
        group_size = 128

        x = paddle.randn([batch_size, seq_len], dtype=paddle.float32)
        x = paddle.to_tensor(x, place=paddle.CUDAPlace(0))

        # Test with use_ue8m0=True
        x_q_ue8m0, x_s_ue8m0 = per_token_group_quant_fp8(x, group_size=group_size, use_ue8m0=True)

        # Test with use_ue8m0=False
        x_q_normal, x_s_normal = per_token_group_quant_fp8(x, group_size=group_size, use_ue8m0=False)

        # Both should produce valid outputs with correct shapes
        self.assertEqual(list(x_q_ue8m0.shape), list(x_q_normal.shape))
        self.assertEqual(list(x_s_ue8m0.shape), list(x_s_normal.shape))

        print("✓ use_ue8m0 flag test passed")
        print(f"  ue8m0=True: x_s sample = {x_s_ue8m0[0, 0].item():.6f}")
        print(f"  ue8m0=False: x_s sample = {x_s_normal[0, 0].item():.6f}")


def run_quick_test():
    """Run a quick smoke test without unittest framework."""
    print("=" * 60)
    print("Quick smoke test for per_token_group_quant_fp8 triton kernel")
    print("=" * 60)

    import sys

    sys.path.insert(0, ".")

    paddle.set_device("gpu")

    from fastdeploy.model_executor.layers.quantization.fp8_utils import (
        per_token_group_quant_fp8,
    )

    # Test case 1: Basic test
    print("\n[Test 1] Basic functionality...")
    x = paddle.randn([4, 128], dtype=paddle.float32)
    x = paddle.to_tensor(x, place=paddle.CUDAPlace(0))

    x_q, x_s = per_token_group_quant_fp8(x, group_size=128)

    print(f"  Input shape: {list(x.shape)}, dtype: {x.dtype}")
    print(f"  Output x_q shape: {list(x_q.shape)}, dtype: {x_q.dtype}")
    print(f"  Output x_s shape: {list(x_s.shape)}, dtype: {x_s.dtype}")
    print("  ✓ PASSED")

    # Test case 2: Different group size
    print("\n[Test 2] Different group sizes...")
    x = paddle.randn([2, 256], dtype=paddle.float32)
    x = paddle.to_tensor(x, place=paddle.CUDAPlace(0))

    for gs in [64, 128, 256]:
        x_q, x_s = per_token_group_quant_fp8(x, group_size=gs)
        print(f"  group_size={gs}: x_s shape = {list(x_s.shape)} ✓")
    print("  ✓ PASSED")

    # Test case 3: use_ue8m0 flag
    print("\n[Test 3] use_ue8m0 flag...")
    x = paddle.randn([4, 128], dtype=paddle.float32)
    x = paddle.to_tensor(x, place=paddle.CUDAPlace(0))

    x_q1, x_s1 = per_token_group_quant_fp8(x, group_size=128, use_ue8m0=True)
    x_q2, x_s2 = per_token_group_quant_fp8(x, group_size=128, use_ue8m0=False)
    print(f"  use_ue8m0=True:  scale sample = {x_s1[0, 0].item():.6f}")
    print(f"  use_ue8m0=False: scale sample = {x_s2[0, 0].item():.6f}")
    print("  ✓ PASSED")

    print("\n" + "=" * 60)
    print("All smoke tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_test()
    else:
        unittest.main(verbosity=2)
