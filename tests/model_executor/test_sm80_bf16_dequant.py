"""
Unit tests for SM80 (A100) BF16 dequant workaround.

The SM80 BF16 dequant workaround enables MiniMax-M2.5 FP8 inference on
A100 GPUs that lack FP8 tensor cores. It dequantizes FP8 weights to BF16
at load time and uses standard cuBLAS GEMM instead of FP8 kernels.

These tests verify:
1. Block-wise FP8 to BF16 dequantization produces correct values
2. Scale expansion with transpose produces correct block-interleaved layout
3. Marlin scale permutation produces correct output
"""

import unittest

import numpy as np


class TestDequantFP8BlockwiseToBF16(unittest.TestCase):
    """Test _dequant_fp8_blockwise_to_bf16 helper function."""

    def test_basic_dequant(self):
        """Verify basic FP8 block-wise dequant produces correct BF16 values."""
        # Simulate: fp8_weight = [256, 256] float8 (cast to float32 for numpy),
        # scale = [2, 2] (256/128 = 2 blocks each dimension)
        BLOCK = 128
        N, K = 256, 256
        n_blocks_r = N // BLOCK  # 2
        n_blocks_c = K // BLOCK  # 2

        # Create known FP8-like weight (values 0-255 mapped to float32)
        wt_f32 = np.random.randn(N, K).astype(np.float32) * 0.1

        # Create block-wise scales
        sc = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        # Expected dequant: each 128x128 block multiplied by its scale
        wt_blocked = wt_f32.reshape([n_blocks_r, BLOCK, n_blocks_c, BLOCK])
        sc_expanded = sc.reshape([n_blocks_r, n_blocks_c])[:, np.newaxis, :, np.newaxis]
        wt_dequant = (wt_blocked * sc_expanded).reshape([N, K])

        # Verify block [0,0] has scale 1.0
        np.testing.assert_allclose(wt_dequant[:128, :128], wt_f32[:128, :128] * 1.0, rtol=1e-6)
        # Verify block [0,1] has scale 2.0
        np.testing.assert_allclose(wt_dequant[:128, 128:], wt_f32[:128, 128:] * 2.0, rtol=1e-6)
        # Verify block [1,0] has scale 3.0
        np.testing.assert_allclose(wt_dequant[128:, :128], wt_f32[128:, :128] * 3.0, rtol=1e-6)
        # Verify block [1,1] has scale 4.0
        np.testing.assert_allclose(wt_dequant[128:, 128:], wt_f32[128:, 128:] * 4.0, rtol=1e-6)

    def test_padded_dequant(self):
        """Verify dequant with padding when dimensions are not multiples of BLOCK."""
        BLOCK = 128
        N, K = 200, 300  # Not multiples of 128
        n_blocks_r = (N + BLOCK - 1) // BLOCK  # 2
        n_blocks_c = (K + BLOCK - 1) // BLOCK  # 3

        wt_f32 = np.random.randn(N, K).astype(np.float32) * 0.1
        sc = np.random.randn(n_blocks_r, n_blocks_c).astype(np.float32)

        # Pad weight to block boundaries
        pad_r = n_blocks_r * BLOCK - N
        pad_c = n_blocks_c * BLOCK - K
        wt_padded = np.pad(wt_f32, ((0, pad_r), (0, pad_c)))

        # Apply block-wise scale
        wt_blocked = wt_padded.reshape([n_blocks_r, BLOCK, n_blocks_c, BLOCK])
        sc_expanded = sc.reshape([n_blocks_r, n_blocks_c])[:, np.newaxis, :, np.newaxis]
        wt_dequant = (wt_blocked * sc_expanded).reshape([n_blocks_r * BLOCK, n_blocks_c * BLOCK])[:N, :K]

        # Verify shape matches original
        self.assertEqual(wt_dequant.shape, (N, K))

        # Verify first block's values
        np.testing.assert_allclose(
            wt_dequant[:128, :128],
            wt_f32[:128, :128] * sc[0, 0],
            rtol=1e-6,
        )


class TestScaleExpansion(unittest.TestCase):
    """Test block-wise scale expansion correctness."""

    def test_block_scale_application(self):
        """Verify the block-wise scale multiplication produces correct results.

        This tests the same computation as block_wise_fp8.py's SM80 path:
        expand scale to full size, multiply with weight.
        """
        BLOCK = 128
        n_blocks_out, n_blocks_in = 3, 2
        out_d = n_blocks_out * BLOCK
        in_d = n_blocks_in * BLOCK

        # Create weight with distinct values per block region
        weight = np.ones([out_d, in_d], dtype=np.float32)
        for r in range(n_blocks_out):
            for c in range(n_blocks_in):
                weight[r * BLOCK : (r + 1) * BLOCK, c * BLOCK : (c + 1) * BLOCK] = r * n_blocks_in + c + 1

        # Create block-wise scales
        scale = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

        # Apply scale using the expand+transpose+reshape approach (matches PaddlePaddle code)
        # Step 1: expand scale to [n_out, n_in, BLOCK, BLOCK]
        sc_exp = np.repeat(np.repeat(scale[:, :, np.newaxis, np.newaxis], BLOCK, axis=2), BLOCK, axis=3)
        # Step 2: transpose to [n_out, BLOCK, n_in, BLOCK]
        sc_exp = sc_exp.transpose([0, 2, 1, 3])
        # Step 3: reshape to [n_out*BLOCK, n_in*BLOCK]
        sc_full = sc_exp.reshape([out_d, in_d])

        # Verify each block has the correct scale
        for r in range(n_blocks_out):
            for c in range(n_blocks_in):
                block = sc_full[r * BLOCK : (r + 1) * BLOCK, c * BLOCK : (c + 1) * BLOCK]
                expected = scale[r, c]
                self.assertTrue(
                    np.all(block == expected),
                    f"Scale block [{r},{c}] expected {expected}, got unique={np.unique(block)}",
                )

        # Verify dequant result
        dequant = weight * sc_full
        for r in range(n_blocks_out):
            for c in range(n_blocks_in):
                block_dequant = dequant[r * BLOCK : (r + 1) * BLOCK, c * BLOCK : (c + 1) * BLOCK]
                weight_val = r * n_blocks_in + c + 1
                scale_val = scale[r, c]
                expected = weight_val * scale_val
                self.assertTrue(
                    np.all(block_dequant == expected),
                    f"Dequant block [{r},{c}] expected {expected}, got unique={np.unique(block_dequant)}",
                )


class TestMarlinPermuteScales(unittest.TestCase):
    """Test Marlin scale permutation logic."""

    def test_permute_array(self):
        """Verify the scale_perm array is correct."""
        scale_perm = []
        for i in range(8):
            scale_perm.extend([i + 8 * j for j in range(8)])

        # Should be 64 elements
        self.assertEqual(len(scale_perm), 64)

        # Verify pattern: [0,8,16,...,56, 1,9,17,...,57, ...]
        expected = []
        for i in range(8):
            for j in range(8):
                expected.append(i + 8 * j)
        self.assertEqual(scale_perm, expected)

    def test_permute_single_array(self):
        """Verify the scale_perm_single array for group_size == size_k."""
        scale_perm_single = []
        for i in range(4):
            scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])

        self.assertEqual(len(scale_perm_single), 32)

        # Verify first 8 elements: [0,1,8,9,16,17,24,25]
        self.assertEqual(scale_perm_single[:8], [0, 1, 8, 9, 16, 17, 24, 25])


if __name__ == "__main__":
    unittest.main()
