"""
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

import unittest

import paddle
import paddle.nn.functional as F

from fastdeploy.model_executor.layers.normalization import RMSNormGated
from fastdeploy.model_executor.ops.triton_ops.rmsnorm_gated_kernel import (
    MAX_ROWS_PER_BLOCK,
    calc_rows_per_block,
    rmsnorm_gated,
)


class MockFDConfig:
    """Mock FDConfig for testing."""

    class ModelConfig:
        rms_norm_eps = 1e-5

    model_config = ModelConfig()


class TestRMSNormGated(unittest.TestCase):
    """Test cases for RMSNormGated layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 128
        self.fd_config = MockFDConfig()
        self.batch_size = 4
        self.seq_len = 32
        paddle.seed(1024)

    def test_init(self):
        """Test layer initialization."""
        layer = RMSNormGated(
            fd_config=self.fd_config,
            hidden_size=self.hidden_size,
            eps=1e-5,
            prefix="test_layer",
            activation="swish",
        )
        self.assertIsNotNone(layer.weight)
        self.assertEqual(layer.weight.shape[0], self.hidden_size)
        self.assertEqual(layer.eps, 1e-5)
        self.assertEqual(layer.activation, "swish")

    def naive_rmsnorm_gated(self, x, gate, weight, eps=1e-5, activation="swish", bias=None):
        """naive RMSNormGated for comparison."""
        input_dtype = x.dtype
        x = x.cast("float32")
        weight = weight.cast("float32")
        variance = paddle.mean(x**2, axis=-1, keepdim=True)
        # Norm before gate
        x = x * paddle.rsqrt(variance + eps)
        if bias is not None:
            x = x + bias.cast("float32")
        x = weight * x
        if gate is not None:
            gate = gate.cast("float32")
            if activation == "swish":
                x = x * F.swish(gate)
            elif activation == "silu":
                x = x * F.silu(gate)
            elif activation == "sigmoid":
                x = x * F.sigmoid(gate)
        return x.cast(input_dtype)

    def test_forward_no_gate(self):
        """Test forward pass without gate."""
        layer = RMSNormGated(
            fd_config=self.fd_config,
            hidden_size=self.hidden_size,
            eps=1e-5,
            prefix="test_layer",
            activation="swish",
        )
        x = paddle.randn([self.batch_size * self.seq_len, self.hidden_size], dtype="float16")
        output_triton = layer(x)
        self.assertEqual(output_triton.shape, x.shape)
        self.assertEqual(output_triton.dtype, x.dtype)

        output_naive = self.naive_rmsnorm_gated(x, None, layer.weight, eps=1e-5, activation="swish")

        assert (output_triton - output_naive).abs().max() < 1e-3

    def test_forward_with_gate_swish(self):
        """Test forward pass with swish gate."""
        layer = RMSNormGated(
            fd_config=self.fd_config,
            hidden_size=self.hidden_size,
            eps=1e-5,
            prefix="test_layer",
            activation="swish",
        )

        x = paddle.randn([self.batch_size * self.seq_len, self.hidden_size], dtype="float16")
        z = paddle.randn([self.batch_size * self.seq_len, self.hidden_size], dtype="float16")

        output_triton = layer(x, z=z)

        self.assertEqual(output_triton.shape, x.shape)
        self.assertEqual(output_triton.dtype, x.dtype)

        output_naive = self.naive_rmsnorm_gated(x, z, layer.weight, 1e-5, "swish")

        assert (output_triton - output_naive).abs().max() < 1e-3

    def test_forward_with_gate_silu(self):
        """Test forward pass with silu gate."""
        layer = RMSNormGated(
            fd_config=self.fd_config,
            hidden_size=self.hidden_size,
            eps=1e-5,
            prefix="test_layer",
            activation="silu",
        )

        x = paddle.randn([self.batch_size * self.seq_len, self.hidden_size], dtype="float16")
        z = paddle.randn([self.batch_size * self.seq_len, self.hidden_size], dtype="float16")

        output_triton = layer(x, z=z)

        self.assertEqual(output_triton.shape, x.shape)
        self.assertEqual(output_triton.dtype, x.dtype)

        output_naive = self.naive_rmsnorm_gated(x, z, layer.weight, 1e-5, "silu")

        assert (output_triton - output_naive).abs().max() < 1e-3

    def test_forward_with_gate_sigmoid(self):
        """Test forward pass with sigmoid gate."""
        layer = RMSNormGated(
            fd_config=self.fd_config,
            hidden_size=self.hidden_size,
            prefix="test_layer",
            activation="sigmoid",
        )

        x = paddle.randn([self.batch_size * self.seq_len, self.hidden_size], dtype="float16")
        z = paddle.randn([self.batch_size * self.seq_len, self.hidden_size], dtype="float16")

        output_triton = layer(x, z=z)

        self.assertEqual(output_triton.shape, x.shape)
        self.assertEqual(output_triton.dtype, x.dtype)

        output_naive = self.naive_rmsnorm_gated(x, z, layer.weight, 1e-5, "sigmoid")

        assert (output_triton - output_naive).abs().max() < 1e-3

    def test_init_no_prefix(self):
        """Test initialization with empty prefix results in no weight (with_weight=False)."""
        layer = RMSNormGated(
            fd_config=self.fd_config,
            hidden_size=self.hidden_size,
            eps=1e-5,
            prefix="",
            activation="swish",
        )
        self.assertIsNone(layer.weight)
        self.assertFalse(layer.with_weight)

    def test_forward_3d_input(self):
        """Test forward with 3D input [B, T, H]; the layer should reshape internally."""
        layer = RMSNormGated(
            fd_config=self.fd_config,
            hidden_size=self.hidden_size,
            eps=1e-5,
            prefix="test_layer",
            activation="swish",
        )
        x = paddle.randn([self.batch_size, self.seq_len, self.hidden_size], dtype="float16")
        z = paddle.randn([self.batch_size, self.seq_len, self.hidden_size], dtype="float16")

        output = layer(x, z=z)

        # Output shape must match input shape exactly (no collapse)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.dtype, x.dtype)

        # Numerical correctness: flatten to 2D and compare with naive
        x_2d = x.reshape([-1, self.hidden_size])
        z_2d = z.reshape([-1, self.hidden_size])
        output_naive = self.naive_rmsnorm_gated(x_2d, z_2d, layer.weight, 1e-5, "swish")
        output_naive_3d = output_naive.reshape([self.batch_size, self.seq_len, self.hidden_size])

        assert (output - output_naive_3d).abs().max() < 1e-3

    def test_forward_single_token_decode(self):
        """Test forward with M=1 (decode phase); exercises rows_per_block=1 boundary."""
        layer = RMSNormGated(
            fd_config=self.fd_config,
            hidden_size=self.hidden_size,
            eps=1e-5,
            prefix="test_layer",
            activation="swish",
        )
        x = paddle.randn([1, self.hidden_size], dtype="float16")
        z = paddle.randn([1, self.hidden_size], dtype="float16")

        output = layer(x, z=z)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.dtype, x.dtype)

        output_naive = self.naive_rmsnorm_gated(x, z, layer.weight, 1e-5, "swish")
        assert (output - output_naive).abs().max() < 1e-3

    def test_forward_float32_input(self):
        """Test that float32 input dtype is preserved in the output."""
        layer = RMSNormGated(
            fd_config=self.fd_config,
            hidden_size=self.hidden_size,
            eps=1e-5,
            prefix="test_layer",
            activation="swish",
        )
        x = paddle.randn([self.batch_size * self.seq_len, self.hidden_size], dtype="float32")
        z = paddle.randn([self.batch_size * self.seq_len, self.hidden_size], dtype="float32")

        output = layer(x, z=z)

        self.assertEqual(output.dtype, paddle.float32)
        self.assertEqual(output.shape, x.shape)

        output_naive = self.naive_rmsnorm_gated(x, z, layer.weight, 1e-5, "swish")
        assert (output - output_naive).abs().max() < 1e-5

    def test_kernel_out_parameter(self):
        """Test that rmsnorm_gated writes into a pre-allocated output tensor."""
        hidden_size = self.hidden_size
        M = self.batch_size * self.seq_len
        x = paddle.randn([M, hidden_size], dtype="float16")
        z = paddle.randn([M, hidden_size], dtype="float16")
        weight = paddle.ones([hidden_size], dtype="float16")
        pre_alloc = paddle.empty([M, hidden_size], dtype="float16")

        out = rmsnorm_gated(x=x, weight=weight, bias=None, eps=1e-5, z=z, out=pre_alloc)

        # Must return the same tensor object that was passed in
        self.assertIs(out, pre_alloc)
        self.assertEqual(out.shape, x.shape)

        ref = rmsnorm_gated(x=x, weight=weight, bias=None, eps=1e-5, z=z)
        assert (out - ref).abs().max() < 1e-3

    def test_kernel_with_bias(self):
        """Test rmsnorm_gated kernel with HAS_BIAS=True path."""
        hidden_size = self.hidden_size
        M = self.batch_size * self.seq_len
        x = paddle.randn([M, hidden_size], dtype="float16")
        z = paddle.randn([M, hidden_size], dtype="float16")
        weight = paddle.ones([hidden_size], dtype="float16")
        bias = paddle.zeros([hidden_size], dtype="float16")

        out = rmsnorm_gated(x=x, weight=weight, bias=bias, eps=1e-5, z=z)

        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

        # Compare with naive: bias=0 so result should match no-bias case
        ref = rmsnorm_gated(x=x, weight=weight, bias=None, eps=1e-5, z=z)
        assert (out - ref).abs().max() < 1e-3

    def test_kernel_runtime_error_large_N(self):
        """Test that a feature dim >= 64KB raises RuntimeError."""
        # Construct x with N just above MAX_FUSED_SIZE for float16 (65536/2=32768)
        M = 2
        N = 32769  # > 65536 // 2 (element_size=2 for float16)
        x = paddle.randn([M, N], dtype="float16")
        weight = paddle.ones([N], dtype="float16")

        with self.assertRaises(RuntimeError):
            rmsnorm_gated(x=x, weight=weight, bias=None, eps=1e-5)

    def test_kernel_unknown_activation_defaults_to_swish(self):
        """Test that an unknown activation string falls back to swish (value 0)."""
        hidden_size = self.hidden_size
        M = self.batch_size * self.seq_len
        x = paddle.randn([M, hidden_size], dtype="float16")
        z = paddle.randn([M, hidden_size], dtype="float16")
        weight = paddle.ones([hidden_size], dtype="float16")

        # Unknown activation should map to 0 (swish) via .get(..., 0)
        out_unknown = rmsnorm_gated(x=x, weight=weight, bias=None, eps=1e-5, z=z, activation="unknown_act")
        out_swish = rmsnorm_gated(x=x, weight=weight, bias=None, eps=1e-5, z=z, activation="swish")

        assert (out_unknown - out_swish).abs().max() < 1e-6

    def test_forward_no_gate_no_triton(self):
        """Test forward pass without gate and without Triton kernel (fallback path)."""
        layer = RMSNormGated(
            fd_config=self.fd_config,
            hidden_size=self.hidden_size,
            eps=1e-5,
            prefix="test_layer",
            activation="swish",
        )
        # Force disable triton kernel to exercise the else branch
        layer.use_triton_kernel = False

        x = paddle.randn([self.batch_size * self.seq_len, self.hidden_size], dtype="float16")
        output = layer(x)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.dtype, x.dtype)

        output_naive = self.naive_rmsnorm_gated(x, None, layer.weight, 1e-5, "swish")
        assert (output - output_naive).abs().max() < 1e-3

    def test_forward_with_gate_no_triton_swish(self):
        """Test forward with swish gate when Triton is disabled (fallback path)."""
        layer = RMSNormGated(
            fd_config=self.fd_config,
            hidden_size=self.hidden_size,
            eps=1e-5,
            prefix="test_layer",
            activation="swish",
        )
        layer.use_triton_kernel = False

        x = paddle.randn([self.batch_size * self.seq_len, self.hidden_size], dtype="float16")
        z = paddle.randn([self.batch_size * self.seq_len, self.hidden_size], dtype="float16")

        output = layer(x, z=z)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.dtype, x.dtype)

        output_naive = self.naive_rmsnorm_gated(x, z, layer.weight, 1e-5, "swish")
        assert (output - output_naive).abs().max() < 1e-3

    def test_forward_with_gate_no_triton_sigmoid(self):
        """Test forward with sigmoid gate when Triton is disabled (fallback path)."""
        layer = RMSNormGated(
            fd_config=self.fd_config,
            hidden_size=self.hidden_size,
            eps=1e-5,
            prefix="test_layer",
            activation="sigmoid",
        )
        layer.use_triton_kernel = False

        x = paddle.randn([self.batch_size * self.seq_len, self.hidden_size], dtype="float16")
        z = paddle.randn([self.batch_size * self.seq_len, self.hidden_size], dtype="float16")

        output = layer(x, z=z)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.dtype, x.dtype)

        output_naive = self.naive_rmsnorm_gated(x, z, layer.weight, 1e-5, "sigmoid")
        assert (output - output_naive).abs().max() < 1e-3

    def test_invalid_activation_raises(self):
        """Test that an invalid activation string raises AssertionError on init."""
        with self.assertRaises(AssertionError):
            RMSNormGated(
                fd_config=self.fd_config,
                hidden_size=self.hidden_size,
                eps=1e-5,
                prefix="test_layer",
                activation="relu",
            )

    def test_kernel_no_gate_no_bias(self):
        """Test rmsnorm_gated kernel without gate (HAS_Z=False) and without bias."""
        hidden_size = self.hidden_size
        M = self.batch_size * self.seq_len
        x = paddle.randn([M, hidden_size], dtype="float16")
        weight = paddle.ones([hidden_size], dtype="float16")

        out = rmsnorm_gated(x=x, weight=weight, bias=None, eps=1e-5, z=None)

        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

        # Compare with naive RMSNorm (no gate)
        ref = self.naive_rmsnorm_gated(x, None, weight, eps=1e-5, activation="swish")
        assert (out - ref).abs().max() < 1e-3


class TestCalcRowsPerBlock(unittest.TestCase):
    """Unit tests for the calc_rows_per_block heuristic."""

    def test_large_N_gives_one_row(self):
        """Large hidden dim: each row already fills the threads, expect 1 row."""
        # BLOCK_N=1024, num_warps=4 => threads=128, ceil(128/1024)=1
        rows = calc_rows_per_block(M=512, BLOCK_N=1024, num_warps=4)
        self.assertEqual(rows, 1)

    def test_small_N_increases_rows(self):
        """Small hidden dim: threads outnumber columns, expect multiple rows."""
        # BLOCK_N=32, num_warps=4 => threads=128, ceil(128/32)=4
        rows = calc_rows_per_block(M=512, BLOCK_N=32, num_warps=4)
        self.assertEqual(rows, 4)

    def test_result_capped_at_max_rows_per_block(self):
        """Result must never exceed MAX_ROWS_PER_BLOCK regardless of params."""
        rows = calc_rows_per_block(M=512, BLOCK_N=1, num_warps=8)
        self.assertLessEqual(rows, MAX_ROWS_PER_BLOCK)

    def test_result_capped_at_M(self):
        """Result must never exceed the actual row count M."""
        rows = calc_rows_per_block(M=1, BLOCK_N=32, num_warps=4)
        self.assertEqual(rows, 1)

    def test_result_is_power_of_two(self):
        """Result must always be a power of two."""
        for BLOCK_N in [32, 64, 128, 256, 512]:
            for num_warps in [1, 2, 4, 8]:
                rows = calc_rows_per_block(M=64, BLOCK_N=BLOCK_N, num_warps=num_warps)
                self.assertTrue(
                    rows > 0 and (rows & (rows - 1)) == 0,
                    f"rows={rows} is not a power of 2 " f"(BLOCK_N={BLOCK_N}, num_warps={num_warps})",
                )

    def test_minimum_is_one(self):
        """Result must be at least 1."""
        rows = calc_rows_per_block(M=1, BLOCK_N=4096, num_warps=1)
        self.assertGreaterEqual(rows, 1)


if __name__ == "__main__":
    unittest.main()
