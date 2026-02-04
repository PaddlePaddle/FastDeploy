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
from unittest.mock import patch

import paddle

from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.layers.normalization import QKRMSNorm


class DummyQuantConfig:
    quant_round_type = 1
    quant_max_bound = 127
    quant_min_bound = -128


class DummyModelConfig:
    rms_norm_eps = 1e-5


class DummyParallelConfig:
    expert_parallel_size = 1
    tensor_parallel_size = 1
    tensor_parallel_rank = 0
    tp_group = None


class DummyFDConfig:
    def __init__(self):
        self.quant_config = DummyQuantConfig()
        self.model_config = DummyModelConfig()
        self.parallel_config = DummyParallelConfig()


class TestQKRMSNorm(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.fd_config = DummyFDConfig()
        self.head_dim = 64
        self.q_size = 512  # 8 heads * 64 head_dim
        self.kv_size = 128  # 2 heads * 64 head_dim
        self.eps = 1e-5

    def create_qkrmsnorm_layer(self, dtype="float16"):
        """Helper method to create QKRMSNorm layer with given dtype."""
        return QKRMSNorm(
            fd_config=self.fd_config,
            head_dim=self.head_dim,
            q_size=self.q_size,
            kv_size=self.kv_size,
            eps=self.eps,
            prefix="test_qk_norm",
            dtype=dtype,
        )

    def create_test_forward_meta(self, step_use_cudagraph=False):
        """Helper method to create ForwardMeta with given cudagraph setting."""
        forward_meta = ForwardMeta(
            ids_remove_padding=paddle.to_tensor([1, 2, 3]), step_use_cudagraph=step_use_cudagraph
        )
        return forward_meta

    def create_test_qkv_tensor(self, batch_size=2, seq_len=10, dtype="float16"):
        """Helper method to create test qkv tensor."""
        total_size = self.q_size + self.kv_size + self.kv_size
        qkv_out = paddle.randn([batch_size, seq_len, total_size], dtype=dtype)
        return qkv_out

    def test_initialization(self):
        """Test that QKRMSNorm initializes correctly with different parameters."""
        # Test with float16 dtype
        layer = self.create_qkrmsnorm_layer(dtype="float16")
        self.assertEqual(layer.head_dim, self.head_dim)
        self.assertEqual(layer.q_size, self.q_size)
        self.assertEqual(layer.kv_size, self.kv_size)
        self.assertEqual(layer.eps, self.eps)
        self.assertIsNotNone(layer.q_norm)
        self.assertIsNotNone(layer.k_norm)

        # Test with float32 dtype
        layer_fp32 = self.create_qkrmsnorm_layer(dtype="float32")
        self.assertEqual(layer_fp32.head_dim, self.head_dim)

        # Test with bfloat16 dtype
        layer_bf16 = self.create_qkrmsnorm_layer(dtype="bfloat16")
        self.assertEqual(layer_bf16.head_dim, self.head_dim)

    def test_invalid_dtype_initialization(self):
        """Test that QKRMSNorm raises error with invalid dtype."""
        with self.assertRaises(AssertionError) as context:
            QKRMSNorm(
                fd_config=self.fd_config,
                head_dim=self.head_dim,
                q_size=self.q_size,
                kv_size=self.kv_size,
                eps=self.eps,
                prefix="test",
                dtype="int8",  # Invalid dtype
            )
        self.assertIn("Unsupported dtype: int8", str(context.exception))

    def test_forward_non_fused_path(self):
        """Test forward computation using non-fused path (split and reassemble)."""
        layer = self.create_qkrmsnorm_layer()
        qkv_out = self.create_test_qkv_tensor()
        forward_meta = self.create_test_forward_meta(step_use_cudagraph=False)

        # Mock the triton availability to ensure non-fused path
        with patch.object(layer, "qk_norm_fused", False):
            output = layer.forward(qkv_out, forward_meta)

        # Verify output shape is same as input
        self.assertEqual(output.shape, qkv_out.shape)
        self.assertEqual(output.dtype, qkv_out.dtype)

        # Verify output is different from input (normalization occurred)
        self.assertFalse(paddle.allclose(output, qkv_out))

    def test_forward_fused_path_cuda_cudagraph(self):
        """Test forward computation using fused path when CUDA and cudagraph are available."""
        layer = self.create_qkrmsnorm_layer()
        qkv_out = self.create_test_qkv_tensor()
        forward_meta = self.create_test_forward_meta(step_use_cudagraph=True)

        # Mock to simulate CUDA environment with triton available
        with patch.object(layer, "qk_norm_fused", True):
            # Mock the qk_rmsnorm_fused function
            with patch("fastdeploy.model_executor.layers.normalization.qk_rmsnorm_fused") as mock_fused:
                mock_fused.return_value = qkv_out  # Return the same tensor for simplicity

                output = layer.forward(qkv_out, forward_meta)

                # Verify fused function was called with correct parameters
                mock_fused.assert_called_once()
                call_args = mock_fused.call_args[0]
                self.assertEqual(call_args[0].shape, qkv_out.shape)  # qkv_out
                self.assertEqual(call_args[3], layer.eps)  # eps
                self.assertEqual(call_args[4], layer.q_size)  # q_size
                self.assertEqual(call_args[5], layer.kv_size)  # kv_size
                self.assertEqual(call_args[6], layer.head_dim)  # head_dim

        self.assertEqual(output.shape, qkv_out.shape)

    def test_forward_fused_path_cuda_no_cudagraph(self):
        """Test that fused path is not used when cudagraph is disabled."""
        layer = self.create_qkrmsnorm_layer()
        qkv_out = self.create_test_qkv_tensor()
        forward_meta = self.create_test_forward_meta(step_use_cudagraph=False)

        # Even if triton is available, should use non-fused path when cudagraph is False
        with patch.object(layer, "qk_norm_fused", True):
            output = layer.forward(qkv_out, forward_meta)

        # Should still work correctly using non-fused path
        self.assertEqual(output.shape, qkv_out.shape)
        self.assertEqual(output.dtype, qkv_out.dtype)

    def test_forward_different_batch_sizes(self):
        """Test forward computation with different batch sizes."""
        layer = self.create_qkrmsnorm_layer()

        # Test with batch_size = 1
        qkv_out_1 = self.create_test_qkv_tensor(batch_size=1, seq_len=5)
        forward_meta = self.create_test_forward_meta(step_use_cudagraph=False)
        output_1 = layer.forward(qkv_out_1, forward_meta)
        self.assertEqual(output_1.shape, qkv_out_1.shape)

        # Test with batch_size = 8
        qkv_out_8 = self.create_test_qkv_tensor(batch_size=8, seq_len=5)
        output_8 = layer.forward(qkv_out_8, forward_meta)
        self.assertEqual(output_8.shape, qkv_out_8.shape)

        # Test with batch_size = 16, seq_len = 20
        qkv_out_16 = self.create_test_qkv_tensor(batch_size=16, seq_len=20)
        output_16 = layer.forward(qkv_out_16, forward_meta)
        self.assertEqual(output_16.shape, qkv_out_16.shape)

    def test_forward_different_sequence_lengths(self):
        """Test forward computation with different sequence lengths."""
        layer = self.create_qkrmsnorm_layer()
        forward_meta = self.create_test_forward_meta(step_use_cudagraph=False)

        # Test with short sequence
        qkv_out_short = self.create_test_qkv_tensor(batch_size=2, seq_len=1)
        output_short = layer.forward(qkv_out_short, forward_meta)
        self.assertEqual(output_short.shape, qkv_out_short.shape)

        # Test with long sequence
        qkv_out_long = self.create_test_qkv_tensor(batch_size=2, seq_len=100)
        output_long = layer.forward(qkv_out_long, forward_meta)
        self.assertEqual(output_long.shape, qkv_out_long.shape)

    def test_forward_different_dtypes(self):
        """Test forward computation with different input dtypes."""
        forward_meta = self.create_test_forward_meta(step_use_cudagraph=False)

        # Test with float16
        layer_fp16 = self.create_qkrmsnorm_layer(dtype="float16")
        qkv_out_fp16 = self.create_test_qkv_tensor(dtype="float16")
        output_fp16 = layer_fp16.forward(qkv_out_fp16, forward_meta)
        self.assertEqual(output_fp16.dtype, paddle.float16)

        # Test with float32
        layer_fp32 = self.create_qkrmsnorm_layer(dtype="float32")
        qkv_out_fp32 = self.create_test_qkv_tensor(dtype="float32")
        output_fp32 = layer_fp32.forward(qkv_out_fp32, forward_meta)
        self.assertEqual(output_fp32.dtype, paddle.float32)

        # Test with bfloat16
        layer_bf16 = self.create_qkrmsnorm_layer(dtype="bfloat16")
        qkv_out_bf16 = self.create_test_qkv_tensor(dtype="bfloat16")
        output_bf16 = layer_bf16.forward(qkv_out_bf16, forward_meta)
        self.assertEqual(output_bf16.dtype, paddle.bfloat16)

    def test_forward_edge_cases(self):
        """Test forward computation with edge cases."""
        layer = self.create_qkrmsnorm_layer()
        forward_meta = self.create_test_forward_meta(step_use_cudagraph=False)

        # Test with very small values
        qkv_out_small = paddle.full([2, 5, self.q_size + 2 * self.kv_size], 1e-6, dtype="float16")
        output_small = layer.forward(qkv_out_small, forward_meta)
        self.assertEqual(output_small.shape, qkv_out_small.shape)

        # Test with very large values
        qkv_out_large = paddle.full([2, 5, self.q_size + 2 * self.kv_size], 1e6, dtype="float16")
        output_large = layer.forward(qkv_out_large, forward_meta)
        self.assertEqual(output_large.shape, qkv_out_large.shape)

        # Test with mixed positive and negative values
        qkv_out_mixed = paddle.randn([2, 5, self.q_size + 2 * self.kv_size], dtype="float16")
        # Ensure some negative values
        qkv_out_mixed = qkv_out_mixed - 0.5
        output_mixed = layer.forward(qkv_out_mixed, forward_meta)
        self.assertEqual(output_mixed.shape, qkv_out_mixed.shape)

    def test_q_k_v_split_correctness(self):
        """Test that Q, K, V splitting in non-fused path is correct."""
        layer = self.create_qkrmsnorm_layer()
        qkv_out = self.create_test_qkv_tensor()
        forward_meta = self.create_test_forward_meta(step_use_cudagraph=False)

        with patch.object(layer, "qk_norm_fused", False):
            output = layer.forward(qkv_out, forward_meta)

        # Manually split and verify the dimensions
        q, k, v = qkv_out.split([layer.q_size, layer.kv_size, layer.kv_size], axis=-1)

        self.assertEqual(q.shape[-1], layer.q_size)
        self.assertEqual(k.shape[-1], layer.kv_size)
        self.assertEqual(v.shape[-1], layer.kv_size)

        # Verify that q and k have been normalized by checking they're different from original
        q_original = qkv_out.split([layer.q_size, layer.kv_size, layer.kv_size], axis=-1)[0]
        self.assertFalse(
            paddle.allclose(q_original, output.split([layer.q_size, layer.kv_size, layer.kv_size], axis=-1)[0])
        )

    def test_load_state_dict(self):
        """Test loading state dictionary."""
        layer = self.create_qkrmsnorm_layer()

        # Create a mock state dict
        state_dict = {
            "test_qk_norm.q_norm.weight": paddle.ones([layer.head_dim], dtype="float16"),
            "test_qk_norm.k_norm.weight": paddle.ones([layer.head_dim], dtype="float16"),
        }

        # This should not raise any errors
        layer.load_state_dict(state_dict)

    def test_forward_with_none_forward_meta(self):
        """Test forward computation when forward_meta is None."""
        layer = self.create_qkrmsnorm_layer()
        qkv_out = self.create_test_qkv_tensor()

        # Should work without forward_meta (use non-fused path)
        with patch.object(layer, "qk_norm_fused", True):  # Even if triton available
            output = layer.forward(qkv_out, None)

        self.assertEqual(output.shape, qkv_out.shape)

    def test_forward_consistency_between_paths(self):
        """Test that both fused and non-fused paths produce consistent results (when applicable)."""
        # Note: This test verifies that both paths work without crashing
        # In practice, the results may differ due to different implementations
        layer = self.create_qkrmsnorm_layer()
        qkv_out = self.create_test_qkv_tensor()

        # Test non-fused path
        forward_meta_no_cuda = self.create_test_forward_meta(step_use_cudagraph=False)
        with patch.object(layer, "qk_norm_fused", False):
            output_non_fused = layer.forward(qkv_out, forward_meta_no_cuda)

        # Test fused path (mocked)
        forward_meta_cuda = self.create_test_forward_meta(step_use_cudagraph=True)
        with patch.object(layer, "qk_norm_fused", True):
            with patch("fastdeploy.model_executor.layers.normalization.qk_rmsnorm_fused") as mock_fused:
                # Make the mock return a tensor with the same shape but different values
                mock_output = qkv_out + 0.1  # Slightly different to simulate actual computation
                mock_fused.return_value = mock_output
                output_fused = layer.forward(qkv_out, forward_meta_cuda)

        # Both should produce valid outputs with correct shapes
        self.assertEqual(output_non_fused.shape, qkv_out.shape)
        self.assertEqual(output_fused.shape, qkv_out.shape)


if __name__ == "__main__":
    unittest.main()
