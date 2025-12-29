"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import os
import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import masked_per_token_quant


def masked_per_token_quant_ref(input_tensor, recv_expert_count, block_size):
    """
    Paddle API implementation of masked_per_token_quant

    Args:
        input_tensor: Input tensor with shape [num_local_expert, num_max_tokens_per_expert, hidden_size]
        recv_expert_count: Expert token count tensor with shape [num_local_expert]
        block_size: Quantization block size

    Returns:
        Tuple of (quantized_tensor, scale_tensor)
    """
    MAX_VALUE = 448.0
    epsilon = 1e-10

    # Get dimensions
    input_shape = input_tensor.shape
    num_local_expert = input_shape[0]
    num_max_tokens_per_expert = input_shape[1]
    hidden_size = input_shape[2]

    # CUDA kernel uses: hidden_size_scale = hidden_size / block_size (integer division)
    # This assumes hidden_size is divisible by block_size
    hidden_size_scale = hidden_size // block_size

    # Check environment variable for fine-grained range
    use_finegrained_range = False
    env_var = os.getenv("PER_TOKEN_QUANT_FP8_USE_FINEGRAINED_RANGE")
    if env_var:
        use_finegrained_range = bool(int(env_var))

    # Create mask for valid tokens based on recv_expert_count
    token_indices = paddle.arange(num_max_tokens_per_expert, dtype="int32").unsqueeze(
        0
    )  # [1, num_max_tokens_per_expert]
    expert_counts = recv_expert_count.unsqueeze(1)  # [num_local_expert, 1]
    valid_mask = token_indices < expert_counts  # [num_local_expert, num_max_tokens_per_expert]

    # Reshape input for block-wise processing
    # [num_local_expert, num_max_tokens_per_expert, hidden_size_scale, block_size]
    reshaped_input = paddle.reshape(
        input_tensor, [num_local_expert, num_max_tokens_per_expert, hidden_size_scale, block_size]
    ).astype("float32")

    # Calculate max absolute values per block
    max_abs_val = paddle.max(
        paddle.abs(reshaped_input), axis=-1, keepdim=True
    )  # [num_local_expert, num_max_tokens_per_expert, hidden_size_scale, 1]
    max_abs_val = paddle.clip(max_abs_val, min=epsilon)

    # Apply valid mask - set invalid tokens' max values to epsilon
    valid_mask_expanded = valid_mask.unsqueeze(2).unsqueeze(3)  # [num_local_expert, num_max_tokens_per_expert, 1, 1]
    max_abs_val = paddle.where(valid_mask_expanded, max_abs_val, paddle.to_tensor(epsilon))

    # Apply fine-grained range if enabled
    if use_finegrained_range:
        max_abs_val *= 7.0

    # Calculate scale
    scale = max_abs_val / MAX_VALUE

    # Quantize
    quanted_value = reshaped_input / scale

    # Convert to float8_e4m3fn and reshape back
    quanted_x_reshaped = quanted_value.astype("float8_e4m3fn")
    quanted_x = paddle.reshape(quanted_x_reshaped, [num_local_expert, num_max_tokens_per_expert, hidden_size])

    # Apply valid mask to quantized output - convert to float32 first, then back to float8_e4m3fn
    valid_mask_full = valid_mask.unsqueeze(2)  # [num_local_expert, num_max_tokens_per_expert, 1]
    quanted_x_float32 = quanted_x.astype("float32")
    quanted_x_masked_float32 = paddle.where(valid_mask_full, quanted_x_float32, paddle.zeros_like(quanted_x_float32))
    quanted_x = quanted_x_masked_float32.astype("float8_e4m3fn")

    # Prepare scale output - squeeze the last dimension
    quanted_scale = paddle.squeeze(scale, axis=-1)  # [num_local_expert, num_max_tokens_per_expert, hidden_size_scale]

    # Apply valid mask to scale
    valid_mask_scale = valid_mask.unsqueeze(2)  # [num_local_expert, num_max_tokens_per_expert, 1]
    quanted_scale = paddle.where(valid_mask_scale, quanted_scale, paddle.zeros_like(quanted_scale))

    return quanted_x, quanted_scale


class TestMaskedPerTokenQuant(unittest.TestCase):
    def setUp(self) -> None:
        paddle.seed(2024)
        self.num_local_expert = 2
        self.num_max_tokens_per_expert = 4
        self.hidden_size = 256
        self.block_size = 128
        self.dtype = paddle.bfloat16

        self.input_tensor = paddle.randn(
            [self.num_local_expert, self.num_max_tokens_per_expert, self.hidden_size], dtype=self.dtype
        )
        self.recv_expert_count = paddle.to_tensor([3, 2], dtype="int32")

        # Get reference results from paddle implementation
        self.quanted_x_ref, self.quanted_scale_ref = masked_per_token_quant_ref(
            self.input_tensor, self.recv_expert_count, self.block_size
        )

    def _mask_invalid_tokens(self, quanted_x, quanted_scale, recv_expert_count):
        """Apply mask to zero out invalid tokens"""
        token_indices = paddle.arange(self.num_max_tokens_per_expert, dtype="int32").unsqueeze(0)
        expert_counts = recv_expert_count.unsqueeze(1)
        valid_mask = token_indices < expert_counts

        # Apply mask to quantized values - convert to float32 first
        valid_mask_full = valid_mask.unsqueeze(2)
        quanted_x_float32 = quanted_x.astype("float32")
        quanted_x_masked_float32 = paddle.where(
            valid_mask_full, quanted_x_float32, paddle.zeros_like(quanted_x_float32)
        )
        quanted_x_masked = quanted_x_masked_float32.astype("float8_e4m3fn")

        # Apply mask to scale values
        valid_mask_scale = valid_mask.unsqueeze(2)
        quanted_scale_masked = paddle.where(valid_mask_scale, quanted_scale, paddle.zeros_like(quanted_scale))

        return quanted_x_masked, quanted_scale_masked

    def test_masked_per_token_quant_basic(self):
        """Test basic functionality against CUDA kernel"""
        quanted_x_cuda, quanted_scale_cuda = masked_per_token_quant(
            self.input_tensor, self.recv_expert_count, self.block_size
        )

        quanted_x_cuda_masked, quanted_scale_cuda_masked = self._mask_invalid_tokens(
            quanted_x_cuda, quanted_scale_cuda, self.recv_expert_count
        )

        # Check output shapes
        self.assertEqual(quanted_x_cuda.shape, self.quanted_x_ref.shape)
        self.assertEqual(quanted_scale_cuda.shape, self.quanted_scale_ref.shape)

        # Check dtypes
        self.assertEqual(quanted_x_cuda.dtype, paddle.float8_e4m3fn)
        self.assertEqual(quanted_scale_cuda.dtype, paddle.float32)

        # Compare scale values (using masked versions)
        np.testing.assert_allclose(
            self.quanted_scale_ref.numpy(), quanted_scale_cuda_masked.numpy(), rtol=1e-5, atol=1e-6
        )

        # Compare quantized values (convert to float32 for comparison, using masked versions)
        quant_diff = paddle.mean(
            paddle.abs(quanted_x_cuda_masked.astype("float32") - self.quanted_x_ref.astype("float32"))
        ) / paddle.mean(paddle.abs(self.quanted_x_ref.astype("float32")) + 1e-9)
        diff_val = float(quant_diff.numpy().item())
        self.assertLess(diff_val, 0.01, msg="Quantized values should be close")


class TestMaskedPerTokenQuantCase1(TestMaskedPerTokenQuant):
    """Test with float16 input"""

    def setUp(self) -> None:
        paddle.seed(2024)
        self.num_local_expert = 3
        self.num_max_tokens_per_expert = 6
        self.hidden_size = 512
        self.block_size = 128
        self.dtype = paddle.float16

        self.input_tensor = paddle.randn(
            [self.num_local_expert, self.num_max_tokens_per_expert, self.hidden_size], dtype=self.dtype
        )
        self.recv_expert_count = paddle.to_tensor([4, 2, 5], dtype="int32")

        self.quanted_x_ref, self.quanted_scale_ref = masked_per_token_quant_ref(
            self.input_tensor, self.recv_expert_count, self.block_size
        )


class TestMaskedPerTokenQuantCase2(TestMaskedPerTokenQuant):
    """Test with different hidden size"""

    def setUp(self) -> None:
        paddle.seed(2024)
        self.num_local_expert = 4
        self.num_max_tokens_per_expert = 8
        self.hidden_size = 384  # 3 * 128
        self.block_size = 128
        self.dtype = paddle.bfloat16

        self.input_tensor = paddle.randn(
            [self.num_local_expert, self.num_max_tokens_per_expert, self.hidden_size], dtype=self.dtype
        )
        self.recv_expert_count = paddle.to_tensor([6, 3, 7, 1], dtype="int32")

        self.quanted_x_ref, self.quanted_scale_ref = masked_per_token_quant_ref(
            self.input_tensor, self.recv_expert_count, self.block_size
        )


class TestMaskedPerTokenQuantCase3(TestMaskedPerTokenQuant):
    """Test with all experts having max tokens"""

    def setUp(self) -> None:
        paddle.seed(2024)
        self.num_local_expert = 2
        self.num_max_tokens_per_expert = 4
        self.hidden_size = 256
        self.block_size = 128
        self.dtype = paddle.bfloat16

        self.input_tensor = paddle.randn(
            [self.num_local_expert, self.num_max_tokens_per_expert, self.hidden_size], dtype=self.dtype
        )
        # All experts use all tokens
        self.recv_expert_count = paddle.to_tensor([4, 4], dtype="int32")

        self.quanted_x_ref, self.quanted_scale_ref = masked_per_token_quant_ref(
            self.input_tensor, self.recv_expert_count, self.block_size
        )


class TestMaskedPerTokenQuantEdgeCases(unittest.TestCase):
    """Test edge cases"""

    def test_zero_tokens_expert(self):
        """Test expert with zero tokens"""
        paddle.seed(2024)
        input_tensor = paddle.randn([2, 4, 256], dtype="bfloat16")
        recv_expert_count = paddle.to_tensor([0, 2], dtype="int32")  # First expert has no tokens

        quanted_x_ref, quanted_scale_ref = masked_per_token_quant_ref(input_tensor, recv_expert_count, 128)

        # First expert should be all zeros - convert to float32 for comparison
        expert_0_quanted = quanted_x_ref[0].astype("float32")
        self.assertTrue(paddle.all(expert_0_quanted == 0), "Expert with zero tokens should be all zero")
        self.assertTrue(paddle.all(quanted_scale_ref[0] == 0), "Expert with zero tokens should have zero scales")

        # Second expert should have valid values - convert to float32 for comparison
        expert_1_quanted = quanted_x_ref[1, :2].astype("float32")
        self.assertTrue(paddle.any(expert_1_quanted != 0), "Expert with tokens should have non-zero values")


if __name__ == "__main__":
    unittest.main()
