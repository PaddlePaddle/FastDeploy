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
Unit tests for get_padding_offset function.

This test file covers the following scenarios:
1. Normal path on XPU device with token_num_data > 0
2. CPU path when input_ids is on CPU
3. Empty token path when token_num_data == 0
4. Speculative decoding path when draft_tokens is provided
5. Edge cases: single batch, maximum sequence length, zero sequence length
"""

import unittest

import numpy as np
import paddle


class TestGetPaddingOffset(unittest.TestCase):
    """Test cases for get_padding_offset function."""

    def setUp(self):
        """Set up test fixtures."""
        # Import here to ensure ops are available
        from fastdeploy.model_executor.ops.xpu import get_padding_offset

        self.get_padding_offset = get_padding_offset
        np.random.seed(2024)

    def test_normal_path_xpu(self):
        """
        Test normal path on XPU with token_num_data > 0.

        This tests the main code path (lines 54-67 in get_padding_offset.cc)
        where the get_padding_offset plugin is called.
        """
        # --------------------
        # Setup: 3 batches with different sequence lengths
        # --------------------
        bsz = 3
        max_seq_len = 10
        seq_lens = np.array([4, 3, 6], dtype=np.int32)
        token_num = int(np.sum(seq_lens))

        # Create input_ids with random token IDs
        input_ids = np.zeros((bsz, max_seq_len), dtype=np.int64)
        for i in range(bsz):
            ids_len = seq_lens[i]
            input_ids[i, 0:ids_len] = np.random.randint(1, 100, seq_lens[i], dtype=np.int64)

        # --------------------
        # Call function (lines 54-67)
        # --------------------
        try:
            paddle.set_device("xpu")
        except Exception:
            self.skipTest("XPU not available, skipping XPU test")

        result = self.get_padding_offset(
            paddle.to_tensor(input_ids),
            paddle.to_tensor(seq_lens),
            None,
            None,
            token_num,
        )

        # Unpack 4 outputs: x_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k
        x_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k = result
        x_remove_padding = x_remove_padding.numpy()
        batch_id_per_token = batch_id_per_token.numpy()
        cu_seqlens_q = cu_seqlens_q.numpy()
        cu_seqlens_k = cu_seqlens_k.numpy()

        # --------------------
        # Verify outputs
        # --------------------
        # Check shape
        self.assertEqual(x_remove_padding.shape[0], token_num)
        self.assertEqual(batch_id_per_token.shape[0], token_num)
        self.assertEqual(cu_seqlens_q.shape[0], bsz + 1)
        self.assertEqual(cu_seqlens_k.shape[0], bsz + 1)

        # Check cu_seqlens_q is cumulative
        expected_cu_seqlens_q = np.cumsum(np.concatenate([[0], seq_lens]))
        np.testing.assert_array_equal(cu_seqlens_q, expected_cu_seqlens_q)

        # Check cu_seqlens_k equals cu_seqlens_q (no decoder tokens in this test)
        np.testing.assert_array_equal(cu_seqlens_k, cu_seqlens_q)

        # Check batch_id_per_token is valid (0, 1, 2 for each token)
        for i in range(token_num):
            self.assertGreaterEqual(batch_id_per_token[i], 0)
            self.assertLess(batch_id_per_token[i], bsz)

    def test_empty_token_path(self):
        """
        Test path when token_num_data == 0.

        This tests the branch at line 54: if (token_num_data > 0)
        where the condition is false.
        """
        bsz = 2
        max_seq_len = 8
        seq_lens = np.array([0, 0], dtype=np.int32)
        token_num = 0
        input_ids = np.zeros((bsz, max_seq_len), dtype=np.int64)

        try:
            paddle.set_device("xpu")
        except Exception:
            self.skipTest("XPU not available, skipping XPU test")

        result = self.get_padding_offset(
            paddle.to_tensor(input_ids),
            paddle.to_tensor(seq_lens),
            None,
            None,
            token_num,
        )

        x_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k = result

        # With token_num == 0, outputs should have shape [0]
        self.assertEqual(x_remove_padding.shape[0], 0)
        self.assertEqual(batch_id_per_token.shape[0], 0)
        self.assertEqual(cu_seqlens_q.shape[0], bsz + 1)
        self.assertEqual(cu_seqlens_k.shape[0], bsz + 1)

    def test_single_batch(self):
        """
        Test edge case with single batch.

        Tests boundary value: bsz == 1
        """
        bsz = 1
        max_seq_len = 5
        seq_lens = np.array([5], dtype=np.int32)
        token_num = 5
        input_ids = np.random.randint(1, 100, (bsz, max_seq_len), dtype=np.int64)

        try:
            paddle.set_device("xpu")
        except Exception:
            self.skipTest("XPU not available, skipping XPU test")

        result = self.get_padding_offset(
            paddle.to_tensor(input_ids),
            paddle.to_tensor(seq_lens),
            None,
            None,
            token_num,
        )

        x_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k = result

        # All tokens should belong to batch 0
        np.testing.assert_array_equal(batch_id_per_token.numpy(), np.array([0, 0, 0, 0, 0], dtype=np.int32))

        # cu_seqlens should be [0, 5]
        np.testing.assert_array_equal(cu_seqlens_q.numpy(), np.array([0, 5], dtype=np.int32))

    def test_max_sequence_length(self):
        """
        Test boundary value with maximum sequence length.

        Tests: all sequences at max_seq_len
        """
        bsz = 4
        max_seq_len = 16
        seq_lens = np.array([16, 16, 16, 16], dtype=np.int32)
        token_num = 64
        input_ids = np.random.randint(1, 1000, (bsz, max_seq_len), dtype=np.int64)

        try:
            paddle.set_device("xpu")
        except Exception:
            self.skipTest("XPU not available, skipping XPU test")

        result = self.get_padding_offset(
            paddle.to_tensor(input_ids),
            paddle.to_tensor(seq_lens),
            None,
            None,
            token_num,
        )

        x_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k = result

        # Check cumulative sequence lengths
        expected_cu = np.array([0, 16, 32, 48, 64], dtype=np.int32)
        np.testing.assert_array_equal(cu_seqlens_q.numpy(), expected_cu)

    def test_mixed_sequence_lengths(self):
        """
        Test various sequence length combinations.

        Tests: varying seq_lens including 0, 1, small, and large values
        """
        bsz = 4
        max_seq_len = 20
        seq_lens = np.array([0, 1, 10, 20], dtype=np.int32)
        token_num = 31
        input_ids = np.random.randint(1, 1000, (bsz, max_seq_len), dtype=np.int64)

        try:
            paddle.set_device("xpu")
        except Exception:
            self.skipTest("XPU not available, skipping XPU test")

        result = self.get_padding_offset(
            paddle.to_tensor(input_ids),
            paddle.to_tensor(seq_lens),
            None,
            None,
            token_num,
        )

        x_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k = result

        # Check cumulative sequence lengths
        expected_cu = np.array([0, 0, 1, 11, 31], dtype=np.int32)
        np.testing.assert_array_equal(cu_seqlens_q.numpy(), expected_cu)

        # Check batch_id for each token
        # Batch 0: 0 tokens
        # Batch 1: 1 token
        # Batch 2: 10 tokens
        # Batch 3: 20 tokens
        expected_batch_ids = [1] * 1 + [2] * 10 + [3] * 20  # Skip batch 0 (0 tokens)
        np.testing.assert_array_equal(batch_id_per_token.numpy(), np.array(expected_batch_ids, dtype=np.int32))

    def test_cpu_path(self):
        """
        Test CPU path when input_ids is on CPU.

        This tests the branch at lines 34-36 in get_padding_offset.cc:
        if (input_ids.is_cpu()) {
            ctx = new baidu::xpu::api::Context(baidu::xpu::api::kCPU);
        }
        """
        bsz = 2
        max_seq_len = 6
        seq_lens = np.array([3, 4], dtype=np.int32)
        token_num = 7
        input_ids = np.random.randint(1, 100, (bsz, max_seq_len), dtype=np.int64)

        # Place on CPU explicitly
        input_ids_tensor = paddle.to_tensor(input_ids, place=paddle.CPUPlace())
        seq_lens_tensor = paddle.to_tensor(seq_lens, place=paddle.CPUPlace())

        result = self.get_padding_offset(
            input_ids_tensor,
            seq_lens_tensor,
            None,
            None,
            token_num,
        )

        x_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k = result

        # Check output shapes
        self.assertEqual(x_remove_padding.shape[0], token_num)
        self.assertEqual(batch_id_per_token.shape[0], token_num)
        self.assertEqual(cu_seqlens_q.shape[0], bsz + 1)

        # Verify correctness - same logic as XPU path
        expected_cu = np.array([0, 3, 7], dtype=np.int32)
        np.testing.assert_array_equal(cu_seqlens_q.numpy(), expected_cu)


class TestGetPaddingOffsetWithSpeculative(unittest.TestCase):
    """Test that speculative decoding (draft_tokens) raises on XPU since it is not yet supported."""

    def setUp(self):
        """Set up test fixtures."""
        from fastdeploy.model_executor.ops.xpu import get_padding_offset

        self.get_padding_offset = get_padding_offset
        np.random.seed(2024)

    def test_with_draft_tokens(self):
        """Passing draft_tokens should raise because XPU does not support speculative decoding."""
        try:
            paddle.set_device("xpu")
        except Exception:
            self.skipTest("XPU not available, skipping XPU test")

        bsz = 2
        max_seq_len = 10
        seq_lens = np.array([4, 6], dtype=np.int32)
        input_ids = np.random.randint(1, 100, (bsz, max_seq_len), dtype=np.int64)
        draft_tokens = np.random.randint(1, 100, (bsz, 3), dtype=np.int64)
        seq_lens_encoder = np.array([2, 3], dtype=np.int32)
        token_num = int(np.sum(seq_lens))

        with self.assertRaises(RuntimeError):
            self.get_padding_offset(
                paddle.to_tensor(input_ids),
                paddle.to_tensor(seq_lens),
                paddle.to_tensor(draft_tokens),
                paddle.to_tensor(seq_lens_encoder),
                token_num,
            )

    def test_draft_tokens_single_batch(self):
        """Passing draft_tokens with single batch should raise on XPU."""
        try:
            paddle.set_device("xpu")
        except Exception:
            self.skipTest("XPU not available, skipping XPU test")

        bsz = 1
        max_seq_len = 8
        seq_lens = np.array([4], dtype=np.int32)
        input_ids = np.random.randint(1, 100, (bsz, max_seq_len), dtype=np.int64)
        draft_tokens = np.random.randint(1, 100, (bsz, 2), dtype=np.int64)
        seq_lens_encoder = np.array([2], dtype=np.int32)
        token_num = 4

        with self.assertRaises(RuntimeError):
            self.get_padding_offset(
                paddle.to_tensor(input_ids),
                paddle.to_tensor(seq_lens),
                paddle.to_tensor(draft_tokens),
                paddle.to_tensor(seq_lens_encoder),
                token_num,
            )

    def test_draft_tokens_multiple_batches(self):
        """Passing draft_tokens with multiple batches should raise on XPU."""
        try:
            paddle.set_device("xpu")
        except Exception:
            self.skipTest("XPU not available, skipping XPU test")

        bsz = 4
        max_seq_len = 12
        seq_lens = np.array([3, 5, 2, 7], dtype=np.int32)
        input_ids = np.random.randint(1, 100, (bsz, max_seq_len), dtype=np.int64)
        draft_tokens = np.random.randint(1, 100, (bsz, 4), dtype=np.int64)
        seq_lens_encoder = np.array([1, 2, 1, 3], dtype=np.int32)
        token_num = int(np.sum(seq_lens))

        with self.assertRaises(RuntimeError):
            self.get_padding_offset(
                paddle.to_tensor(input_ids),
                paddle.to_tensor(seq_lens),
                paddle.to_tensor(draft_tokens),
                paddle.to_tensor(seq_lens_encoder),
                token_num,
            )


class TestGetPaddingOffsetEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for get_padding_offset."""

    def setUp(self):
        """Set up test fixtures."""
        from fastdeploy.model_executor.ops.xpu import get_padding_offset

        self.get_padding_offset = get_padding_offset

    def test_single_token_per_batch(self):
        """
        Test edge case with exactly 1 token per batch.

        Tests: min token_num == bsz
        """
        try:
            paddle.set_device("xpu")
        except Exception:
            self.skipTest("XPU not available, skipping XPU test")

        bsz = 4
        max_seq_len = 8
        seq_lens = np.array([1, 1, 1, 1], dtype=np.int32)
        token_num = 4
        input_ids = np.random.randint(1, 100, (bsz, max_seq_len), dtype=np.int64)

        result = self.get_padding_offset(
            paddle.to_tensor(input_ids),
            paddle.to_tensor(seq_lens),
            None,
            None,
            token_num,
        )

        x_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k = result

        # Each batch has exactly 1 token
        expected_batch_ids = np.array([0, 1, 2, 3], dtype=np.int32)
        np.testing.assert_array_equal(batch_id_per_token.numpy(), expected_batch_ids)

    def test_large_batch_size(self):
        """
        Test performance with large batch size.

        Tests: large bsz value for stress testing
        """
        try:
            paddle.set_device("xpu")
        except Exception:
            self.skipTest("XPU not available, skipping XPU test")

        bsz = 128
        max_seq_len = 64
        seq_lens = np.full(bsz, 32, dtype=np.int32)
        token_num = 4096
        input_ids = np.random.randint(1, 1000, (bsz, max_seq_len), dtype=np.int64)

        result = self.get_padding_offset(
            paddle.to_tensor(input_ids),
            paddle.to_tensor(seq_lens),
            None,
            None,
            token_num,
        )

        x_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k = result

        # Verify output shapes
        self.assertEqual(x_remove_padding.shape[0], token_num)
        self.assertEqual(batch_id_per_token.shape[0], token_num)
        self.assertEqual(cu_seqlens_q.shape[0], bsz + 1)

        # Verify cumulative sum
        expected_cu = np.cumsum(np.concatenate([[0], seq_lens]))
        np.testing.assert_array_equal(cu_seqlens_q.numpy(), expected_cu)

    def test_large_vocab_sequence(self):
        """
        Test with large sequence lengths.

        Tests: max_seq_len boundary
        """
        try:
            paddle.set_device("xpu")
        except Exception:
            self.skipTest("XPU not available, skipping XPU test")

        bsz = 2
        max_seq_len = 128
        seq_lens = np.array([64, 128], dtype=np.int32)
        token_num = 192
        input_ids = np.random.randint(1, 10000, (bsz, max_seq_len), dtype=np.int64)

        result = self.get_padding_offset(
            paddle.to_tensor(input_ids),
            paddle.to_tensor(seq_lens),
            None,
            None,
            token_num,
        )

        x_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k = result

        # Check cumulative lengths
        expected_cu = np.array([0, 64, 192], dtype=np.int32)
        np.testing.assert_array_equal(cu_seqlens_q.numpy(), expected_cu)

    def test_zero_first_batches(self):
        """
        Test with some batches having zero tokens at the start.

        Tests: seq_lens with leading zeros
        """
        try:
            paddle.set_device("xpu")
        except Exception:
            self.skipTest("XPU not available, skipping XPU test")

        bsz = 4
        max_seq_len = 16
        seq_lens = np.array([0, 0, 5, 8], dtype=np.int32)
        token_num = 13
        input_ids = np.random.randint(1, 100, (bsz, max_seq_len), dtype=np.int64)

        result = self.get_padding_offset(
            paddle.to_tensor(input_ids),
            paddle.to_tensor(seq_lens),
            None,
            None,
            token_num,
        )

        x_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k = result

        # Check cumulative: [0, 0, 0, 5, 13]
        expected_cu = np.array([0, 0, 0, 5, 13], dtype=np.int32)
        np.testing.assert_array_equal(cu_seqlens_q.numpy(), expected_cu)


if __name__ == "__main__":
    unittest.main()
