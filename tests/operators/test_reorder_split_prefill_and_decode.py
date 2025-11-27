import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.layers.attention.ops import (
    reorder_split_prefill_and_decode,
)


class TestReorderSplitPrefillAndDecode(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)

    def test_basic_functionality(self):
        """
        Test basic functionality: 3 sequences, each with 1 prefill and 1 decode token
        """
        # Input data: 3 sequences, 2 tokens each
        x_remove_padding = paddle.to_tensor([1, 2, 3, 4, 5, 6], dtype="int64", place=self.place)
        batch_id_per_token = paddle.to_tensor([0, 0, 1, 1, 2, 2], dtype="int32", place=self.place)
        cu_seqlens_q = paddle.to_tensor([0, 2, 4, 6], dtype="int32", place=self.place)
        seq_lens_encoder = paddle.to_tensor([1, 1, 1], dtype="int32", place=self.place)

        # Call the operator
        x_reorder, batch_id_reorder, num_decode = reorder_split_prefill_and_decode(
            x_remove_padding, batch_id_per_token, cu_seqlens_q, seq_lens_encoder
        )

        # Verify outputs
        self.assertEqual(num_decode.numpy()[0], 3)  # 3 decode tokens expected
        np.testing.assert_array_equal(x_reorder.numpy(), [2, 4, 6, 1, 3, 5])  # decode tokens first
        np.testing.assert_array_equal(batch_id_reorder.numpy(), [0, 1, 2, 0, 1, 2])

    def test_mixed_prefill_decode_ratio(self):
        """
        Test different prefill/decode ratios
        """
        x_remove_padding = paddle.to_tensor([10, 11, 12, 20, 21, 22], dtype="int64", place=self.place)
        batch_id_per_token = paddle.to_tensor([0, 0, 0, 1, 1, 1], dtype="int32", place=self.place)
        cu_seqlens_q = paddle.to_tensor([0, 3, 6], dtype="int32", place=self.place)
        seq_lens_encoder = paddle.to_tensor([1, 2], dtype="int32", place=self.place)

        x_reorder, _, num_decode = reorder_split_prefill_and_decode(
            x_remove_padding, batch_id_per_token, cu_seqlens_q, seq_lens_encoder
        )

        self.assertEqual(num_decode.numpy()[0], 3)
        np.testing.assert_array_equal(x_reorder.numpy(), [11, 12, 22, 10, 20, 21])

    def test_empty_input(self):
        """Test empty input case"""
        with self.assertRaises(Exception, msg="empty input is not detected"):
            reorder_split_prefill_and_decode(
                paddle.to_tensor([], dtype="int64"),
                paddle.to_tensor([], dtype="int32"),
                paddle.to_tensor([0], dtype="int32"),
                paddle.to_tensor([], dtype="int64"),
            )

    def test_varied_sequence_lengths(self):
        """
        Test with multiple sequences of varying lengths
        """
        # Input data: 5 sequences with varying lengths
        x_remove_padding = paddle.to_tensor(
            [10, 11, 12, 20, 21, 22, 23, 30, 31, 40, 41, 42, 43, 44, 50], dtype="int64", place=self.place
        )
        batch_id_per_token = paddle.to_tensor(
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4], dtype="int32", place=self.place
        )
        cu_seqlens_q = paddle.to_tensor([0, 3, 7, 9, 14, 15], dtype="int32", place=self.place)
        seq_lens_encoder = paddle.to_tensor([1, 2, 1, 3, 1], dtype="int32", place=self.place)

        x_reorder, batch_id_reorder, num_decode = reorder_split_prefill_and_decode(
            x_remove_padding, batch_id_per_token, cu_seqlens_q, seq_lens_encoder
        )

        # Verify outputs
        self.assertEqual(num_decode.numpy()[0], 7)
        np.testing.assert_array_equal(x_reorder.numpy(), [11, 12, 22, 23, 31, 43, 44, 10, 20, 21, 30, 40, 41, 42, 50])
        np.testing.assert_array_equal(batch_id_reorder.numpy(), [0, 0, 1, 1, 2, 3, 3, 0, 1, 1, 2, 3, 3, 3, 4])

    def test_performance_with_30000_elements(self):
        """
        Performance test with 30000 tokens to measure execution time
        """
        import time

        # Create input data with 30000 tokens
        total_tokens = 30000
        batch_size = 30  # 30 sequences, each with 1000 tokens

        # Generate test data
        prefill_ratio = 0.4
        sequence_length = 1000

        x_remove_padding = paddle.arange(total_tokens, dtype="int64")
        batch_id_per_token = []
        cu_seqlens_q = [0]
        seq_lens_encoder = []

        for i in range(batch_size):
            seq_start = i * sequence_length
            seq_end = (i + 1) * sequence_length
            cu_seqlens_q.append(seq_end)
            prefill_len = int(sequence_length * (prefill_ratio + 0.1 * (i % 10) / 10))
            seq_lens_encoder.append(prefill_len)
            batch_id_per_token.extend([i] * sequence_length)

        cu_seqlens_q = paddle.to_tensor(cu_seqlens_q, dtype="int32", place=self.place)
        seq_lens_encoder = paddle.to_tensor(seq_lens_encoder, dtype="int32", place=self.place)
        batch_id_per_token = paddle.to_tensor(batch_id_per_token, dtype="int32", place=self.place)

        # Warm up
        for _ in range(3):
            reorder_split_prefill_and_decode(x_remove_padding, batch_id_per_token, cu_seqlens_q, seq_lens_encoder)

        # Timing measurement
        start_time = time.time()
        num_runs = 10
        for _ in range(num_runs):
            x_reorder, batch_id_reorder, num_decode = reorder_split_prefill_and_decode(
                x_remove_padding, batch_id_per_token, cu_seqlens_q, seq_lens_encoder
            )

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_runs

        print("\nPerformance test for unified reorder_split_prefill_and_decode:")
        print(f"Total tokens: {total_tokens}")
        print(f"Batch size: {batch_size}")
        print(f"Average time per run: {avg_time:.4f} seconds")
        print(f"Tokens processed per second: {total_tokens / avg_time:.0f}")

        # Verify results
        expected_decode_tokens = 0
        for i in range(batch_size):
            seq_start = cu_seqlens_q[i]
            seq_end = cu_seqlens_q[i + 1]
            prefill_len = seq_lens_encoder[i]
            expected_decode_tokens += seq_end - seq_start - prefill_len
        self.assertEqual(num_decode.numpy()[0], expected_decode_tokens)


if __name__ == "__main__":
    unittest.main()
