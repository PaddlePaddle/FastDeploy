import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import reorder_split_prefill_and_decode


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
        prompt_lens = paddle.to_tensor([1, 1, 1], dtype="int64", place=self.place)

        # Call the operator
        x_reorder, batch_id_reorder, num_decode = reorder_split_prefill_and_decode(
            x_remove_padding, batch_id_per_token, cu_seqlens_q, prompt_lens
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
        prompt_lens = paddle.to_tensor([1, 2], dtype="int64", place=self.place)

        x_reorder, _, num_decode = reorder_split_prefill_and_decode(
            x_remove_padding, batch_id_per_token, cu_seqlens_q, prompt_lens
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


if __name__ == "__main__":
    unittest.main()
