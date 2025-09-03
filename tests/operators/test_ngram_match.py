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

import unittest

import paddle

from fastdeploy.model_executor.ops.gpu import ngram_match


class TestNgramMatchOp(unittest.TestCase):

    def setUp(self):
        paddle.set_device("cpu")

    def test_basic_match(self):
        """
        Case 1: input_ids overlaps with pre_ids, and can extract draft tokens.
        """
        batch_size = 1
        seq_len = 6

        # Input IDs
        input_ids = paddle.to_tensor([[10, 20, 30, 40, 50, 60]], dtype="int64")
        # Length of input IDs
        input_ids_len = paddle.to_tensor([6], dtype="int64")
        # Previous IDs
        pre_ids = paddle.to_tensor([[10, 20, 30, 40, 0, 0]], dtype="int64")
        # Current step index
        step_idx = paddle.to_tensor([3], dtype="int64")
        # Number of draft tokens
        draft_token_num = paddle.to_tensor([3], dtype="int32")
        # Placeholder for draft tokens
        draft_tokens = paddle.zeros([batch_size, seq_len], dtype="int64")

        # Sequence lengths for this time step
        seq_lens_this_time = paddle.zeros([batch_size], dtype="int32")
        # Sequence lengths for encoder
        seq_lens_encoder = paddle.zeros([batch_size], dtype="int32")
        # Sequence lengths for decoder
        seq_lens_decoder = paddle.ones([batch_size], dtype="int32")
        # Maximum decoding length
        max_dec_len = paddle.to_tensor([10], dtype="int64")

        ngram_match(
            input_ids,
            input_ids_len,
            pre_ids,
            step_idx,
            draft_token_num,
            draft_tokens,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            max_dec_len,
            3,
            4,
        )

        # Extract non-zero tokens and assert the results.
        nonzero_tokens = draft_tokens.numpy()[0][draft_tokens.numpy()[0] != 0]
        expected_tokens = [50, 60]
        self.assertTrue((nonzero_tokens == expected_tokens).all())

        # Check length
        self.assertEqual(seq_lens_this_time.numpy()[0], 3)

    def test_no_match(self):
        """
        Case 2: pre_ids does not match input_ids, should only keep the current token.
        """
        batch_size = 1
        input_ids = paddle.to_tensor([[100, 200, 300, 400]], dtype="int64")
        input_ids_len = paddle.to_tensor([4], dtype="int64")
        pre_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        step_idx = paddle.to_tensor([3], dtype="int64")
        draft_token_num = paddle.to_tensor([2], dtype="int32")
        draft_tokens = paddle.zeros([batch_size, 4], dtype="int64")

        seq_lens_this_time = paddle.zeros([batch_size], dtype="int32")
        seq_lens_encoder = paddle.zeros([batch_size], dtype="int32")
        seq_lens_decoder = paddle.ones([batch_size], dtype="int32")
        max_dec_len = paddle.to_tensor([6], dtype="int64")

        ngram_match(
            input_ids,
            input_ids_len,
            pre_ids,
            step_idx,
            draft_token_num,
            draft_tokens,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            max_dec_len,
            3,
            3,
        )

        # No match → should only keep 1 token
        self.assertEqual(seq_lens_this_time.numpy()[0], 1)


if __name__ == "__main__":
    unittest.main()
