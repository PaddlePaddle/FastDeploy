# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import draft_model_set_value_by_flags


class TestDraftModelSetValueByFlags(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def test_basic_update(self):
        """
        Test normal update behavior:
        batch0 performs a decoder step, batch1 performs an encoder step
        """
        bs = 2
        pre_id_length = 5
        draft_tokens = paddle.to_tensor([[10, 11, 12], [20, 21, 22]], dtype="int64")
        pre_ids_all = paddle.zeros([bs, pre_id_length], dtype="int64")
        stop_flags = paddle.to_tensor([False, False], dtype="bool")
        seq_lens_this_time = paddle.to_tensor([3, 1], dtype="int32")
        seq_lens_encoder = paddle.to_tensor([0, 0], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0, 0], dtype="int32")
        step_idx = paddle.to_tensor([3, 1], dtype="int64")  # batch0 decoder, batch1 encoder

        """ Call custom op """
        draft_model_set_value_by_flags(
            draft_tokens, pre_ids_all, stop_flags, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, step_idx
        )

        """
        batch0: 3 tokens updated at decoder step
        batch1: 1 token updated at encoder step
        """
        expected = np.array([[0, 10, 11, 12, 0], [0, 20, 0, 0, 0]], dtype=np.int64)

        np.testing.assert_array_equal(pre_ids_all.numpy(), expected)
        np.testing.assert_array_equal(seq_lens_this_time.numpy(), [1, 1])

    def test_stop_flags(self):
        """
        batch0 is skipped (stop_flags=True), batch1 updates normally
        """
        bs = 2
        pre_id_length = 4
        draft_tokens = paddle.to_tensor([[5, 6], [7, 8]], dtype="int64")
        pre_ids_all = paddle.zeros([bs, pre_id_length], dtype="int64")
        stop_flags = paddle.to_tensor([True, False], dtype="bool")
        seq_lens_this_time = paddle.to_tensor([2, 2], dtype="int32")
        seq_lens_encoder = paddle.to_tensor([0, 0], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0, 0], dtype="int32")
        step_idx = paddle.to_tensor([1, 2], dtype="int64")

        draft_model_set_value_by_flags(
            draft_tokens, pre_ids_all, stop_flags, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, step_idx
        )

        """
        batch0: no update due to stop flag
        batch1: 2 tokens updated at decoder step
        """
        expected = np.array([[0, 0, 0, 0], [0, 7, 8, 0]], dtype=np.int64)

        np.testing.assert_array_equal(pre_ids_all.numpy(), expected)
        np.testing.assert_array_equal(seq_lens_this_time.numpy(), [2, 1])


if __name__ == "__main__":
    unittest.main()
