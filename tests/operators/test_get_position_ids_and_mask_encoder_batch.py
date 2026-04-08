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

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import get_position_ids_and_mask_encoder_batch


class TestGetPositionIdsAndMaskEncoderBatch(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        paddle.set_device("gpu")

    def test_basic_functionality(self):
        # Test normal case with batch size 2
        seq_lens_encoder = paddle.to_tensor([3, 2], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([1, 2], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([1, 2], dtype="int32")

        total_len = int(seq_lens_encoder.numpy().sum() + seq_lens_this_time.numpy().sum())
        position_ids = paddle.zeros([total_len], dtype="int32")
        mask_encoder_batch = paddle.zeros([total_len], dtype="int32")

        # Call the custom operator
        get_position_ids_and_mask_encoder_batch(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, position_ids, mask_encoder_batch
        )

        expected_position_ids = np.array([0, 1, 2, 1, 0, 1, 2, 3], dtype=np.int32)

        expected_mask = np.array([1, 1, 1, 0, 1, 1, 0, 0], dtype=np.int32)

        # Convert to numpy for comparison
        position_ids_np = position_ids.numpy()
        mask_encoder_batch_np = mask_encoder_batch.numpy()

        # Assert equality
        np.testing.assert_array_equal(position_ids_np, expected_position_ids)
        np.testing.assert_array_equal(mask_encoder_batch_np, expected_mask)

    def test_empty_decoder(self):
        # Test case where decoder length is 0
        seq_lens_encoder = paddle.to_tensor([2], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([0], dtype="int32")

        position_ids = paddle.zeros([2], dtype="int32")
        mask_encoder_batch = paddle.zeros([2], dtype="int32")

        get_position_ids_and_mask_encoder_batch(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, position_ids, mask_encoder_batch
        )

        expected_position_ids = np.array([0, 1], dtype=np.int32)
        expected_mask = np.array([1, 1], dtype=np.int32)

        np.testing.assert_array_equal(position_ids.numpy(), expected_position_ids)
        np.testing.assert_array_equal(mask_encoder_batch.numpy(), expected_mask)


if __name__ == "__main__":
    unittest.main()
