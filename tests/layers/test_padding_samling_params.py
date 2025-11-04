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

from fastdeploy.model_executor.layers.sample.sampler import padding_sampling_params


class TestPaddingSamplingParams(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32

    def test_all_decode(self):
        top_p = paddle.to_tensor([0.8, 0.9, 0.95], dtype="float32")
        top_k = paddle.to_tensor([10, 20, 30], dtype="int64")
        seq_lens_this_time = paddle.to_tensor([2, 3, 1], dtype="int64")
        seq_lens_encoder = paddle.to_tensor([0, 0, 0], dtype="int64")

        top_p_padding, top_k_padding = padding_sampling_params(top_p, top_k, seq_lens_this_time, seq_lens_encoder)

        expected_len = sum(seq_lens_this_time.numpy())
        self.assertEqual(top_p_padding.shape[0], expected_len)
        self.assertEqual(top_k_padding.shape[0], expected_len)

        expected_top_p = np.repeat([0.8, 0.9, 0.95], [2, 3, 1]).reshape(-1, 1)
        np.testing.assert_allclose(top_p_padding.numpy(), expected_top_p, rtol=1e-6)

    def test_partial_decode(self):
        top_p = paddle.to_tensor([0.7, 0.6, 0.5], dtype="float32")
        top_k = paddle.to_tensor([15, 25, 35], dtype="int64")
        seq_lens_this_time = paddle.to_tensor([3, 2, 4], dtype="int64")
        seq_lens_encoder = paddle.to_tensor([0, 1, 0], dtype="int64")

        top_p_padding, top_k_padding = padding_sampling_params(top_p, top_k, seq_lens_this_time, seq_lens_encoder)

        expected_repeats = [3, 1, 4]
        expected_top_p = np.repeat([0.7, 0.6, 0.5], expected_repeats).reshape(-1, 1)
        expected_top_k = np.repeat([15, 25, 35], expected_repeats).reshape(-1, 1)

        np.testing.assert_allclose(top_p_padding.numpy(), expected_top_p, rtol=1e-6)
        np.testing.assert_array_equal(top_k_padding.numpy(), expected_top_k)

    def test_all_prefill(self):
        top_p = paddle.to_tensor([0.5, 0.6], dtype="float32")
        top_k = paddle.to_tensor([5, 6], dtype="int64")
        seq_lens_this_time = paddle.to_tensor([4, 3], dtype="int64")
        seq_lens_encoder = paddle.to_tensor([1, 2], dtype="int64")

        top_p_padding, top_k_padding = padding_sampling_params(top_p, top_k, seq_lens_this_time, seq_lens_encoder)

        expected_top_p = np.array([[0.5], [0.6]])
        expected_top_k = np.array([[5], [6]])

        np.testing.assert_allclose(top_p_padding.numpy(), expected_top_p)
        np.testing.assert_array_equal(top_k_padding.numpy(), expected_top_k)


if __name__ == "__main__":
    unittest.main()
