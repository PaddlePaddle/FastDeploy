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

from fastdeploy.model_executor.ops.gpu import rejection_top_p_sampling


class TestRejectionTopPSampling(unittest.TestCase):
    def setUp(self):
        """Initialize common test data"""
        self.batch_size = 10
        self.vocab_size = 103424
        paddle.seed(2023)

        # Generate test data once for all tests
        self.pre_norm_prob_np = np.random.rand(self.batch_size, self.vocab_size).astype(np.float32)
        self.paddle_pre_norm_prob = paddle.to_tensor(self.pre_norm_prob_np)
        self.paddle_norm_prob = self.paddle_pre_norm_prob / self.paddle_pre_norm_prob.sum(axis=-1, keepdim=True)

    def test_top_p_sampling_reject_case1(self):
        """Test with fixed top_p=0.8 and different random seeds"""
        top_p_paddle = paddle.full((self.batch_size,), 0.8)
        top_k_paddle = paddle.full((self.batch_size,), 20).cast("int64")

        # Test with different seeds
        for seed in [1024, 2033, 2033]:
            samples = rejection_top_p_sampling(self.paddle_norm_prob, top_p_paddle, top_k_paddle, seed)
            self._validate_samples(samples)

            # Basic validation
            self.assertTrue(paddle.all(samples >= 0))
            self.assertTrue(paddle.all(samples < self.vocab_size))

    def test_top_p_sampling_reject_case2(self):
        """Test with varying top_p values across batch"""
        top_p_paddle = paddle.uniform(shape=[self.batch_size], min=0.1, max=1.0)
        top_k_paddle = paddle.full((self.batch_size,), 20).cast("int64")
        samples = rejection_top_p_sampling(self.paddle_norm_prob, top_p_paddle, top_k_paddle, -1)
        self._validate_samples(samples)

        # Additional check that we're getting different results for different top_p
        unique_samples = len(paddle.unique(samples))
        self.assertGreater(unique_samples, 1)  # Should have some diversity

    def _validate_samples(self, samples):
        """Common validation for all test cases"""
        self.assertTrue(paddle.all(samples >= 0))
        self.assertTrue(paddle.all(samples < self.vocab_size))

        # Check dtype
        self.assertEqual(samples.dtype, paddle.int64)


if __name__ == "__main__":
    unittest.main()
