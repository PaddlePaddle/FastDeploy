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

from fastdeploy.model_executor.ops.gpu import top_k_renorm_probs


class TestTopKRenormProbs(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def _check_output(self, probs, top_k):
        probs_tensor = paddle.to_tensor(probs)
        top_k_tensor = paddle.to_tensor(top_k)
        renorm_probs = top_k_renorm_probs(probs_tensor, top_k_tensor).numpy()

        self.assertEqual(renorm_probs.shape, probs.shape)

        batch_size, vocab_size = probs.shape
        for b in range(batch_size):
            self.assertAlmostEqual(renorm_probs[b].sum(), 1.0, places=6)
            top_indices = np.argsort(probs[b])[::-1][: top_k[b]]
            for j in range(vocab_size):
                if j not in top_indices:
                    self.assertAlmostEqual(renorm_probs[b, j], 0.0, places=6)

    def test_single_batch_basic(self):
        """Test with batch_size = 1"""
        probs = np.random.rand(1, 5).astype("float32")
        probs /= probs.sum(axis=1, keepdims=True)
        top_k = np.array([2], dtype="int64")
        self._check_output(probs, top_k)

    def test_single_batch_edge_cases(self):
        """Test edge cases with batch_size = 1"""
        probs = np.array([[0.1, 0.3, 0.4, 0.2]], dtype="float32")

        # top_k = 1
        self._check_output(probs, np.array([1], dtype="int64"))

        # top_k = vocab_size
        renorm_probs = top_k_renorm_probs(
            paddle.to_tensor(probs), paddle.to_tensor(np.array([4], dtype="int64"))
        ).numpy()
        np.testing.assert_allclose(renorm_probs, probs, rtol=1e-6, atol=1e-6)

    def test_batch_size_two(self):
        """Test with batch_size = 2"""
        probs = np.random.rand(2, 5).astype("float32")
        probs /= probs.sum(axis=1, keepdims=True)
        top_k = np.array([2, 3], dtype="int64")
        self._check_output(probs, top_k)

    def test_batch_size_three(self):
        """Test with batch_size = 3"""
        probs = np.random.rand(3, 6).astype("float32")
        probs /= probs.sum(axis=1, keepdims=True)
        top_k = np.array([1, 2, 4], dtype="int64")
        self._check_output(probs, top_k)


if __name__ == "__main__":
    unittest.main()
