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

from fastdeploy.model_executor.ops.gpu import top_k_renorm_probs


class TestTopKRenormProbs(unittest.TestCase):

    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def test_basic_functionality(self):
        """Test the operator"""
        batch_size = 1
        vocab_size = 5
        # Construct a valid probability distribution
        probs = np.random.rand(batch_size, vocab_size).astype("float32")
        probs /= probs.sum(axis=1, keepdims=True)
        top_k = np.array([2], dtype="int64")

        probs_tensor = paddle.to_tensor(probs)
        top_k_tensor = paddle.to_tensor(top_k)

        # Call the operator
        renorm_probs = top_k_renorm_probs(probs_tensor, top_k_tensor)[0].numpy()
        renorm_probs = renorm_probs.reshape(batch_size, vocab_size)

        self.assertEqual(renorm_probs.shape, probs.shape)

        # Non top-k positions must be zero
        top_indices = np.argsort(probs[0])[::-1][: top_k[0]]
        for j in range(vocab_size):
            if j not in top_indices:
                self.assertAlmostEqual(renorm_probs[0, j], 0.0, places=6)

        self.assertAlmostEqual(renorm_probs[0].sum(), 1.0, places=6)

    def test_edge_cases(self):
        """Test the operator with edge-case top_k values"""
        probs = np.array([[0.1, 0.3, 0.4, 0.2]], dtype="float32")

        # Case 1: top_k = 1, only the max element remains
        top_k_tensor = paddle.to_tensor(np.array([1], dtype="int64"))
        renorm_probs = top_k_renorm_probs(paddle.to_tensor(probs), top_k_tensor)[0].numpy()
        renorm_probs = renorm_probs.reshape(1, -1)
        self.assertEqual((renorm_probs > 0).sum(), 1)  # Only one non-zero element
        self.assertAlmostEqual(renorm_probs.sum(), 1.0, places=6)

        # Case 2: top_k = vocab_size, output should match input
        top_k_tensor = paddle.to_tensor(np.array([4], dtype="int64"))
        renorm_probs = top_k_renorm_probs(paddle.to_tensor(probs), top_k_tensor)[0].numpy()
        renorm_probs = renorm_probs.reshape(1, -1)
        np.testing.assert_allclose(renorm_probs, probs, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
