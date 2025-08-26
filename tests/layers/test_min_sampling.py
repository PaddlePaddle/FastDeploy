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
import paddle.nn.functional as F

from fastdeploy.model_executor.ops.gpu import min_p_sampling


class TestMinPSampling(unittest.TestCase):
    def setUp(self):
        self.sample_time = 1000000
        self.vocab_size = 1000
        self.min_p_value = 0.5
        self.batch_size = 3
        self.batch_min_p_values = [0.1, 0.0, 0.9]
        self.additional_batch_min_p_values = [0.1, 0.0, 0.3]

    # min_p:0.5：FastDeploy
    def min_p_sampling_cpu(self, min_p):
        logits = paddle.ones(shape=[1, self.vocab_size], dtype="float32")
        logits[0][0] = 10
        logits[0][1] = 8
        low_prob_tensor = paddle.linspace(2.0, 0.0, self.vocab_size - 2)
        logits[0][2:] = low_prob_tensor

        probs = F.softmax(logits)
        max_probabilities = paddle.amax(probs, axis=-1, keepdim=True)
        adjusted_min_p = max_probabilities * min_p.reshape([-1, 1])
        invalid_token_mask = probs < adjusted_min_p
        probs = paddle.where(invalid_token_mask, paddle.full_like(probs, 0.0), probs)
        return probs

    # min_p:0.5：FastDeploy
    def fastdeploy_min_p_sampling(self, min_p):
        logits = paddle.ones(shape=[1, self.vocab_size], dtype="float32")
        logits[0][0] = 10
        logits[0][1] = 8
        low_prob_tensor = paddle.linspace(2.0, 0.0, self.vocab_size - 2)
        logits[0][2:] = low_prob_tensor

        probs = F.softmax(logits)
        probs = min_p_sampling(probs, min_p)
        return probs

    # batch:[0.1.0.0,0.9]：FastDeploy
    def fastdeploy_batch_min_p_sampling(self, batch_size, min_p_values):
        logits = paddle.ones(shape=[batch_size, self.vocab_size], dtype="float32")
        for b in range(batch_size):
            logits[b][0] = 10
            logits[b][1] = 8
            logits[b][2:] = paddle.linspace(2.0, 0.0, self.vocab_size - 2)

        probs = F.softmax(logits, axis=-1)
        min_p_arr = paddle.to_tensor(min_p_values, dtype="float32")

        probs = min_p_sampling(probs, min_p_arr)

        return probs

    def compare_results(self, probs, probs_cpu, atol=1e-6, rtol=1e-6):
        probs_np = probs.numpy()
        probs_cpu_np = probs_cpu.numpy()
        try:
            np.testing.assert_allclose(
                probs_np,
                probs_cpu_np,
                rtol=rtol,
                atol=atol,
            )
            print("The results are same between fastdeploy_min_p_sampling and min_p_sampling_cpu")
        except AssertionError as e:
            raise AssertionError(
                f"The results are different between fastdeploy_min_p_sampling and min_p_sampling_cpu:\n{str(e)}"
            )

    def test_single_min_p_sampling(self):
        min_p = paddle.to_tensor([self.min_p_value], dtype="float32")
        probs = self.fastdeploy_min_p_sampling(min_p)
        probs_cpu = self.min_p_sampling_cpu(min_p)
        self.compare_results(probs, probs_cpu)

    def test_batch_min_p_sampling(self):
        batch_min_p = paddle.to_tensor(self.batch_min_p_values, dtype="float32")
        batch_probs = self.fastdeploy_batch_min_p_sampling(self.batch_size, batch_min_p)
        batch_probs_cpu = self.min_p_sampling_cpu(batch_min_p)
        self.compare_results(batch_probs, batch_probs_cpu)

    def test_additional_batch_min_p_sampling(self):
        additional_batch_min_p = paddle.to_tensor(self.additional_batch_min_p_values, dtype="float32")
        additional_batch_probs = self.fastdeploy_batch_min_p_sampling(self.batch_size, additional_batch_min_p)
        additional_batch_probs_cpu = self.min_p_sampling_cpu(additional_batch_min_p)
        self.compare_results(additional_batch_probs, additional_batch_probs_cpu)


if __name__ == "__main__":
    if paddle.is_compiled_with_cuda():
        unittest.main()
