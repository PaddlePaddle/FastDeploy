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

from fastdeploy.model_executor.layers.sample.logprobs import batched_count_greater_than


class TestBatchedCountGreaterThan(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def naive_impl(self, x, y):
        return (x >= y).sum(-1)

    def test_batched_count_greater_than(self):
        vocab_size_list = [151552, 566]
        test_token_nums = [1, 32, 128, 1024, 8192]
        for idx, num_tokens in enumerate(test_token_nums):
            for vocab_size in vocab_size_list:
                x = paddle.randn([num_tokens, vocab_size], dtype="float32")
                y = paddle.randn([num_tokens, 1], dtype="float32")
                x[0, 0] = -float("inf")
                y[0, 0] = -float("inf")
                out = self.naive_impl(x, y)
                out_triton = batched_count_greater_than(x, y)
                self.assertTrue(np.allclose(out.numpy(), out_triton.numpy()))

        return out


if __name__ == "__main__":
    unittest.main()
