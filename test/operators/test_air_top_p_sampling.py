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
"""UT for air_top_p_sampling kernel"""

import subprocess
import unittest

import numpy as np
import paddle

import fastdeploy.model_executor.ops.gpu


class Test(unittest.TestCase):
    def setUp(self):
        """
        Initialize.
        """
        paddle.seed(2024)
        np.random.seed(42)
        print(paddle.device.cuda.get_device_properties())
        print(paddle.__git_commit__)
        nvcc_output = subprocess.check_output(["nvcc", "--version"], universal_newlines=True)
        output = nvcc_output.split()
        release_idx = output.index("release") + 1
        self.nvcc_cuda_version = float(output[release_idx].split(",")[0])

    def test_air_top_p_sampling(self):
        """
        Check air_top_p_sampling output with paddle.tensor.top_p_sampling.
        """
        if self.nvcc_cuda_version < 12.0:
            self.skipTest("air_top_p_sampling only support cu12+")
        bsz = 8
        vocab_size = 103424
        x = paddle.randn([bsz, vocab_size])
        x = paddle.nn.functional.softmax(x)
        x = paddle.cast(x, "float32")
        top_ps = paddle.to_tensor(np.random.uniform(0, 1, [bsz]).astype(np.float32))
        _, next_tokens = fastdeploy.model_executor.ops.gpu.air_top_p_sampling(
            x.cuda(), top_ps.cuda(), None, None, seed=0, k=1, mode="truncated"
        )
        print(next_tokens)
        less_than_zero = next_tokens >= 0
        greater_than_vocab_size = next_tokens <= vocab_size
        accuracy = paddle.logical_and(less_than_zero, greater_than_vocab_size)
        print(f"Accuracy of results: {accuracy}")


if __name__ == "__main__":
    unittest.main()
