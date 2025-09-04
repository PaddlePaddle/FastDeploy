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

from fastdeploy.model_executor.ops.gpu import speculate_get_seq_lens_output


class TestSpeculateGetSeqLensOutput(unittest.TestCase):

    def run_seq_lens(self, input_values):
        paddle.seed(42)
        np.random.seed(42)
        seq_lens_this_time = paddle.to_tensor(input_values[0], dtype="int32")
        seq_lens_encoder = paddle.to_tensor(input_values[1], dtype="int32")
        seq_lens_decoder = paddle.to_tensor(input_values[2], dtype="int32")
        seq_lens_output = speculate_get_seq_lens_output(seq_lens_this_time, seq_lens_encoder, seq_lens_decoder)[0]
        return seq_lens_output

    def test_speculate_get_seq_lens_output1(self):
        input_values = [[7], [0], [0]]
        output_value = 7
        result = self.run_seq_lens(input_values)
        np.testing.assert_allclose(result.numpy(), output_value)

    def test_speculate_get_seq_lens_output2(self):
        input_values = [[7], [1], [0]]
        output_value = 1
        result = self.run_seq_lens(input_values)
        np.testing.assert_allclose(result.numpy(), output_value)

    def test_speculate_get_seq_lens_output3(self):
        input_values = [[1], [1], [0]]
        output_value = 1
        result = self.run_seq_lens(input_values)
        np.testing.assert_allclose(result.numpy(), output_value)

    def test_speculate_get_seq_lens_output4(self):
        input_values = [[0], [1], [0]]
        output_value = 0
        result = self.run_seq_lens(input_values)
        np.testing.assert_allclose(result.numpy(), output_value)


if __name__ == "__main__":
    unittest.main()
