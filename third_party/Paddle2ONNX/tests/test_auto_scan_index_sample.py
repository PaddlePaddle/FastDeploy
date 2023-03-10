# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from auto_scan_test import OPConvertAutoScanTest, BaseNet
from hypothesis import reproduce_failure
import hypothesis.strategies as st
import numpy as np
import unittest
import paddle
from onnxbase import randtool


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, input, index):
        """
        forward
        """
        x = paddle.index_sample(input, index=index)
        return x


class TestIndexSampleConvert(OPConvertAutoScanTest):
    """
    api: paddle.index_sample
    OPset version: 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=20), min_size=2, max_size=2))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))

        index_dtype = draw(st.sampled_from(["int32", "int64"]))

        def generator_data():
            import random
            n = random.randint(1, input_shape[1])
            input_data = randtool("int", 0.0, input_shape[1] - 1,
                                  [input_shape[0], n])
            return input_data

        config = {
            "op_names": ["index_sample"],
            "test_data_shapes": [input_shape, generator_data],
            "test_data_types": [[dtype], [index_dtype]],
            "opset_version": [11, 15],
            "input_spec_shape": [],
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
