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
import random


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs1, inputs2):
        """
        forward
        """
        x = paddle.expand_as(inputs1, inputs2)
        return x + inputs2


class TestStackConvert(OPConvertAutoScanTest):
    """
    api: paddle.expand_as
    OPset version: 8, 9, 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape1 = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=8), min_size=1, max_size=5))

        n = random.randint(1, 6 - len(input_shape1))
        pre_shape = random.sample([1, 1, 2, 2, 3, 3], n)
        input_shape2 = pre_shape + input_shape1
        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))

        config = {
            "op_names": ["expand_as_v2"],
            "test_data_shapes": [input_shape1, input_shape2],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": [8, 9, 10, 11, 12, 13, 14, 15],
            "input_spec_shape": [],
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
