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


class Net0(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.reshape(inputs, [1, -1])
        return x


class Net1(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs, shape):
        """
        forward
        """
        x = paddle.reshape(inputs, shape)
        return x


class TestReshapeConvert0(OPConvertAutoScanTest):
    """
    api: paddle.reshape
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=20), min_size=1, max_size=4))

        dtype = draw(st.sampled_from(["float32"]))

        config = {
            "op_names": ["reshape2"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
        }

        models = Net0(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class TestReshapeConvert1(OPConvertAutoScanTest):
    """
    api: paddle.reshape
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=20), min_size=1, max_size=4))

        dtype = draw(st.sampled_from(["float32"]))

        def generator_shape():
            shape_list = input_shape
            shape_list[-1] = -1
            return np.array(shape_list)

        config = {
            "op_names": ["reshape2"],
            "test_data_shapes": [input_shape, generator_shape],
            "test_data_types": [[dtype], ["int32"]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
        }

        models = Net1(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
