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

    def forward(self, input, k):
        """
        forward
        """
        x = paddle.fluid.layers.topk(input, k=k)
        return x


class TestTopkConvert(OPConvertAutoScanTest):
    """
    api: paddle.fluid.layers.topk
    OPset version: 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=10), min_size=1, max_size=5))

        dtype = draw(st.sampled_from(["float32", "float64"]))
        k = random.randint(1, min(input_shape))
        isTensor = draw(st.booleans())

        def generator_k():
            input_data = np.array([k])
            return input_data

        config = {
            "op_names": ["top_k"],
            "test_data_shapes": [input_shape, generator_k],
            "test_data_types": [[dtype], ["int32"]],
            "opset_version": [11, 15],
            "input_spec_shape": [],
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class Net1(BaseNet):
    """
    simple Net
    """

    def forward(self, input):
        """
        forward
        """
        x = paddle.fluid.layers.topk(input, k=self.config['k'])
        return x


class TestTopkConvert1(OPConvertAutoScanTest):
    """
    api: paddle.fluid.layers.topk
    OPset version: 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=10), min_size=1, max_size=5))

        dtype = draw(st.sampled_from(["float32", "float64"]))
        k = random.randint(1, min(input_shape))

        config = {
            "op_names": ["top_k"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [11, 15],
            "input_spec_shape": [],
            "k": k,
        }

        models = Net1(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
