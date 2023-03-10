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
from onnxbase import randtool
import numpy as np
import unittest
import paddle
from random import sample


class Net0(BaseNet):
    """
    simple Net
    """

    def forward(self, x, index):
        """
        forward
        """
        x = paddle.gather(x, index, axis=self.config["axis"])
        return x


class Net1(BaseNet):
    """
    simple Net
    """

    def forward(self, x, index):
        """
        forward
        """
        axis = self.config["axis"]
        axis = paddle.to_tensor(axis, dtype="int64")
        x = paddle.gather(x, index, axis=axis)
        shape = paddle.shape(x)
        x = paddle.reshape(x, shape)
        return x


class TestGatherConvert0(OPConvertAutoScanTest):
    """
    api: paddle.gather
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=20), min_size=1, max_size=4))

        dtype = draw(st.sampled_from(["int32", "int64", "float32", "float64"]))
        index_dtype = draw(st.sampled_from(["int32", "int64"]))

        axis = draw(st.integers(min_value=0, max_value=len(input_shape) - 1))

        def generator_index():
            index_list = [i for i in range(input_shape[axis])]
            index_select = sample(index_list, 2)
            return np.array(index_select)

        config = {
            "op_names": ["gather"],
            "test_data_shapes": [input_shape, generator_index],
            "test_data_types": [[dtype], [index_dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "axis": axis,
        }

        models = Net0(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class TestGatherConvert1(OPConvertAutoScanTest):
    """
    api: paddle.gather
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=20), min_size=1, max_size=4))

        dtype = draw(st.sampled_from(["int32", "int64", "float32", "float64"]))
        index_dtype = draw(st.sampled_from(["int32", "int64"]))

        axis = draw(st.integers(min_value=0, max_value=len(input_shape) - 1))

        def generator_index():
            index_list = [i for i in range(input_shape[axis])]
            index_select = sample(index_list, 2)
            return np.array(index_select)

        config = {
            "op_names": ["gather"],
            "test_data_shapes": [input_shape, generator_index],
            "test_data_types": [[dtype], [index_dtype]],
            "opset_version": [7, 9],
            "input_spec_shape": [],
            "axis": axis,
        }

        models = Net1(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
