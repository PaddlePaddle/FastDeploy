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


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        repeat_times = self.config['repeat_times']
        if self.config['repeat_times_dtype'] == "list":
            repeat_times = repeat_times
        elif self.config['repeat_times_dtype'] == "Tensor":
            repeat_times = paddle.to_tensor(
                np.array(repeat_times).astype(self.config['shape_dtype']))
        elif self.config['repeat_times_dtype'] == "int":
            repeat_times = [repeat_times[0]]
        x = paddle.tile(inputs, repeat_times=repeat_times)
        return x


class TestTileConvert(OPConvertAutoScanTest):
    """
    api: paddle.tile
    OPset version: 7, 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=5), min_size=2, max_size=5))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        # when repeat_times_dtype is tensor has a bug
        repeat_times_dtype = draw(st.sampled_from(["list", "Tensor", "int"]))
        shape_dtype = draw(st.sampled_from(["int32", "int64"]))

        config = {
            "op_names": ["tile"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 11, 15],
            "input_spec_shape": [],
            "repeat_times_dtype": repeat_times_dtype,
            "repeat_times": input_shape,
            "shape_dtype": shape_dtype,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class Net1(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        repeat_times = [
            4, paddle.to_tensor(
                3, dtype=self.config['shape_dtype']), 2, 1
        ]
        # repeat_times = [4, 3, 2, 1]
        # repeat_times = paddle.to_tensor(
        #                     np.array([4, 3, 2, 1]).astype('int32'))
        # not work
        # repeat_times = [4, 3, paddle.to_tensor(np.array(2).astype("int64")), 1]
        x = paddle.tile(inputs, repeat_times=repeat_times)
        return x


class TestTileConvert1(OPConvertAutoScanTest):
    """
    api: paddle.tile
    OPset version: 7, 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=5), min_size=2, max_size=5))
        input_shape = [4, 3, 2, 1]
        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        shape_dtype = draw(st.sampled_from(["int32", "int64"]))

        # when repeat_times_dtype is tensor has a bug
        repeat_times_dtype = draw(st.sampled_from(["list", "Tensor"]))
        config = {
            "op_names": ["tile"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 11, 15],
            "input_spec_shape": [],
            "repeat_times_dtype": repeat_times_dtype,
            "repeat_times": input_shape,
            "shape_dtype": shape_dtype,
        }

        models = Net1(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
