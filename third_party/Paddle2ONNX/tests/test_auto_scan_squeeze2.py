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
        if self.config["tensor_attr"] and self.config['axis'] is not None:
            if isinstance(self.config['axis'], list):
                axis = [paddle.to_tensor(i) for i in self.config['axis']]
            else:
                axis = paddle.to_tensor(self.config['axis'])
        else:
            axis = self.config['axis']
        x = paddle.squeeze(inputs, axis=axis)
        return x


class TestSqueezeConvert(OPConvertAutoScanTest):
    """
    api: paddle.squeeze
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=10), min_size=3, max_size=5))

        dtype = draw(
            st.sampled_from(["bool", "float32", "float64", "int32", "int64"]))
        axis = None
        axis_dtype = draw(st.sampled_from(["None", "int", "list"]))
        if axis_dtype == "list":
            axis = draw(
                st.integers(
                    min_value=-len(input_shape), max_value=len(input_shape) -
                    1))
            if axis == 0:
                axis = [0, -1]
            else:
                axis = [0, axis]
            input_shape[axis[0]] = 1
            input_shape[axis[1]] = 1
        elif axis_dtype == "int":
            axis = draw(
                st.integers(
                    min_value=-len(input_shape), max_value=len(input_shape) -
                    1))
            input_shape[axis] = 1
        else:
            input_shape[0] = 1

        tensor_attr = draw(st.booleans())

        if draw(st.booleans()):
            input_spec_shape = []
        else:
            input_spec_shape = [len(input_shape) * [-1]]

        config = {
            "op_names": ["squeeze2"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": input_spec_shape,
            "axis": axis,
            "tensor_attr": tensor_attr
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
