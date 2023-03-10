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
        axis = self.config['axis']
        if self.config['isTensor']:
            axis = paddle.to_tensor(axis)
            if self.config['isTensor13']:
                axis = axis * 1
        x = paddle.unsqueeze(inputs, axis=axis)
        return x


class TestUnsqueezeConvert(OPConvertAutoScanTest):
    """
    api: paddle.unsqueeze
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=6), min_size=2, max_size=5))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        axis = draw(
            st.integers(
                min_value=-len(input_shape), max_value=len(input_shape) - 1))
        isTensor = draw(st.booleans())

        axis_dtype = draw(st.sampled_from(["int", "list"]))
        if axis_dtype == "list":
            if len(input_shape) == 5:
                axis = [0]
            if len(input_shape) == 4:
                axis = [0, 1]
            if len(input_shape) == 3:
                axis = [1, 2, 3]
            if len(input_shape) == 2:
                if draw(st.booleans()):
                    axis = [0, 1, 2]
                else:
                    axis = [1, 3]
        isTensor13 = draw(st.booleans())
        opset_version = [7, 9, 10, 11, 12, 13, 14, 15]
        if isTensor13 or isTensor:
            opset_version = [13, 14, 15]
        config = {
            "op_names": ["unsqueeze2"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "axis": axis,
            "isTensor": isTensor,
            "isTensor13": isTensor13,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
