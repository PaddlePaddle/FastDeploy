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
        shifts = self.config['shifts']
        # axis = [0, -1]
        # TODO not work
        # shifts = [paddle.to_tensor(-2), -2]
        if self.config['is_shifts_tensor']:
            shifts = paddle.to_tensor(shifts).astype(self.config['shift_dtype'])
        x = paddle.roll(inputs, shifts=shifts, axis=axis)
        return x


class TestRollConvert(OPConvertAutoScanTest):
    """
    api: paddle.roll
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=2, max_size=5))

        dtype = draw(st.sampled_from(["float32"]))
        axis_dtype = draw(st.sampled_from(["None", "int", "list"]))
        shift_dtype = draw(st.sampled_from(["int32", "int64"]))
        if axis_dtype == "int":
            axis = draw(
                st.integers(
                    min_value=-len(input_shape), max_value=len(input_shape) -
                    1))
            axis_idx = axis + len(input_shape) if axis < 0 else axis
            shifts = draw(
                st.integers(
                    min_value=-input_shape[axis_idx],
                    max_value=-input_shape[axis_idx]))
        elif axis_dtype == "list":
            axis = [0, -1]
            axis_idx = [
                axis + len(input_shape) if axis < 0 else axis
                for i, axis in enumerate(axis)
            ]
            shifts = []
            sf0 = draw(
                st.integers(
                    min_value=-input_shape[axis_idx[0]],
                    max_value=-input_shape[axis_idx[0]]))
            sf1 = draw(
                st.integers(
                    min_value=-input_shape[axis_idx[1]],
                    max_value=-input_shape[axis_idx[1]]))
            shifts.append(sf0)
            shifts.append(sf1)
        else:
            axis = None
            shifts = draw(
                st.integers(
                    min_value=-input_shape[0], max_value=-input_shape[0]))

        is_shifts_tensor = draw(st.booleans())

        if is_shifts_tensor:
            opset_version = [10, 11, 12, 13, 14, 15]
        else:
            opset_version = [7, 8, 9, 10, 11, 12, 13, 14, 15]
        config = {
            "op_names": ["roll"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "axis": axis,
            "shifts": shifts,
            "is_shifts_tensor": is_shifts_tensor,
            "shift_dtype": shift_dtype,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=80)


if __name__ == "__main__":
    unittest.main()
