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
        if self.config["tensor_attr"]:
            axis = paddle.to_tensor(
                self.config["axis"], dtype=self.config["axis_dtype"])
        else:
            axis = self.config["axis"]
        x = paddle.cumsum(inputs, axis=axis, dtype=self.config["dtype"])
        return x


class TestCumsumConvert(OPConvertAutoScanTest):
    """
    api: paddle.cumsum
    OPset version: 11, 14, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=1, max_size=4))

        axis = draw(
            st.integers(
                min_value=-len(input_shape), max_value=len(input_shape) - 1))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))

        axis_dtype = draw(st.sampled_from(["int32", "int64"]))

        tensor_attr = draw(st.booleans())

        config = {
            "op_names": ["cumsum"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [11, 15],
            "input_spec_shape": [],
            "axis": axis,
            "dtype": dtype,
            "axis_dtype": axis_dtype,
            "use_gpu": False,
            "tensor_attr": tensor_attr
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
