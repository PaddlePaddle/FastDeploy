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

    def forward(self):
        """
        forward
        """
        num_rows = self.config["num_rows"]
        num_columns = self.config["num_columns"]
        if self.config["tensor_attr"]:
            num_rows = paddle.assign(self.config["num_rows"])
            if self.config["num_columns"] is not None:
                num_columns = paddle.assign(self.config["num_columns"])
        dtype = self.config["dtype"]
        x = paddle.eye(num_rows, num_columns=num_columns, dtype=dtype)
        return x


class TestEyeConvert(OPConvertAutoScanTest):
    """
    api: paddle.eye
    OPset version: 9, 13, 15
    """

    def sample_convert_config(self, draw):
        num_rows = draw(st.integers(min_value=5, max_value=20))

        num_columns = None
        if draw(st.booleans()):
            num_columns = draw(st.integers(min_value=5, max_value=20))
        dtype = None
        if draw(st.booleans()):
            dtype = draw(
                st.sampled_from(["float32", "float64", "int32", "int64"]))

        tensor_attr = draw(st.booleans())

        config = {
            "op_names": ["eye"],
            "test_data_shapes": [],
            "test_data_types": [],
            "opset_version": [9, 13, 15],
            "input_spec_shape": [],
            "num_rows": num_rows,
            "num_columns": num_columns,
            "dtype": dtype,
            "tensor_attr": tensor_attr
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
