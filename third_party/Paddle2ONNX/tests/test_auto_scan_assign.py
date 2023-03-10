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
        np.random.seed(13)
        # float64 has a bug
        x1 = np.random.random(self.config['input_shape']).astype("float32")
        if self.config['input_dtype'] == "ndarray":
            x = x1
        elif self.config['input_dtype'] == "list":
            x = x1.tolist()
        elif self.config['input_dtype'] == "tensor":
            x = paddle.to_tensor(x1)
        x = paddle.assign(x)
        return x + inputs


class TestAssignConvert(OPConvertAutoScanTest):
    """
    api: paddle.assign
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=8), min_size=2, max_size=5))

        dtype = draw(
            st.sampled_from(
                ["float16", "float32", "float64", "int32", "int64"]))
        # "list" has a bug
        input_dtype = draw(st.sampled_from(["tensor", "ndarray"]))

        config = {
            "op_names": ["assign_value"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "dtype": dtype,
            "input_dtype": input_dtype,
            "input_shape": input_shape,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
