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

    def forward(self, x, y):
        """
        forward
        """
        x = paddle.dist(x, y, p=self.config["p"])
        return x


class TestDistConvert(OPConvertAutoScanTest):
    """
    api: paddle.dist
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input1_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=3, max_size=3))

        input2_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=2, max_size=2))

        input2_shape[0] = input1_shape[1]
        input2_shape[1] = input1_shape[2]

        p = 0.0
        p_type = draw(st.sampled_from(["str", "float"]))
        if p_type == "str":
            p = draw(st.sampled_from(["inf", "-inf"]))
            p = float(p)
        elif p_type == "float":
            p = draw(st.floats(min_value=0, max_value=4.0))

        dtype = draw(st.sampled_from(["float32", "float64"]))
        opset_version = [7, 9, 15]
        if p == 0.0:
            opset_version = [9, 15]
        config = {
            "op_names": ["dist"],
            "test_data_shapes": [input1_shape, input2_shape],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "p": p,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
