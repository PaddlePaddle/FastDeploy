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
        x = paddle.fluid.layers.matmul(
            x,
            y,
            transpose_x=self.config["transpose_x"],
            transpose_y=self.config["transpose_y"])
        return x


class TestMatmulConvert(OPConvertAutoScanTest):
    """
    api: paddle.fluid.layers.matmul
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape1 = draw(
            st.lists(
                st.integers(
                    min_value=5, max_value=20), min_size=3, max_size=5))
        # broadcast
        input_shape2 = input_shape1[-2:]
        input_shape2.reverse()
        dtype = draw(st.sampled_from(["float32", "float64"]))
        transpose = draw(st.booleans())

        config = {
            "op_names": ["matmul"],
            "test_data_shapes": [input_shape1, input_shape2],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "transpose_x": transpose,
            "transpose_y": transpose,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
