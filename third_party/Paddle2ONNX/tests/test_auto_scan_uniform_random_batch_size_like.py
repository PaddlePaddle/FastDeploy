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
from onnxbase import randtool
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
        x = paddle.fluid.layers.uniform_random_batch_size_like(
            inputs,
            shape=self.config["shape"],
            min=self.config["min"],
            max=self.config["max"],
            dtype=self.config["out_dtype"])
        return x


class TestUniformRandomBaTchSizeLikeConvert(OPConvertAutoScanTest):
    """
    api: paddle.fluid.layers.uniform_random_batch_size_like
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=9), min_size=1, max_size=1))

        min = draw(st.floats(min_value=0, max_value=1.0))

        max = draw(st.floats(min_value=1.0, max_value=2.0))

        dtype = draw(st.sampled_from(["float32", "float64"]))
        out_dtype = draw(st.sampled_from(["float32", "float64"]))

        config = {
            "op_names": ["uniform_random_batch_size_like"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "shape": input_shape,
            "min": min,
            "max": max,
            "out_dtype": out_dtype,
            "delta": 1e11,
            "rtol": 1e11
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
