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
from onnxbase import randtool
import numpy as np
import unittest
import paddle


class Net(BaseNet):
    def forward(self):
        start = self.config["start"]
        stop = self.config["stop"]
        num = self.config["num"]
        dtype = self.config["dtype"]
        x = paddle.linspace(start=start, stop=stop, num=num, dtype=dtype)
        return x


class TestLinspaceConvert(OPConvertAutoScanTest):
    """
    api: linspace
    OPset version:
    """

    def sample_convert_config(self, draw):
        start = draw(st.integers(min_value=1, max_value=10))

        stop = draw(st.integers(min_value=20, max_value=30))

        num = draw(st.integers(min_value=2, max_value=40))
        if draw(st.booleans()):
            dtype = draw(
                st.sampled_from(["float32", "float64", "int32", "int64"]))
        else:
            dtype = None

        config = {
            "op_names": ["linspace"],
            "test_data_shapes": [],
            "test_data_types": [],
            "opset_version": [9, 15],
            "input_spec_shape": [],
            "start": start,
            "stop": stop,
            "num": num,
            "dtype": dtype
        }

        model = Net(config)

        return (config, model)

    def test(self):
        self.run_and_statis(max_examples=30, max_duration=-1)


if __name__ == "__main__":
    unittest.main()
