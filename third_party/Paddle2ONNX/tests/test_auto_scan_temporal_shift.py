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
        x = paddle.nn.functional.temporal_shift(
            inputs,
            seg_num=self.config["seg_num"],
            shift_ratio=self.config["shift_ratio"])
        return x


class TestTemporal_shiftConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.temporal_shift
    OPset version: 7, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=30), min_size=4, max_size=4))

        seg_num = draw(st.integers(min_value=1, max_value=10))

        batch = draw(st.integers(min_value=1, max_value=5))

        input_shape[0] = batch * seg_num

        shift_ratio = draw(st.floats(min_value=0.01, max_value=0.49))

        dtype = draw(st.sampled_from(["float32", "float64"]))

        config = {
            "op_names": ["temporal_shift"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 15],
            "input_spec_shape": [],
            "seg_num": seg_num,
            "shift_ratio": shift_ratio
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=50)


if __name__ == "__main__":
    unittest.main()
