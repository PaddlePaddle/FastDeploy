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
import random


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, x):
        """
        forward
        """
        scale = self.config["scale"]
        if self.config['isTensor']:
            scale = paddle.to_tensor(scale)
        x = paddle.scale(
            x,
            scale=scale,
            bias=self.config["bias"],
            bias_after_scale=self.config["bias_after_scale"])
        return x


class TestScaleConvert(OPConvertAutoScanTest):
    """
    api: paddle.scale
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=20), min_size=2, max_size=5))
        # int32, int64 has a bug
        dtype = draw(st.sampled_from(["float32", "float64"]))

        scale = draw(st.floats(min_value=-20, max_value=20))
        isTensor = draw(st.booleans())

        bias = draw(st.floats(min_value=-20, max_value=20))
        bias_after_scale = draw(st.booleans())

        config = {
            "op_names": ["scale"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "scale": scale,
            "bias": bias,
            "bias_after_scale": bias_after_scale,
            "isTensor": isTensor,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
