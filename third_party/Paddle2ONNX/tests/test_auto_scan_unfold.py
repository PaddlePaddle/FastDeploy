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
        x = paddle.nn.functional.unfold(
            inputs,
            self.config["kernel_size"],
            strides=self.config["strides"],
            paddings=self.config["paddings"],
            dilations=self.config["dilations"])
        return x


class TestUnfoldConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.unfold
    OPset version: 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=20, max_value=30), min_size=4, max_size=4))

        kernel_size = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=5), min_size=1, max_size=2))
        if len(kernel_size) == 1:
            kernel_size = kernel_size[0]

        strides = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=5), min_size=1, max_size=2))
        if len(strides) == 1:
            strides = strides[0]

        if draw(st.booleans()):
            paddings = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=5),
                    min_size=1,
                    max_size=2))
            if len(paddings) == 1:
                paddings = paddings[0]
        else:
            paddings = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=5),
                    min_size=4,
                    max_size=4))

        dilations = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=3), min_size=1, max_size=2))

        if len(dilations) == 1:
            dilations = dilations[0]

        dtype = draw(st.sampled_from(["float32", "float64"]))

        input_spec_shape = []
        if draw(st.booleans()):
            input_spec_shape = [[-1, input_shape[1], -1, -1]]

        config = {
            "op_names": ["unfold"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [11, 12, 13, 14, 15],
            "input_spec_shape": input_spec_shape,
            "kernel_size": kernel_size,
            "strides": strides,
            "dilations": dilations,
            "paddings": paddings,
            "use_gpu": False,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=25)


if __name__ == "__main__":
    unittest.main()
