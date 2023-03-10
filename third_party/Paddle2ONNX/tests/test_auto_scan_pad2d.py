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
    def forward(self, inputs):
        paddings = self.config["paddings"]
        mode = self.config["mode"]
        pad_value = self.config["pad_value"]
        data_format = self.config["data_format"]
        x = paddle.fluid.layers.pad2d(
            inputs,
            paddings=paddings,
            mode=mode,
            pad_value=pad_value,
            data_format=data_format)
        return x


class TestPadopsConvert(OPConvertAutoScanTest):
    """
    api: pad2d
    OPset version:
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=4, max_size=4))

        dtype = "float32"

        paddings = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=4), min_size=4, max_size=4))

        mode = draw(st.sampled_from(["constant", "reflect", "edge"]))

        pad_value = draw(st.floats(min_value=10, max_value=20))

        data_format = draw(st.sampled_from(["NCHW", "NHWC"]))

        config = {
            "op_names": ["pad2d"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 11, 15],
            "input_spec_shape": [],
            "mode": mode,
            "pad_value": pad_value,
            "paddings": paddings,
            "data_format": data_format
        }

        model = Net(config)

        return (config, model)

    def test(self):
        self.run_and_statis(max_examples=30, max_duration=-1)


class Net2(BaseNet):
    def forward(self, inputs, padding):
        mode = self.config["mode"]
        pad_value = self.config["pad_value"]
        data_format = self.config["data_format"]
        x = paddle.fluid.layers.pad2d(
            inputs,
            padding,
            mode=mode,
            pad_value=pad_value,
            data_format=data_format)
        return x


class TestPadopsConvert_Paddingtensor(OPConvertAutoScanTest):
    """
    api: pad2d
    OPset version:
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=4, max_size=4))

        dtype = "float32"

        paddings = [4]

        mode = draw(st.sampled_from(["constant", "reflect", "edge"]))

        pad_value = draw(st.floats(min_value=10, max_value=20))

        data_format = draw(st.sampled_from(["NCHW", "NHWC"]))

        def generator_data():
            input_data = randtool("int", 1, 10, paddings)
            return input_data

        config = {
            "op_names": ["pad2d"],
            "test_data_shapes": [input_shape, generator_data],
            "test_data_types": [[dtype], ["int32"]],
            "opset_version": [11, 15],
            "input_spec_shape": [],
            "mode": mode,
            "pad_value": pad_value,
            "paddings": paddings,
            "data_format": data_format
        }

        model = Net2(config)

        return (config, model)

    def test(self):
        self.run_and_statis(max_examples=30, max_duration=-1)


class Net3(BaseNet):
    def forward(self, inputs):
        data = np.ones(shape=[4], dtype="int32")
        padding = paddle.to_tensor(data, dtype='int32')
        mode = self.config["mode"]
        pad_value = self.config["pad_value"]
        data_format = self.config["data_format"]
        x = paddle.fluid.layers.pad2d(
            inputs,
            padding,
            mode=mode,
            pad_value=pad_value,
            data_format=data_format)
        return x


class TestPadopsConvert_Constanttensor(OPConvertAutoScanTest):
    """
    api: pad2d
    OPset version:
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=4, max_size=4))

        dtype = "float32"

        paddings = [4]

        mode = draw(st.sampled_from(["constant", "reflect", "edge"]))

        pad_value = draw(st.floats(min_value=10, max_value=20))

        data_format = draw(st.sampled_from(["NCHW", "NHWC"]))

        config = {
            "op_names": ["pad2d"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [9, 11],
            "input_spec_shape": [],
            "mode": mode,
            "pad_value": pad_value,
            "paddings": paddings,
            "data_format": data_format
        }

        model = Net3(config)

        return (config, model)

    def test(self):
        self.run_and_statis(max_examples=25, max_duration=-1)


if __name__ == "__main__":
    unittest.main()
