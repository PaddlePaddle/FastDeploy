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


class Net0(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.clip(inputs, min=self.config["min"], max=self.config["max"])
        return x


class Net1(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs, max_value):
        """
        forward
        """
        x = paddle.clip(inputs, min=self.config["min"], max=max_value)
        return x


class Net2(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs, min_value):
        """
        forward
        """
        x = paddle.clip(inputs, min=min_value, max=self.config["max"])
        return x


class Net3(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs, min_value, max_value):
        """
        forward
        """
        x = paddle.clip(inputs, min=min_value, max=max_value)
        return x


class Net4(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.clip(inputs)
        return x


class TestClipConvert0(OPConvertAutoScanTest):
    """
    api: paddle.clip
    OPset version: 7, 9, 13, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=1, max_size=4))

        input_spec = [-1] * len(input_shape)

        dtype = draw(st.sampled_from(["float32", "float64"]))

        min_num = draw(st.integers(min_value=-4.0, max_value=-1.0))
        max_num = draw(st.floats(min_value=0, max_value=4.0))

        models = list()
        config0 = {
            "op_names": ["clip"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 13, 15],
            "input_spec_shape": [],
            "min": min_num,
            "max": max_num,
        }

        models.append(Net0(config0))

        return (config0, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class TestClipConvert1(OPConvertAutoScanTest):
    """
    api: paddle.clip
    OPset version: 13, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=1, max_size=4))

        input_spec = [-1] * len(input_shape)

        dtype = draw(st.sampled_from(["float32", "float64"]))

        min_num = draw(st.floats(min_value=-4.0, max_value=-2.0))

        def generator_max():
            input_data = randtool("int", 0, 10, [1])
            return input_data

        models = list()
        config1 = {
            "op_names": ["clip"],
            "test_data_shapes": [input_shape, generator_max],
            "test_data_types": [[dtype], ["int32"]],
            "opset_version": [13, 15],
            "input_spec_shape": [],
            "min": min_num,
        }
        models.append(Net1(config1))

        return (config1, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class TestClipConvert2(OPConvertAutoScanTest):
    """
    api: paddle.clip
    OPset version: 13, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=1, max_size=4))

        input_spec = [-1] * len(input_shape)

        dtype = draw(st.sampled_from(["float32", "float64"]))

        max_num = draw(st.floats(min_value=2.0, max_value=4.0))

        models = list()
        config2 = {
            "op_names": ["clip"],
            "test_data_shapes": [input_shape, [1]],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": [13, 15],
            "input_spec_shape": [],
            "max": max_num,
        }
        models.append(Net2(config2))

        return (config2, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class TestClipConvert3(OPConvertAutoScanTest):
    """
    api: paddle.clip
    OPset version: 13, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=1, max_size=4))

        input_spec = [-1] * len(input_shape)

        dtype = draw(st.sampled_from(["float32", "float64"]))

        def generator_min():
            input_data = randtool("float", -10, -1, [1])
            return input_data

        def generator_max():
            input_data = randtool("int", 0, 10, [1])
            return input_data

        models = list()
        config3 = {
            "op_names": ["clip"],
            "test_data_shapes": [input_shape, generator_min, generator_max],
            "test_data_types": [[dtype], [dtype], [dtype]],
            "opset_version": [13, 15],
            "input_spec_shape": [],
        }
        models.append(Net3(config3))

        return (config3, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class TestClipConvert4(OPConvertAutoScanTest):
    """
    api: paddle.clip
    OPset version: 7, 9, 13, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=1, max_size=4))

        input_spec = [-1] * len(input_shape)

        dtype = draw(st.sampled_from(["float32", "float64"]))

        models = list()
        config0 = {
            "op_names": ["clip"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 13, 15],
            "input_spec_shape": [],
        }

        models.append(Net4(config0))

        return (config0, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
