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


class NetAvgPool1d(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        output_size = self.config['output_size']
        x = paddle.nn.functional.adaptive_max_pool1d(
            inputs, output_size=output_size, return_mask=False)
        return x


class TestAdaptiveAvgPool1dConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.adaptive_avg_pool1d
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=12), min_size=3, max_size=3))

        if input_shape[2] % 2 != 0:
            input_shape[2] = input_shape[2] + 1

        dtype = draw(st.sampled_from(["float32", "float64"]))

        output_size = draw(st.integers(min_value=1, max_value=3))

        config = {
            "op_names": ["max_pool2d_with_index"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "output_size": output_size,
        }

        models = NetAvgPool1d(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


# TODO max_pool2d_with_index not supported yet
class NetAvgPool2d(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        output_size = self.config['output_size']
        x = paddle.nn.functional.adaptive_max_pool2d(
            inputs, output_size, return_mask=False)
        return x


class TestAdaptiveAvgPool2dConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.adaptive_avg_pool2d
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=12), min_size=4, max_size=4))

        if input_shape[2] % 2 != 0:
            input_shape[2] = input_shape[2] + 1
        if input_shape[3] % 2 != 0:
            input_shape[3] = input_shape[3] + 1
        input_shape = [10, 10, 10, 10]
        dtype = draw(st.sampled_from(["float32", "float64"]))
        data_format = draw(st.sampled_from(["NCHW"]))

        output_type = draw(st.sampled_from(["int", "list"]))
        if output_type == "int":
            output_size = draw(st.integers(min_value=1, max_value=3))
        elif output_type == "list":
            output_size = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=3),
                    min_size=2,
                    max_size=2))
        config = {
            "op_names": ["max_pool2d_with_index"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "output_size": output_size,
            "data_format": data_format,
        }

        models = NetAvgPool2d(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


# pool3d not supported yet
class NetAvgPool3d(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        output_size = self.config['output_size']
        x = paddle.nn.functional.adaptive_max_pool3d(
            inputs, output_size=output_size, return_mask=False)
        return x


class TestAdaptiveAvgPool3dConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.adaptive_avg_pool3d
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=12), min_size=5, max_size=5))

        if input_shape[2] % 2 != 0:
            input_shape[2] = input_shape[2] + 1
        if input_shape[3] % 2 != 0:
            input_shape[3] = input_shape[3] + 1
        if input_shape[4] % 2 != 0:
            input_shape[4] = input_shape[4] + 1

        dtype = draw(st.sampled_from(["float32", "float64"]))
        data_format = draw(st.sampled_from(["NCDHW"]))

        output_type = draw(st.sampled_from(["int", "list"]))
        if output_type == "int":
            output_size = draw(st.integers(min_value=1, max_value=3))
        elif output_type == "list":
            output_size = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=3),
                    min_size=3,
                    max_size=3))

        config = {
            "op_names": ["max_pool3d_with_index"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "output_size": output_size,
            "data_format": data_format,
        }

        models = NetAvgPool3d(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
