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

    def forward(self, inputs1, inputs2):
        """
        forward
        """
        x = paddle.meshgrid([inputs1, inputs2])
        return x


class TestMeshgridConvert(OPConvertAutoScanTest):
    """
    api: paddle.meshgrid
    OPset version: 8, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape1 = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=8), min_size=1, max_size=1))

        input_shape2 = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=8), min_size=1, max_size=1))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))

        config = {
            "op_names": ["meshgrid"],
            "test_data_shapes": [input_shape1, input_shape2],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": [8, 9, 15],
            "input_spec_shape": [],
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class Net1(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs1, inputs2, inputs3):
        """
        forward
        """
        x = paddle.meshgrid([inputs1, inputs2, inputs3])
        return x


class TestMeshgridConvert1(OPConvertAutoScanTest):
    """
    api: paddle.meshgrid
    OPset version: 8, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape1 = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=8), min_size=1, max_size=1))

        input_shape2 = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=8), min_size=1, max_size=1))

        input_shape3 = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=8), min_size=1, max_size=1))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))

        config = {
            "op_names": ["meshgrid"],
            "test_data_shapes": [input_shape1, input_shape2, input_shape3],
            "test_data_types": [[dtype], [dtype], [dtype]],
            "opset_version": [8, 9, 15],
            "input_spec_shape": [],
        }

        models = Net1(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class Net2(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs1, inputs2, inputs3, inputs4, inputs5):
        """
        forward
        """
        x = paddle.meshgrid([inputs1, inputs2, inputs3, inputs4, inputs5])
        return x


class TestMeshgridConvert2(OPConvertAutoScanTest):
    """
    api: paddle.meshgrid
    OPset version: 8, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape1 = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=8), min_size=1, max_size=1))

        input_shape2 = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=8), min_size=1, max_size=1))

        input_shape3 = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=8), min_size=1, max_size=1))

        input_shape4 = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=8), min_size=1, max_size=1))

        input_shape5 = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=8), min_size=1, max_size=1))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))

        config = {
            "op_names": ["meshgrid"],
            "test_data_shapes": [
                input_shape1, input_shape2, input_shape3, input_shape4,
                input_shape5
            ],
            "test_data_types": [[dtype], [dtype], [dtype], [dtype], [dtype]],
            "opset_version": [8, 9, 15],
            "input_spec_shape": [],
        }

        models = Net2(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
