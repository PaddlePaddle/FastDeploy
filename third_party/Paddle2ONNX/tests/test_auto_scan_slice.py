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
        axes = self.config['axes']
        starts = self.config['starts']
        ends = self.config['ends']
        if self.config['isStartsTensor']:
            starts = paddle.to_tensor(starts)
        if self.config['isEndsTensor']:
            ends = paddle.to_tensor(ends)
        x = paddle.slice(inputs, axes=axes, starts=starts, ends=ends)
        return x


class TestSliceConvert(OPConvertAutoScanTest):
    """
    api: paddle.slice
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=6), min_size=2, max_size=5))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        isStartsTensor = draw(st.booleans())
        isEndsTensor = draw(st.booleans())

        starts = [1, 0]
        ends = [4, 7]
        axes = [0, 1]
        opset_version = [7, 9, 15]
        if isStartsTensor or isEndsTensor:
            opset_version = [10, 15]
        config = {
            "op_names": ["slice"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "isStartsTensor": isStartsTensor,
            "isEndsTensor": isEndsTensor,
            "axes": axes,
            "starts": starts,
            "ends": ends,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class Net1(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        axes = self.config['axes']
        starts = self.config['starts']
        ends = self.config['ends']
        if self.config['isStartsTensor']:
            starts = paddle.to_tensor(starts)
        if self.config['isEndsTensor']:
            ends = paddle.to_tensor(ends)
        x = paddle.slice(inputs, axes=axes, starts=starts, ends=ends)
        return x


class TestSliceConvert1(OPConvertAutoScanTest):
    """
    api: paddle.slice
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=6), min_size=4, max_size=4))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        isStartsTensor = draw(st.booleans())
        isEndsTensor = draw(st.booleans())

        axes = [0, 1, 2, 3]
        starts = [1, 0, 0, 0]
        ends = [
            input_shape[0] + 10, input_shape[1] + 10, input_shape[2] + 10,
            input_shape[3] + 10
        ]
        if draw(st.booleans()):
            starts = [1, 0]
            ends = [input_shape[0], input_shape[2]]
            axes = [0, 3]
        opset_version = [7, 9, 15]
        if isStartsTensor or isEndsTensor:
            opset_version = [10, 15]
        config = {
            "op_names": ["slice"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "isStartsTensor": isStartsTensor,
            "isEndsTensor": isEndsTensor,
            "axes": axes,
            "starts": starts,
            "ends": ends,
        }

        models = Net1(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class Net2(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        axes = self.config['axes']
        starts = self.config['starts']
        starts = [1, 0, paddle.to_tensor(0), 0]
        ends = self.config['ends']
        ends = [10, 10, paddle.to_tensor(10), 10]
        x = paddle.slice(inputs, axes=axes, starts=starts, ends=ends)
        return x


class TestSliceConvert2(OPConvertAutoScanTest):
    """
    api: paddle.slice
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=6), min_size=4, max_size=4))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        isStartsTensor = draw(st.booleans())
        isEndsTensor = draw(st.booleans())

        axes = [0, 1, 2, 3]
        starts = [1, 0, 0, 0]
        ends = [10, 10, 10, 10]
        config = {
            "op_names": ["slice"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [10, 15],
            "input_spec_shape": [],
            "isStartsTensor": isStartsTensor,
            "isEndsTensor": isEndsTensor,
            "axes": axes,
            "starts": starts,
            "ends": ends,
        }

        models = Net2(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class Net3(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        axes = self.config['axes']
        starts = self.config['starts']
        starts = [
            1, paddle.to_tensor(np.array(0).astype("int64")),
            paddle.to_tensor(0), 0
        ]
        ends = self.config['ends']
        # ends = [10, 10, paddle.to_tensor(np.array(10).astype("int64")), 10]
        ends = [
            paddle.to_tensor(np.array(10).astype("int64")),
            paddle.to_tensor(np.array(10).astype("int64")),
            paddle.to_tensor(np.array(10).astype("int64")),
            paddle.to_tensor(np.array(10).astype("int64"))
        ]
        x = paddle.slice(inputs, axes=axes, starts=starts, ends=ends)
        return x


class TestSliceConvert3(OPConvertAutoScanTest):
    """
    api: paddle.slice
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=6), min_size=4, max_size=4))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        isStartsTensor = draw(st.booleans())
        isEndsTensor = draw(st.booleans())

        axes = [0, 1, 2, 3]
        starts = [1, 0, 0, 0]
        ends = [10, 10, 10, 10]
        config = {
            "op_names": ["slice"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [10, 15],
            "input_spec_shape": [],
            "isStartsTensor": isStartsTensor,
            "isEndsTensor": isEndsTensor,
            "axes": axes,
            "starts": starts,
            "ends": ends,
        }

        models = Net3(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class Net4(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        x = inputs[1:2, 2, :]
        return x


class TestSliceConvert4(OPConvertAutoScanTest):
    """
    api: paddle.slice
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=6), min_size=2, max_size=5))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        input_shape = [5, 5, 5]

        config = {
            "op_names": ["slice"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 12, 15],
            "input_spec_shape": [],
        }

        models = Net4(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
