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
        strides = self.config['strides']
        if self.config['isStartsTensor']:
            starts = paddle.to_tensor(np.array(starts).astype('int32'))
        if self.config['isEndsTensor']:
            ends = paddle.to_tensor(np.array(ends).astype('int32'))
        if self.config['isStridesTensor']:
            strides = paddle.to_tensor(np.array(strides).astype('int32'))
        x = paddle.strided_slice(
            inputs, axes=axes, starts=starts, ends=ends, strides=strides)
        return x


class TestStridedsliceConvert(OPConvertAutoScanTest):
    """
    api: paddle.strided_slice
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=10), min_size=4, max_size=6))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        isStartsTensor = draw(st.booleans())
        isEndsTensor = draw(st.booleans())
        isStridesTensor = draw(st.booleans())

        axes = [1, 2, 3]
        if draw(st.booleans()):
            starts = [-100, 0, 0]
        else:
            starts = [-input_shape[axes[0]], 0, -input_shape[axes[2]] - 22]

        if draw(st.booleans()):
            ends = [3, 2, 40000]
        else:
            ends = [-1, 2, 4]

        if draw(st.booleans()):
            strides = [2, 1, 2]
        else:
            strides = [1, 1, 1]

        tmp = [i for i, val in enumerate(strides) if val == 1]
        if len(tmp) == len(strides) and isStridesTensor is False:
            opset_version = [10, 15]
        else:
            opset_version = [10, 15]

        config = {
            "op_names": ["strided_slice"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "axes": axes,
            "starts": starts,
            "ends": ends,
            "strides": strides,
            "isStartsTensor": isStartsTensor,
            "isEndsTensor": isEndsTensor,
            "isStridesTensor": isStridesTensor,
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
        # strides = self.config['strides']
        # strides = [1, paddle.to_tensor(1).astype('int32'), 1]
        # strides = [1, paddle.to_tensor(1, dtype='int32'), 1]
        strides = [1, paddle.to_tensor(np.array(1).astype("int32")), 1]
        x = paddle.strided_slice(
            inputs, axes=axes, starts=starts, ends=ends, strides=strides)
        return x


class TestStridedsliceConvert1(OPConvertAutoScanTest):
    """
    api: paddle.strided_slice
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=10), min_size=4, max_size=6))
        input_shape = [4, 4, 4, 4]
        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        isStartsTensor = False  # draw(st.booleans())
        isEndsTensor = False  #draw(st.booleans())
        isStridesTensor = False  #draw(st.booleans())

        axes = [1, 2, 3]
        if draw(st.booleans()):
            starts = [-100, 0, 0]
        else:
            starts = [-input_shape[axes[0]], 0, -input_shape[axes[2]] - 22]

        if draw(st.booleans()):
            ends = [3, 2, 40000]
        else:
            ends = [-1, 2, 4]

        # if draw(st.booleans()):
        #     strides = [2, 1, 2]
        # else:
        strides = [1, 1, 1]

        tmp = [i for i, val in enumerate(strides) if val == 1]
        if len(tmp) == len(strides) and isStridesTensor is False:
            opset_version = [10, 15]
        else:
            opset_version = [10, 15]

        config = {
            "op_names": ["strided_slice"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "axes": axes,
            "starts": starts,
            "ends": ends,
            "strides": strides,
            "isStartsTensor": isStartsTensor,
            "isEndsTensor": isEndsTensor,
            "isStridesTensor": isStridesTensor,
        }

        models = Net1(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
