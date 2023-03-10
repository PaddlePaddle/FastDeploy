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

    def forward(self, inputs):
        """
        forward
        """
        shape = self.config['shape']
        if self.config['isTensor']:
            shape = paddle.to_tensor(
                np.array(shape).astype(self.config['shape_dtype']))
        x = paddle.expand(inputs, shape=shape)
        # TODO there's bug with expand operator
        x = paddle.reshape(
            x, shape=paddle.to_tensor(np.array([-1]).astype("int32")))
        return x


class TestExpandConvert(OPConvertAutoScanTest):
    """
    api: paddle.expand
    OPset version: 8, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=6), min_size=2, max_size=5))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        isTensor = draw(st.booleans())  # future to valid
        shape_dtype = draw(st.sampled_from(["int32", "int64"]))
        n = random.randint(1, 6 - len(input_shape))
        pre_shape = random.sample([1, 1, 2, 2, 3, 3], n)

        config = {
            "op_names": ["expand_v2"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [8, 9, 15],
            "input_spec_shape": [],
            "isTensor": isTensor,
            "shape": pre_shape + input_shape,
            "shape_dtype": shape_dtype,
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
        shape = [
            2, 1, paddle.to_tensor(
                2, dtype=self.config['shape_dtype']), 3, 2, 2
        ]
        # not supported
        # shape = [paddle.to_tensor(2), paddle.to_tensor(np.array(1).astype("int64")), paddle.to_tensor(2), paddle.to_tensor(3), paddle.to_tensor(2), paddle.to_tensor(2)]
        x = paddle.expand(inputs, shape=shape)
        # TODO there's bug with expand operator
        x = paddle.reshape(
            x, shape=paddle.to_tensor(np.array([-1]).astype("int32")))
        return x


class TestExpandConvert1(OPConvertAutoScanTest):
    """
    api: paddle.expand
    OPset version: 8, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=6), min_size=2, max_size=5))
        input_shape = [2, 2]
        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        isTensor = draw(st.booleans())  # future to valid

        n = random.randint(1, 6 - len(input_shape))
        pre_shape = random.sample([1, 1, 2, 2, 3, 3], n)
        shape_dtype = draw(st.sampled_from(["int32", "int64"]))
        config = {
            "op_names": ["expand_v2"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [8, 9, 15],
            "input_spec_shape": [],
            "isTensor": isTensor,
            "shape": pre_shape + input_shape,
            "shape_dtype": shape_dtype,
        }

        models = Net1(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
