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

    def forward(self, inputs, weight):
        """
        forward
        """
        x = paddle.nn.functional.embedding(
            inputs,
            weight,
            padding_idx=self.config["padding_idx"],
            sparse=self.config["sparse"])
        return x


class TestKookuptablev2Convert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.embedding
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=30), min_size=2, max_size=2))

        weight_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=30), min_size=2, max_size=2))

        def generator_data():
            input_data = randtool("int", 0, weight_shape[0] - 1, input_shape)
            return input_data

        padding_idx = None
        if draw(st.booleans()):
            padding_idx = draw(
                st.integers(
                    min_value=-1 * weight_shape[0] + 1,
                    max_value=weight_shape[0] - 1))

        sparse = draw(st.booleans())

        dtype1 = draw(st.sampled_from(["int32", "int64"]))
        dtype = draw(st.sampled_from(["float32", "float64"]))

        config = {
            "op_names": ["lookup_table_v2"],
            "test_data_shapes": [generator_data, weight_shape],
            "test_data_types": [[dtype1], [dtype]],
            "opset_version": [7, 9, 11, 15],
            "input_spec_shape": [],
            "padding_idx": padding_idx,
            "sparse": sparse
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=50)


if __name__ == "__main__":
    unittest.main()
