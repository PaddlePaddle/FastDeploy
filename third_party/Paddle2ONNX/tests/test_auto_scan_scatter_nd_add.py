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
from onnxbase import randtool


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs, index, updates):
        """
        forward
        """
        x = paddle.scatter_nd_add(inputs, index, updates)
        return x


class TestScatterNdAddConvert(OPConvertAutoScanTest):
    """
    api: paddle.scatter_nd_add
    OPset version: 16
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=5, max_value=15), min_size=2, max_size=4))

        index_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=10), min_size=2, max_size=4))

        index_shape[-1] = draw(
            st.integers(
                min_value=1, max_value=len(input_shape)))

        update_shape = index_shape[:-1] + input_shape[index_shape[-1]:]

        dtype = draw(st.sampled_from(["float32", "float64"]))
        index_dtype = draw(st.sampled_from(["int32", "int64"]))

        def generator_index():
            min_val = np.min(input_shape)
            index_list = randtool("int", 0, min_val, index_shape)
            return index_list

        config = {
            "op_names": ["scatter_nd_add"],
            "test_data_shapes": [input_shape, generator_index, update_shape],
            "test_data_types": [[dtype], [index_dtype], [dtype]],
            "opset_version": [16],
            "input_spec_shape": [],
            "use_gpu": False,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
