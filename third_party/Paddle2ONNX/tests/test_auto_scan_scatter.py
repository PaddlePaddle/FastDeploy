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
        x = paddle.scatter(
            inputs, index, updates, overwrite=self.config['overwrite'])
        return x


class TestScatterConvert(OPConvertAutoScanTest):
    """
    api: paddle.scatter
    OPset version: 11, 12, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=10), min_size=1, max_size=5))

        index_shape = draw(st.integers(min_value=1, max_value=input_shape[0]))

        update_shape = input_shape
        update_shape[0] = index_shape

        dtype = draw(st.sampled_from(["float32", "float64"]))
        index_dtype = draw(st.sampled_from(["int32", "int64"]))
        overwrite = draw(st.booleans())

        opset_version = [16]
        if overwrite:
            opset_version = [11, 15]

        def generator_index():
            index_list = randtool("int", 0, input_shape[0], index_shape)
            return index_list

        config = {
            "op_names": ["scatter"],
            "test_data_shapes": [input_shape, generator_index, update_shape],
            "test_data_types": [[dtype], [index_dtype], [dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "overwrite": overwrite,
            "use_gpu": False,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
