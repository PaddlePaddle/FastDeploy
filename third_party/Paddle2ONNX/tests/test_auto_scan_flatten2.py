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

    def forward(self, x):
        """
        forward
        """
        import paddle.fluid as fluid
        x = fluid.layers.flatten(x=x, axis=self.config['axis'])
        return x


class TestFlattenConvert(OPConvertAutoScanTest):
    """
    api: paddle.flatten
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=20), min_size=2, max_size=5))

        dtype = draw(st.sampled_from(["int32", "int64", "float32", "float64"]))

        axis = draw(st.integers(min_value=0, max_value=len(input_shape) - 1))

        config = {
            "op_names": ["flatten2"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [9, 15],
            "input_spec_shape": [],
            "axis": axis,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
