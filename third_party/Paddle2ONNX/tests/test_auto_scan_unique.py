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

    def forward(self, input):
        """
        forward
        """
        x = paddle.unique(
            input,
            return_index=self.config['return_index'],
            return_inverse=self.config['return_inverse'],
            return_counts=self.config['return_counts'],
            axis=self.config['axis'],
            dtype=self.config['dtype'])

        return x


class TestUniqueConvert(OPConvertAutoScanTest):
    """
    api: paddle.unique
    OPset version: 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=10), min_size=1, max_size=4))

        return_index = draw(st.booleans())
        return_inverse = draw(st.booleans())
        return_counts = draw(st.booleans())

        axis = None
        if draw(st.booleans()):
            axis = draw(
                st.integers(
                    min_value=0, max_value=len(input_shape) - 1))
        dtype = draw(st.sampled_from(["float32", "int64"]))
        xdtype = draw(st.sampled_from(["int64", "int32"]))
        config = {
            "op_names": ["unique"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [11, 15],
            "input_spec_shape": [],
            "return_index": return_index,
            "return_inverse": return_inverse,
            "return_counts": return_counts,
            "axis": axis,
            "dtype": xdtype,
            "use_gpu": False,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
