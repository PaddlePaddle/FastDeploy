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

        x = paddle.argsort(
            input,
            axis=self.config['axis'],
            descending=self.config['descending'])
        return x


class TestArgsortConvert(OPConvertAutoScanTest):
    """
    api: paddle.argsort
    OPset version: 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=5), min_size=2, max_size=5))

        axis = draw(
            st.integers(
                min_value=-len(input_shape), max_value=len(input_shape) - 1))

        dtype = draw(st.sampled_from(["float32", "float64"]))
        descending = draw(st.booleans())

        def generator_data():
            import random
            import numpy as np
            t = 1
            for i in range(len(input_shape)):
                t = t * input_shape[i]
            input_data = np.array(random.sample(range(-5000, 5000), t))
            input_data = input_data.reshape(input_shape)
            return input_data

        if descending:
            opset_version = [7, 10, 11, 15]
        else:
            opset_version = [11, 15]
        config = {
            "op_names": ["argsort"],
            "test_data_shapes": [generator_data],
            "test_data_types": [[dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "axis": axis,
            "descending": descending,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
