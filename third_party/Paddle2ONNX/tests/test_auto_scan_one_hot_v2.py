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
from onnxbase import randtool
import paddle


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        num_classes = self.config["num_classes"]
        if self.config["is_tensor"]:
            num_classes = paddle.to_tensor([num_classes])
        x = paddle.nn.functional.one_hot(inputs, num_classes)
        return x


class TestOneHotV2Convert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.one_hot
    OPset version: 9, 13, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=1, max_size=4))

        num_classes = draw(st.integers(min_value=10, max_value=20))

        def generator_data():
            input_data = randtool("int", 0, num_classes - 1, input_shape)
            return input_data

        dtype = draw(st.sampled_from(["int32", "int64"]))

        is_tensor = draw(st.booleans())
        config = {
            "op_names": ["one_hot_v2"],
            "test_data_shapes": [generator_data],
            "test_data_types": [[dtype]],
            "opset_version": [9, 13, 15],
            "input_spec_shape": [],
            "num_classes": num_classes,
            "is_tensor": is_tensor
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
