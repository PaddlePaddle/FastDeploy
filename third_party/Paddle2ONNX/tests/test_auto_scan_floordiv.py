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
from onnxbase import randtool
import numpy as np
import unittest
import paddle

op_api_map = {"elementwise_floordiv": paddle.floor_divide}

opset_version_map = {"elementwise_floordiv": [7, 9, 15], }


class Net(BaseNet):
    def forward(self, inputs1, inputs2):
        x = op_api_map[self.config["op_names"]](inputs1, inputs2)
        return x


class TestfloordivConvert(OPConvertAutoScanTest):
    """
    api: paddle.floor_divide
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input1_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=2, max_size=4))

        if draw(st.booleans()):
            input2_shape = [input1_shape[-1]]
        else:
            input2_shape = input1_shape

        dtype = draw(st.sampled_from(["int32", "int64"]))

        def generator_data():
            input_data = randtool("int", 1.0, 20.0, input2_shape)
            return input_data

        config = {
            "op_names": ["elementwise_floordiv"],
            "test_data_shapes": [input1_shape, generator_data],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": []
        }

        models = list()
        op_names = list()
        opset_versions = list()
        for op_name, i in op_api_map.items():
            config["op_names"] = op_name
            models.append(Net(config))
            op_names.append(op_name)
        for op_name, i in op_api_map.items():
            opset_versions.append(opset_version_map[op_name])
        config["op_names"] = op_names
        config["opset_version"] = opset_versions

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
