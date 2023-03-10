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

op_api_map = {
    "greater_equal": paddle.greater_equal,
    "equal": paddle.equal,
    "not_equal": paddle.not_equal,
    "greater_than": paddle.greater_than,
    "logical_and": paddle.logical_and,
    "logical_or": paddle.logical_or,
    "logical_xor": paddle.logical_xor,
    "less_equal": paddle.less_equal,
    "less_than": paddle.less_than,
}

opset_version_map = {
    "greater_equal": [12, 15],
    "equal": [11, 15],
    "not_equal": [11, 15],
    "greater_than": [11, 15],
    "logical_and": [7, 15],
    "logical_or": [7, 12, 15],
    "logical_xor": [7, 15],
    "less_equal": [12, 15],
    "less_than": [9, 15],
}


class Net(BaseNet):
    def forward(self, inputs1, inputs2):
        if self.config[
                "op_names"] in ["logical_and", "logical_or", "logical_xor"]:
            inputs1 = inputs1.astype('bool')
            inputs2 = inputs2.astype('bool')
        if self.config["op_names"] in [
                "greater_equal", "greater_than", "less_equal", "less_than"
        ] and inputs1.dtype == paddle.bool:
            inputs1 = inputs1.astype('int32')
            inputs2 = inputs2.astype('int32')
        x = op_api_map[self.config["op_names"]](inputs1, inputs2)
        return x


class TestLogicopsConvert(OPConvertAutoScanTest):
    """
    api: logic ops
    OPset version:
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

        if draw(st.booleans()):
            input2_shape = [1]

        dtype = draw(st.sampled_from(["float32", "int32", "int64", "bool"]))

        config = {
            "op_names": ["elementwise_add"],
            "test_data_shapes": [input1_shape, input2_shape],
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
            opset_versions.append(opset_version_map[op_name])
        config["op_names"] = op_names
        config["opset_version"] = opset_versions

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30, max_duration=-1)


class NetNot(BaseNet):
    def forward(self, inputs):
        x = paddle.logical_not(inputs)
        return x.astype('float32')


class TestLogicNotConvert(OPConvertAutoScanTest):
    """
    api: logical_not ops
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input1_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=2, max_size=4))

        dtype = "bool"
        config = {
            "op_names": ["logical_not"],
            "test_data_shapes": [input1_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": []
        }

        model = NetNot(config)

        return (config, model)

    def test(self):
        self.run_and_statis(max_examples=30, max_duration=-1)


if __name__ == "__main__":
    unittest.main()
