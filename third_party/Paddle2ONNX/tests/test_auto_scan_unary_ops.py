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

op_api_map = {
    "abs": paddle.abs,
    "acos": paddle.acos,
    "asin": paddle.asin,
    "atan": paddle.atan,
    "ceil": paddle.ceil,
    "cos": paddle.cos,
    "cosh": paddle.cosh,
    "erf": paddle.erf,
    "exp": paddle.exp,
    "floor": paddle.floor,
    "hard_shrink": paddle.nn.functional.hardshrink,
    "brelu": paddle.nn.functional.hardtanh,
    "log": paddle.log,
    "log1p": paddle.log1p,
    "log2": paddle.log2,
    "log10": paddle.log10,
    "reciprocal": paddle.reciprocal,
    "relu": paddle.nn.functional.relu,
    "relu6": paddle.nn.functional.relu6,
    "round": paddle.round,
    "rsqrt": paddle.rsqrt,
    "selu": paddle.nn.functional.selu,
    "sigmoid": paddle.nn.functional.sigmoid,
    "sign": paddle.sign,
    "sin": paddle.sin,
    "sinh": paddle.sinh,
    "softplus": paddle.nn.functional.softplus,
    "softsign": paddle.nn.functional.softsign,
    "sqrt": paddle.sqrt,
    "square": paddle.square,
    "swish": paddle.nn.functional.swish,
    "silu": paddle.nn.functional.silu,
    "tanh": paddle.tanh,
    "tan": paddle.tan,
}

opset_version_map = {
    "abs": [7, 13, 15],
    "acos": [7, 15],
    "asin": [7, 15],
    "atan": [7, 15],
    "ceil": [7, 13, 15],
    "cos": [7, 15],
    "cosh": [9, 15],
    "erf": [9, 13, 15],
    "exp": [7, 13, 15],
    "floor": [7, 13, 15],
    "hard_shrink": [9, 15],
    "brelu": [9, 15],
    "log": [7, 13, 15],
    "log1p": [7, 13, 14, 15],
    "log2": [7, 13, 15],
    "log10": [7, 13, 15],
    "reciprocal": [7, 13, 15],
    "relu": [7, 13, 14, 15],
    "relu6": [7, 13, 14, 15],
    "round": [11, 15],
    "rsqrt": [7, 13, 15],
    "selu": [7, 15],
    "sigmoid": [7, 13, 15],
    "sign": [9, 13, 15],
    "sin": [7, 15],
    "sinh": [9, 15],
    "softplus": [7, 15],
    "softsign": [7, 15],
    "sqrt": [7, 13, 15],
    "square": [7, 13, 14, 15],
    "swish": [7, 13, 14, 15],
    "tanh": [7, 13, 15],
    "tan": [7, 15],
    "silu": [7, 15],
}


class Net(BaseNet):
    def forward(self, inputs):
        if self.config["op_names"].count("log") > 0:
            inputs = paddle.abs(inputs) + 0.01
        return op_api_map[self.config["op_names"]](inputs)


class TestUnaryOPConvert(OPConvertAutoScanTest):
    """Testcases for all the unary operators."""

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=20), min_size=4, max_size=4))

        data_shapes = input_shape
        input_specs = [-1, input_shape[1], -1, -1]
        dtype = draw(st.sampled_from(["float32"]))
        config = {
            "op_names": "",
            "test_data_shapes": [data_shapes],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [input_specs],
        }
        models = list()
        op_names = list()
        opset_versions = list()
        for op_name, i in op_api_map.items():
            config["op_names"] = op_name
            models.append(Net(config))
            op_names.append(op_name)
        for op_name, vs in opset_version_map.items():
            opset_versions.append(vs)
        config["op_names"] = op_names
        config["opset_version"] = opset_versions
        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=40, max_duration=600)


if __name__ == "__main__":
    unittest.main()
