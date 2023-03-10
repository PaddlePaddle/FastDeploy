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
    "hard_shrink": paddle.nn.functional.hardshrink,
    "softshrink": paddle.nn.functional.softshrink,
    "tanh_shrink": paddle.nn.functional.tanhshrink
}

opset_version_map = {
    "hard_shrink": [9, 15],
    "softshrink": [9, 15],
    "tanh_shrink": [7, 15]
}


class Net(BaseNet):
    def forward(self, inputs):
        if self.config["op_names"] in ["tanh_shrink"]:
            x = op_api_map[self.config["op_names"]](inputs)
        else:
            x = op_api_map[self.config["op_names"]](
                inputs, threshold=self.config["threshold"])
        return x


class TestShrinkopsConvert(OPConvertAutoScanTest):
    """
    api: shrink ops
    OPset version:
    """

    def sample_convert_config(self, draw):
        input1_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=2, max_size=4))

        threshold = draw(st.floats(min_value=0.1, max_value=1.0))

        dtype = draw(st.sampled_from(["float32", "float64"]))

        config = {
            "op_names": ["shrink"],
            "test_data_shapes": [input1_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "threshold": threshold
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


if __name__ == "__main__":
    unittest.main()
