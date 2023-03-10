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
import random

op_api_map = {
    "arg_min": paddle.argmin,
    "arg_max": paddle.argmax,
}

opset_version_map = {
    "arg_min": [7, 9, 15],
    "arg_max": [7, 9, 15],
}


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        if self.config["tensor_attr"]:
            axis = paddle.assign(self.config["axis"])
        else:
            axis = self.config["axis"]
        x = op_api_map[self.config["op_names"]](inputs,
                                                axis=axis,
                                                keepdim=self.config["keep_dim"],
                                                dtype=self.config["out_dtype"])
        return x


class TestArgMinMaxConvert(OPConvertAutoScanTest):
    """
    api: paddle.argmin/argmax
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=10), min_size=2, max_size=4))

        input_spec = [-1] * len(input_shape)

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))

        axis = draw(
            st.integers(
                min_value=-len(input_shape), max_value=len(input_shape) - 1))

        keep_dim = draw(st.booleans())

        out_dtype = draw(st.sampled_from(["int32", "int64"]))

        tensor_attr = draw(st.booleans())

        config = {
            "op_names": ["reduce_max"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "axis": axis,
            "out_dtype": out_dtype,
            "keep_dim": keep_dim,
            "tensor_attr": tensor_attr,
            "input_spec_shape": [],
            "delta": 1e-4,
            "rtol": 1e-4
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
