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
    "reduce_max": paddle.max,
    "reduce_min": paddle.min,
    "reduce_mean": paddle.mean,
    "reduce_sum": paddle.sum,
    "reduce_prod": paddle.prod,
}

opset_version_map = {
    "reduce_max": [7, 9, 15],
    "reduce_min": [7, 9, 15],
    "reduce_mean": [7, 9, 15],
    "reduce_sum": [7, 9, 15],
    "reduce_prod": [7, 9, 15],
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
            axis = paddle.to_tensor(
                self.config["dim"], dtype=self.config["axis_dtype"])
        else:
            axis = self.config["dim"]

        x = op_api_map[self.config["op_names"]](inputs,
                                                axis=axis,
                                                keepdim=self.config["keep_dim"])
        x = paddle.unsqueeze(x, axis=[0])
        return x


class TestReduceAllConvert(OPConvertAutoScanTest):
    """
    api: paddle.fluid.layers.reduce_max/min/mean/sum/prod/
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=10), min_size=1, max_size=4))

        input_spec = [-1] * len(input_shape)

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        axis_type = draw(st.sampled_from([
            "list",
            "int",
        ]))
        if axis_type == "int":
            axes = draw(
                st.integers(
                    min_value=-len(input_shape), max_value=len(input_shape) -
                    1))
        elif axis_type == "list":
            lenSize = random.randint(1, len(input_shape))
            axes = []
            for i in range(lenSize):
                axes.append(random.choice([i, i - len(input_shape)]))
            # paddle.max/min has a bug when aixs < 0
            axes = [
                axis + len(input_shape) if axis < 0 else axis
                for i, axis in enumerate(axes)
            ]
        keep_dim = draw(st.booleans())
        tensor_attr = draw(st.booleans())
        # Must be int64, otherwise cast will be added after const and the value cannot be obtained
        axis_dtype = draw(st.sampled_from(["int64"]))
        config = {
            "op_names": ["reduce_max"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "dim": axes,
            "keep_dim": keep_dim,
            "input_spec_shape": [],
            "delta": 1e-4,
            "rtol": 1e-4,
            "tensor_attr": tensor_attr,
            "axis_dtype": axis_dtype
        }

        models = list()
        op_names = list()
        opset_versions = list()
        for op_name, i in op_api_map.items():
            config["op_names"] = op_name
            if op_name == "reduce_mean":
                dtype_mean = draw(st.sampled_from(["float32", "float64"]))
                config["test_data_types"] = [[dtype_mean]]
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
