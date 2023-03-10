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
    "isfinite_v2": paddle.isfinite,
    "isinf_v2": paddle.isinf,
    "isnan_v2": paddle.isnan,
    "isnan": paddle.fluid.layers.has_nan,
}

opset_version_map = {
    "isfinite_v2": [10, 15],
    "isinf_v2": [10, 15],
    "isnan_v2": [9, 15],
    "isnan": [9, 13, 15],
}


class Net(BaseNet):
    def forward(self, inputs):
        x = op_api_map[self.config["op_names"]](inputs)
        return x.astype('float32')


class TestLogicopsConvert(OPConvertAutoScanTest):
    """
    api: inX ops
    OPset version:
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=1, max_size=4))

        dtype = "float32"

        def generator_data():
            print(input_shape)
            input_data = np.ones(shape=input_shape)
            inf_data = np.ones(shape=input_shape)
            inf_data = np.ones(shape=input_shape)
            inf_data[:] = float('inf')
            inf_condition = np.random.randint(-2, 2, input_shape).astype("bool")
            input_data = np.where(inf_condition, input_data, inf_data)
            nan_data = np.ones(shape=input_shape)
            nan_data = np.ones(shape=input_shape)
            nan_data[:] = float('nan')
            nan_condition = np.random.randint(-2, 2, input_shape).astype("bool")
            input_data = np.where(nan_condition, input_data, nan_data)
            minus_inf_data = np.ones(shape=input_shape)
            minus_inf_data = np.ones(shape=input_shape)
            minus_inf_data[:] = float('-inf')
            minus_inf_condition = np.random.randint(-2, 2,
                                                    input_shape).astype("bool")
            input_data = np.where(minus_inf_condition, input_data,
                                  minus_inf_data)
            minus_nan_data = np.ones(shape=input_shape)
            minus_nan_data = np.ones(shape=input_shape)
            minus_nan_data[:] = float('-nan')
            minus_nan_condition = np.random.randint(-2, 2,
                                                    input_shape).astype("bool")
            input_data = np.where(minus_nan_condition, input_data,
                                  minus_nan_data)
            return input_data

        config = {
            "op_names": ["isX"],
            "test_data_shapes": [generator_data],
            "test_data_types": [[dtype]],
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
        self.run_and_statis(max_examples=40, max_duration=-1)


if __name__ == "__main__":
    unittest.main()
