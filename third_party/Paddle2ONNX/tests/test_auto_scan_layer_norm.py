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

    def __init__(self, config=None):
        super(Net, self).__init__(config)
        param_shape = [np.prod(self.config["normalized_shape"])]
        self.weight = self.create_parameter(
            attr=None,
            shape=param_shape,
            default_initializer=paddle.nn.initializer.Constant(1.0))

        self.bias = self.create_parameter(
            attr=None, shape=param_shape, is_bias=True)

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.nn.functional.layer_norm(
            inputs,
            weight=self.weight if self.config['has_weight_bias'] else None,
            bias=self.bias if self.config['has_weight_bias'] else None,
            normalized_shape=self.config["normalized_shape"],
            epsilon=self.config["epsilon"])
        return x


class TestLayerNormConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.layer_norm
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=8), min_size=2, max_size=5))

        input_spec = [-1] * len(input_shape)

        # When the dims is 5 and the last dimension is too small, an error will be reported due to the optimization of ONNXRuntime
        if len(input_shape) == 5:
            input_shape[4] = 10
        axis = draw(st.integers(min_value=1, max_value=len(input_shape) - 1))

        axis_type = draw(st.sampled_from(["int", "list"]))
        if axis_type == "int":
            normalized_shape = input_shape[-1]
        else:
            normalized_shape = input_shape[axis:]

        dtype = draw(st.sampled_from(["float32"]))
        epsilon = draw(st.floats(min_value=1e-12, max_value=1e-5))
        has_weight_bias = draw(st.booleans())

        config = {
            "op_names": ["layer_norm"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 15],
            "input_spec_shape": [],
            "epsilon": epsilon,
            "normalized_shape": normalized_shape,
            "has_weight_bias": has_weight_bias,
            "use_gpu": False
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
