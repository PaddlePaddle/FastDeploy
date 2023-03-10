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
        groups = self.config['groups']
        epsilon = self.config['epsilon']
        num_channels = self.config['num_channels']
        data_format = self.config['data_format']
        self.group_norm = paddle.nn.GroupNorm(
            num_groups=groups,
            num_channels=num_channels,
            epsilon=epsilon,
            weight_attr=None if self.config['has_weight_attr'] else False,
            bias_attr=None if self.config['has_bias_attr'] else False,
            data_format=data_format)

    def forward(self, inputs):
        """
        forward
        """
        x = self.group_norm(inputs)
        return x


class TestGroupNormConvert(OPConvertAutoScanTest):
    """
    api: paddle.fluid.layers.nn.group_norm
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=10), min_size=4, max_size=4))

        dtype = draw(st.sampled_from(["float32"]))
        data_format = draw(st.sampled_from(["NCHW"]))
        groups = input_shape[1]
        epsilon = draw(st.floats(min_value=1e-12, max_value=1e-5))
        has_weight_attr = draw(st.booleans())
        has_bias_attr = draw(st.booleans())

        config = {
            "op_names": ["group_norm"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "epsilon": epsilon,
            "data_format": data_format,
            "groups": groups,
            "num_channels": input_shape[1],
            "has_weight_attr": has_weight_attr,
            "has_bias_attr": has_bias_attr,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
