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
from paddle import ParamAttr


class Net(BaseNet):
    """
    simple Net
    """

    def __init__(self, config=None):
        super(Net, self).__init__(config)
        if self.config['data_format'] in ["NC", "NCL", "NCHW", "NCDHW", "NCHW"]:
            param_shape = [self.config['input_shape'][1]]
        else:
            param_shape = [self.config['input_shape'][-1]]
        dtype = self.config['dtype']

        self.mean = self.create_parameter(
            dtype=dtype,
            attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0),
                trainable=False,
                do_model_average=True),
            shape=param_shape)

        self.variance = self.create_parameter(
            dtype=dtype,
            attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(1.0),
                trainable=False,
                do_model_average=True),
            shape=param_shape)

        self.weight = self.create_parameter(
            shape=param_shape,
            dtype=dtype,
            default_initializer=paddle.nn.initializer.Constant(1.0))

        self.bias = self.create_parameter(
            shape=param_shape, dtype=dtype, is_bias=True)

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.nn.functional.batch_norm(
            inputs,
            running_mean=self.mean,
            running_var=self.variance,
            weight=self.weight,
            bias=self.bias,
            momentum=self.config['momentum'],
            epsilon=self.config['epsilon'],
            data_format=self.config['data_format'],
            use_global_stats=self.config['use_global_stats'])
        return x


class TestBatchNormConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.batch_norm
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=8), min_size=2, max_size=5))

        input_spec = [-1] * len(input_shape)
        dtype = draw(st.sampled_from(["float32", "float64"]))
        epsilon = draw(st.floats(min_value=1e-12, max_value=1e-5))
        momentum = draw(st.floats(min_value=0.1, max_value=0.9))
        if len(input_shape) == 2:
            data_format = "NC"
        elif len(input_shape) == 3:
            data_format = draw(st.sampled_from(["NCL"]))
        elif len(input_shape) == 4:
            data_format = draw(st.sampled_from(["NCHW"]))
        else:
            data_format = "NCDHW"
        use_global_stats = None
        is_use_global_stats = draw(st.sampled_from(["None", "False", "True"]))
        if is_use_global_stats == "False":
            use_global_stats = None  # has a diff
        elif is_use_global_stats == "True":
            use_global_stats = True

        config = {
            "op_names": ["batch_norm"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "epsilon": epsilon,
            "momentum": momentum,
            "input_shape": input_shape,
            "dtype": dtype,
            "data_format": data_format,
            "use_global_stats": use_global_stats,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
