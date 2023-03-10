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

    def forward(self, inputs, weight):
        """
        forward
        """
        x = paddle.nn.functional.conv2d(
            inputs,
            weight,
            stride=self.config["stride"],
            padding=self.config["padding"],
            dilation=self.config["dilation"],
            groups=self.config["groups"],
            data_format=self.config["data_format"])
        return x


class TestConv2dConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.Conv2d
    OPset version: 9
    1.OPset version需要根据op_mapper中定义的version来设置。
    2.测试中所有OP对应升级到Opset version 15。
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=20, max_value=30), min_size=4, max_size=4))

        kernel_size = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=7), min_size=4, max_size=4))

        data_format = "NCHW"

        groups = draw(st.integers(min_value=1, max_value=4))
        muti1 = draw(st.integers(min_value=1, max_value=4))
        kernel_size[0] = groups * muti1
        input_shape[1] = kernel_size[1] * groups

        strides = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=5), min_size=1, max_size=2))
        if len(strides) == 1:
            strides = strides[0]
            if strides > kernel_size[2]:
                strides = kernel_size[2]
            if strides > kernel_size[3]:
                strides = kernel_size[3]
        else:
            if strides[0] > kernel_size[2]:
                strides[0] = kernel_size[2]
            if strides[1] > kernel_size[3]:
                strides[1] = kernel_size[3]

        padding_type = draw(st.sampled_from(["str", "list", "int", "tuple"]))
        padding = None
        if padding_type == "str":
            padding = draw(st.sampled_from(["SAME", "VALID"]))
        elif padding_type == "int":
            padding = draw(st.integers(min_value=1, max_value=5))
        elif padding_type == "tuple":
            padding1 = np.expand_dims(
                np.array(
                    draw(
                        st.lists(
                            st.integers(
                                min_value=1, max_value=5),
                            min_size=2,
                            max_size=2))),
                axis=0).tolist()
            padding2 = np.expand_dims(
                np.array(
                    draw(
                        st.lists(
                            st.integers(
                                min_value=1, max_value=5),
                            min_size=2,
                            max_size=2))),
                axis=0).tolist()
            if data_format == "NCHW":
                padding = [[0, 0]] + [[0, 0]] + padding1 + padding2
            else:
                padding = [[0, 0]] + padding1 + padding2 + [[0, 0]]
        elif padding_type == "list":
            if draw(st.booleans()):
                padding = draw(
                    st.lists(
                        st.integers(
                            min_value=1, max_value=5),
                        min_size=2,
                        max_size=2))
            else:
                padding = draw(
                    st.lists(
                        st.integers(
                            min_value=1, max_value=5),
                        min_size=4,
                        max_size=4))

        dilations = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=3), min_size=1, max_size=2))
        if len(dilations) == 1:
            dilations = dilations[0]
        if padding == "SAME":
            dilations = 1

        config = {
            "op_names": ["conv2d"],
            "test_data_shapes": [input_shape, kernel_size],
            "test_data_types": [['float32'], ['float32']],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [[-1, input_shape[1], -1, -1], kernel_size],
            "data_format": data_format,
            "stride": strides,
            "dilation": dilations,
            "padding": padding,
            "groups": groups,
            "input_shape": input_shape,
            "kernel_size": kernel_size,
            "delta": 1e-4,
            "rtol": 1e-4
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


#class Net1(BaseNet):
#    """
#    simple Net
#    """
#
#    def forward(self, inputs, weight):
#        """
#        forward
#        """
#        x = paddle.nn.functional.conv2d(
#            inputs,
#            weight,
#            stride=[1, 2],
#            padding='SAME',
#            dilation=[1, 2],
#            groups=1)
#        return x
#
#
#class TestConv2dConvert1(OPConvertAutoScanTest):
#    """
#    api: paddle.nn.Conv2d
#    OPset version: 9
#    1.OPset version需要根据op_mapper中定义的version来设置。
#    2.测试中所有OP对应升级到Opset version 15。
#    """
#
#    def sample_convert_config(self, draw):
#        config = {
#            "op_names": ["conv2d"],
#            "test_data_shapes": [[2, 5, 20, 20], [7, 5, 5, 5]],
#            "test_data_types": [['float32'], ['float32']],
#            "opset_version": [7, 9, 15],
#            "input_spec_shape": [[-1, 5, -1, -1], [7, 5, 5, 5]],
#            "delta": 1e-4,
#            "rtol": 1e-4
#        }
#
#        models = Net1(config)
#
#        return (config, models)
#
#    def test(self):
#        self.run_and_statis(max_examples=30, min_success_num=-1)

if __name__ == "__main__":
    unittest.main()
