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
        x = paddle.nn.functional.conv3d(
            inputs,
            weight,
            stride=self.config["stride"],
            padding=self.config["padding"],
            dilation=self.config["dilation"],
            groups=self.config["groups"],
            data_format=self.config["data_format"])
        return x


class TestConv3dConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.Conv3d
    OPset version: 9
    1.OPset version需要根据op_mapper中定义的version来设置。
    2.测试中所有OP对应升级到Opset version 15。
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=15, max_value=25), min_size=5, max_size=5))

        kernel_size = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=5), min_size=5, max_size=5))

        data_format = "NCDHW"

        groups = draw(st.integers(min_value=1, max_value=4))
        muti1 = draw(st.integers(min_value=1, max_value=4))
        kernel_size[0] = groups * muti1
        input_shape[1] = kernel_size[1] * groups

        strides_type = draw(st.sampled_from(["list", "int"]))

        strides = None
        if strides_type == "int":
            strides = draw(st.integers(min_value=1, max_value=5))
            if strides > kernel_size[2]:
                strides = kernel_size[2]
            if strides > kernel_size[3]:
                strides = kernel_size[3]
            if strides > kernel_size[4]:
                strides = kernel_size[4]
        else:
            strides = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=5),
                    min_size=3,
                    max_size=3))
            if strides[0] > kernel_size[2]:
                strides[0] = kernel_size[2]
            if strides[1] > kernel_size[3]:
                strides[1] = kernel_size[3]
            if strides[2] > kernel_size[4]:
                strides[2] = kernel_size[4]

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
            padding3 = np.expand_dims(
                np.array(
                    draw(
                        st.lists(
                            st.integers(
                                min_value=1, max_value=5),
                            min_size=2,
                            max_size=2))),
                axis=0).tolist()
            padding = [[0, 0]] + [[0, 0]] + padding1 + padding2 + padding3
        elif padding_type == "list":
            if draw(st.booleans()):
                padding = draw(
                    st.lists(
                        st.integers(
                            min_value=1, max_value=5),
                        min_size=3,
                        max_size=3))
            else:
                padding = draw(
                    st.lists(
                        st.integers(
                            min_value=1, max_value=5),
                        min_size=6,
                        max_size=6))

        dilations_type = draw(st.sampled_from(["int", "tuple"]))
        dilations = None
        if dilations_type == "int":
            dilations = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=3),
                    min_size=1,
                    max_size=1))
        else:
            dilations = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=3),
                    min_size=3,
                    max_size=3))
        if len(dilations) == 1:
            dilations = dilations[0]
        if padding == "SAME":
            dilations = 1

        config = {
            "op_names": ["conv3d"],
            "test_data_shapes": [input_shape, kernel_size],
            "test_data_types": [['float32'], ['float32']],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [[-1, input_shape[1], -1, -1, -1], kernel_size],
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


if __name__ == "__main__":
    unittest.main()
