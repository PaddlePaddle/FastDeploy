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
        if self.config["tensor_attr"]:
            output_size = self.config["output_size"]
        else:
            output_size = self.config["output_size"]
        x = paddle.nn.functional.conv2d_transpose(
            inputs,
            weight,
            stride=self.config["stride"],
            padding=self.config["padding"],
            dilation=self.config["dilation"],
            groups=self.config["groups"],
            output_size=output_size,
            data_format=self.config["data_format"])
        return x


class TestConv2dTransposeConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.conv2d_transpose
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
        kernel_size[1] = groups * muti1
        kernel_size[0] = kernel_size[1]
        input_shape[1] = kernel_size[0]
        if draw(st.booleans()):
            kernel_size[1] = 1
            groups = draw(st.integers(min_value=2, max_value=4))
            input_shape[1] = groups
            kernel_size[0] = groups

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
            stride_1 = strides
            stride_2 = strides
        else:
            if strides[0] > kernel_size[2]:
                strides[0] = kernel_size[2]
            if strides[1] > kernel_size[3]:
                strides[1] = kernel_size[3]
            stride_1 = strides[0]
            stride_2 = strides[1]

        # ORT have bug in SAME and Valid
        # padding_type = draw(st.sampled_from(["str", "list", "int", "tuple"]))
        padding_type = draw(st.sampled_from(["list", "int", "tuple"]))
        padding = None
        if padding_type == "str":
            padding = draw(st.sampled_from(["SAME", "VALID"]))
        elif padding_type == "int":
            padding = draw(st.integers(min_value=1, max_value=5))
            padding_1_1 = padding
            padding_1_2 = padding
            padding_2_1 = padding
            padding_2_2 = padding
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
            padding_1_1 = padding[2][0]
            padding_1_2 = padding[2][1]
            padding_2_1 = padding[3][0]
            padding_2_2 = padding[3][1]
        elif padding_type == "list":
            if draw(st.booleans()):
                padding = draw(
                    st.lists(
                        st.integers(
                            min_value=1, max_value=5),
                        min_size=2,
                        max_size=2))
                padding_1_1 = padding[0]
                padding_1_2 = padding[0]
                padding_2_1 = padding[1]
                padding_2_2 = padding[1]
            else:
                padding = draw(
                    st.lists(
                        st.integers(
                            min_value=1, max_value=5),
                        min_size=4,
                        max_size=4))
                padding_1_1 = padding[0]
                padding_1_2 = padding[1]
                padding_2_1 = padding[2]
                padding_2_2 = padding[3]

        dilations = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=3), min_size=1, max_size=2))
        if len(dilations) == 1:
            dilations = dilations[0]
            dilations_1 = dilations
            dilations_2 = dilations
        else:
            dilations_1 = dilations[0]
            dilations_2 = dilations[1]
        if padding == "SAME":
            dilations = 1

        output_size = None
        if draw(st.booleans()):
            output_size_1 = (
                input_shape[2] - 1
            ) * stride_1 - padding_1_1 - padding_1_2 + dilations_1 * (
                kernel_size[2] - 1) + 1
            output_size_2 = (
                input_shape[3] - 1
            ) * stride_2 - padding_2_1 - padding_2_2 + dilations_2 * (
                kernel_size[3] - 1) + 1
            if output_size_1 == output_size_2:
                output_size = output_size_1
            else:
                output_size = [output_size_1, output_size_2]

        tensor_attr = draw(st.booleans())

        config = {
            "op_names": ["conv2d_transpose"],
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
            "rtol": 1e-4,
            "output_size": output_size,
            "tensor_attr": tensor_attr
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
