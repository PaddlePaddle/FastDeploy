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


class NetAvgpool1d(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        kernel_size = self.config['kernel_size']
        stride = self.config['stride']
        padding = self.config['padding']
        ceil_mode = self.config['ceil_mode']
        x = paddle.nn.functional.avg_pool1d(
            inputs,
            kernel_size,
            stride=stride,
            padding=padding,
            exclusive=True,
            ceil_mode=ceil_mode)
        return x


class TestAvgpool1dConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.max_pool1d
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=3, max_size=3))

        # input_shape = [3, 1, 10]
        dtype = draw(st.sampled_from(["float32", "float64"]))

        ceil_mode = draw(st.booleans())

        kernel_type = draw(st.sampled_from(["int", "list"]))
        if kernel_type == "int":
            kernel_size = draw(st.integers(min_value=7, max_value=10))
        elif kernel_type == "list":
            kernel_size = draw(
                st.lists(
                    st.integers(
                        min_value=7, max_value=10),
                    min_size=1,
                    max_size=1))

        stride_type = draw(st.sampled_from(["None", "int", "list"]))
        if stride_type == "int":
            stride = draw(st.integers(min_value=1, max_value=5))
        elif stride_type == "list":
            stride = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=5),
                    min_size=1,
                    max_size=1))
        else:
            stride = None

        padding = 0

        opset_version = [[7, 9, 15]]
        if ceil_mode:
            opset_version = [10, 15]

        if padding == "VALID":
            ceil_mode = False

        op_names = 'pool2d'

        config = {
            "op_names": [op_names],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "ceil_mode": ceil_mode,
        }

        models = NetAvgpool1d(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class NetAvgpool2d(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        kernel_size = self.config['kernel_size']
        stride = self.config['stride']
        padding = self.config['padding']
        ceil_mode = self.config['ceil_mode']
        data_format = self.config['data_format']
        x = paddle.nn.functional.avg_pool2d(
            inputs,
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            exclusive=True,
            divisor_override=None,
            data_format=data_format)

        return x


class TestAvgpool2dConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.max_pool2d
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=4, max_size=4))

        dtype = draw(st.sampled_from(["float32", "float64"]))
        data_format = draw(st.sampled_from(["NCHW"]))

        ceil_mode = draw(st.booleans())

        kernel_type = draw(st.sampled_from(["int", "list"]))
        if kernel_type == "int":
            kernel_size = draw(st.integers(min_value=5, max_value=7))
        elif kernel_type == "list":
            kernel_size = draw(
                st.lists(
                    st.integers(
                        min_value=5, max_value=7),
                    min_size=2,
                    max_size=2))

        stride_type = draw(st.sampled_from(["None", "int", "list"]))
        if stride_type == "int":
            stride = draw(st.integers(min_value=1, max_value=5))
        elif stride_type == "list":
            stride = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=5),
                    min_size=2,
                    max_size=2))
        else:
            stride = None

        padding_type = draw(
            st.sampled_from(["None", "str", "int", "list2", "list4", "list8"]))
        if padding_type == "str":
            padding = draw(st.sampled_from(["SAME", "VALID"]))
        elif padding_type == "int":
            padding = draw(st.integers(min_value=1, max_value=3))
        elif padding_type == "list2":
            padding = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=3),
                    min_size=2,
                    max_size=2))
        elif padding_type == "list4":
            padding = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=3),
                    min_size=4,
                    max_size=4))
        elif padding_type == "list8":
            padding1 = np.expand_dims(
                np.array(
                    draw(
                        st.lists(
                            st.integers(
                                min_value=1, max_value=3),
                            min_size=2,
                            max_size=2))),
                axis=0).tolist()
            padding2 = np.expand_dims(
                np.array(
                    draw(
                        st.lists(
                            st.integers(
                                min_value=1, max_value=3),
                            min_size=2,
                            max_size=2))),
                axis=0).tolist()
            if data_format == "NCHW":
                padding = [[0, 0]] + [[0, 0]] + padding1 + padding2
            else:
                padding = [[0, 0]] + padding1 + padding2 + [[0, 0]]
        else:
            padding = 0

        opset_version = [[7, 9, 15]]
        if ceil_mode:
            opset_version = [10, 15]

        if padding == "VALID":
            ceil_mode = False

        op_names = 'pool2d'
        config = {
            "op_names": [op_names],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "ceil_mode": ceil_mode,
            "data_format": data_format
        }

        models = NetAvgpool2d(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class NetAvgpool3d(BaseNet):
    """
   simple Net
   """

    def forward(self, inputs):
        """
       forward
       """
        kernel_size = self.config['kernel_size']
        stride = self.config['stride']
        padding = self.config['padding']
        ceil_mode = self.config['ceil_mode']
        data_format = self.config['data_format']
        x = paddle.nn.functional.avg_pool3d(
            inputs,
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            exclusive=True,
            divisor_override=None,
            data_format=data_format)
        return x


class TestAvgpool3dConvert(OPConvertAutoScanTest):
    """
   api: paddle.nn.functional.max_pool3d
   OPset version: 7, 9, 15
   """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=5, max_size=5))

        dtype = draw(st.sampled_from(["float32", "float64"]))
        data_format = draw(st.sampled_from(["NCDHW"]))

        ceil_mode = draw(st.booleans())

        kernel_type = draw(st.sampled_from(["int", "list"]))
        if kernel_type == "int":
            kernel_size = draw(st.integers(min_value=7, max_value=10))
        elif kernel_type == "list":
            kernel_size = draw(
                st.lists(
                    st.integers(
                        min_value=7, max_value=10),
                    min_size=3,
                    max_size=3))

        stride_type = draw(st.sampled_from(["None", "int", "list"]))
        if stride_type == "int":
            stride = draw(st.integers(min_value=1, max_value=5))
        elif stride_type == "list":
            stride = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=5),
                    min_size=3,
                    max_size=3))
        else:
            stride = None

        padding_type = draw(
            st.sampled_from(["None", "str", "int", "list3", "list6", "list10"]))
        if padding_type == "str":
            padding = draw(st.sampled_from(["SAME", "VALID"]))
        elif padding_type == "int":
            padding = draw(st.integers(min_value=1, max_value=5))
        elif padding_type == "list3":
            padding = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=5),
                    min_size=3,
                    max_size=3))
        elif padding_type == "list6":
            padding = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=5),
                    min_size=6,
                    max_size=6))
        elif padding_type == "list10":
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
            if data_format == "NCDHW":
                padding = [[0, 0]] + [[0, 0]] + padding1 + padding2 + padding3
            else:
                padding = [[0, 0]] + padding1 + padding2 + padding3 + [[0, 0]]
        else:
            padding = 0

        opset_version = [[7, 9, 15]]
        if ceil_mode:
            opset_version = [10, 15]

        if padding == "VALID":
            ceil_mode = False

        op_names = 'pool3d'

        config = {
            "op_names": [op_names],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "ceil_mode": ceil_mode,
            "data_format": data_format,
        }

        models = NetAvgpool3d(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
