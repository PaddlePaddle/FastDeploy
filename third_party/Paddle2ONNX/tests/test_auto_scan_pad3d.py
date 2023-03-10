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


class Net(BaseNet):
    def forward(self, inputs):
        pad = self.config["pad"]
        mode = self.config["mode"]
        value = self.config["value"]
        data_format = self.config["data_format"]
        x = paddle.nn.functional.pad(inputs,
                                     pad=pad,
                                     mode=mode,
                                     value=value,
                                     data_format=data_format)
        shape = paddle.shape(x)
        x = paddle.reshape(x, shape)

        return x


class TestPadopsConvert(OPConvertAutoScanTest):
    """
    api: pad3d
    OPset version:
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=4, max_size=5))

        dtype = "float32"

        mode = draw(st.sampled_from(["constant", "reflect", "replicate"]))

        value = draw(st.floats(min_value=0, max_value=10))

        data_format = None
        if len(input_shape) == 3:
            #            data_format = draw(st.sampled_from(["NCL", "NLC"]))
            data_format = "NCL"
        elif len(input_shape) == 4:
            #            data_format = draw(st.sampled_from(["NCHW", "NHWC"]))
            data_format = "NCHW"
        else:
            #            data_format = draw(st.sampled_from(["NCDHW", "NDHWC"]))
            data_format = "NCDHW"

        pad = None
        if len(input_shape) == 3:
            pad = draw(
                st.lists(
                    st.integers(
                        min_value=0, max_value=4),
                    min_size=2,
                    max_size=2))
        elif len(input_shape) == 4:
            pad = draw(
                st.lists(
                    st.integers(
                        min_value=0, max_value=4),
                    min_size=4,
                    max_size=4))
        else:
            pad = draw(
                st.lists(
                    st.integers(
                        min_value=0, max_value=4),
                    min_size=6,
                    max_size=6))

        config = {
            "op_names": ["pad3d"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 11, 15],
            "input_spec_shape": [],
            "mode": mode,
            "value": value,
            "pad": pad,
            "data_format": data_format
        }

        model = Net(config)

        return (config, model)

    def test(self):
        self.run_and_statis(max_examples=25, max_duration=-1)


class Net2(BaseNet):
    def forward(self, inputs):
        data = np.ones(shape=[6], dtype="int32")
        pad = paddle.to_tensor(data, dtype='int32')
        mode = self.config["mode"]
        value = self.config["value"]
        data_format = self.config["data_format"]
        x = paddle.nn.functional.pad(inputs,
                                     pad,
                                     mode=mode,
                                     value=value,
                                     data_format=data_format)
        shape = paddle.shape(x)
        x = paddle.reshape(x, shape)

        return x


class TestPadopsConvert_Constanttensor(OPConvertAutoScanTest):
    """
    api: pad2d
    OPset version:
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=10), min_size=5, max_size=5))

        dtype = "float32"

        mode = draw(st.sampled_from(["constant", "reflect", "replicate"]))

        value = draw(st.floats(min_value=0, max_value=10))

        data_format = None
        #data_format = draw(st.sampled_from(["NCDHW", "NDHWC"]))
        data_format = "NCDHW"

        pad = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=4), min_size=6, max_size=6))

        config = {
            "op_names": ["pad3d"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [11, 12, 13, 14, 15],
            "input_spec_shape": [],
            "mode": mode,
            "value": value,
            "pad": pad,
            "data_format": data_format
        }

        model = Net2(config)

        return (config, model)

    def test(self):
        self.run_and_statis(max_examples=25, max_duration=-1)


if __name__ == "__main__":
    unittest.main()
