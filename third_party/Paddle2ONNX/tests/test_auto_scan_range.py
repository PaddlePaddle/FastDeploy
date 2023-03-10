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
    def forward(self):
        start = self.config["start"]
        if self.config['is_tensor_start']:
            start = paddle.to_tensor(start).astype(self.config['index_dtype'])

        end = self.config["end"]
        if self.config['is_tensor_end']:
            end = paddle.to_tensor(end).astype(self.config['index_dtype'])

        step = self.config["step"]
        if self.config['is_tensor_step']:
            step = paddle.to_tensor(step).astype(self.config['index_dtype'])

        dtype = self.config["dtype"]
        x = paddle.arange(start=start, end=end, step=step, dtype=dtype)
        return x


class TestArangeConvert(OPConvertAutoScanTest):
    """
    api: paddle.arange
    OPset version:
    """

    def sample_convert_config(self, draw):
        start = draw(st.integers(min_value=1, max_value=10))
        is_tensor_start = draw(st.booleans())

        end = None
        is_tensor_end = False
        if draw(st.booleans()):
            end = draw(st.integers(min_value=20, max_value=30))
            is_tensor_end = draw(st.booleans())

        step = draw(st.integers(min_value=1, max_value=4))
        is_tensor_step = draw(st.booleans())

        dtype = None
        if draw(st.booleans()):
            dtype = draw(
                st.sampled_from(["float32", "float64", "int32", "int64"]))
        index_dtype = draw(
            st.sampled_from(["float32", "float64", "int32", "int64"]))

        config = {
            "op_names": ["range"],
            "test_data_shapes": [],
            "test_data_types": [],
            "opset_version": [11, 12, 13, 14, 15],
            "input_spec_shape": [],
            "start": start,
            "end": end,
            "step": step,
            "dtype": dtype,
            "is_tensor_start": is_tensor_start,
            "is_tensor_step": is_tensor_step,
            "is_tensor_end": is_tensor_end,
            "index_dtype": index_dtype,
        }

        model = Net(config)

        return (config, model)

    def test(self):
        self.run_and_statis(max_examples=30, max_duration=-1)


if __name__ == "__main__":
    unittest.main()
