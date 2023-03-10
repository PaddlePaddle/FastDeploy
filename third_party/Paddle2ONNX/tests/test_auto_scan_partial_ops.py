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

name2fun_dict = {}
name2fun_dict["partial_sum"] = paddle.fluid.contrib.layers.partial_sum
name2fun_dict["partial_concat"] = paddle.fluid.contrib.layers.partial_concat


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs1, inputs2):
        """
        forward
        """
        inputs_list = [inputs1]
        for i in range(self.config["repeat_times"]):
            inputs_list.append(inputs2)
        x = name2fun_dict[self.config["op_names"][0]](
            inputs_list,
            start_index=self.config["start_index"],
            length=self.config["length"])
        return x


class TestConcatConvert(OPConvertAutoScanTest):
    """
    api: paddle.fluid.contrib.layers.partial_*
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=8), min_size=2, max_size=2))

        dtype = draw(st.sampled_from(["float32", "float64", "int64"]))

        start_index = draw(
            st.integers(
                min_value=0, max_value=len(input_shape) - 1))

        length = draw(
            st.integers(
                min_value=-1, max_value=len(input_shape) - start_index))
        if length == 0:
            length = 1

        repeat_times = draw(st.integers(min_value=1, max_value=3))

        op_name = draw(st.sampled_from(["partial_sum", "partial_concat"]))

        config = {
            "op_names": [op_name],
            "test_data_shapes": [input_shape, input_shape],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "start_index": start_index,
            "length": length,
            "repeat_times": repeat_times,
            "use_gpu": False
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=60)


if __name__ == "__main__":
    unittest.main()
