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


class Net0(BaseNet):
    """
    simple Net
    """

    def __init__(self, config=None):
        super(Net0, self).__init__(config)
        self.lstm = paddle.nn.LSTM(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            direction=self.config["direction"],
            time_major=self.config["time_major"])

    def forward(self, inputs, prev_h, prev_c):
        """
        forward
        """
        y, (h, c) = self.lstm(inputs, (prev_h, prev_c))
        return y


class Net1(BaseNet):
    """
    simple Net
    """

    def __init__(self, config=None):
        super(Net1, self).__init__(config)
        self.gru = paddle.nn.GRU(input_size=self.config["input_size"],
                                 hidden_size=self.config["hidden_size"],
                                 num_layers=self.config["num_layers"],
                                 direction=self.config["direction"],
                                 time_major=self.config["time_major"])

    def forward(self, inputs, prev_h):
        """
        forward
        """
        y, h = self.gru(inputs, prev_h)
        return y


class TestRNNConvert0(OPConvertAutoScanTest):
    """
    api: paddle.nn.LSTM
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=10), min_size=3, max_size=3))

        dtype = draw(st.sampled_from(["float32"]))
        hidden_size = 32
        num_layers = 2
        time_major = draw(st.booleans())
        if time_major == True:
            t, b, input_size = input_shape
        else:
            b, t, input_size = input_shape
        direction = draw(st.sampled_from(["forward", "bidirect"]))
        if direction == "forward":
            num_directions = 1
        else:
            num_directions = 2

        prev_h_shape = [num_layers * num_directions, b, hidden_size]

        prev_c_shape = [num_layers * num_directions, b, hidden_size]

        config = {
            "op_names": ["rnn"],
            "test_data_shapes": [input_shape, prev_h_shape, prev_c_shape],
            "test_data_types": [[dtype], [dtype], [dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "direction": direction,
            "time_major": time_major,
        }

        models = Net0(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class TestRNNConvert1(OPConvertAutoScanTest):
    """
    api: paddle.nn.GRU
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=10), min_size=3, max_size=3))

        dtype = draw(st.sampled_from(["float32"]))
        hidden_size = 32
        num_layers = 2
        time_major = draw(st.booleans())
        if time_major == True:
            t, b, input_size = input_shape
        else:
            b, t, input_size = input_shape
        direction = draw(st.sampled_from(["forward", "bidirect"]))
        if direction == "forward":
            num_directions = 1
        else:
            num_directions = 2

        prev_h_shape = [num_layers * num_directions, b, hidden_size]

        config = {
            "op_names": ["rnn"],
            "test_data_shapes": [input_shape, prev_h_shape],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "direction": direction,
            "time_major": time_major,
        }

        models = Net1(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
