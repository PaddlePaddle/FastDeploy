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
from random import sample


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, logits, label):
        """
        forward
        """
        x = paddle.nn.functional.softmax_with_cross_entropy(
            logits=logits,
            label=label,
            soft_label=self.config["soft_label"],
            return_softmax=self.config["return_softmax"],
            axis=self.config["axis"])
        return x


class TestSoftmaxWithCrossEntropyConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.softmax_with_cross_entropy
    OPset version: 12, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=3, max_value=20), min_size=2, max_size=3))
        axis = draw(
            st.integers(
                min_value=-len(input_shape) + 1, max_value=len(input_shape) -
                1))
        soft_label = draw(st.booleans())
        return_softmax = draw(st.booleans())

        dtype = draw(st.sampled_from(["float32"]))
        if soft_label:
            label_dtype = draw(st.sampled_from(["float32"]))
        else:
            label_dtype = draw(st.sampled_from(["int64"]))

        def generator_label():
            label_shape = input_shape
            if soft_label:
                label_dtype = draw(st.sampled_from(["float32"]))
                label = np.random.random(label_shape)
            else:
                label_shape[axis] = 1
                label = np.random.randint(
                    0, input_shape[axis], size=label_shape)
            return label

        print("input:", input_shape)
        config = {
            "op_names": ["softmax_with_cross_entropy"],
            "test_data_shapes": [input_shape, generator_label],
            "test_data_types": [[dtype], [label_dtype]],
            "opset_version": [12, 15],
            "input_spec_shape": [],
            "axis": axis,
            "soft_label": soft_label,
            "return_softmax": return_softmax,
            "use_gpu": False,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
