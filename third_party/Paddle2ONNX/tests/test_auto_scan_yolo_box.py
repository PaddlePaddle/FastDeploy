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
    """
    simple Net
    """

    def forward(self, inputs_1, inputs_2):
        """
        forward
        """
        anchors = self.config["anchors"]
        class_num = self.config["class_num"]
        conf_thresh = self.config["conf_thresh"]
        downsample_ratio = self.config["downsample_ratio"]
        clip_bbox = self.config["clip_bbox"]
        scale_x_y = self.config["scale_x_y"]
        x = paddle.vision.ops.yolo_box(
            inputs_1,
            inputs_2,
            anchors=anchors,
            class_num=class_num,
            conf_thresh=conf_thresh,
            downsample_ratio=downsample_ratio,
            clip_bbox=clip_bbox,
            scale_x_y=scale_x_y)
        return x


class TestYoloBoxConvert(OPConvertAutoScanTest):
    """
    api: paddle.vision.ops.yolo_box
    OPset version: 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=30), min_size=4, max_size=4))
        input_shape[0] = 1
        input_shape[2] = input_shape[3]
        img_size = [input_shape[0], 2]

        num = draw(st.integers(min_value=2, max_value=4))

        anchors = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=5),
                min_size=num * 2,
                max_size=num * 2))

        class_num = draw(st.integers(min_value=2, max_value=5))

        conf_thresh = draw(st.floats(min_value=0.01, max_value=0.9))

        downsample_ratio = draw(st.sampled_from([32, 16, 8]))

        clip_bbox = draw(st.booleans())

        scale_x_y = draw(st.floats(min_value=1.0, max_value=2.0))

        dtype = draw(st.sampled_from(["float32", "float64"]))

        def generator_data():
            input_data = randtool("int", 320, 640, img_size)
            return input_data

        input_shape[1] = num * (5 + class_num)

        config = {
            "op_names": ["yolo_box"],
            "test_data_shapes": [input_shape, generator_data],
            "test_data_types": [[dtype], ["int32"]],
            "opset_version": [11, 12, 13, 14, 15],
            "input_spec_shape": [],
            "anchors": anchors,
            "class_num": class_num,
            "conf_thresh": conf_thresh,
            "downsample_ratio": downsample_ratio,
            "clip_bbox": clip_bbox,
            "scale_x_y": scale_x_y,
            "use_gpu": False
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
