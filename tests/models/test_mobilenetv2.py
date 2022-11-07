# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import fastdeploy as fd
import cv2
import os
import pickle
import numpy as np
import runtime_config as rc


def test_classification_mobilenetv2():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV1_x0_25_infer.tgz"
    input_url = "https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg"
    fd.download_and_decompress(model_url, "resources")
    fd.download(input_url, "resources")
    model_path = "resources/MobileNetV1_x0_25_infer"

    model_file = "resources/MobileNetV1_x0_25_infer/inference.pdmodel"
    params_file = "resources/MobileNetV1_x0_25_infer/inference.pdiparams"
    config_file = "resources/MobileNetV1_x0_25_infer/inference_cls.yaml"
    model = fd.vision.classification.PaddleClasModel(
        model_file, params_file, config_file, runtime_option=rc.test_option)

    expected_label_ids = [153, 333, 259, 338, 265, 154]
    expected_scores = [
        0.221088, 0.109457, 0.078668, 0.076814, 0.052401, 0.048206
    ]
    # compare diff
    im = cv2.imread("./resources/ILSVRC2012_val_00000010.jpeg")
    for i in range(2):
        result = model.predict(im, topk=6)
        diff_label = np.fabs(
            np.array(result.label_ids) - np.array(expected_label_ids))
        diff_scores = np.fabs(
            np.array(result.scores) - np.array(expected_scores))
        assert diff_label.max() < 1e-06, "There's difference in classify label."
        assert diff_scores.max(
        ) < 1e-05, "There's difference in classify score."
