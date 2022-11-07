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
    input_url1 = "https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg"
    input_url2 = "https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00030010.jpeg"
    fd.download_and_decompress(model_url, "resources")
    fd.download(input_url1, "resources")
    fd.download(input_url2, "resources")
    model_path = "resources/MobileNetV1_x0_25_infer"

    model_file = "resources/MobileNetV1_x0_25_infer/inference.pdmodel"
    params_file = "resources/MobileNetV1_x0_25_infer/inference.pdiparams"
    config_file = "resources/MobileNetV1_x0_25_infer/inference_cls.yaml"
    model = fd.vision.classification.PaddleClasModel(
        model_file, params_file, config_file, runtime_option=rc.test_option)

    expected_label_ids_1 = [153, 333, 259, 338, 265, 154]
    expected_scores_1 = [
        0.221088, 0.109457, 0.078668, 0.076814, 0.052401, 0.048206
    ]
    expected_label_ids_2 = [80, 23, 93, 99, 143, 7]
    expected_scores_2 = [
        0.975599, 0.014083, 0.003821, 0.001571, 0.001233, 0.000924
    ]

    # compare diff
    im1 = cv2.imread("./resources/ILSVRC2012_val_00000010.jpeg")
    im2 = cv2.imread("./resources/ILSVRC2012_val_00030010.jpeg")

    #    for i in range(3000000):
    while True:
        # test single predict
        model.postprocessor.topk = 6
        result1 = model.predict(im1)
        result2 = model.predict(im2)

        diff_label_1 = np.fabs(
            np.array(result1.label_ids) - np.array(expected_label_ids_1))
        diff_label_2 = np.fabs(
            np.array(result2.label_ids) - np.array(expected_label_ids_2))

        diff_scores_1 = np.fabs(
            np.array(result1.scores) - np.array(expected_scores_1))
        diff_scores_2 = np.fabs(
            np.array(result2.scores) - np.array(expected_scores_2))
        assert diff_label_1.max(
        ) < 1e-06, "There's difference in classify label 1."
        assert diff_scores_1.max(
        ) < 1e-05, "There's difference in classify score 1."
        assert diff_label_2.max(
        ) < 1e-06, "There's difference in classify label 2."
        assert diff_scores_2.max(
        ) < 1e-05, "There's difference in classify score 2."

        # test batch predict
        results = model.batch_predict([im1, im2])
        result1 = results[0]
        result2 = results[1]

        diff_label_1 = np.fabs(
            np.array(result1.label_ids) - np.array(expected_label_ids_1))
        diff_label_2 = np.fabs(
            np.array(result2.label_ids) - np.array(expected_label_ids_2))

        diff_scores_1 = np.fabs(
            np.array(result1.scores) - np.array(expected_scores_1))
        diff_scores_2 = np.fabs(
            np.array(result2.scores) - np.array(expected_scores_2))
        assert diff_label_1.max(
        ) < 1e-06, "There's difference in classify label 1."
        assert diff_scores_1.max(
        ) < 1e-05, "There's difference in classify score 1."
        assert diff_label_2.max(
        ) < 1e-06, "There's difference in classify label 2."
        assert diff_scores_2.max(
        ) < 1e-05, "There's difference in classify score 2."


if __name__ == "__main__":
    test_classification_mobilenetv2()
