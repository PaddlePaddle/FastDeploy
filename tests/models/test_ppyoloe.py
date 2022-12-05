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


def test_detection_ppyoloe():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz"
    input_url1 = "https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg"
    result_url = "https://bj.bcebos.com/fastdeploy/tests/data/ppyoloe_baseline.pkl"
    fd.download_and_decompress(model_url, "resources")
    fd.download(input_url1, "resources")
    fd.download(result_url, "resources")
    model_path = "resources/ppyoloe_crn_l_300e_coco"

    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    config_file = os.path.join(model_path, "infer_cfg.yml")
    rc.test_option.use_ort_backend()
    model = fd.vision.detection.PPYOLOE(
        model_file, params_file, config_file, runtime_option=rc.test_option)

    # compare diff
    im1 = cv2.imread("./resources/000000014439.jpg")
    for i in range(2):
        result = model.predict(im1)
        with open("resources/ppyoloe_baseline.pkl", "rb") as f:
            boxes, scores, label_ids = pickle.load(f)
        pred_boxes = np.array(result.boxes)
        pred_scores = np.array(result.scores)
        pred_label_ids = np.array(result.label_ids)

        diff_boxes = np.fabs(boxes - pred_boxes)
        diff_scores = np.fabs(scores - pred_scores)
        diff_label_ids = np.fabs(label_ids - pred_label_ids)

        print(diff_boxes.max(), diff_scores.max(), diff_label_ids.max())

        score_threshold = 0.0
        assert diff_boxes[scores > score_threshold].max(
        ) < 1e-04, "There's diff in boxes."
        assert diff_scores[scores > score_threshold].max(
        ) < 1e-04, "There's diff in scores."
        assert diff_label_ids[scores > score_threshold].max(
        ) < 1e-04, "There's diff in label_ids."


def test_detection_ppyoloe1():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz"
    input_url1 = "https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg"
    result_url = "https://bj.bcebos.com/fastdeploy/tests/data/ppyoloe_baseline.pkl"
    fd.download_and_decompress(model_url, "resources")
    fd.download(input_url1, "resources")
    fd.download(result_url, "resources")
    model_path = "resources/ppyoloe_crn_l_300e_coco"

    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    config_file = os.path.join(model_path, "infer_cfg.yml")

    preprocessor = fd.vision.detection.PaddleDetPreprocessor(config_file)
    postprocessor = fd.vision.detection.PaddleDetPostprocessor()

    rc.test_option.set_model_path(model_file, params_file)
    runtime = fd.Runtime(rc.test_option)

    # compare diff
    im1 = cv2.imread("./resources/000000014439.jpg")
    for i in range(2):
        input_tensors = preprocessor.run([im1])
        output_tensors = runtime.infer({
            "image": input_tensors[0],
            "scale_factor": input_tensors[1]
        })
        results = postprocessor.run(output_tensors)
        result = results[0]
        with open("resources/ppyoloe_baseline.pkl", "rb") as f:
            boxes, scores, label_ids = pickle.load(f)
        pred_boxes = np.array(result.boxes)
        pred_scores = np.array(result.scores)
        pred_label_ids = np.array(result.label_ids)

        diff_boxes = np.fabs(boxes - pred_boxes)
        diff_scores = np.fabs(scores - pred_scores)
        diff_label_ids = np.fabs(label_ids - pred_label_ids)

        print(diff_boxes.max(), diff_scores.max(), diff_label_ids.max())

        score_threshold = 0.0
        assert diff_boxes[scores > score_threshold].max(
        ) < 1e-01, "There's diff in boxes."
        assert diff_scores[scores > score_threshold].max(
        ) < 1e-02, "There's diff in scores."
        assert diff_label_ids[scores > score_threshold].max(
        ) < 1e-04, "There's diff in label_ids."


if __name__ == "__main__":
    test_detection_ppyoloe()
    test_detection_ppyoloe1()
