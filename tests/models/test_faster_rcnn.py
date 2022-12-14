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
print(fd.__path__)
import cv2
import os
import pickle
import numpy as np
import runtime_config as rc


def test_detection_faster_rcnn():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/faster_rcnn_r50_vd_fpn_2x_coco.tgz"
    input_url1 = "https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg"
    result_url = "https://bj.bcebos.com/fastdeploy/tests/data/faster_rcnn_baseline.pkl"
    fd.download_and_decompress(model_url, "resources")
    fd.download(input_url1, "resources")
    fd.download(result_url, "resources")
    model_path = "resources/faster_rcnn_r50_vd_fpn_2x_coco"

    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    config_file = os.path.join(model_path, "infer_cfg.yml")
    rc.test_option.use_paddle_backend()
    model = fd.vision.detection.FasterRCNN(
        model_file, params_file, config_file, runtime_option=rc.test_option)

    # compare diff
    im1 = cv2.imread("./resources/000000014439.jpg")
    for i in range(2):
        result = model.predict(im1)
        with open("resources/faster_rcnn_baseline.pkl", "rb") as f:
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


#    result = model.predict(im1)
#    with open("faster_rcnn_baseline.pkl", "wb") as f:
#        pickle.dump([np.array(result.boxes), np.array(result.scores), np.array(result.label_ids)], f)


def test_detection_faster_rcnn1():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/faster_rcnn_r50_vd_fpn_2x_coco.tgz"
    input_url1 = "https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg"
    result_url = "https://bj.bcebos.com/fastdeploy/tests/data/faster_rcnn_baseline.pkl"
    fd.download_and_decompress(model_url, "resources")
    fd.download(input_url1, "resources")
    fd.download(result_url, "resources")
    model_path = "resources/faster_rcnn_r50_vd_fpn_2x_coco"

    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    config_file = os.path.join(model_path, "infer_cfg.yml")
    preprocessor = fd.vision.detection.PaddleDetPreprocessor(config_file)
    postprocessor = fd.vision.detection.PaddleDetPostprocessor()

    option = rc.test_option
    option.set_model_path(model_file, params_file)
    option.use_paddle_infer_backend()
    runtime = fd.Runtime(option)

    # compare diff
    for i in range(2):
        im1 = cv2.imread("./resources/000000014439.jpg")
        input_tensors = preprocessor.run([im1])
        output_tensors = runtime.infer({
            "image": input_tensors[0],
            "scale_factor": input_tensors[1],
            "im_shape": input_tensors[2]
        })
        results = postprocessor.run(output_tensors)
        result = results[0]

        with open("resources/faster_rcnn_baseline.pkl", "rb") as f:
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


# test runtime.zero_copy_infer and bind_input_tensor get_output_tensor
def test_detection_faster_rcnn2():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/faster_rcnn_r50_vd_fpn_2x_coco.tgz"
    input_url1 = "https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg"
    result_url = "https://bj.bcebos.com/fastdeploy/tests/data/faster_rcnn_baseline.pkl"
    fd.download_and_decompress(model_url, "resources")
    fd.download(input_url1, "resources")
    fd.download(result_url, "resources")
    model_path = "resources/faster_rcnn_r50_vd_fpn_2x_coco"

    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    config_file = os.path.join(model_path, "infer_cfg.yml")
    preprocessor = fd.vision.detection.PaddleDetPreprocessor(config_file)
    postprocessor = fd.vision.detection.PaddleDetPostprocessor()

    option = rc.test_option
    option.set_model_path(model_file, params_file)
    option.use_paddle_infer_backend()
    runtime = fd.Runtime(option)

    # compare diff
    input_names = ["image", "scale_factor", "im_shape"]
    output_names = ["concat_12.tmp_0", "concat_8.tmp_0"]
    for i in range(2):
        im1 = cv2.imread("./resources/000000014439.jpg")
        input_tensors = preprocessor.run([im1.copy(), ])
        for i, input_tensor in enumerate(input_tensors):
            runtime.bind_input_tensor(input_names[i], input_tensor)
        runtime.zero_copy_infer()
        output_tensors = []
        for name in output_names:
            output_tensor = runtime.get_output_tensor(name)
            output_tensors.append(output_tensor)
        results = postprocessor.run(output_tensors)
        result = results[0]

        with open("resources/faster_rcnn_baseline.pkl", "rb") as f:
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


if __name__ == "__main__":
    test_detection_faster_rcnn()
    test_detection_faster_rcnn1()
    test_detection_faster_rcnn2()
