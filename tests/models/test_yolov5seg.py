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

from fastdeploy import ModelFormat
import fastdeploy as fd
import cv2
import os
import pickle
import numpy as np
import runtime_config as rc


def test_detection_yolov5seg():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s-seg.onnx"
    input_url1 = "https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg"
    input_url2 = "https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000570688.jpg"
    result_url1 = "https://bj.bcebos.com/paddlehub/fastdeploy/yolov5seg_result1.pkl"
    result_url2 = "https://bj.bcebos.com/paddlehub/fastdeploy/yolov5seg_result2.pkl"
    fd.download(model_url, "resources")
    fd.download(input_url1, "resources")
    fd.download(input_url2, "resources")
    fd.download(result_url1, "resources")
    fd.download(result_url2, "resources")

    model_file = "resources/yolov5s-seg.onnx"
    rc.test_option.use_ort_backend()
    model = fd.vision.detection.YOLOv5Seg(
        model_file, runtime_option=rc.test_option)

    with open("resources/yolov5seg_result1.pkl", "rb") as f:
        expect1 = pickle.load(f)

    with open("resources/yolov5seg_result2.pkl", "rb") as f:
        expect2 = pickle.load(f)

    # compare diff
    im1 = cv2.imread("./resources/000000014439.jpg")
    im2 = cv2.imread("./resources/000000570688.jpg")

    for i in range(3):
        # test single predict
        result1 = model.predict(im1)
        result2 = model.predict(im2)

        diff_boxes_1 = np.fabs(
            np.array(result1.boxes) - np.array(expect1["boxes"]))
        diff_boxes_2 = np.fabs(
            np.array(result2.boxes) - np.array(expect2["boxes"]))

        diff_label_1 = np.fabs(
            np.array(result1.label_ids) - np.array(expect1["label_ids"]))
        diff_label_2 = np.fabs(
            np.array(result2.label_ids) - np.array(expect2["label_ids"]))

        diff_scores_1 = np.fabs(
            np.array(result1.scores) - np.array(expect1["scores"]))
        diff_scores_2 = np.fabs(
            np.array(result2.scores) - np.array(expect2["scores"]))

        # for masks
        for j in range(np.array(result1.boxes).shape[0]):
            result_mask_1 = np.array(result1.masks[j].data).reshape(
                result1.masks[j].shape)
            diff_mask_1 = np.fabs(result_mask_1 - np.array(expect1["mask_" +
                                                                   str(j)]))
            nonzero_nums = np.count_nonzero(diff_mask_1)
            nonzero_count = nonzero_nums / (diff_mask_1.shape[0] *
                                            diff_mask_1.shape[1])
            assert nonzero_count < 1e-02, "The different pixel ratio of mask1 is greater than 1%."

        for k in range(np.array(result2.boxes).shape[0]):
            result_mask_2 = np.array(result2.masks[k].data).reshape(
                result2.masks[k].shape)
            diff_mask_2 = np.fabs(result_mask_2 - np.array(expect2["mask_" +
                                                                   str(k)]))
            nonzero_nums = np.count_nonzero(diff_mask_2)
            nonzero_count = nonzero_nums / (diff_mask_2.shape[0] *
                                            diff_mask_2.shape[1])
            assert nonzero_count < 1e-02, "The different pixel ratio of mask2 is greater than 1%."

        assert diff_boxes_1.max(
        ) < 1e-01, "There's difference in detection boxes 1."
        assert diff_label_1.max(
        ) < 1e-02, "There's difference in detection label 1."
        assert diff_scores_1.max(
        ) < 1e-04, "There's difference in detection score 1."

        assert diff_boxes_2.max(
        ) < 1e-01, "There's difference in detection boxes 2."
        assert diff_label_2.max(
        ) < 1e-02, "There's difference in detection label 2."
        assert diff_scores_2.max(
        ) < 1e-04, "There's difference in detection score 2."

        # test batch predict
        results = model.batch_predict([im1, im2])
        result1 = results[0]
        result2 = results[1]

        diff_boxes_1 = np.fabs(
            np.array(result1.boxes) - np.array(expect1["boxes"]))
        diff_boxes_2 = np.fabs(
            np.array(result2.boxes) - np.array(expect2["boxes"]))

        diff_label_1 = np.fabs(
            np.array(result1.label_ids) - np.array(expect1["label_ids"]))
        diff_label_2 = np.fabs(
            np.array(result2.label_ids) - np.array(expect2["label_ids"]))

        diff_scores_1 = np.fabs(
            np.array(result1.scores) - np.array(expect1["scores"]))
        diff_scores_2 = np.fabs(
            np.array(result2.scores) - np.array(expect2["scores"]))

        # for masks
        for j in range(np.array(result1.boxes).shape[0]):
            result_mask_1 = np.array(result1.masks[j].data).reshape(
                result1.masks[j].shape)
            diff_mask_1 = np.fabs(result_mask_1 - np.array(expect1["mask_" +
                                                                   str(j)]))
            nonzero_nums = np.count_nonzero(diff_mask_1)
            nonzero_count = nonzero_nums / (diff_mask_1.shape[0] *
                                            diff_mask_1.shape[1])
            assert nonzero_count < 1e-02, "The different pixel ratio of mask1 is greater than 1%."

        for k in range(np.array(result2.boxes).shape[0]):
            result_mask_2 = np.array(result2.masks[k].data).reshape(
                result2.masks[k].shape)
            diff_mask_2 = np.fabs(result_mask_2 - np.array(expect2["mask_" +
                                                                   str(k)]))
            nonzero_nums = np.count_nonzero(diff_mask_2)
            nonzero_count = nonzero_nums / (diff_mask_2.shape[0] *
                                            diff_mask_2.shape[1])
            assert nonzero_count < 1e-02, "The different pixel ratio of mask2 is greater than 1%."

        assert diff_boxes_1.max(
        ) < 1e-01, "There's difference in detection boxes 1."
        assert diff_label_1.max(
        ) < 1e-02, "There's difference in detection label 1."
        assert diff_scores_1.max(
        ) < 1e-03, "There's difference in detection score 1."

        assert diff_boxes_2.max(
        ) < 1e-01, "There's difference in detection boxes 2."
        assert diff_label_2.max(
        ) < 1e-02, "There's difference in detection label 2."
        assert diff_scores_2.max(
        ) < 1e-04, "There's difference in detection score 2."


def test_detection_yolov5seg_runtime():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s-seg.onnx"
    input_url1 = "https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg"
    result_url1 = "https://bj.bcebos.com/paddlehub/fastdeploy/yolov5seg_result1.pkl"
    fd.download(model_url, "resources")
    fd.download(input_url1, "resources")
    fd.download(result_url1, "resources")

    model_file = "resources/yolov5s-seg.onnx"

    preprocessor = fd.vision.detection.YOLOv5SegPreprocessor()
    postprocessor = fd.vision.detection.YOLOv5SegPostprocessor()

    rc.test_option.set_model_path(model_file, model_format=ModelFormat.ONNX)
    rc.test_option.use_ort_backend()
    runtime = fd.Runtime(rc.test_option)

    with open("resources/yolov5seg_result1.pkl", "rb") as f:
        expect1 = pickle.load(f)

    # compare diff
    im1 = cv2.imread("./resources/000000014439.jpg")

    for i in range(3):
        # test runtime
        input_tensors, ims_info = preprocessor.run([im1.copy()])
        output_tensors = runtime.infer({"images": input_tensors[0]})
        results = postprocessor.run(output_tensors, ims_info)
        result1 = results[0]

        diff_boxes_1 = np.fabs(
            np.array(result1.boxes) - np.array(expect1["boxes"]))
        diff_label_1 = np.fabs(
            np.array(result1.label_ids) - np.array(expect1["label_ids"]))
        diff_scores_1 = np.fabs(
            np.array(result1.scores) - np.array(expect1["scores"]))

        # for masks
        for j in range(np.array(result1.boxes).shape[0]):
            result_mask_1 = np.array(result1.masks[j].data).reshape(
                result1.masks[j].shape)
            diff_mask_1 = np.fabs(result_mask_1 - np.array(expect1["mask_" +
                                                                   str(j)]))
            nonzero_nums = np.count_nonzero(diff_mask_1)
            nonzero_count = nonzero_nums / (diff_mask_1.shape[0] *
                                            diff_mask_1.shape[1])
            assert nonzero_count < 1e-02, "The different pixel ratio of mask1 is greater than 1%."

        assert diff_boxes_1.max(
        ) < 1e-01, "There's difference in detection boxes 1."
        assert diff_label_1.max(
        ) < 1e-02, "There's difference in detection label 1."
        assert diff_scores_1.max(
        ) < 1e-04, "There's difference in detection score 1."


if __name__ == "__main__":
    test_detection_yolov5seg()
    test_detection_yolov5seg_runtime()
