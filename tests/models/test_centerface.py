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


def test_facedet_centerface():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/CenterFace.onnx"
    input_url1 = "https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg"
    result_url1 = "https://bj.bcebos.com/paddlehub/fastdeploy/centerface_result1.pkl"
    fd.download(model_url, "resources")
    fd.download(input_url1, "resources")
    fd.download(result_url1, "resources")

    model_file = "resources/CenterFace.onnx"
    model = fd.vision.facedet.CenterFace(
        model_file, runtime_option=rc.test_option)

    with open("resources/centerface_result1.pkl", "rb") as f:
        expect1 = pickle.load(f)

    # compare diff
    im1 = cv2.imread("./resources/test_lite_face_detector_3.jpg")
    print(expect1)
    for i in range(3):
        # test single predict
        result1 = model.predict(im1)

        diff_boxes_1 = np.fabs(
            np.array(result1.boxes) - np.array(expect1["boxes"]))
        diff_scores_1 = np.fabs(
            np.array(result1.scores) - np.array(expect1["scores"]))

        assert diff_boxes_1.max(
        ) < 1e-04, "There's difference in detection boxes 1."
        assert diff_scores_1.max(
        ) < 1e-05, "There's difference in detection score 1."

def test_facedet_centerface_runtime():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/CenterFace.onnx"
    input_url1 = "https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg"
    result_url1 = "https://bj.bcebos.com/paddlehub/fastdeploy/centerface_result1.pkl"
    fd.download(model_url, "resources")
    fd.download(input_url1, "resources")
    fd.download(result_url1, "resources")

    model_file = "resources/CenterFace.onnx"

    preprocessor = fd.vision.facedet.CenterFacePreprocessor()
    postprocessor = fd.vision.facedet.CenterFacePostprocessor()

    rc.test_option.set_model_path(model_file, model_format=ModelFormat.ONNX)
    rc.test_option.use_openvino_backend()
    runtime = fd.Runtime(rc.test_option)

    with open("resources/centerface_result1.pkl", "rb") as f:
        expect1 = pickle.load(f)

    # compare diff
    im1 = cv2.imread("./resources/test_lite_face_detector_3.jpg")

    for i in range(3):
        # test runtime
        input_tensors, ims_info = preprocessor.run([im1.copy()])
        output_tensors = runtime.infer({"input.1": input_tensors[0]})
        results = postprocessor.run(output_tensors, ims_info)
        result1 = results[0]

        diff_boxes_1 = np.fabs(
            np.array(result1.boxes) - np.array(expect1["boxes"]))
        diff_scores_1 = np.fabs(
            np.array(result1.scores) - np.array(expect1["scores"]))

        assert diff_boxes_1.max(
        ) < 1e-04, "There's difference in detection boxes 1."
        assert diff_scores_1.max(
        ) < 1e-05, "There's difference in detection score 1."


if __name__ == "__main__":
    test_facedet_centerface()
    test_facedet_centerface_runtime()