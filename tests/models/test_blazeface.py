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


def test_detection_blazeface():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/blazeface_1000e.tgz"
    input_url1 = "https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg"
    input_url2 = "https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000570688.jpg"
    result_url1 = "https://bj.bcebos.com/paddlehub/fastdeploy/blazeface_result1.pkl"
    result_url2 = "https://bj.bcebos.com/paddlehub/fastdeploy/blazeface_result2.pkl"
    fd.download_and_decompress(model_url, "resources")
    fd.download(input_url1, "resources")
    fd.download(input_url2, "resources")


    model_dir = "resources/blazeface_1000e"
    model_file = os.path.join(model_dir, "model.pdmodel")
    params_file = os.path.join(model_dir, "model.pdiparams")
    config_file = os.path.join(model_dir, "infer_cfg.yml")
    model = fd.vision.facedet.BlazeFace(
        model_file, params_file, config_file, runtime_option=rc.test_option)
    model.postprocessor.conf_threshold = 0.5

    with open("resources/blazeface_result1.pkl", "rb") as f:
        expect1 = pickle.load(f)

    with open("resources/blazeface_result2.pkl", "rb") as f:
        expect2 = pickle.load(f)

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

        diff_scores_1 = np.fabs(
            np.array(result1.scores) - np.array(expect1["scores"]))
        diff_scores_2 = np.fabs(
            np.array(result2.scores) - np.array(expect2["scores"]))

        assert diff_boxes_1.max(
        ) < 1e-04, "There's difference in detection boxes 1."
        assert diff_scores_1.max(
        ) < 1e-04, "There's difference in detection score 1."

        assert diff_boxes_2.max(
        ) < 1e-03, "There's difference in detection boxes 2."
        assert diff_scores_2.max(
        ) < 1e-04, "There's difference in detection score 2."

        print("one image test success!")

        # test batch predict
        results = model.batch_predict([im1, im2])
        result1 = results[0]
        result2 = results[1]

        diff_boxes_1 = np.fabs(
            np.array(result1.boxes) - np.array(expect1["boxes"]))
        diff_boxes_2 = np.fabs(
            np.array(result2.boxes) - np.array(expect2["boxes"]))

        diff_scores_1 = np.fabs(
            np.array(result1.scores) - np.array(expect1["scores"]))
        diff_scores_2 = np.fabs(
            np.array(result2.scores) - np.array(expect2["scores"]))
        assert diff_boxes_1.max(
        ) < 1e-04, "There's difference in detection boxes 1."
        assert diff_scores_1.max(
        ) < 1e-03, "There's difference in detection score 1."

        assert diff_boxes_2.max(
        ) < 1e-04, "There's difference in detection boxes 2."
        assert diff_scores_2.max(
        ) < 1e-04, "There's difference in detection score 2."

        print("batch predict success!")


def test_detection_blazeface_runtime():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/blazeface_1000e.tgz"
    input_url1 = "https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg"
    result_url1 = "https://bj.bcebos.com/paddlehub/fastdeploy/blazeface_result1.pkl"
    fd.download_and_decompress(model_url, "resources")
    fd.download(input_url1, "resources")
    fd.download(result_url1, "resources")

    model_dir = "resources/blazeface_1000e"
    model_file = os.path.join(model_dir, "model.pdmodel")
    params_file = os.path.join(model_dir, "model.pdiparams")
    config_file = os.path.join(model_dir, "infer_cfg.yml")

    preprocessor = fd.vision.facedet.BlazeFacePreprocessor()
    postprocessor = fd.vision.facedet.BlazeFacePostprocessor()

    rc.test_option.set_model_path(model_file, params_file, config_file, model_format=ModelFormat.PADDLE)
    rc.test_option.use_openvino_backend()
    runtime = fd.Runtime(rc.test_option)

    with open("resources/blazeface_result1.pkl", "rb") as f:
        expect1 = pickle.load(f)

    im1 = cv2.imread("resources/000000014439.jpg")

    for i in range(3):
        # test runtime
        input_tensors, ims_info = preprocessor.run([im1.copy()])
        output_tensors = runtime.infer({"images": input_tensors[0]})
        results = postprocessor.run(output_tensors, ims_info)
        result1 = results[0]

        diff_boxes_1 = np.fabs(
            np.array(result1.boxes) - np.array(expect1["boxes"]))
        diff_scores_1 = np.fabs(
            np.array(result1.scores) - np.array(expect1["scores"]))

        assert diff_boxes_1.max(
        ) < 1e-03, "There's difference in detection boxes 1."
        assert diff_scores_1.max(
        ) < 1e-04, "There's difference in detection score 1."


if __name__ == "__main__":
    test_detection_blazeface()
    test_detection_blaze_runtime()
