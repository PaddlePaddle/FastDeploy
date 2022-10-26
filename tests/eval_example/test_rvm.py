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


def test_matting_rvm_cpu():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/rvm.tgz"
    input_url = "https://bj.bcebos.com/paddlehub/fastdeploy/video.mp4"
    fd.download_and_decompress(model_url, ".")
    fd.download(input_url, ".")
    model_path = "rvm/rvm_mobilenetv3_fp32.onnx"
    # use ORT
    runtime_option = fd.RuntimeOption()
    runtime_option.use_ort_backend()
    model = fd.vision.matting.RobustVideoMatting(
        model_path, runtime_option=runtime_option)

    cap = cv2.VideoCapture(input_url)

    frame_id = 0
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        result = model.predict(frame)
        # compare diff
        expect_alpha = np.load("rvm/result_alpha_" + str(frame_id) + ".npy")
        result_alpha = np.array(result.alpha).reshape(1920, 1080)
        diff = np.fabs(expect_alpha - result_alpha)
        thres = 1e-05
        assert diff.max(
        ) < thres, "The label diff is %f, which is bigger than %f" % (
            diff.max(), thres)
        frame_id = frame_id + 1
        cv2.waitKey(30)
        if frame_id >= 10:
            cap.release()
            cv2.destroyAllWindows()
            break


def test_matting_rvm_gpu_trt():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/rvm.tgz"
    input_url = "https://bj.bcebos.com/paddlehub/fastdeploy/video.mp4"
    fd.download_and_decompress(model_url, ".")
    fd.download(input_url, ".")
    model_path = "rvm/rvm_mobilenetv3_trt.onnx"
    # use TRT
    runtime_option = fd.RuntimeOption()
    runtime_option.use_gpu()
    runtime_option.use_trt_backend()
    runtime_option.set_trt_input_shape("src", [1, 3, 1920, 1080])
    runtime_option.set_trt_input_shape("r1i", [1, 1, 1, 1], [1, 16, 240, 135],
                                       [1, 16, 240, 135])
    runtime_option.set_trt_input_shape("r2i", [1, 1, 1, 1], [1, 20, 120, 68],
                                       [1, 20, 120, 68])
    runtime_option.set_trt_input_shape("r3i", [1, 1, 1, 1], [1, 40, 60, 34],
                                       [1, 40, 60, 34])
    runtime_option.set_trt_input_shape("r4i", [1, 1, 1, 1], [1, 64, 30, 17],
                                       [1, 64, 30, 17])
    model = fd.vision.matting.RobustVideoMatting(
        model_path, runtime_option=runtime_option)

    cap = cv2.VideoCapture("./video.mp4")

    frame_id = 0
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        result = model.predict(frame)
        # compare diff
        expect_alpha = np.load("rvm/result_alpha_" + str(frame_id) + ".npy")
        result_alpha = np.array(result.alpha).reshape(1920, 1080)
        diff = np.fabs(expect_alpha - result_alpha)
        thres = 1e-04
        assert diff.max(
        ) < thres, "The label diff is %f, which is bigger than %f" % (
            diff.max(), thres)
        frame_id = frame_id + 1
        cv2.waitKey(30)
        if frame_id >= 10:
            cap.release()
            cv2.destroyAllWindows()
            break
