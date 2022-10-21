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
import numpy as np


def test_pptracking_cpu():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/fairmot_hrnetv2_w18_dlafpn_30e_576x320.tgz"
    input_url = "https://bj.bcebos.com/paddlehub/fastdeploy/person.mp4"
    fd.download_and_decompress(model_url, ".")
    fd.download(input_url, ".")
    model_path = "pptracking/fairmot_hrnetv2_w18_dlafpn_30e_576x320"
    # use default backend
    runtime_option = fd.RuntimeOption()
    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    config_file = os.path.join(model_path, "infer_cfg.yml")
    model = fd.vision.tracking.PPTracking(model_file, params_file, config_file, runtime_option=runtime_option)
    cap = cv2.VideoCapture(input_url)
    frame_id = 0
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        result = model.predict(frame)
        # compare diff
        expect_scores = np.load("pptracking/tracking_result" + str(frame_id) + ".npy")
        result_scores = np.array(result.scores)
        diff = np.fabs(expect_scores - result_scores)
        thres = 1e-05
        assert diff.max() < thres, "The label diff is %f, which is bigger than %f" % (diff.max(), thres)
        frame_id = frame_id + 1
        cv2.waitKey(30)
        if frame_id >= 10:
            cap.release()
            cv2.destroyAllWindows()
            break


def test_pptracking_gpu_trt():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/fairmot_hrnetv2_w18_dlafpn_30e_576x320.tgz"
    input_url = "https://bj.bcebos.com/paddlehub/fastdeploy/person.mp4"
    fd.download_and_decompress(model_url, ".")
    fd.download(input_url, ".")
    model_path = "pptracking/fairmot_hrnetv2_w18_dlafpn_30e_576x320"
    runtime_option = fd.RuntimeOption()
    runtime_option.use_gpu()
    runtime_option.use_trt_backend()
    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    config_file = os.path.join(model_path, "infer_cfg.yml")
    model = fd.vision.tracking.PPTracking(model_file, params_file, config_file, runtime_option=runtime_option)
    cap = cv2.VideoCapture(input_url)
    frame_id = 0
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        result = model.predict(frame)
        # compare diff
        expect_scores = np.load("pptracking/tracking_result" + str(frame_id) + ".npy")
        result_scores = np.array(result.scores)
        diff = np.fabs(expect_scores - result_scores)
        thres = 1e-05
        assert diff.max() < thres, "The label diff is %f, which is bigger than %f" % (diff.max(), thres)
        frame_id = frame_id + 1
        cv2.waitKey(30)
        if frame_id >= 10:
            cap.release()
            cv2.destroyAllWindows()
            break
