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
import pickle
import runtime_config as rc


def test_pptracking():
    model_url = "https://bj.bcebos.com/fastdeploy/tests/pptracking.tgz"
    input_url = "https://bj.bcebos.com/paddlehub/fastdeploy/person.mp4"
    fd.download_and_decompress(model_url, "resources")
    fd.download(input_url, "resources")
    model_path = "resources/pptracking/fairmot_hrnetv2_w18_dlafpn_30e_576x320"
    # use default backend
    runtime_option = fd.RuntimeOption()
    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    config_file = os.path.join(model_path, "infer_cfg.yml")
    model = fd.vision.tracking.PPTracking(model_file, params_file, config_file, runtime_option=rc.test_option)
    cap = cv2.VideoCapture("./resources/person.mp4")
    frame_id = 0
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        result = model.predict(frame)
        # compare diff
        expect = pickle.load(open("resources/pptracking/frame" + str(frame_id) + ".pkl", "rb"))
        diff_boxes = np.fabs(np.array(expect["boxes"]) - np.array(result.boxes))
        diff_scores = np.fabs(np.array(expect["scores"]) - np.array(result.scores))
        diff = max(diff_boxes.max(), diff_scores.max())
        thres = 1e-05
        assert diff < thres, "The label diff is %f, which is bigger than %f" % (diff, thres)
        frame_id = frame_id + 1
        cv2.waitKey(30)
        if frame_id >= 10:
            cap.release()
            cv2.destroyAllWindows()
            break
