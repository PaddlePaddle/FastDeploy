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
import runtime_config as rc

def test_facealignment_pfld():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/pfld-106-lite.onnx"
    input_url = "https://bj.bcebos.com/paddlehub/fastdeploy/facealign_input.png"
    output_url = "https://bj.bcebos.com/paddlehub/fastdeploy/result_landmarks.npy"
    fd.download(model_url, "resources")
    fd.download(input_url, "resources")
    fd.download(output_url, "resources")
    model_path = "resources/pfld-106-lite.onnx"
    # use ORT
    model = fd.vision.facealign.PFLD(model_path, runtime_option=rc.test_option)

    # compare diff
    im = cv2.imread("resources/facealign_input.png")
    for i in range(2):
        result = model.predict(im)
        expect = np.load("resources/result_landmarks.npy")
    
        diff = np.fabs(np.array(result.landmarks) - expect)
        thres = 1e-04
        assert diff.max() < thres, "The diff is %f, which is bigger than %f" % (
            diff.max(), thres)
