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

def test_classification_yolov5cls():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/yolov5n-cls.tgz"
    input_url = "https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg"
    fd.download_and_decompress(model_url, ".")
    fd.download(input_url, ".")
    model_path = "yolov5n-cls/yolov5n-cls.onnx"
    # use ORT
    runtime_option = fd.RuntimeOption()
    runtime_option.use_ort_backend()
    model = fd.vision.classification.YOLOv5Cls(
        model_path, runtime_option=rc.test_option)

    # compare diff
    im = cv2.imread("./ILSVRC2012_val_00000010.jpeg")
    result = model.predict(im.copy(), topk=5)
    with open("yolov5n-cls/result.pkl", "rb") as f:
        expect = pickle.load(f)

    diff_label = np.fabs(
        np.array(result.label_ids) - np.array(expect["labels"]))
    diff_score = np.fabs(np.array(result.scores) - np.array(expect["scores"]))
    thres = 1e-05
    assert diff_label.max(
    ) < thres, "The label diff is %f, which is bigger than %f" % (
        diff_label.max(), thres)
    assert diff_score.max(
    ) < thres, "The score diff is %f, which is bigger than %f" % (
        diff_score.max(), thres)
