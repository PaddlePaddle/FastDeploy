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


def test_basicvsr():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/basicvsr.tgz"
    input_url = "https://bj.bcebos.com/paddlehub/fastdeploy/vsr_src.mp4"
    fd.download_and_decompress(model_url, "resources")
    fd.download(input_url, "resources")
    model_path = "resources/basicvsr/BasicVSR_reds_x4"
    # use default backend
    runtime_option = fd.RuntimeOption()
    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    model = fd.vision.sr.PPMSVSR(
        model_file, params_file, runtime_option=rc.test_option)
    # 该处应该与你导出模型的第二个维度一致模型输入shape=[b,n,c,h,w]
    capture = cv2.VideoCapture("./resources/vsr_src.mp4")
    frame_id = 0
    reach_end = False
    t = 0
    while capture.isOpened():
        imgs = []
        for i in range(2):
            _, frame = capture.read()
            if frame is not None:
                imgs.append(frame)
            else:
                reach_end = True
        if reach_end:
            break
        results = model.predict(imgs)
        for item in results:
            if t < 10:
                ret = pickle.load(
                    open("./resources/basicvsr/frame_" + str(t) + ".pkl",
                         "rb"))
                mean_diff = np.fabs(ret["mean"] - item.mean())
                std_diff = np.fabs(ret["std"] - item.std())
                shape_diff = max(
                    np.fabs(np.array(ret["shape"]) - np.array(item.shape)))
                thres = 1e-02
                assert mean_diff < thres, "The mean diff is %f, which is bigger than %f" % (
                    mean_diff, thres)
                assert std_diff < thres, "The std diff is %f, which is bigger than %f" % (
                    std_diff, thres)
                assert shape_diff <= 0, "The shape diff is %f, which is bigger than %f" % (
                    shape_diff, 0)
            t = t + 1
            frame_id += 1
        if t >= 10:
            break
    capture.release()


if __name__ == "__main__":
    test_basicvsr()
