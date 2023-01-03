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


def test_edvr():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/edvr.tgz"
    input_url = "https://bj.bcebos.com/paddlehub/fastdeploy/vsr_src.mp4"
    fd.download_and_decompress(model_url, "resources")
    fd.download(input_url, "resources")
    model_path = "resources/edvr/EDVR_M_wo_tsa_SRx4"
    # use default backend
    runtime_option = fd.RuntimeOption()
    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    model = fd.vision.sr.EDVR(
        model_file, params_file, runtime_option=rc.test_option)

    # 该处应该与你导出模型的第二个维度一致模型输入shape=[b,n,c,h,w]
    capture = cv2.VideoCapture("./resources/vsr_src.mp4")
    # Create VideoWriter for output
    frame_id = 0
    imgs = []
    t = 0
    while capture.isOpened():
        ret, frame = capture.read()
        if frame_id < 5 and frame is not None:
            imgs.append(frame)
            frame_id += 1
            continue
        # 始终保持imgs队列中具有frame_num帧
        imgs.pop(0)
        imgs.append(frame)
        frame_id += 1
        # 视频读取完毕退出
        if not ret:
            break
        results = model.predict(imgs)
        for item in results:
            if frame_id <= 4:
                continue
            if t < 10:
                ret = pickle.load(
                    open("./resources/edvr/frame_" + str(t) + ".pkl", "rb"))
                mean_diff = np.fabs(ret["mean"] - item.mean())
                std_diff = np.fabs(ret["std"] - item.std())
                shape_diff = max(
                    np.fabs(np.array(ret["shape"]) - np.array(item.shape)))
                thres = 1e-03
                assert mean_diff < thres, "The mean diff is %f, which is bigger than %f" % (
                    mean_diff, thres)
                assert std_diff < thres, "The std diff is %f, which is bigger than %f" % (
                    std_diff, thres)
                assert shape_diff <= 0, "The shape diff is %f, which is bigger than %f" % (
                    shape_diff, 0)
            t = t + 1
        if t >= 10:
            break
    capture.release()


if __name__ == "__main__":
    test_edvr()
