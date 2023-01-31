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


def test_animegan():
    model_name = 'animegan_v1_hayao_60'
    model_path = fd.download_model(
        name=model_name, path='./resources', format='paddle')
    test_img = 'https://bj.bcebos.com/paddlehub/fastdeploy/style_transfer_testimg.jpg'
    label_img = 'https://bj.bcebos.com/paddlehub/fastdeploy/style_transfer_result.png'
    fd.download(test_img, "./resources")
    fd.download(label_img, "./resources")
    # use default backend
    runtime_option = fd.RuntimeOption()
    runtime_option.set_paddle_mkldnn(False)
    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    animegan = fd.vision.generation.AnimeGAN(
        model_file, params_file, runtime_option=runtime_option)

    src_img = cv2.imread("./resources/style_transfer_testimg.jpg")
    label_img = cv2.imread("./resources/style_transfer_result.png")
    res = animegan.predict(src_img)

    diff = np.fabs(res.astype(np.float32) - label_img.astype(np.float32)) / 255
    assert diff.max() < 1e-04, "There's diff in prediction."


if __name__ == "__main__":
    test_animegan()
