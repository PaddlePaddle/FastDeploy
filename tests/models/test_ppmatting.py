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

def test_matting_ppmatting():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-512.tgz"
    input_url = "https://bj.bcebos.com/paddlehub/fastdeploy/matting_input.jpg"
    fd.download_and_decompress(model_url, "resources")
    fd.download(input_url, "resources")
    model_path = "./resources/PP-Matting-512"
    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    config_file = os.path.join(model_path, "deploy.yaml")
    model = fd.vision.matting.PPMatting(
        model_file, params_file, config_file, runtime_option=rc.test_option)

    # 预测图片抠图结果
    im = cv2.imread("./resources/matting_input.jpg")
    for i in range(2):
        result = model.predict(im)
        pkl_url = "https://bj.bcebos.com/fastdeploy/tests/ppmatting_result.pkl"
        if pkl_url:
            fd.download(pkl_url, "resources")
        with open("./resources/ppmatting_result.pkl", "rb") as f:
            baseline = pickle.load(f)
    
        diff = np.fabs(np.array(result.alpha) - np.array(baseline))
        thres = 1e-05
        assert diff.max() < thres, "The diff is %f, which is bigger than %f" % (
            diff.max(), thres)


def test_matting_ppmodnet():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/PPModnet_MobileNetV2.tgz"
    input_url = "https://bj.bcebos.com/paddlehub/fastdeploy/matting_input.jpg"
    fd.download_and_decompress(model_url, "resources")
    fd.download(input_url, "resources")
    model_path = "./resources/PPModnet_MobileNetV2"
    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    config_file = os.path.join(model_path, "deploy.yaml")
    model = fd.vision.matting.PPMatting(
        model_file, params_file, config_file, runtime_option=rc.test_option)

    # 预测图片抠图结果
    im = cv2.imread("./resources/matting_input.jpg")

    for i in range(2):
        result = model.predict(im)
    
        pkl_url = "https://bj.bcebos.com/fastdeploy/tests/ppmodnet_result.pkl"
        if pkl_url:
            fd.download(pkl_url, "resources")
        with open("./resources/ppmodnet_result.pkl", "rb") as f:
            baseline = pickle.load(f)
    
        diff = np.fabs(np.array(result.alpha) - np.array(baseline))
        thres = 1e-05
        assert diff.max() < thres, "The diff is %f, which is bigger than %f" % (
            diff.max(), thres)


def test_matting_pphumanmatting():
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/PPHumanMatting.tgz"
    input_url = "https://bj.bcebos.com/paddlehub/fastdeploy/matting_input.jpg"
    fd.download_and_decompress(model_url, "resources")
    fd.download(input_url, "resources")
    model_path = "./resources/PPHumanMatting"
    # 配置runtime，加载模型
    runtime_option = fd.RuntimeOption()
    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    config_file = os.path.join(model_path, "deploy.yaml")
    model = fd.vision.matting.PPMatting(
        model_file, params_file, config_file, runtime_option=rc.test_option)

    # 预测图片抠图结果
    im = cv2.imread("./resources/matting_input.jpg")
    for i in range(2):
        result = model.predict(im)
    
        pkl_url = "https://bj.bcebos.com/fastdeploy/tests/pphumanmatting_result.pkl"
        if pkl_url:
            fd.download(pkl_url, "resources")
    
        with open("./resources/pphumanmatting_result.pkl", "rb") as f:
            baseline = pickle.load(f)
    
        diff = np.fabs(np.array(result.alpha) - np.array(baseline))
        thres = 1e-05
        assert diff.max() < thres, "The diff is %f, which is bigger than %f" % (
            diff.max(), thres)
