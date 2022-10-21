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


def test_keypointdetection_pptinypose():
    pp_tinypose_model_url = "https://bj.bcebos.com/fastdeploy/tests/PP_TinyPose_256x192_test.tgz"
    fd.download_and_decompress(pp_tinypose_model_url, ".")
    model_path = "./PP_TinyPose_256x192_test"
    # 配置runtime，加载模型
    runtime_option = fd.RuntimeOption()
    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    config_file = os.path.join(model_path, "infer_cfg.yml")
    image_file = os.path.join(model_path, "hrnet_demo.jpg")
    baseline_file = os.path.join(model_path, "baseline.npy")
    model = fd.vision.keypointdetection.PPTinyPose(
        model_file, params_file, config_file, runtime_option=runtime_option)

    # 预测图片关键点
    im = cv2.imread(image_file)
    result = model.predict(im)
    result = np.concatenate(
        (np.array(result.keypoints), np.array(result.scores)[:, np.newaxis]),
        axis=1)
    baseline = np.load(baseline_file)
    diff = np.fabs(result - np.array(baseline))
    thres = 1e-05
    assert diff.max() < thres, "The diff is %f, which is bigger than %f" % (
        diff.max(), thres)
    print("No diff")


def test_keypointdetection_det_keypoint_unite():
    det_keypoint_unite_model_url = "https://bj.bcebos.com/fastdeploy/tests/PicoDet_320x320_TinyPose_256x192_test.tgz"
    fd.download_and_decompress(det_keypoint_unite_model_url, ".")
    model_path = "./PicoDet_320x320_TinyPose_256x192_test"
    # 配置runtime，加载模型
    runtime_option = fd.RuntimeOption()
    tinypose_model_file = os.path.join(
        model_path, "PP_TinyPose_256x192_infer/model.pdmodel")
    tinypose_params_file = os.path.join(
        model_path, "PP_TinyPose_256x192_infer/model.pdiparams")
    tinypose_config_file = os.path.join(
        model_path, "PP_TinyPose_256x192_infer/infer_cfg.yml")
    picodet_model_file = os.path.join(
        model_path, "PP_PicoDet_V2_S_Pedestrian_320x320_infer/model.pdmodel")
    picodet_params_file = os.path.join(
        model_path, "PP_PicoDet_V2_S_Pedestrian_320x320_infer/model.pdiparams")
    picodet_config_file = os.path.join(
        model_path, "PP_PicoDet_V2_S_Pedestrian_320x320_infer/infer_cfg.yml")
    image_file = os.path.join(model_path, "000000018491.jpg")
    # image_file = os.path.join(model_path, "hrnet_demo.jpg")

    baseline_file = os.path.join(model_path, "baseline.npy")

    tinypose_model = fd.vision.keypointdetection.PPTinyPose(
        tinypose_model_file,
        tinypose_params_file,
        tinypose_config_file,
        runtime_option=runtime_option)

    det_model = fd.vision.detection.PicoDet(
        picodet_model_file,
        picodet_params_file,
        picodet_config_file,
        runtime_option=runtime_option)

    # 预测图片关键点
    im = cv2.imread(image_file)
    pipeline = fd.pipeline.PPTinyPose(det_model, tinypose_model)
    pipeline.detection_model_score_threshold = 0.5
    result = pipeline.predict(im)
    print(result)
    result = np.concatenate(
        (np.array(result.keypoints), np.array(result.scores)[:, np.newaxis]),
        axis=1)
    print(result)
    np.save("baseline.npy", result)
    baseline = np.load(baseline_file)
    diff = np.fabs(result - np.array(baseline))
    thres = 1e-05
    assert diff.max() < thres, "The diff is %f, which is bigger than %f" % (
        diff.max(), thres)
    print("No diff")
