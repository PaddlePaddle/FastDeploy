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
from fastdeploy import ModelFormat
import cv2
import os
import numpy as np
import runtime_config as rc
import pickle


def test_segmentation_ppliteseg():
    pp_liteseg_model_url = "https://bj.bcebos.com/fastdeploy/tests/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_test.tgz"
    fd.download_and_decompress(pp_liteseg_model_url, "resources")
    model_path = "./resources/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_test"
    # 配置runtime，加载模型
    runtime_option = fd.RuntimeOption()
    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    config_file = os.path.join(model_path, "deploy.yaml")
    image_file_1 = os.path.join(model_path, "cityscapes_demo_1.png")
    image_file_2 = os.path.join(model_path, "cityscapes_demo_2.png")
    result_file_1 = os.path.join(model_path, "ppliteseg_result1.pkl")
    result_file_2 = os.path.join(model_path, "ppliteseg_result2.pkl")
    model = fd.vision.segmentation.PaddleSegModel(
        model_file, params_file, config_file, runtime_option=rc.test_option)
    model.postprocessor.store_score_map = True

    im1 = cv2.imread(image_file_1)
    im2 = cv2.imread(image_file_2)

    with open(result_file_1, "rb") as f:
        expect1 = pickle.load(f)

    with open(result_file_2, "rb") as f:
        expect2 = pickle.load(f)

    for i in range(3):
        # test single predict
        result1 = model.predict(im1)
        result2 = model.predict(im2)

        diff_label_map_1 = np.fabs(
            np.array(result1.label_map) - np.array(expect1["label_map"]))
        diff_label_map_2 = np.fabs(
            np.array(result2.label_map) - np.array(expect2["label_map"]))

        diff_score_map_1 = np.fabs(
            np.array(result1.score_map) - np.array(expect1["score_map"]))
        diff_score_map_2 = np.fabs(
            np.array(result2.score_map) - np.array(expect2["score_map"]))

        thres = 1e-05
        assert diff_label_map_1.max(
        ) < thres, "The label_map diff is %f, which is bigger than %f" % (
            diff_label_map_1.max(), thres)
        assert diff_score_map_1.max(
        ) < thres, "The score map diff is %f, which is bigger than %f" % (
            diff_score_map_1.max(), thres)
        assert diff_label_map_2.max(
        ) < thres, "The label_map diff is %f, which is bigger than %f" % (
            diff_label_map_2.max(), thres)
        assert diff_score_map_2.max(
        ) < thres, "The score map diff is %f, which is bigger than %f" % (
            diff_score_map_2.max(), thres)
        print("Single image No diff")

        # test batch predict
        results = model.batch_predict([im1, im2])
        result1 = results[0]
        result2 = results[1]
        diff_label_map_1 = np.fabs(
            np.array(result1.label_map) - np.array(expect1["label_map"]))
        diff_label_map_2 = np.fabs(
            np.array(result2.label_map) - np.array(expect2["label_map"]))

        diff_score_map_1 = np.fabs(
            np.array(result1.score_map) - np.array(expect1["score_map"]))
        diff_score_map_2 = np.fabs(
            np.array(result2.score_map) - np.array(expect2["score_map"]))

        thres = 1e-05
        assert diff_label_map_1.max(
        ) < thres, "The label_map diff is %f, which is bigger than %f" % (
            diff_label_map_1.max(), thres)
        assert diff_score_map_1.max(
        ) < thres, "The score map diff is %f, which is bigger than %f" % (
            diff_score_map_1.max(), thres)
        assert diff_label_map_2.max(
        ) < thres, "The label_map diff is %f, which is bigger than %f" % (
            diff_label_map_2.max(), thres)
        assert diff_score_map_2.max(
        ) < thres, "The score map diff is %f, which is bigger than %f" % (
            diff_score_map_2.max(), thres)
        print("Batch images No diff")


def test_segmentation_ppliteseg_runtime():
    pp_liteseg_model_url = "https://bj.bcebos.com/fastdeploy/tests/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_test.tgz"
    fd.download_and_decompress(pp_liteseg_model_url, "resources")
    model_path = "./resources/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_test"
    # 配置runtime，加载模型
    runtime_option = fd.RuntimeOption()
    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")
    config_file = os.path.join(model_path, "deploy.yaml")
    image_file_1 = os.path.join(model_path, "cityscapes_demo_1.png")
    result_file_1 = os.path.join(model_path, "ppliteseg_result1.pkl")

    preprocessor = fd.vision.segmentation.PaddleSegPreprocessor(config_file)
    postprocessor = fd.vision.segmentation.PaddleSegPostprocessor(config_file)
    postprocessor.store_score_map = True

    rc.test_option.set_model_path(
        model_file, params_file, model_format=ModelFormat.PADDLE)
    rc.test_option.use_paddle_backend()
    runtime = fd.Runtime(rc.test_option)

    with open(result_file_1, "rb") as f:
        expect1 = pickle.load(f)
    im1 = cv2.imread(image_file_1)
    print(image_file_1)

    for i in range(3):
        # test runtime
        input_tensors, ims_info = preprocessor.run([im1])
        output_tensors = runtime.infer({"x": input_tensors[0]})
        results = postprocessor.run(output_tensors, ims_info)
        result1 = results[0]
        diff_label_map_1 = np.fabs(
            np.array(result1.label_map) - np.array(expect1["label_map"]))

        diff_score_map_1 = np.fabs(
            np.array(result1.score_map) - np.array(expect1["score_map"]))

        thres = 1e-05
        assert diff_label_map_1.max(
        ) < thres, "The label_map diff is %f, which is bigger than %f" % (
            diff_label_map_1.max(), thres)
        assert diff_score_map_1.max(
        ) < thres, "The score map diff is %f, which is bigger than %f" % (
            diff_score_map_1.max(), thres)
        print("Runtime images No diff")


if __name__ == "__main__":
    test_segmentation_ppliteseg()
    test_segmentation_ppliteseg_runtime()
