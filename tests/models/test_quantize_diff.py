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

model_url = "https://bj.bcebos.com/fastdeploy/tests/yolov6_quant.tgz"
fd.download_and_decompress(model_url, ".")


def test_quant_mkldnn():
    model_path = "./yolov6_quant"
    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")

    input_file = os.path.join(model_path, "input.npy")
    output_file = os.path.join(model_path, "mkldnn_output.npy")

    option = fd.RuntimeOption()
    option.use_paddle_backend()
    option.use_cpu()

    option.set_model_path(model_file, params_file)
    runtime = fd.Runtime(option)
    input_name = runtime.get_input_info(0).name
    data = np.load(input_file)
    outs = runtime.infer({input_name: data})
    expected = np.load(output_file)
    diff = np.fabs(outs[0] - expected)
    thres = 1e-05
    assert diff.max() < thres, "The diff is %f, which is bigger than %f" % (
        diff.max(), thres)


def test_quant_ort():
    model_path = "./yolov6_quant"
    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")

    input_file = os.path.join(model_path, "input.npy")
    output_file = os.path.join(model_path, "ort_output.npy")

    option = fd.RuntimeOption()
    option.use_ort_backend()
    option.use_cpu()

    option.set_ort_graph_opt_level(1)

    option.set_model_path(model_file, params_file)
    runtime = fd.Runtime(option)
    input_name = runtime.get_input_info(0).name
    data = np.load(input_file)
    outs = runtime.infer({input_name: data})
    expected = np.load(output_file)
    diff = np.fabs(outs[0] - expected)
    thres = 1e-05
    assert diff.max() < thres, "The diff is %f, which is bigger than %f" % (
        diff.max(), thres)


def test_quant_trt():
    model_path = "./yolov6_quant"
    model_file = os.path.join(model_path, "model.pdmodel")
    params_file = os.path.join(model_path, "model.pdiparams")

    input_file = os.path.join(model_path, "input.npy")
    output_file = os.path.join(model_path, "trt_output.npy")

    option = fd.RuntimeOption()
    option.use_trt_backend()
    option.use_gpu()

    option.set_model_path(model_file, params_file)
    runtime = fd.Runtime(option)
    input_name = runtime.get_input_info(0).name
    data = np.load(input_file)
    outs = runtime.infer({input_name: data})
    expected = np.load(output_file)
    diff = np.fabs(outs[0] - expected)
    thres = 1e-05
    assert diff.max() < thres, "The diff is %f, which is bigger than %f" % (
        diff.max(), thres)
