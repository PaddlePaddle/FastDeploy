# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--det_model", required=True, help="Path of Detection model of PPOCR.")
    parser.add_argument(
        "--cls_model",
        required=True,
        help="Path of Classification model of PPOCR.")
    parser.add_argument(
        "--rec_model",
        required=True,
        help="Path of Recognization model of PPOCR.")
    parser.add_argument(
        "--rec_label_file",
        required=True,
        help="Path of Recognization model of PPOCR.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--backend",
        type=str,
        default="default",
        help="Type of inference backend, support ort/trt/paddle/openvino, default 'openvino' for cpu, 'tensorrt' for gpu"
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Define which GPU card used to run model.")
    parser.add_argument(
        "--cpu_thread_num",
        type=int,
        default=9,
        help="Number of threads while inference on CPU.")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()
    if args.device.lower() == "kunlunxin":
        option.use_kunlunxin()

    if args.device.lower() == "ascend":
        option.use_ascend()

    if args.device.lower() == "gpu":
        option.use_gpu()

    return option


args = parse_arguments()

# Detection模型, 检测文字框
det_model_file = os.path.join(args.det_model, "inference.pdmodel")
det_params_file = os.path.join(args.det_model, "inference.pdiparams")
# Classification模型，方向分类，可选
cls_model_file = os.path.join(args.cls_model, "inference.pdmodel")
cls_params_file = os.path.join(args.cls_model, "inference.pdiparams")
# Recognition模型，文字识别模型
rec_model_file = os.path.join(args.rec_model, "inference.pdmodel")
rec_params_file = os.path.join(args.rec_model, "inference.pdiparams")
rec_label_file = args.rec_label_file

# 对于三个模型，均采用同样的部署配置
# 用户也可根据自行需求分别配置
runtime_option = build_option(args)

det_model = fd.vision.ocr.DBDetector(
    det_model_file, det_params_file, runtime_option=runtime_option)
cls_model = fd.vision.ocr.Classifier(
    cls_model_file, cls_params_file, runtime_option=runtime_option)
rec_model = fd.vision.ocr.Recognizer(
    rec_model_file,
    rec_params_file,
    rec_label_file,
    runtime_option=runtime_option)

# PPOCR的Rec模型开启静态推理, 其他硬件不需要的话请注释掉.
rec_model.preprocessor.static_shape_infer = True

# 创建PP-OCR，串联3个模型，其中cls_model可选，如无需求，可设置为None
ppocr_v3 = fd.vision.ocr.PPOCRv3(
    det_model=det_model, cls_model=cls_model, rec_model=rec_model)

#####
#准备输入图片数据
img_dir = args.image
imgs_file_lists = []
if os.path.isdir(img_dir):
    for single_file in os.listdir(img_dir):
        if 'jpg' in single_file:
            file_path = os.path.join(img_dir, single_file)
            if os.path.isfile(file_path):
                imgs_file_lists.append(file_path)

imgs_file_lists.sort()

fd_result = []
for idx, image in enumerate(imgs_file_lists):
    img = cv2.imread(image)
    result = ppocr_v3.predict(img)
    for i in range(len(result.boxes)):
        one_res = result.boxes[i] + [
            result.rec_scores[i]
        ] + [result.cls_labels[i]] + [result.cls_scores[i]]
        fd_result.append(one_res)

local_result = []
with open('PPOCRv3_ICDAR10_BS116_1221.txt', 'r') as f:
    for line in f:
        local_result.append(list(map(float, line.split(','))))

# Begin to Diff Compare
total_num_res = len(local_result) * 11
total_diff_num = 0

print("==== Begin to check OCR diff ====")
for list_local, list_fd in zip(local_result, fd_result):

    for i in range(len(list_local)):

        if (i < 8):
            #Det
            diff = list_local[i] - list_fd[i]
            assert (
                abs(diff) < 1
            ), "Diff exist in Det box result, where is {} - {} .".format(
                list_local, list_fd)
        elif (i == 8):
            #rec
            diff = round(list_local[i], 6) - round(list_fd[i], 6)
            assert (
                abs(diff) < 0.001
            ), "Diff exist in rec scores result, where is {} - {} .".format(
                list_local, list_fd)
        elif (i == 9):
            diff = list_local[i] - list_fd[i]
            assert (
                abs(diff) != 1
            ), "Diff exist in cls label result, where is {} - {} .".format(
                list_local, list_fd)
        else:
            diff = round(list_local[i], 6) - round(list_fd[i], 6)
            assert (
                abs(diff) < 0.001
            ), "Diff exist in cls score result, where is {} - {} .".format(
                list_local, list_fd)
