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

from tqdm import trange
import numpy as np
import collections


def eval_segmentation(model, data_dir):
    import cv2
    from utils import Cityscapes
    from utils import f1_score, calculate_area, mean_iou, accuracy, kappa
    eval_dataset = Cityscapes(dataset_root=data_dir, mode="val")
    file_list = eval_dataset.file_list
    image_num = eval_dataset.num_samples
    num_classes = eval_dataset.num_classes
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0
    conf_mat_all = []
    for image_label_path, i in zip(file_list,
                                   trange(
                                       image_num, desc="Inference Progress")):
        im = cv2.imread(image_label_path[0])
        label = cv2.imread(image_label_path[1], cv2.IMREAD_GRAYSCALE)
        result = model.predict(im)
        pred = np.array(result.label_map).reshape(result.shape[0],
                                                  result.shape[1])
        intersect_area, pred_area, label_area = calculate_area(pred, label,
                                                               num_classes)
        intersect_area_all = intersect_area_all + intersect_area
        pred_area_all = pred_area_all + pred_area
        label_area_all = label_area_all + label_area

    class_iou, miou = mean_iou(intersect_area_all, pred_area_all,
                               label_area_all)
    class_acc, oacc = accuracy(intersect_area_all, pred_area_all)
    kappa_res = kappa(intersect_area_all, pred_area_all, label_area_all)
    category_f1score = f1_score(intersect_area_all, pred_area_all,
                                label_area_all)

    eval_metrics = collections.OrderedDict(
        zip([
            'miou', 'category_iou', 'oacc', 'category_acc', 'kappa',
            'category_F1-score'
        ], [miou, class_iou, oacc, class_acc, kappa_res, category_f1score]))
    return eval_metrics


import fastdeploy as fd
#model = fd.vision.segmentation.PaddleSegModel("/huangjianhui/temp/FastDeploy/model_zoo/vision/ppseg/unet/unet_Cityscapes/model.pdmodel",
#                              "/huangjianhui/temp/FastDeploy/model_zoo/vision/ppseg/unet/unet_Cityscapes//model.pdiparams",
#                              "/huangjianhui/temp/FastDeploy/model_zoo/vision/ppseg/unet/unet_Cityscapes/deploy.yaml")
#
option = fd.RuntimeOption()
option.use_paddle_backend()
option.use_gpu(3)
model = fd.vision.segmentation.PaddleSegModel(
    "/huangjianhui/temp/FastDeploy/model_zoo/vision/ppseg/PP-LiteSeg/output_no_static_size/model.pdmodel",
    "/huangjianhui/temp/FastDeploy/model_zoo/vision/ppseg/PP-LiteSeg/output_no_static_size/model.pdiparams",
    "/huangjianhui/temp/FastDeploy/model_zoo/vision/ppseg/PP-LiteSeg/output_no_static_size/deploy.yaml",
    option)

result = eval_segmentation(model, "/huangjianhui/PaddleSeg/data/cityscapes/")
