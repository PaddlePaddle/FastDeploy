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

import numpy as np
import copy
import collections


def eval_detection(model,
                   data_dir,
                   ann_file,
                   conf_threshold=None,
                   nms_iou_threshold=None,
                   plot=False):
    from .utils import CocoDetection
    from .utils import COCOMetric
    import cv2
    from tqdm import trange 

    if conf_threshold is not None or nms_iou_threshold is not None:
        assert conf_threshold is not None and nms_iou_threshold is not None, "The conf_threshold and nms_iou_threshold should be setted at the same time"
        assert isinstance(conf_threshold, (
            float,
            int)), "The conf_threshold:{} need to be int or float".format(
                conf_threshold)
        assert isinstance(nms_iou_threshold, (
            float,
            int)), "The nms_iou_threshold:{} need to be int or float".format(
                nms_iou_threshold)
    eval_dataset = CocoDetection(
        data_dir=data_dir, ann_file=ann_file, shuffle=False)
    all_image_info = eval_dataset.file_list
    image_num = eval_dataset.num_samples
    eval_dataset.data_fields = {
        'im_id', 'image_shape', 'image', 'gt_bbox', 'gt_class', 'is_crowd'
    }
    eval_metric = COCOMetric(
        coco_gt=copy.deepcopy(eval_dataset.coco_gt), classwise=False)
    scores = collections.OrderedDict()
    for image_info, i in zip(all_image_info,
                             trange(
                                 image_num, desc="Inference Progress")):
        im = cv2.imread(image_info["image"])
        im_id = image_info["im_id"]
        if conf_threshold is None and nms_iou_threshold is None:
            result = model.predict(im)
        else:
            result = model.predict(im, conf_threshold, nms_iou_threshold)
        pred = {
            'bbox':
            [[c] + [s] + b
             for b, s, c in zip(result.boxes, result.scores, result.label_ids)
             ],
            'bbox_num': len(result.boxes),
            'im_id': im_id
        }
        eval_metric.update(im_id, pred)
    eval_metric.accumulate()
    eval_details = eval_metric.details
    scores.update(eval_metric.get())
    eval_metric.reset()
    return scores
