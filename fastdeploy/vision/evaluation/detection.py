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
from pathlib import Path
import cv2
import os
import numpy as np
import glob
from .utils import box_iou, scale_coords, xywhn2xyxy, xyxy2xywhn, xywh2xyxy, calculate_padding
from .metric import ap_per_class

# The implementation refers to
# https://github.com/ultralytics/yolov5/blob/master/val.py

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = np.where((iou >= iouv[i]) &
                     correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = np.concatenate((np.stack(
                x, 1), iou[x[0], x[1]][:, None]), 1)  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(
                    matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(
                    matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [
        sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths
    ]


def eval_detection(model,
                   conf_threshold,
                   nms_iou_threshold,
                   image_file_path,
                   plot=False):
    assert isinstance(conf_threshold, (
        float, int
    )), "The conf_threshold:{} need to be int or float".format(conf_threshold)
    assert isinstance(nms_iou_threshold, (
        float,
        int)), "The nms_iou_threshold:{} need to be int or float".format(
            nms_iou_threshold)
    try:
        f = []  # image files
        for p in image_file_path if isinstance(image_file_path,
                                               list) else [image_file_path]:
            p = Path(p)
            if p.is_dir():  # dir
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [
                        x.replace('./', parent) if x.startswith('./') else x
                        for x in t
                    ]  # local to global path
            else:
                raise Exception(f'{p} does not exist')
        image_files = sorted(
            x.replace('/', os.sep) for x in f
            if x.split('.')[-1].lower() in IMG_FORMATS)
        assert image_files, f'No images found'
    except Exception:
        raise Exception(f'Error loading data from {image_file_path}')
    label_files = img2label_paths(image_files)
    image_label_dict = {}
    for label_file, image_file in zip(label_files, image_files):
        if os.path.isfile(label_file):
            with open(label_file) as f:
                lb = [
                    x.split() for x in f.read().strip().splitlines() if len(x)
                ]
                lb = np.array(lb, dtype=np.float32)
                image_label_dict[image_file] = lb
    stats = []
    image_num = len(image_files)
    for image, i in zip(image_files,
                        trange(
                            image_num,
                            desc="Inference Progress")):
        im = cv2.imread(image)
        h0, w0 = im.shape[:2]
        h = h0
        w = w0
        new_shape = 640  # resize image shape
        r = new_shape / max(h0, w0)
        if r != 1:
            w = w * r
            h = h * r
        result = model.predict(im, conf_threshold, nms_iou_threshold)
        max_det = 300  # max number of detection boxes
        pred = [
            b + [s] + [c]
            for b, s, c in zip(result.boxes, result.scores, result.label_ids)
        ]
        pred = np.array(pred, dtype='f')
        if pred.shape[0] > max_det:
            pred = pred[:300]

        old_shape = (h, w)
        pad, ratio = calculate_padding(old_shape, new_shape, False)
        shapes = (h0, w0), ((h / h0, w / w0), pad)
        if image not in image_label_dict:
            continue
        labels = image_label_dict[image]
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(
                labels[:, 1:],
                ratio[0] * w,
                ratio[1] * h,
                padw=pad[0],
                padh=pad[1])
        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(
                labels[:, 1:5], w=new_shape, h=new_shape, clip=True, eps=1E-3)
        labels[:, 1:] *= np.array([new_shape, new_shape, new_shape, new_shape])
        npr = pred.shape[0]
        iouv = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        niou = iouv.shape[0]
        correct = np.zeros((npr, niou))
        predn = np.copy(pred)
        if npr == 0:
            if nl:
                stats.append((correct, *np.zeros((3, 0))))
                continue
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])
            scale_coords(
                np.array([new_shape, new_shape]), tbox, shapes[0], shapes[1])
            labelsn = np.append(labels[:, 0:1], tbox, axis=1)
            correct = process_batch(predn, labelsn, iouv)
        stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))
    stats = [np.concatenate(x, 0) for x in zip(*stats)]

    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(
            *stats, plot=plot, save_dir='.')
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = round(p.mean(), 3), round(r.mean(), 3), round(
            ap50.mean(), 3), round(ap.mean(), 3)
        return {"Precision": mp, "Recall": mr, "mAP@.5": map50, "mAP@.5:.95": map}
