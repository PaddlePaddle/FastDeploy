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

import cv2
import os
import numpy as np
import paddle


def generate_scale(im, target_shape):
    origin_shape = im.shape[:2]
    im_size_min = np.min(origin_shape)
    im_size_max = np.max(origin_shape)
    target_size_min = np.min(target_shape)
    target_size_max = np.max(target_shape)
    im_scale = float(target_size_min) / float(im_size_min)
    if np.round(im_scale * im_size_max) > target_size_max:
        im_scale = float(target_size_max) / float(im_size_max)
    im_scale_x = im_scale
    im_scale_y = im_scale

    return im_scale_y, im_scale_x


def yolo_image_preprocess(img, target_shape=[640, 640]):
    # Resize image
    im_scale_y, im_scale_x = generate_scale(img, target_shape)
    img = cv2.resize(
        img,
        None,
        None,
        fx=im_scale_x,
        fy=im_scale_y,
        interpolation=cv2.INTER_LINEAR)
    # Pad
    im_h, im_w = img.shape[:2]
    h, w = target_shape[:]
    if h != im_h or w != im_w:
        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array([114.0, 114.0, 114.0], dtype=np.float32)
        canvas[0:im_h, 0:im_w, :] = img.astype(np.float32)
        img = canvas
    img = np.transpose(img / 255, [2, 0, 1])

    return img.astype(np.float32)


def cls_resize_short(img, target_size):

    img_h, img_w = img.shape[:2]
    percent = float(target_size) / min(img_w, img_h)
    w = int(round(img_w * percent))
    h = int(round(img_h * percent))

    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def crop_image(img, target_size, center):

    height, width = img.shape[:2]
    size = target_size

    if center == True:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size

    return img[h_start:h_end, w_start:w_end, :]


def cls_image_preprocess(img):

    # resize
    img = cls_resize_short(img, target_size=256)
    # crop
    img = crop_image(img, target_size=224, center=True)

    #ToCHWImage & Normalize
    img = np.transpose(img / 255, [2, 0, 1])

    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    return img.astype(np.float32)


def ppdet_resize_no_keepratio(img, target_shape=[640, 640]):
    im_shape = img.shape

    resize_h, resize_w = target_shape
    im_scale_y = resize_h / im_shape[0]
    im_scale_x = resize_w / im_shape[1]

    scale_factor = np.asarray([im_scale_y, im_scale_x], dtype=np.float32)
    return cv2.resize(
        img, None, None, fx=im_scale_x, fy=im_scale_y,
        interpolation=2), scale_factor


def ppdet_normliaze(img, is_scale=True):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = img.astype(np.float32, copy=False)

    if is_scale:
        scale = 1.0 / 255.0
        img *= scale

    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    img -= mean
    img /= std
    return img


def hwc_to_chw(img):
    img = img.transpose((2, 0, 1))
    return img


def ppdet_image_preprocess(img):

    img, scale_factor = ppdet_resize_no_keepratio(img, target_shape=[640, 640])

    img = np.transpose(img / 255, [2, 0, 1])

    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    return img.astype(np.float32), scale_factor
