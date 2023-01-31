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
import random
from PIL import Image, ImageEnhance
import paddle
"""
Preprocess for Yolov5/v6/v7 Series
"""


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


"""
Preprocess for PaddleClas model
"""


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


"""
Preprocess for PPYOLOE
"""


def ppdet_resize_no_keepratio(img, target_shape=[640, 640]):
    im_shape = img.shape

    resize_h, resize_w = target_shape
    im_scale_y = resize_h / im_shape[0]
    im_scale_x = resize_w / im_shape[1]

    scale_factor = np.asarray([im_scale_y, im_scale_x], dtype=np.float32)
    return cv2.resize(
        img, None, None, fx=im_scale_x, fy=im_scale_y,
        interpolation=2), scale_factor


def ppyoloe_withNMS_image_preprocess(img):

    img, scale_factor = ppdet_resize_no_keepratio(img, target_shape=[640, 640])

    img = np.transpose(img / 255, [2, 0, 1])

    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    return img.astype(np.float32), scale_factor


def ppyoloe_plus_withNMS_image_preprocess(img):

    img, scale_factor = ppdet_resize_no_keepratio(img, target_shape=[640, 640])

    img = np.transpose(img / 255, [2, 0, 1])

    return img.astype(np.float32), scale_factor


"""
Preprocess for PP_LiteSeg

"""


def ppseg_cityscapes_ptq_preprocess(img):

    #ToCHWImage & Normalize
    img = np.transpose(img / 255.0, [2, 0, 1])

    img_mean = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
    img_std = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    return img.astype(np.float32)


def ResizeStepScaling(img,
                      min_scale_factor=0.75,
                      max_scale_factor=1.25,
                      scale_step_size=0.25):
    # refer form ppseg
    if min_scale_factor == max_scale_factor:
        scale_factor = min_scale_factor
    elif scale_step_size == 0:
        scale_factor = np.random.uniform(min_scale_factor, max_scale_factor)
    else:
        num_steps = int((max_scale_factor - min_scale_factor) / scale_step_size
                        + 1)
        scale_factors = np.linspace(min_scale_factor, max_scale_factor,
                                    num_steps).tolist()
        np.random.shuffle(scale_factors)
        scale_factor = scale_factors[0]

    w = int(round(scale_factor * img.shape[1]))
    h = int(round(scale_factor * img.shape[0]))

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    return img


def RandomPaddingCrop(img,
                      crop_size=(512, 512),
                      im_padding_value=(127.5, 127.5, 127.5),
                      label_padding_value=255):

    if isinstance(crop_size, list) or isinstance(crop_size, tuple):
        if len(crop_size) != 2:
            raise ValueError(
                'Type of `crop_size` is list or tuple. It should include 2 elements, but it is {}'
                .format(crop_size))
    else:
        raise TypeError(
            "The type of `crop_size` is invalid. It should be list or tuple, but it is {}"
            .format(type(crop_size)))

    if isinstance(crop_size, int):
        crop_width = crop_size
        crop_height = crop_size
    else:
        crop_width = crop_size[0]
        crop_height = crop_size[1]

    img_height = img.shape[0]
    img_width = img.shape[1]

    if img_height == crop_height and img_width == crop_width:
        return img
    else:
        pad_height = max(crop_height - img_height, 0)
        pad_width = max(crop_width - img_width, 0)
        if (pad_height > 0 or pad_width > 0):
            img = cv2.copyMakeBorder(
                img,
                0,
                pad_height,
                0,
                pad_width,
                cv2.BORDER_CONSTANT,
                value=im_padding_value)

            img_height = img.shape[0]
            img_width = img.shape[1]

        if crop_height > 0 and crop_width > 0:
            h_off = np.random.randint(img_height - crop_height + 1)
            w_off = np.random.randint(img_width - crop_width + 1)

            img = img[h_off:(crop_height + h_off), w_off:(w_off + crop_width
                                                          ), :]

        return img


def RandomHorizontalFlip(img, prob=0.5):
    if random.random() < prob:

        if len(img.shape) == 3:
            img = img[:, ::-1, :]
        elif len(img.shape) == 2:
            img = img[:, ::-1]

        return img
    else:
        return img


def brightness(im, brightness_lower, brightness_upper):
    brightness_delta = np.random.uniform(brightness_lower, brightness_upper)
    im = ImageEnhance.Brightness(im).enhance(brightness_delta)
    return im


def contrast(im, contrast_lower, contrast_upper):
    contrast_delta = np.random.uniform(contrast_lower, contrast_upper)
    im = ImageEnhance.Contrast(im).enhance(contrast_delta)
    return im


def saturation(im, saturation_lower, saturation_upper):
    saturation_delta = np.random.uniform(saturation_lower, saturation_upper)
    im = ImageEnhance.Color(im).enhance(saturation_delta)
    return im


def hue(im, hue_lower, hue_upper):
    hue_delta = np.random.uniform(hue_lower, hue_upper)
    im = np.array(im.convert('HSV'))
    im[:, :, 0] = im[:, :, 0] + hue_delta
    im = Image.fromarray(im, mode='HSV').convert('RGB')
    return im


def sharpness(im, sharpness_lower, sharpness_upper):
    sharpness_delta = np.random.uniform(sharpness_lower, sharpness_upper)
    im = ImageEnhance.Sharpness(im).enhance(sharpness_delta)
    return im


def RandomDistort(img,
                  brightness_range=0.5,
                  brightness_prob=0.5,
                  contrast_range=0.5,
                  contrast_prob=0.5,
                  saturation_range=0.5,
                  saturation_prob=0.5,
                  hue_range=18,
                  hue_prob=0.5,
                  sharpness_range=0.5,
                  sharpness_prob=0):

    brightness_lower = 1 - brightness_range
    brightness_upper = 1 + brightness_range
    contrast_lower = 1 - contrast_range
    contrast_upper = 1 + contrast_range
    saturation_lower = 1 - saturation_range
    saturation_upper = 1 + saturation_range
    hue_lower = -hue_range
    hue_upper = hue_range
    sharpness_lower = 1 - sharpness_range
    sharpness_upper = 1 + sharpness_range
    ops = [brightness, contrast, saturation, hue, sharpness]
    random.shuffle(ops)
    params_dict = {
        'brightness': {
            'brightness_lower': brightness_lower,
            'brightness_upper': brightness_upper
        },
        'contrast': {
            'contrast_lower': contrast_lower,
            'contrast_upper': contrast_upper
        },
        'saturation': {
            'saturation_lower': saturation_lower,
            'saturation_upper': saturation_upper
        },
        'hue': {
            'hue_lower': hue_lower,
            'hue_upper': hue_upper
        },
        'sharpness': {
            'sharpness_lower': sharpness_lower,
            'sharpness_upper': sharpness_upper,
        }
    }
    prob_dict = {
        'brightness': brightness_prob,
        'contrast': contrast_prob,
        'saturation': saturation_prob,
        'hue': hue_prob,
        'sharpness': sharpness_prob
    }

    img = img.astype('uint8')
    img = Image.fromarray(img)

    for id in range(len(ops)):
        params = params_dict[ops[id].__name__]
        prob = prob_dict[ops[id].__name__]
        params['im'] = img
        if np.random.uniform(0, 1) < prob:
            img = ops[id](**params)
    img = np.asarray(img).astype('float32')
    return img


def ppseg_cityscapes_qat_preprocess(img):

    min_scale_factor = 0.5
    max_scale_factor = 2.0
    scale_step_size = 0.25

    crop_size = (1024, 512)

    brightness_range = 0.5
    contrast_range = 0.5
    saturation_range = 0.5

    img = ResizeStepScaling(
        img, min_scale_factor=0.5, max_scale_factor=2.0, scale_step_size=0.25)
    img = RandomPaddingCrop(img, crop_size=(1024, 512))
    img = RandomHorizontalFlip(img)
    img = RandomDistort(
        img, brightness_range=0.5, contrast_range=0.5, saturation_range=0.5)

    img = np.transpose(img / 255.0, [2, 0, 1])
    img_mean = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
    img_std = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    return img.astype(np.float32)
