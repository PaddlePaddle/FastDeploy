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

from __future__ import absolute_import
import logging
from ... import c_lib_wrap as C
import cv2


def vis_detection(im_data,
                  det_result,
                  score_threshold=0.0,
                  line_size=1,
                  font_size=0.5):
    return C.vision.vis_detection(im_data, det_result, score_threshold,
                                  line_size, font_size)


def vis_keypoint_detection(im_data, keypoint_det_result, conf_threshold=0.5):
    return C.vision.Visualize.vis_keypoint_detection(
        im_data, keypoint_det_result, conf_threshold)


def vis_face_detection(im_data, face_det_result, line_size=1, font_size=0.5):
    return C.vision.vis_face_detection(im_data, face_det_result, line_size,
                                       font_size)


def vis_segmentation(im_data, seg_result, weight=0.5):
    return C.vision.vis_segmentation(im_data, seg_result, weight)


def vis_matting_alpha(im_data,
                      matting_result,
                      remove_small_connected_area=False):
    logging.warning(
        "DEPRECATED: fastdeploy.vision.vis_matting_alpha is deprecated, please use fastdeploy.vision.vis_matting function instead."
    )
    return C.vision.vis_matting(im_data, matting_result,
                                remove_small_connected_area)


def vis_matting(im_data, matting_result, remove_small_connected_area=False):
    return C.vision.vis_matting(im_data, matting_result,
                                remove_small_connected_area)


def swap_background_matting(im_data,
                            background,
                            result,
                            remove_small_connected_area=False):
    logging.warning(
        "DEPRECATED: fastdeploy.vision.swap_background_matting is deprecated, please use fastdeploy.vision.swap_background function instead."
    )
    assert isinstance(
        result, C.vision.MattingResult), "The result must be MattingResult type"
    return C.vision.Visualize.swap_background_matting(
        im_data, background, result, remove_small_connected_area)


def swap_background_segmentation(im_data, background, background_label, result):
    logging.warning(
        "DEPRECATED: fastdeploy.vision.swap_background_segmentation is deprecated, please use fastdeploy.vision.swap_background function instead."
    )
    assert isinstance(
        result, C.vision.
        SegmentationResult), "The result must be SegmentaitonResult type"
    return C.vision.Visualize.swap_background_segmentation(
        im_data, background, background_label, result)


def swap_background(im_data,
                    background,
                    result,
                    remove_small_connected_area=False,
                    background_label=0):
    if isinstance(result, C.vision.MattingResult):
        return C.vision.swap_background(im_data, background, result,
                                        remove_small_connected_area)
    elif isinstance(result, C.vision.SegmentationResult):
        return C.vision.swap_background(im_data, background, result,
                                        background_label)
    else:
        raise Exception(
            "Only support result type of MattingResult or SegmentationResult, but now the data type is {}.".
            format(type(result)))


def vis_ppocr(im_data, det_result):
    return C.vision.vis_ppocr(im_data, det_result)


def vis_mot(im_data, mot_result, score_threshold=0.0, is_draw_trail=False, trails: dict = None):
    if is_draw_trail:
        if trails is None:
            print("[WARNING] <trails> must be a global dict, when <is_draw_trail> is True ,trails will be hidden!")
        else:
            draw_trail(im_data, mot_result, trails)
    return C.vision.vis_mot(im_data, mot_result, score_threshold, False)


def get_mot_box_color(idx: int):
    idx = idx * 3
    color = [(37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255]
    return color


def draw_trail(frame, mot_result, trails: dict):
    n = len(mot_result.boxes)
    for i in range(n):
        id_ = mot_result.ids[i]
        color = get_mot_box_color(id_)
        x = int((mot_result.boxes[i][0] + mot_result.boxes[i][2]) / 2)
        y = int((mot_result.boxes[i][1] + mot_result.boxes[i][3]) / 2)
        if id_ in trails:
            trails[id_].append((x, y))
        else:
            trails.update({id_:[(x, y)]})
        if trails[id_] is None:
            continue
        for point in trails[id_]:
            cv2.circle(frame, point, 2, color)
