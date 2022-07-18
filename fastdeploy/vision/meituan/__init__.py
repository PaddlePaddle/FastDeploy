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
from ... import FastDeployModel, Frontend
from ... import fastdeploy_main as C


class YOLOv6(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file="",
                 runtime_option=None,
                 model_format=Frontend.ONNX):
        # 调用基函数进行backend_option的初始化
        # 初始化后的option保存在self._runtime_option
        super(YOLOv6, self).__init__(runtime_option)

        self._model = C.vision.meituan.YOLOv6(
            model_file, params_file, self._runtime_option, model_format)
        # 通过self.initialized判断整个模型的初始化是否成功
        assert self.initialized, "YOLOv6 initialize failed."

    def predict(self, input_image, conf_threshold=0.25, nms_iou_threshold=0.5):
        return self._model.predict(input_image, conf_threshold,
                                   nms_iou_threshold)

    # BOOL: 查看输入的模型是否为动态维度的 
    def is_dynamic_shape(self):
        return self._model.is_dynamic_shape()                               

    # 一些跟YOLOv6模型有关的属性封装
    # 多数是预处理相关，可通过修改如model.size = [1280, 1280]改变预处理时resize的大小（前提是模型支持）
    @property
    def size(self):
        return self._model.size

    @property
    def padding_value(self):
        return self._model.padding_value

    @property
    def is_no_pad(self):
        return self._model.is_no_pad

    @property
    def is_mini_pad(self):
        return self._model.is_mini_pad

    @property
    def is_scale_up(self):
        return self._model.is_scale_up

    @property
    def stride(self):
        return self._model.stride

    @property
    def max_wh(self):
        return self._model.max_wh

    @size.setter
    def size(self, wh):
        assert isinstance(wh, [list, tuple]),\
            "The value to set `size` must be type of tuple or list."
        assert len(wh) == 2,\
            "The value to set `size` must contatins 2 elements means [width, height], but now it contains {} elements.".format(
            len(wh))
        self._model.size = wh

    @padding_value.setter
    def padding_value(self, value):
        assert isinstance(
            value,
            list), "The value to set `padding_value` must be type of list."
        self._model.padding_value = value

    @is_no_pad.setter
    def is_no_pad(self, value):
        assert isinstance(
            value, bool), "The value to set `is_no_pad` must be type of bool."
        self._model.is_no_pad = value

    @is_mini_pad.setter
    def is_mini_pad(self, value):
        assert isinstance(
            value,
            bool), "The value to set `is_mini_pad` must be type of bool."
        self._model.is_mini_pad = value

    @is_scale_up.setter
    def is_scale_up(self, value):
        assert isinstance(
            value,
            bool), "The value to set `is_scale_up` must be type of bool."
        self._model.is_scale_up = value

    @stride.setter
    def stride(self, value):
        assert isinstance(
            value, int), "The value to set `stride` must be type of int."
        self._model.stride = value

    @max_wh.setter
    def max_wh(self, value):
        assert isinstance(
            value, float), "The value to set `max_wh` must be type of float."
        self._model.max_wh = value
