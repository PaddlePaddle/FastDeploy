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


class YOLOX(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file="",
                 runtime_option=None,
                 model_format=Frontend.ONNX):
        # 调用基函数进行backend_option的初始化
        # 初始化后的option保存在self._runtime_option
        super(YOLOX, self).__init__(runtime_option)

        self._model = C.vision.megvii.YOLOX(
            model_file, params_file, self._runtime_option, model_format)
        # 通过self.initialized判断整个模型的初始化是否成功
        assert self.initialized, "YOLOX initialize failed."

    def predict(self, input_image, conf_threshold=0.25, nms_iou_threshold=0.5):
        return self._model.predict(input_image, conf_threshold,
                                   nms_iou_threshold)

    # 一些跟YOLOX模型有关的属性封装
    # 多数是预处理相关，可通过修改如model.size = [1280, 1280]改变预处理时resize的大小（前提是模型支持）
    @property
    def size(self):
        return self._model.size

    @property
    def padding_value(self):
        return self._model.padding_value

    @property
    def is_decode_exported(self):
        return self._model.is_decode_exported

    @property
    def downsample_strides(self):
        return self._model.downsample_strides        
    
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

    @is_decode_exported.setter
    def is_decode_exported(self, value):
        assert isinstance(
            value, 
            bool), "The value to set `is_decode_exported` must be type of bool."
        self._model.max_wh = value

    @downsample_strides.setter
    def downsample_strides(self, value):
        assert isinstance(
            value,
            list), "The value to set `downsample_strides` must be type of list."
        self._model.downsample_strides = value          

    @max_wh.setter
    def max_wh(self, value):
        assert isinstance(
            value, float), "The value to set `max_wh` must be type of float."
        self._model.max_wh = value
