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
from .... import FastDeployModel, Frontend
from .... import c_lib_wrap as C


class PartialFC(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file="",
                 runtime_option=None,
                 model_format=Frontend.ONNX):
        # 调用基函数进行backend_option的初始化
        # 初始化后的option保存在self._runtime_option
        super(PartialFC, self).__init__(runtime_option)

        self._model = C.vision.faceid.PartialFC(
            model_file, params_file, self._runtime_option, model_format)
        # 通过self.initialized判断整个模型的初始化是否成功
        assert self.initialized, "PartialFC initialize failed."

    def predict(self, input_image):
        return self._model.predict(input_image)

    # 一些跟模型有关的属性封装
    # 多数是预处理相关，可通过修改如model.size = [112, 112]改变预处理时resize的大小（前提是模型支持）
    @property
    def size(self):
        return self._model.size

    @property
    def alpha(self):
        return self._model.alpha

    @property
    def beta(self):
        return self._model.beta

    @property
    def swap_rb(self):
        return self._model.swap_rb

    @property
    def l2_normalize(self):
        return self._model.l2_normalize

    @size.setter
    def size(self, wh):
        assert isinstance(wh, (list, tuple)),\
            "The value to set `size` must be type of tuple or list."
        assert len(wh) == 2,\
            "The value to set `size` must contatins 2 elements means [width, height], but now it contains {} elements.".format(
            len(wh))
        self._model.size = wh

    @alpha.setter
    def alpha(self, value):
        assert isinstance(value, (list, tuple)),\
            "The value to set `alpha` must be type of tuple or list."
        assert len(value) == 3,\
            "The value to set `alpha` must contatins 3 elements for each channels, but now it contains {} elements.".format(
            len(value))
        self._model.alpha = value

    @beta.setter
    def beta(self, value):
        assert isinstance(value, (list, tuple)),\
            "The value to set `beta` must be type of tuple or list."
        assert len(value) == 3,\
            "The value to set `beta` must contatins 3 elements for each channels, but now it contains {} elements.".format(
            len(value))
        self._model.beta = value

    @swap_rb.setter
    def swap_rb(self, value):
        assert isinstance(
            value, bool), "The value to set `swap_rb` must be type of bool."
        self._model.swap_rb = value

    @l2_normalize.setter
    def l2_normalize(self, value):
        assert isinstance(
            value,
            bool), "The value to set `l2_normalize` must be type of bool."
        self._model.l2_normalize = value
