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


class SCRFD(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file="",
                 runtime_option=None,
                 model_format=Frontend.ONNX):
        # 调用基函数进行backend_option的初始化
        # 初始化后的option保存在self._runtime_option
        super(SCRFD, self).__init__(runtime_option)

        self._model = C.vision.deepinsight.SCRFD(
            model_file, params_file, self._runtime_option, model_format)
        # 通过self.initialized判断整个模型的初始化是否成功
        assert self.initialized, "SCRFD initialize failed."

    def predict(self, input_image, conf_threshold=0.7, nms_iou_threshold=0.3):
        return self._model.predict(input_image, conf_threshold,
                                   nms_iou_threshold)

    # 一些跟SCRFD模型有关的属性封装
    # 多数是预处理相关，可通过修改如model.size = [640, 640]改变预处理时resize的大小（前提是模型支持）
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
    def downsample_strides(self):
        return self._model.downsample_strides

    @property
    def landmarks_per_face(self):
        return self._model.landmarks_per_face

    @property
    def use_kps(self):
        return self._model.use_kps

    @property
    def max_nms(self):
        return self._model.max_nms

    @property
    def num_anchors(self):
        return self._model.num_anchors

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

    @downsample_strides.setter
    def downsample_strides(self, value):
        assert isinstance(
            value,
            list), "The value to set `downsample_strides` must be type of list."
        self._model.downsample_strides = value

    @landmarks_per_face.setter
    def landmarks_per_face(self, value):
        assert isinstance(
            value,
            int), "The value to set `landmarks_per_face` must be type of int."
        self._model.landmarks_per_face = value

    @use_kps.setter
    def use_kps(self, value):
        assert isinstance(
            value, bool), "The value to set `use_kps` must be type of bool."
        self._model.use_kps = value

    @max_nms.setter
    def max_nms(self, value):
        assert isinstance(
            value, int), "The value to set `max_nms` must be type of int."
        self._model.max_nms = value

    @num_anchors.setter
    def num_anchors(self, value):
        assert isinstance(
            value, int), "The value to set `num_anchors` must be type of int."
        self._model.num_anchors = value


class InsightFaceRecognitionModel(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file="",
                 runtime_option=None,
                 model_format=Frontend.ONNX):
        # 调用基函数进行backend_option的初始化
        # 初始化后的option保存在self._runtime_option
        super(InsightFaceRecognitionModel, self).__init__(runtime_option)

        self._model = C.vision.deepinsight.InsightFaceRecognitionModel(
            model_file, params_file, self._runtime_option, model_format)
        # 通过self.initialized判断整个模型的初始化是否成功
        assert self.initialized, "InsightFaceRecognitionModel initialize failed."

    def predict(self, input_image):
        return self._model.predict(input_image)

    # 一些跟InsightFaceRecognitionModel模型有关的属性封装
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
        assert isinstance(wh, [list, tuple]),\
            "The value to set `size` must be type of tuple or list."
        assert len(wh) == 2,\
            "The value to set `size` must contatins 2 elements means [width, height], but now it contains {} elements.".format(
            len(wh))
        self._model.size = wh

    @alpha.setter
    def alpha(self, value):
        assert isinstance(value, [list, tuple]),\
            "The value to set `alpha` must be type of tuple or list."
        assert len(value) == 3,\
            "The value to set `alpha` must contatins 3 elements for each channels, but now it contains {} elements.".format(
            len(value))
        self._model.alpha = value

    @beta.setter
    def beta(self, value):
        assert isinstance(value, [list, tuple]),\
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


class ArcFace(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file="",
                 runtime_option=None,
                 model_format=Frontend.ONNX):
        # 调用基函数进行backend_option的初始化
        # 初始化后的option保存在self._runtime_option
        super(ArcFace, self).__init__(runtime_option)

        self._model = C.vision.deepinsight.ArcFace(
            model_file, params_file, self._runtime_option, model_format)
        # 通过self.initialized判断整个模型的初始化是否成功
        assert self.initialized, "ArcFace initialize failed."

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
        assert isinstance(wh, [list, tuple]),\
            "The value to set `size` must be type of tuple or list."
        assert len(wh) == 2,\
            "The value to set `size` must contatins 2 elements means [width, height], but now it contains {} elements.".format(
            len(wh))
        self._model.size = wh

    @alpha.setter
    def alpha(self, value):
        assert isinstance(value, [list, tuple]),\
            "The value to set `alpha` must be type of tuple or list."
        assert len(value) == 3,\
            "The value to set `alpha` must contatins 3 elements for each channels, but now it contains {} elements.".format(
            len(value))
        self._model.alpha = value

    @beta.setter
    def beta(self, value):
        assert isinstance(value, [list, tuple]),\
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


class CosFace(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file="",
                 runtime_option=None,
                 model_format=Frontend.ONNX):
        # 调用基函数进行backend_option的初始化
        # 初始化后的option保存在self._runtime_option
        super(CosFace, self).__init__(runtime_option)

        self._model = C.vision.deepinsight.CosFace(
            model_file, params_file, self._runtime_option, model_format)
        # 通过self.initialized判断整个模型的初始化是否成功
        assert self.initialized, "CosFace initialize failed."

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
        assert isinstance(wh, [list, tuple]),\
            "The value to set `size` must be type of tuple or list."
        assert len(wh) == 2,\
            "The value to set `size` must contatins 2 elements means [width, height], but now it contains {} elements.".format(
            len(wh))
        self._model.size = wh

    @alpha.setter
    def alpha(self, value):
        assert isinstance(value, [list, tuple]),\
            "The value to set `alpha` must be type of tuple or list."
        assert len(value) == 3,\
            "The value to set `alpha` must contatins 3 elements for each channels, but now it contains {} elements.".format(
            len(value))
        self._model.alpha = value

    @beta.setter
    def beta(self, value):
        assert isinstance(value, [list, tuple]),\
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


class PartialFC(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file="",
                 runtime_option=None,
                 model_format=Frontend.ONNX):
        # 调用基函数进行backend_option的初始化
        # 初始化后的option保存在self._runtime_option
        super(PartialFC, self).__init__(runtime_option)

        self._model = C.vision.deepinsight.PartialFC(
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
        assert isinstance(wh, [list, tuple]),\
            "The value to set `size` must be type of tuple or list."
        assert len(wh) == 2,\
            "The value to set `size` must contatins 2 elements means [width, height], but now it contains {} elements.".format(
            len(wh))
        self._model.size = wh

    @alpha.setter
    def alpha(self, value):
        assert isinstance(value, [list, tuple]),\
            "The value to set `alpha` must be type of tuple or list."
        assert len(value) == 3,\
            "The value to set `alpha` must contatins 3 elements for each channels, but now it contains {} elements.".format(
            len(value))
        self._model.alpha = value

    @beta.setter
    def beta(self, value):
        assert isinstance(value, [list, tuple]),\
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


class VPL(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file="",
                 runtime_option=None,
                 model_format=Frontend.ONNX):
        # 调用基函数进行backend_option的初始化
        # 初始化后的option保存在self._runtime_option
        super(VPL, self).__init__(runtime_option)

        self._model = C.vision.deepinsight.VPL(
            model_file, params_file, self._runtime_option, model_format)
        # 通过self.initialized判断整个模型的初始化是否成功
        assert self.initialized, "VPL initialize failed."

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
        assert isinstance(wh, [list, tuple]),\
            "The value to set `size` must be type of tuple or list."
        assert len(wh) == 2,\
            "The value to set `size` must contatins 2 elements means [width, height], but now it contains {} elements.".format(
            len(wh))
        self._model.size = wh

    @alpha.setter
    def alpha(self, value):
        assert isinstance(value, [list, tuple]),\
            "The value to set `alpha` must be type of tuple or list."
        assert len(value) == 3,\
            "The value to set `alpha` must contatins 3 elements for each channels, but now it contains {} elements.".format(
            len(value))
        self._model.alpha = value

    @beta.setter
    def beta(self, value):
        assert isinstance(value, [list, tuple]),\
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
