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


class DBDetector(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file="",
                 runtime_option=None,
                 model_format=Frontend.PADDLE):
        # 调用基函数进行backend_option的初始化
        # 初始化后的option保存在self._runtime_option
        super(DBDetector, self).__init__(runtime_option)

        self._model = C.vision.ppocr.DBDetector(
            model_file, params_file, self._runtime_option, model_format)
        # 通过self.initialized判断整个模型的初始化是否成功
        assert self.initialized, "DBDetector initialize failed."

    def predict(self, input_image):
        return self._model.predict(input_image)

    # 一些跟DBDetector模型有关的属性封装
    # 多数是预处理相关，可通过修改如model.size = [1280, 1280]改变预处理时resize的大小（前提是模型支持）
    @property
    def max_side_len(self):
        return self._model.max_side_len

    @property
    def det_db_thresh(self):
        return self._model.det_db_thresh

    @property
    def det_db_box_thresh(self):
        return self._model.det_db_box_thresh

    @property
    def det_db_unclip_ratio(self):
        return self._model.det_db_unclip_ratio

    @property
    def det_db_score_mode(self):
        return self._model.det_db_score_mode

    @property
    def use_dilation(self):
        return self._model.use_dilation

    @property
    def is_scale(self):
        return self._model.max_wh

    @max_side_len.setter
    def max_side_len(self, value):
        assert isinstance(
            value, int), "The value to set `max_side_len` must be type of int."
        self._model.max_side_len = value

    @det_db_thresh.setter
    def det_db_thresh(self, value):
        assert isinstance(
            value,
            float), "The value to set `det_db_thresh` must be type of float."
        self._model.det_db_thresh = value

    @det_db_box_thresh.setter
    def det_db_box_thresh(self, value):
        assert isinstance(
            value, float
        ), "The value to set `det_db_box_thresh` must be type of float."
        self._model.det_db_box_thresh = value

    @det_db_unclip_ratio.setter
    def det_db_unclip_ratio(self, value):
        assert isinstance(
            value, float
        ), "The value to set `det_db_unclip_ratio` must be type of float."
        self._model.det_db_unclip_ratio = value

    @det_db_score_mode.setter
    def det_db_score_mode(self, value):
        assert isinstance(
            value,
            str), "The value to set `det_db_score_mode` must be type of str."
        self._model.det_db_score_mode = value

    @use_dilation.setter
    def use_dilation(self, value):
        assert isinstance(
            value,
            bool), "The value to set `use_dilation` must be type of bool."
        self._model.use_dilation = value

    @is_scale.setter
    def is_scale(self, value):
        assert isinstance(
            value, bool), "The value to set `is_scale` must be type of bool."
        self._model.is_scale = value
