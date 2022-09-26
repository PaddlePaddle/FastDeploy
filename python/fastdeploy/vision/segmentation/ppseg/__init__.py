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
from .... import FastDeployModel, ModelFormat
from .... import c_lib_wrap as C


class PaddleSegModel(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        super(PaddleSegModel, self).__init__(runtime_option)

        assert model_format == ModelFormat.PADDLE, "PaddleSeg only support model format of ModelFormat.Paddle now."
        self._model = C.vision.segmentation.PaddleSegModel(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PaddleSeg model initialize failed."

    def predict(self, input_image):
        return self._model.predict(input_image)

    @property
    def apply_softmax(self):
        return self._model.apply_softmax

    @apply_softmax.setter
    def apply_softmax(self, value):
        assert isinstance(
            value,
            bool), "The value to set `apply_softmax` must be type of bool."
        self._model.apply_softmax = value

    @property
    def is_vertical_screen(self):
        return self._model.is_vertical_screen

    @is_vertical_screen.setter
    def is_vertical_screen(self, value):
        assert isinstance(
            value,
            bool), "The value to set `is_vertical_screen` must be type of bool."
        self._model.is_vertical_screen = value
