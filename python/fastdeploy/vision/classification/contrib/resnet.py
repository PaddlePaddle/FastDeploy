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


class ResNet(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file="",
                 runtime_option=None,
                 model_format=ModelFormat.ONNX):

        # call super() to initialize the backend_option
        # the result of initialization will be saved in self._runtime_option
        super(ResNet, self).__init__(runtime_option)

        self._model = C.vision.classification.ResNet(
            model_file, params_file, self._runtime_option, model_format)
        # self.initialized shows the initialization of the model is successful or not

        assert self.initialized, "ResNet initialize failed."

    def predict(self, input_image, topk=1):
        return self._model.predict(input_image, topk)

    # Implement the setter and getter method for variables
    @property
    def size(self):
        return self._model.size

    @property
    def mean_vals(self):
        return self._model.mean_vals

    @property
    def std_vals(self):
        return self._model.std_vals

    @size.setter
    def size(self, wh):
        assert isinstance(wh, (list, tuple)),\
            "The value to set `size` must be type of tuple or list."
        assert len(wh) == 2,\
            "The value to set `size` must contatins 2 elements means [width, height], but now it contains {} elements.".format(
            len(wh))
        self._model.size = wh

    @mean_vals.setter
    def mean_vals(self, value):
        assert isinstance(
            value, list), "The value to set `mean_vals` must be type of list."
        self._model.mean_vals = value

    @std_vals.setter
    def std_vals(self, value):
        assert isinstance(
            value, list), "The value to set `std_vals` must be type of list."
        self._model.std_vals = value
