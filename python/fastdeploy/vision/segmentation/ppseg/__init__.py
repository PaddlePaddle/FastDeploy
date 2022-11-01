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
        """Load a image segmentation model exported by PaddleSeg.

        :param model_file: (str)Path of model file, e.g unet/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g unet/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str) Path of configuration file for deploy, e.g unet/deploy.yml
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model
        """
        super(PaddleSegModel, self).__init__(runtime_option)

        # assert model_format == ModelFormat.PADDLE, "PaddleSeg only support model format of ModelFormat.Paddle now."
        self._model = C.vision.segmentation.PaddleSegModel(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PaddleSeg model initialize failed."

    def predict(self, input_image):
        """Predict the segmentation result for an input image

        :param im: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: SegmentationResult
        """
        return self._model.predict(input_image)

    def disable_normalize_and_permute(self):
        return self._model.disable_normalize_and_permute()

    @property
    def apply_softmax(self):
        """Atrribute of PaddleSeg model. Stating Whether applying softmax operator in the postprocess, default value is False

        :return: value of apply_softmax(bool)
        """
        return self._model.apply_softmax

    @apply_softmax.setter
    def apply_softmax(self, value):
        """Set attribute apply_softmax of PaddleSeg model.

        :param value: (bool)The value to set apply_softmax
        """
        assert isinstance(
            value,
            bool), "The value to set `apply_softmax` must be type of bool."
        self._model.apply_softmax = value

    @property
    def is_vertical_screen(self):
        """Atrribute of PP-HumanSeg model. Stating Whether the input image is vertical image(height > width), default value is False

        :return: value of is_vertical_screen(bool)
        """
        return self._model.is_vertical_screen

    @is_vertical_screen.setter
    def is_vertical_screen(self, value):
        """Set attribute is_vertical_screen of PP-HumanSeg model.

        :param value: (bool)The value to set is_vertical_screen
        """
        assert isinstance(
            value,
            bool), "The value to set `is_vertical_screen` must be type of bool."
        self._model.is_vertical_screen = value
