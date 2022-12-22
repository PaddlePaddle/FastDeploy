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


class AnimeGAN(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file="",
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load a AnimeGAN model.

        :param model_file: (str)Path of model file, e.g ./model.pdmodel
        :param params_file: (str)Path of parameters file, e.g ./model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model
        """
        # 调用基函数进行backend_option的初始化
        # 初始化后的option保存在self._runtime_option
        super(AnimeGAN, self).__init__(runtime_option)

        self._model = C.vision.generation.AnimeGAN(
            model_file, params_file, self._runtime_option, model_format)
        # 通过self.initialized判断整个模型的初始化是否成功
        assert self.initialized, "AnimeGAN initialize failed."

    def predict(self, input_image):
        """ Predict the style transfer result for an input image

        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: style transfer result
        """
        return self._model.predict(input_image)

    def batch_predict(self, input_images):
        """ Predict the style transfer result for multiple input images

        :param input_images: (list of numpy.ndarray)The list of input image data, each image is a 3-D array with layout HWC, BGR format
        :return: a list of style transfer results
        """
        return self._model.batch_predict(input_images)
