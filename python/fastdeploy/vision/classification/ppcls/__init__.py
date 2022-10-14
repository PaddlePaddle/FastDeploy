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


class PaddleClasModel(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load a image classification model exported by PaddleClas.

        :param model_file: (str)Path of model file, e.g resnet50/inference.pdmodel
        :param params_file: (str)Path of parameters file, e.g resnet50/inference.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str) Path of configuration file for deploy, e.g resnet50/inference_cls.yaml
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model
        """

        super(PaddleClasModel, self).__init__(runtime_option)

        assert model_format == ModelFormat.PADDLE, "PaddleClasModel only support model format of ModelFormat.PADDLE now."
        self._model = C.vision.classification.PaddleClasModel(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PaddleClas model initialize failed."

    def predict(self, im, topk=1):
        """Classify an input image

        :param im: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :param topk: (int)The topk result by the classify confidence score, default 1
        :return: ClassifyResult
        """

        return self._model.predict(im, topk)
