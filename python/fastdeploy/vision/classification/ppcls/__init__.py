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
from ...common import ProcessorManager


class PaddleClasPreprocessor(ProcessorManager):
    def __init__(self, config_file):
        """Create a preprocessor for PaddleClasModel from configuration file

        :param config_file: (str)Path of configuration file, e.g resnet50/inference_cls.yaml
        """
        super(PaddleClasPreprocessor, self).__init__()
        self._manager = C.vision.classification.PaddleClasPreprocessor(
            config_file)

    def disable_normalize(self):
        """
        This function will disable normalize in preprocessing step.
        """
        self._manager.disable_normalize()

    def disable_permute(self):
        """
        This function will disable hwc2chw in preprocessing step.
        """
        self._manager.disable_permute()

    def initial_resize_on_cpu(self, v):
        """
        When the initial operator is Resize, and input image size is large,
        maybe it's better to run resize on CPU, because the HostToDevice memcpy
        is time consuming. Set this True to run the initial resize on CPU.

        :param: v: True or False
        """
        self._manager.initial_resize_on_cpu(v)


class PaddleClasPostprocessor:
    def __init__(self, topk=1):
        """Create a postprocessor for PaddleClasModel

        :param topk: (int)Filter the top k classify label
        """
        self._postprocessor = C.vision.classification.PaddleClasPostprocessor(
            topk)

    def run(self, runtime_results):
        """Postprocess the runtime results for PaddleClasModel

        :param: runtime_results: (list of FDTensor)The output FDTensor results from runtime
        :return: list of ClassifyResult(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results)


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
        self._model = C.vision.classification.PaddleClasModel(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PaddleClas model initialize failed."

    def clone(self):
        """Clone PaddleClasModel object

        :return: a new PaddleClasModel object
        """

        class PaddleClasCloneModel(PaddleClasModel):
            def __init__(self, model):
                self._model = model

        clone_model = PaddleClasCloneModel(self._model.clone())
        return clone_model

    def predict(self, im, topk=1):
        """Classify an input image

        :param im: (numpy.ndarray) The input image data, a 3-D array with layout HWC, BGR format
        :param topk: (int) Filter the topk classify result, default 1
        :return: ClassifyResult
        """

        self.postprocessor.topk = topk
        return self._model.predict(im)

    def batch_predict(self, images):
        """Classify a batch of input image

        :param im: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return list of ClassifyResult
        """

        return self._model.batch_predict(images)

    @property
    def preprocessor(self):
        """Get PaddleClasPreprocessor object of the loaded model

        :return PaddleClasPreprocessor
        """
        return self._model.preprocessor

    @property
    def postprocessor(self):
        """Get PaddleClasPostprocessor object of the loaded model

        :return PaddleClasPostprocessor
        """
        return self._model.postprocessor
