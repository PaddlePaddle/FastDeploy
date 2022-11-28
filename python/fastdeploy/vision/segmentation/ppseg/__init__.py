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

    def predict(self, image):
        """Predict the segmentation result for an input image

        :param im: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: SegmentationResult
        """
        return self._model.predict(image)

    def batch_predict(self, image_list):
        """Predict the segmentation results for a batch of input image
        :param image_list: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return list of SegmentationResult
        """
        return self._model.batch_predict(image_list)

    @property
    def preprocessor(self):
        """Get PaddleSegPreprocessor object of the loaded model
        :return PaddleSegPreprocessor
        """
        return self._model.preprocessor

    @property
    def postprocessor(self):
        """Get PaddleSegPostprocessor object of the loaded model
        :return PaddleSegPostprocessor
        """
        return self._model.postprocessor


class PaddleSegPreprocessor:
    def __init__(self, config_file):
        """Create a preprocessor for PaddleSegModel from configuration file
        :param config_file: (str)Path of configuration file, e.g ppliteseg/deploy.yaml
        """
        self._preprocessor = C.vision.segmentation.PaddleSegPreprocessor(
            config_file)

    def run(self, input_ims):
        """Preprocess input images for PaddleSegModel
        :param: input_ims: (list of numpy.ndarray)The input image
        :return: list of FDTensor
        """
        return self._preprocessor.run(input_ims)

    def disable_normalize_and_permute(self):
        """To disable normalize and hwc2chw in preprocessing step.
        """
        return self._preprocessor.disable_normalize_and_permute()

    @property
    def is_vertical_screen(self):
        """Atrribute of PP-HumanSeg model. Stating Whether the input image is vertical image(height > width), default value is False

        :return: value of is_vertical_screen(bool)
        """
        return self._preprocessor.is_vertical_screen

    @is_vertical_screen.setter
    def is_vertical_screen(self, value):
        """Set attribute is_vertical_screen of PP-HumanSeg model.

        :param value: (bool)The value to set is_vertical_screen
        """
        assert isinstance(
            value,
            bool), "The value to set `is_vertical_screen` must be type of bool."
        self._preprocessor.is_vertical_screen = value


class PaddleSegPostprocessor:
    def __init__(self, config_file):
        """Create a postprocessor for PaddleSegModel from configuration file
        :param config_file: (str)Path of configuration file, e.g ppliteseg/deploy.yaml
        """
        self._postprocessor = C.vision.segmentation.PaddleSegPostprocessor(
            config_file)

    def run(self, runtime_results, imgs_info):
        """Postprocess the runtime results for PaddleSegModel
        :param: runtime_results: (list of FDTensor)The output FDTensor results from runtime
        :param: imgs_info: The original input images shape info map, key is "shape_info", value is [[image_height, image_width]]
        :return: list of SegmentationResult(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results, imgs_info)

    @property
    def apply_softmax(self):
        """Atrribute of PaddleSeg model. Stating Whether applying softmax operator in the postprocess, default value is False

        :return: value of apply_softmax(bool)
        """
        return self._postprocessor.apply_softmax

    @apply_softmax.setter
    def apply_softmax(self, value):
        """Set attribute apply_softmax of PaddleSeg model.

        :param value: (bool)The value to set apply_softmax
        """
        assert isinstance(
            value,
            bool), "The value to set `apply_softmax` must be type of bool."
        self._postprocessor.apply_softmax = value

    @property
    def store_score_map(self):
        """Atrribute of PaddleSeg model. Stating Whether storing score map in the SegmentationResult, default value is False

        :return: value of store_score_map(bool)
        """
        return self._postprocessor.store_score_map

    @store_score_map.setter
    def store_score_map(self, value):
        """Set attribute store_score_map of PaddleSeg model.

        :param value: (bool)The value to set store_score_map
        """
        assert isinstance(
            value,
            bool), "The value to set `store_score_map` must be type of bool."
        self._postprocessor.store_score_map = value
