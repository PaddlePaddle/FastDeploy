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
from typing import Union, List
import logging
from .... import FastDeployModel, ModelFormat
from .... import c_lib_wrap as C


class PaddleDetPreprocessor:
    def __init__(self, config_file):
        """Create a preprocessor for PaddleDetection Model from configuration file

        :param config_file: (str)Path of configuration file, e.g ppyoloe/infer_cfg.yml
        """
        self._preprocessor = C.vision.detection.PaddleDetPreprocessor(
            config_file)

    def run(self, input_ims):
        """Preprocess input images for PaddleDetection Model

        :param: input_ims: (list of numpy.ndarray)The input image
        :return: list of FDTensor, include image, scale_factor, im_shape
        """
        return self._preprocessor.run(input_ims)


class PaddleDetPostprocessor:
    def __init__(self):
        """Create a postprocessor for PaddleDetection Model

        """
        self._postprocessor = C.vision.detection.PaddleDetPostprocessor()

    def run(self, runtime_results):
        """Postprocess the runtime results for PaddleDetection Model

        :param: runtime_results: (list of FDTensor)The output FDTensor results from runtime
        :return: list of ClassifyResult(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results)

    def apply_decode_and_nms(self):
        """This function will enable decode and nms in postprocess step.
        """
        return self._postprocessor.apply_decode_and_nms()


class PPYOLOE(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load a PPYOLOE model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g ppyoloe/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g ppyoloe/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model
        """
        super(PPYOLOE, self).__init__(runtime_option)

        self._model = C.vision.detection.PPYOLOE(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PPYOLOE model initialize failed."

    def predict(self, im):
        """Detect an input image

        :param im: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: DetectionResult
        """

        assert im is not None, "The input image data is None."
        return self._model.predict(im)

    def batch_predict(self, images):
        """Detect a batch of input image list

        :param im: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return list of DetectionResult
        """

        return self._model.batch_predict(images)

    @property
    def preprocessor(self):
        """Get PaddleDetPreprocessor object of the loaded model

        :return PaddleDetPreprocessor
        """
        return self._model.preprocessor

    @property
    def postprocessor(self):
        """Get PaddleDetPostprocessor object of the loaded model

        :return PaddleDetPostprocessor
        """
        return self._model.postprocessor


class PPYOLO(PPYOLOE):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load a PPYOLO model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g ppyolo/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g ppyolo/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == ModelFormat.PADDLE, "PPYOLO model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.PPYOLO(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PPYOLO model initialize failed."


class PaddleYOLOX(PPYOLOE):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load a YOLOX model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g yolox/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g yolox/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == ModelFormat.PADDLE, "PaddleYOLOX model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.PaddleYOLOX(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PaddleYOLOX model initialize failed."


class PicoDet(PPYOLOE):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load a PicoDet model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g picodet/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g picodet/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        self._model = C.vision.detection.PicoDet(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PicoDet model initialize failed."


class FasterRCNN(PPYOLOE):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load a FasterRCNN model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g fasterrcnn/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g fasterrcnn/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == ModelFormat.PADDLE, "FasterRCNN model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.FasterRCNN(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "FasterRCNN model initialize failed."


class YOLOv3(PPYOLOE):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load a YOLOv3 model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g yolov3/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g yolov3/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == ModelFormat.PADDLE, "YOLOv3 model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.YOLOv3(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "YOLOv3 model initialize failed."


class MaskRCNN(PPYOLOE):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load a MaskRCNN model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g fasterrcnn/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g fasterrcnn/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == ModelFormat.PADDLE, "MaskRCNN model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.MaskRCNN(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "MaskRCNN model initialize failed."

    def batch_predict(self, images):
        """Detect a batch of input image list, batch_predict is not supported for maskrcnn now.

        :param im: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return list of DetectionResult
        """

        raise Exception(
            "batch_predict is not supported for MaskRCNN model now.")


class SSD(PPYOLOE):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load a SSD model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g ssd/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g ssd/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == ModelFormat.PADDLE, "SSD model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.SSD(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "SSD model initialize failed."
