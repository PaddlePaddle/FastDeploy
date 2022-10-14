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

        assert model_format == ModelFormat.PADDLE, "PPYOLOE model only support model format of ModelFormat.Paddle now."
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


class PPYOLOv2(PPYOLOE):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load a PPYOLOv2 model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g ppyolov2/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g ppyolov2/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == ModelFormat.PADDLE, "PPYOLOv2 model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.PPYOLOv2(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PPYOLOv2 model initialize failed."


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

        assert model_format == ModelFormat.PADDLE, "PicoDet model only support model format of ModelFormat.Paddle now."
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


class MaskRCNN(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=ModelFormat.PADDLE):
        """Load a MaskRCNN model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g maskrcnn/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g maskrcnn/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model
        """

        super(MaskRCNN, self).__init__(runtime_option)

        assert model_format == ModelFormat.PADDLE, "MaskRCNN model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.MaskRCNN(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "MaskRCNN model initialize failed."

    def predict(self, input_image):
        assert input_image is not None, "The input image data is None."
        return self._model.predict(input_image)
