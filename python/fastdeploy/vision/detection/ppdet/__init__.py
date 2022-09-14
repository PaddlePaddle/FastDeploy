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
from .... import FastDeployModel, Frontend
from .... import c_lib_wrap as C


class PPYOLOE(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=Frontend.PADDLE):
        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "PPYOLOE model only support model format of Frontend.Paddle now."
        self._model = C.vision.detection.PPYOLOE(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PPYOLOE model initialize failed."

    def predict(self, input_image):
        assert input_image is not None, "The input image data is None."
        return self._model.predict(input_image)


class PPYOLO(PPYOLOE):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=Frontend.PADDLE):
        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "PPYOLO model only support model format of Frontend.Paddle now."
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
                 model_format=Frontend.PADDLE):
        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "PPYOLOv2 model only support model format of Frontend.Paddle now."
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
                 model_format=Frontend.PADDLE):
        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "PaddleYOLOX model only support model format of Frontend.Paddle now."
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
                 model_format=Frontend.PADDLE):
        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "PicoDet model only support model format of Frontend.Paddle now."
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
                 model_format=Frontend.PADDLE):
        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "FasterRCNN model only support model format of Frontend.Paddle now."
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
                 model_format=Frontend.PADDLE):
        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "YOLOv3 model only support model format of Frontend.Paddle now."
        self._model = C.vision.detection.YOLOv3(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "YOLOv3 model initialize failed."


class PPTinyPose(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=Frontend.PADDLE):
        super(PPTinyPose, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "PPTinyPose model only support model format of Frontend.Paddle now."
        self._model = C.vision.detection.PPTinyPose(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "PPTinyPose model initialize failed."

    def predict(self, input_image, detection_result):
        assert input_image is not None, "The input image data is None."
        if detection_result is None:
            import warnings
            warnings.warn(
                "No Detection boxes input. Please make sure the input image has been cropped by Detection model's boxes before"
            )
        return self._model.predict(input_image, detection_result)

    @property
    def use_dark(self):
        return self._model.use_dark


class MaskRCNN(FastDeployModel):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=Frontend.PADDLE):
        super(MaskRCNN, self).__init__(runtime_option)

        assert model_format == Frontend.PADDLE, "MaskRCNN model only support model format of Frontend.Paddle now."
        self._model = C.vision.detection.MaskRCNN(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "MaskRCNN model initialize failed."

    def predict(self, input_image):
        assert input_image is not None, "The input image data is None."
        return self._model.predict(input_image)
