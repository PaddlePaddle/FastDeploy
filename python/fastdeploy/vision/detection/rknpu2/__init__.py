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
from .. import PPYOLOE


class RKPicoDet(PPYOLOE):
    def __init__(self,
                 model_file,
                 params_file,
                 config_file,
                 runtime_option=None,
                 model_format=ModelFormat.RKNN):
        """Load a PicoDet model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g picodet/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g picodet/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (fastdeploy.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (fastdeploy.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert model_format == ModelFormat.RKNN, "RKPicoDet model only support model format of ModelFormat.RKNN now."
        self._model = C.vision.detection.RKPicoDet(
            model_file, params_file, config_file, self._runtime_option,
            model_format)
        assert self.initialized, "RKPicoDet model initialize failed."
