# coding:utf-8
# copyright (c) 2022  paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license"
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import os
import time
import json
import logging
from .predictor import Predictor
from .handler import BaseModelHandler
from .utils import lock_predictor


class ModelManager:
    def __init__(self, task_name, model_path, runtime_option, model_handler):
        self._task_name = task_name
        self._model_path = model_path
        self._runtime_option = runtime_option
        self._model_handler = model_handler
        self._register()

    def _register(self):
        # Get the model handler
        if not issubclass(self._model_handler, BaseModelHandler):
            raise TypeError(
                "The model_handler must be subclass of BaseModelHandler, please check the type."
            )
        self._model_handler = self._model_handler.process

        # Create the model predictor
        # TODO: Create multiple predictors to run on different GPUs or different CPU threads
        device = get_env_device()
        predictor_list = []
        predictor = Predictor(self._model_path, self._runtime_option)
        predictor_list.append(predictor)

        self._predictor_list = predictor_list

    def _get_predict_id(self):
        t = time.time()
        t = int(round(t * 1000))
        predictor_id = t % len(self._predictor_list)
        logging.info("The predictor id: {} is selected by running the model.".format(predictor_id))
        return predictor_id

    def predict(self, data, parameters):
        predictor_id = self._get_predict_id()
        with lock_predictor(self._predictor_list[predictor_id]._lock):
            model_output = self._model_handler(self._predictor_list[predictor_id], data, parameters)
            return model_output

