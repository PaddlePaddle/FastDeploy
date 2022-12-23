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
import threading
# from .predictor import Predictor
from .handler import BaseModelHandler
from .utils import lock_predictor


class ModelManager:
    def __init__(self, model_handler, predictor):
        self._model_handler = model_handler
        self._predictors = []
        self._predictor_locks = []
        self._register(predictor)

    def _register(self, predictor):
        # Get the model handler
        if not issubclass(self._model_handler, BaseModelHandler):
            raise TypeError(
                "The model_handler must be subclass of BaseModelHandler, please check the type."
            )

        # TODO: Create multiple predictors to run on different GPUs or different CPU threads
        self._predictors.append(predictor)
        self._predictor_locks.append(threading.Lock())

    def _get_predict_id(self):
        t = time.time()
        t = int(round(t * 1000))
        predictor_id = t % len(self._predictors)
        logging.info("The predictor id: {} is selected by running the model.".
                     format(predictor_id))
        return predictor_id

    def predict(self, data, parameters):
        predictor_id = self._get_predict_id()
        with lock_predictor(self._predictor_locks[predictor_id]):
            model_output = self._model_handler.process(
                self._predictors[predictor_id], data, parameters)
            return model_output
