# coding:utf-8
# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import fastdeploy as fd
import threading


class Predictor:
    def __init__(self, model_name, **kwargs):
        self._model_name = model_name
        # self._model = self._create_model(model_name, **kwargs)
        self._model = fd.vision.detection.PPYOLOE(**kwargs)
        self._lock = threading.Lock()
        print("init", self._model)

    def _create_model(self, model_name, **kwargs):
        # self._model = eval(model_name)(**kwargs)
        self._model = fd.vision.detection.PPYOLOE(**kwargs)
        print(self._model)

    def predict(self, data):
        print(self._model)
        return self._model.predict(data)
