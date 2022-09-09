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
from ... import Frontend
from ... import c_lib_wrap as C


class UIEModel(object):
    def __init__(self,
                 model_file,
                 params_file,
                 vocab_file,
                 position_prob,
                 max_length,
                 schema=None,
                 runtime_option=None,
                 model_format=Frontend.PADDLE):
        self._model = C.text.UIEModel(model_file, params_file, vocab_file,
                                      position_prob, max_length, schema,
                                      runtime_option, model_format)

    def set_schema(self, schema):
        pass

    def predict(self, texts):
        return self._model.predict(texts)
