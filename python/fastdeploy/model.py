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
from . import c_lib_wrap as C


class FastDeployModel:
    def __init__(self, option):
        self._model = None
        if option is None:
            self._runtime_option = C.RuntimeOption()
        else:
            self._runtime_option = option._option

    def model_name(self):
        return self._model.model_name()

    def num_inputs_of_runtime(self):
        return self._model.num_inputs_of_runtime()

    def num_outputs_of_runtime(self):
        return self._model.num_outputs_of_runtime()

    def input_info_of_runtime(self, index):
        assert index < self.num_inputs_of_runtime(
        ), "The index:{} must be less than number of inputs:{}.".format(
            index, self.num_inputs_of_runtime())
        return self._model.input_info_of_runtime(index)

    def output_info_of_runtime(self, index):
        assert index < self.num_outputs_of_runtime(
        ), "The index:{} must be less than number of outputs:{}.".format(
            index, self.num_outputs_of_runtime())
        return self._model.output_info_of_runtime(index)

    def enable_record_time_of_runtime(self):
        self._model.enable_record_time_of_runtime()

    def disable_record_time_of_runtime(self):
        self._model.disable_record_time_of_runtime()

    def print_statis_info_of_runtime(self):
        return self._model.print_statis_info_of_runtime()

    @property
    def runtime_option(self):
        return self._model.runtime_option if self._model is not None else None

    @property
    def initialized(self):
        if self._model is None:
            return False
        return self._model.initialized()
