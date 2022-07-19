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
from . import fastdeploy_main as C


class FastDeployModel:
    def __init__(self, option):
        self._model = None
        self._runtime_option = option
        if self._runtime_option is None:
            self._runtime_option = C.RuntimeOption()

    def model_name(self):
        return self._model.model_name()

    def num_inputs(self):
        return self._model.num_inputs()

    def num_outputs(self):
        return self._model.num_outputs()

    def get_input_info(self, index):
        assert index < self.num_inputs(
        ), "The index:{} must be less than number of inputs:{}.".format(
            index, self.num_inputs())
        return self._model.get_input_info(index)

    def get_output_info(self, index):
        assert index < self.num_outputs(
        ), "The index:{} must be less than number of outputs:{}.".format(
            index, self.num_outputs())
        return self._model.get_output_info(index)

    @property
    def runtime_option(self):
        return self._model.runtime_option if self._model is not None else None

    @property
    def initialized(self):
        if self._model is None:
            return False
        return self._model.initialized()


class FastDeployRuntime:
    def __init__(self, runtime_option):
        self._runtime = C.Runtime();
        assert self._runtime.init(runtime_option), "Initialize FastDeployRuntime Failed!"

    def infer(self, data):
        assert isinstance(data, dict), "The input data should be type of dict."
        return self._runtime.infer(data)

    def num_inputs(self):
        return self._runtime.num_inputs();

    def num_outputs(self):
        return self._runtime.num_outputs();

    def get_input_info(self, index):
        assert isinstance(index, int), "The input parameter index should be type of int."
        assert index < self.num_inputs(), "The input parameter index:{} should less than number of inputs:{}.".format(index, self.num_inputs)
        return self._runtime.get_input_info(index)

    def get_output_info(self, index):
        assert isinstance(index, int), "The input parameter index should be type of int."
        assert index < self.num_outputs(), "The input parameter index:{} should less than number of outputs:{}.".format(index, self.num_outputs)
        return self._runtime.get_output_info(index)
