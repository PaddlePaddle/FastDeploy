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


class Runtime:
    def __init__(self, runtime_option):
        self._runtime = C.Runtime()
        assert self._runtime.init(
            runtime_option._option), "Initialize Runtime Failed!"

    def infer(self, data):
        assert isinstance(data, dict), "The input data should be type of dict."
        return self._runtime.infer(data)

    def num_inputs(self):
        return self._runtime.num_inputs()

    def num_outputs(self):
        return self._runtime.num_outputs()

    def get_input_info(self, index):
        assert isinstance(
            index, int), "The input parameter index should be type of int."
        assert index < self.num_inputs(
        ), "The input parameter index:{} should less than number of inputs:{}.".format(
            index, self.num_inputs)
        return self._runtime.get_input_info(index)

    def get_output_info(self, index):
        assert isinstance(
            index, int), "The input parameter index should be type of int."
        assert index < self.num_outputs(
        ), "The input parameter index:{} should less than number of outputs:{}.".format(
            index, self.num_outputs)
        return self._runtime.get_output_info(index)


class RuntimeOption:
    def __init__(self):
        self._option = C.RuntimeOption()

    def set_model_path(self, model_path, params_path="",
                       model_format="paddle"):
        return self._option.set_model_path(model_path, params_path,
                                           model_format)

    def use_gpu(self, device_id=0):
        return self._option.use_gpu(device_id)

    def use_cpu(self):
        return self._option.use_cpu()

    def set_cpu_thread_num(self, thread_num=-1):
        return self._option.set_cpu_thread_num(thread_num)

    def use_paddle_backend(self):
        return self._option.use_paddle_backend()

    def use_ort_backend(self):
        return self._option.use_ort_backend()

    def use_trt_backend(self):
        return self._option.use_trt_backend()

    def use_openvino_backend(self):
        return self._option.use_openvino_backend()

    def enable_paddle_mkldnn(self):
        return self._option.enable_paddle_mkldnn()

    def disable_paddle_mkldnn(self):
        return self._option.disable_paddle_mkldnn()

    def enable_paddle_log_info(self):
        return self._option.enable_paddle_log_info()

    def disable_paddle_log_info(self):
        return self._option.disable_paddle_log_info()

    def set_paddle_mkldnn_cache_size(self, cache_size):
        return self._option.set_paddle_mkldnn_cache_size(cache_size)

    def set_trt_input_shape(self,
                            tensor_name,
                            min_shape,
                            opt_shape=None,
                            max_shape=None):
        if opt_shape is None and max_shape is None:
            opt_shape = min_shape
            max_shape = min_shape
        else:
            assert opt_shape is not None and max_shape is not None, "Set min_shape only, or set min_shape, opt_shape, max_shape both."
        return self._option.set_trt_input_shape(tensor_name, min_shape,
                                                opt_shape, max_shape)

    def set_trt_cache_file(self, cache_file_path):
        return self._option.set_trt_cache_file(cache_file_path)

    def enable_trt_fp16(self):
        return self._option.enable_trt_fp16()

    def disable_trt_fp16(self):
        return self._option.disable_trt_fp16()

    def enable_paddle_to_trt(self):
        return self._option.enable_paddle_to_trt()

    def __repr__(self):
        attrs = dir(self._option)
        message = "RuntimeOption(\n"
        for attr in attrs:
            if attr.startswith("__"):
                continue
            if hasattr(getattr(self._option, attr), "__call__"):
                continue
            message += "  {} : {}\t\n".format(attr,
                                              getattr(self._option, attr))
        message.strip("\n")
        message += ")"
        return message
