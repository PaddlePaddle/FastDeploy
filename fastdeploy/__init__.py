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
from .fastdeploy_main import Frontend, Backend, FDDataType, TensorInfo, Device
from .fastdeploy_runtime import *
from . import fastdeploy_main as C
from . import vision
from .download import download, download_and_decompress


def TensorInfoStr(tensor_info):
    message = "TensorInfo(name : '{}', dtype : '{}', shape : '{}')".format(
        tensor_info.name, tensor_info.dtype, tensor_info.shape)
    return message


class RuntimeOption:
    def __init__(self):
    	self._option = C.RuntimeOption()
    
    def set_model_path(self, model_path, params_path="", model_format="paddle"):
        return self._option.set_model_path(model_path, params_path, model_format)

    def use_gpu(self, device_id=0):
        return self._option.use_gpu(device_id)

    def use_cpu(self):
        return self._option.use_cpu()

    def set_cpu_thread_num(self, thread_num=8):
        return self._option.set_cpu_thread_num(thread_num)

    def use_paddle_backend(self):
        return self._option.use_paddle_backend()

    def use_ort_backend(self):
        return self._option.use_ort_backend()

    def use_trt_backend(self):
        return self._option.use_trt_backend()

    def enable_paddle_mkldnn(self):
        return self._option.enable_paddle_mkldnn()

    def disable_paddle_mkldnn(self):
        return self._option.disable_paddle_mkldnn()

    def set_paddle_mkldnn_cache_size(self, cache_size):
        return self._option.set_paddle_mkldnn_cache_size(cache_size)

    def set_trt_input_shape(self, tensor_name, min_shape, opt_shape=None, max_shape=None):
        if opt_shape is None and max_shape is None:
            opt_shape = min_shape
            max_shape = min_shape
        else:
            assert opt_shape is not None and max_shape is not None, "Set min_shape only, or set min_shape, opt_shape, max_shape both."
        return self._option.set_trt_input_shape(tensor_name, min_shape, opt_shape, max_shape)

    def set_trt_cache_file(self, cache_file_path):
        return self._option.set_trt_cache_file(cache_file_path)

    def enable_trt_fp16(self):
        return self._option.enable_trt_fp16()

    def dissable_trt_fp16(self):
        return self._option.disable_trt_fp16()

    def __repr__(self):
        attrs = dir(self._option)
        message = "RuntimeOption(\n"
        for attr in attrs:
            if attr.startswith("__"):
                continue
            if hasattr(getattr(self._option, attr), "__call__"):
                continue
            message += "  {} : {}\t\n".format(attr, getattr(self._option, attr))
        message.strip("\n")
        message += ")"
        return message


def RuntimeOptionStr(runtime_option):
    attrs = dir(runtime_option)
    message = "RuntimeOption(\n"
    for attr in attrs:
        if attr.startswith("__"):
            continue
        if hasattr(getattr(runtime_option, attr), "__call__"):
            continue
        message += "  {} : {}\t\n".format(attr, getattr(runtime_option, attr))
    message.strip("\n")
    message += ")"
    return message
C.TensorInfo.__repr__ = TensorInfoStr
C.RuntimeOption.__repr__ = RuntimeOptionStr
