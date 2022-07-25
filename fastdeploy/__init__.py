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
from .fastdeploy_main import Frontend, Backend, FDDataType, TensorInfo, RuntimeOption, Device
from .fastdeploy_runtime import *
from . import fastdeploy_main as C
from . import vision
from .download import download, download_and_decompress


def TensorInfoStr(tensor_info):
    message = "TensorInfo(name : '{}', dtype : '{}', shape : '{}')".format(
        tensor_info.name, tensor_info.dtype, tensor_info.shape)
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
