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
import os
import sys


def add_dll_search_dir(dir_path):
    os.environ["path"] = dir_path + ";" + os.environ["path"]
    sys.path.insert(0, dir_path)
    if sys.version_info[:2] >= (3, 8):
        os.add_dll_directory(dir_path)


if os.name == "nt":
    current_path = os.path.abspath(__file__)
    dirname = os.path.dirname(current_path)
    third_libs_dir = os.path.join(dirname, "libs")
    add_dll_search_dir(third_libs_dir)
    for root, dirs, filenames in os.walk(third_libs_dir):
        for d in dirs:
            if d == "lib" or d == "bin":
                add_dll_search_dir(os.path.join(dirname, root, d))

from .fastdeploy_main import Frontend, Backend, FDDataType, TensorInfo, Device
from .runtime import Runtime, RuntimeOption
from .model import FastDeployModel
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
