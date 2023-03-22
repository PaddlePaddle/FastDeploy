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

# Create a symbol link to tensorrt library.
trt_directory = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "libs/third_libs/tensorrt/lib/")
if os.name != "nt" and os.path.exists(trt_directory):
    logging.basicConfig(level=logging.INFO)
    for trt_lib in [
            "libnvcaffe_parser.so", "libnvinfer_plugin.so", "libnvinfer.so",
            "libnvonnxparser.so", "libnvparsers.so"
    ]:
        dst = os.path.join(trt_directory, trt_lib)
        src = os.path.join(trt_directory, trt_lib + ".8")
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
                logging.info(
                    f"Create a symbolic link pointing to {src} named {dst}.")
            except OSError as e:
                logging.warning(
                    f"Failed to create a symbolic link pointing to {src} by an unprivileged user. "
                    "It may failed when you use Paddle TensorRT backend. "
                    "Please use administator privilege to import fastdeploy at first time."
                )
                break
    logging.basicConfig(level=logging.NOTSET)

# Note(zhoushunjie): Fix the import order of paddle and fastdeploy library.
# This solution will be removed it when the confilct of paddle and
# fastdeploy is fixed.
try:
    import paddle
except:
    pass

from .c_lib_wrap import (
    ModelFormat,
    Backend,
    FDDataType,
    TensorInfo,
    Device,
    is_built_with_gpu,
    is_built_with_ort,
    ModelFormat,
    is_built_with_paddle,
    is_built_with_trt,
    get_default_cuda_directory, )


def set_logger(enable_info=True, enable_warning=True):
    """Set behaviour of logger while using FastDeploy

    :param enable_info: (boolean)Whether to print out log level of INFO
    :param enable_warning: (boolean)Whether to print out log level of WARNING, recommend to set to True
    """
    from .c_lib_wrap import set_logger
    set_logger(enable_info, enable_warning)


from .runtime import Runtime, RuntimeOption
from .model import FastDeployModel
from . import c_lib_wrap as C
from . import vision
from . import pipeline
from . import text
from . import encryption
from .download import download, download_and_decompress, download_model, get_model_list
from . import serving
from .code_version import version, git_version
__version__ = version
