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

from .c_lib_wrap import (
    ModelFormat,
    Backend,
    rknpu2,
    FDDataType,
    TensorInfo,
    Device,
    is_built_with_gpu,
    is_built_with_ort,
    ModelFormat,
    is_built_with_paddle,
    is_built_with_trt,
    get_default_cuda_directory, )

from .runtime import Runtime, RuntimeOption
from .model import FastDeployModel
from . import c_lib_wrap as C
from . import vision
from . import pipeline
from . import text
from .download import download, download_and_decompress, download_model
