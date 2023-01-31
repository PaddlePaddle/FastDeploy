# Copyright (c) 2022 Baidu, Inc. All Rights Reserved.
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
"""
init file for poros
"""

import os
import sys

if sys.version_info < (3, 6):
    raise Exception("Poros can only work on Python 3.6+")

import ctypes
import torch

from poros._compile import *
from poros._module import PorosOptions

def _register_with_torch():
    poros_dir = os.path.dirname(__file__)
    torch.ops.load_library(poros_dir + '/lib/libporos.so')

_register_with_torch()