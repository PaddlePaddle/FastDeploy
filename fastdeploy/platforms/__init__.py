# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
platform module
"""

import paddle

from .base import _Backend  # noqa: F401
from .cpu import CPUPlatform
from .cuda import CUDAPlatform
from .dcu import DCUPlatform
from .gcu import GCUPlatform
from .iluvatar import IluvatarPlatform
from .intel_hpu import INTEL_HPUPlatform
from .maca import MACAPlatform
from .npu import NPUPlatform
from .xpu import XPUPlatform

_current_platform = None


def __getattr__(name: str):
    if name == "current_platform":
        # lazy init current_platform.
        global _current_platform
        if _current_platform is None:
            if paddle.is_compiled_with_rocm():
                _current_platform = DCUPlatform()
            elif paddle.is_compiled_with_cuda():
                _current_platform = CUDAPlatform()
            elif paddle.is_compiled_with_xpu():
                _current_platform = XPUPlatform()
            elif paddle.is_compiled_with_custom_device("npu"):
                _current_platform = NPUPlatform()
            elif paddle.is_compiled_with_custom_device("intel_hpu"):
                _current_platform = INTEL_HPUPlatform()
            elif paddle.is_compiled_with_custom_device("iluvatar_gpu"):
                _current_platform = IluvatarPlatform()
            elif paddle.is_compiled_with_custom_device("gcu"):
                _current_platform = GCUPlatform()
            elif paddle.is_compiled_with_custom_device("metax_gpu"):
                _current_platform = MACAPlatform()
            else:
                _current_platform = CPUPlatform()
        return _current_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")
