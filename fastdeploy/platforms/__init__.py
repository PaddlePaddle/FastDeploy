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
from .cuda import CUDAPlatform
from .cpu import CPUPlatform
from .xpu import XPUPlatform
from .npu import NPUPlatform
from .dcu import DCUPlatform
from .base import _Backend

_current_platform = None


def __getattr__(name: str):
    if name == "current_platform":
        # lazy init current_platform.
        global _current_platform
        if _current_platform is None:
            if paddle.is_compiled_with_cuda():
                _current_platform = CUDAPlatform()
            elif paddle.is_compiled_with_xpu():
                _current_platform = XPUPlatform()
            elif paddle.is_compiled_with_custom_device("npu"):
                _current_platform = NPUPlatform()
            elif paddle.is_compiled_with_rocm():
                _current_platform = DCUPlatform()
            else:
                _current_platform = CPUPlatform()
        return _current_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")
