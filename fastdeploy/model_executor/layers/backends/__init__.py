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
all backends methods
"""

from fastdeploy.platforms import current_platform

__all__ = []

if current_platform.is_xpu():
    from . import xpu

    # fix: F403 `from .xpu import *` used; unable to detect undefined names
    if hasattr(xpu, "__all__"):
        globals().update({name: getattr(xpu, name) for name in xpu.__all__})
        __all__.extend(xpu.__all__)

if current_platform.is_npu():
    from . import npu

    if hasattr(npu, "__all__"):
        globals().update({name: getattr(npu, name) for name in npu.__all__})
        __all__.extend(npu.__all__)

if current_platform.is_gcu():
    from . import gcu

    if hasattr(gcu, "__all__"):
        globals().update({name: getattr(gcu, name) for name in gcu.__all__})
        __all__.extend(gcu.__all__)

if current_platform.is_dcu():
    from . import dcu

    if hasattr(dcu, "__all__"):
        globals().update({name: getattr(dcu, name) for name in dcu.__all__})
        __all__.extend(dcu.__all__)

if current_platform.is_maca():
    from . import metax

    if hasattr(metax, "__all__"):
        globals().update({name: getattr(metax, name) for name in metax.__all__})
        __all__.extend(metax.__all__)

if current_platform.is_intel_hpu():
    from . import intel_hpu

    if hasattr(intel_hpu, "__all__"):
        globals().update({name: getattr(intel_hpu, name) for name in intel_hpu.__all__})
        __all__.extend(intel_hpu.__all__)
