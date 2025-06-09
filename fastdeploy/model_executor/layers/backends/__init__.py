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

from .xpu import *
from .npu import *

__all__ = []
from . import npu
if hasattr(npu, '__all__'):
    __all__.extend(npu.__all__)
    
from . import xpu
if hasattr(xpu, '__all__'):
    __all__.extend(xpu.__all__)