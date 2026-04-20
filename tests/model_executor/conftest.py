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
Lightweight stubs so model-executor tests can be collected on machines
without GPU drivers or compiled C++ custom ops.  CI machines
have the full stack -- these stubs are transparent when real drivers
and ops are already loaded.
"""

import sys
import types

# ---------- Force CPU platform on GPU machines without compiled ops ----------
# On GPU boxes (e.g. A800) paddle.is_compiled_with_cuda() → True, which
# makes fastdeploy.platforms pick CUDAPlatform.  That triggers import of
# compiled C++ ops (fastdeploy_ops.so) which don't exist in a pure-Python
# source tree.  Force CPUPlatform so the guarded `if is_cuda()` branches
# in the import chain are skipped — our unit tests use only CPU tensors.
# NOTE: Must be unconditional — fastdeploy/__init__.py eagerly imports
# modules that may access current_platform, setting it to CUDAPlatform
# before conftest gets a chance to override.
import fastdeploy.platforms as _plat
from fastdeploy.platforms.cpu import CPUPlatform

_cpu = CPUPlatform()
# CPUPlatform inherits Platform.is_cuda() which calls
# paddle.is_compiled_with_cuda() — True on GPU boxes.
# Override to prevent guarded compiled-ops imports.
_cpu.is_cuda = lambda: False
_cpu.is_cuda_alike = lambda: False
_plat._current_platform = _cpu

# triton_utils.py calls triton.runtime.driver._create_driver() at module
# level when torch is installed.  On CPU-only machines this crashes with
# "RuntimeError: 0 active drivers".  Pre-inject a minimal stub so the
# import chain never triggers that code path.
_TRITON_UTILS = "fastdeploy.model_executor.ops.triton_ops.triton_utils"
if _TRITON_UTILS not in sys.modules:
    _stub = types.ModuleType(_TRITON_UTILS)
    _stub.enable_compat_on_triton_kernel = lambda fn: fn
    sys.modules[_TRITON_UTILS] = _stub
