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
without GPU drivers.  CI machines have the full stack — these stubs are
transparent when real drivers are already loaded.
"""

import sys
import types

# triton_utils.py calls triton.runtime.driver._create_driver() at module
# level when torch is installed.  On CPU-only machines this crashes with
# "RuntimeError: 0 active drivers".  Pre-inject a minimal stub so the
# import chain never triggers that code path.
_TRITON_UTILS = "fastdeploy.model_executor.ops.triton_ops.triton_utils"
if _TRITON_UTILS not in sys.modules:
    _stub = types.ModuleType(_TRITON_UTILS)
    _stub.enable_compat_on_triton_kernel = lambda fn: fn
    sys.modules[_TRITON_UTILS] = _stub
