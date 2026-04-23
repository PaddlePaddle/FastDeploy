# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
"""fastdeploy gpu ops"""

import os
import sys
import warnings

PACKAGE = "fastdeploy.model_executor.ops.gpu"

# Track whether ops loaded successfully
_ops_loaded = False


def decide_module():
    import paddle

    try:
        prop = paddle.device.get_device_properties()
    except AttributeError:
        prop = paddle.device.cuda.get_device_properties()
    sm_version = prop.major * 10 + prop.minor
    print(f"current sm_version={sm_version}")

    curdir = os.path.dirname(os.path.abspath(__file__))
    sm_version_path = os.path.join(curdir, f"fastdeploy_ops_{sm_version}")
    if os.path.exists(sm_version_path):
        return f".fastdeploy_ops_{sm_version}.fastdeploy_ops"
    return ".fastdeploy_ops"


module_path = ".fastdeploy_ops"
try:
    module_path = decide_module()
except Exception as e:
    print(f"decide_module error, load custom_ops from .fastdeploy_ops: {e}")
    pass

# TensorFlow import is now blocked in fastdeploy/__init__.py, so we can safely
# load the ops .so with RTLD_LAZY (via import_custom_ops). The segfault was
# caused by TensorFlow+PaddlePaddle CUDA context conflict, not undefined symbols.
from fastdeploy.import_ops import import_custom_ops
import_custom_ops(PACKAGE, module_path, globals())

_ops_loaded = any(
    not k.startswith("_") and k not in ("sys", "warnings", "os",
                                          "PACKAGE", "decide_module", "module_path",
                                          "tolerant_import_error", "_ops_loaded")
    for k in globals()
)

if not _ops_loaded:
    warnings.warn(
        "Custom GPU ops could not be loaded. Some features will not work. "
        "Ensure custom ops are compiled for your platform (run: cd custom_ops && python setup_ops.py build)."
    )


def tolerant_import_error():
    class NoneModule:
        def __getattr__(self, name):
            return None
    sys.modules[__name__] = NoneModule()


def __getattr__(name):
    """Return None for any missing op when ops failed to load."""
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return None
