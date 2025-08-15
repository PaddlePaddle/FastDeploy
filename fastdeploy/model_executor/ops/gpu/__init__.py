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

import sys

from fastdeploy.import_ops import import_custom_ops

PACKAGE = "fastdeploy.model_executor.ops.gpu"

import_custom_ops(PACKAGE, ".fastdeploy_ops", globals())


def tolerant_import_error():
    class NoneModule:
        def __getattr__(self, name):
            return None

    sys.modules[__name__] = NoneModule()
