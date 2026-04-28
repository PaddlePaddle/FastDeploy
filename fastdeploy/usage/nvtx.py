"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

try:
    import nvtx

    _NVTX_AVAILABLE = True
except ImportError:
    _NVTX_AVAILABLE = False


def nvtx_range(name, color="blue"):
    if _NVTX_AVAILABLE:
        return nvtx.annotate(name, color=color)
    import contextlib

    return contextlib.nullcontext()
