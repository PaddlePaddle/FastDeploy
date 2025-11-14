"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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


def cache_byte_size(raw_cache_dtype):
    """
    Convert the cache dtype to the corresponding byte size.
    """
    if "int4" in raw_cache_dtype.lower() or "float4" in raw_cache_dtype.lower():
        byte_size = 0.5
    elif "int8" in raw_cache_dtype.lower() or "float8" in raw_cache_dtype.lower():
        byte_size = 1
    else:
        byte_size = 2
    return byte_size


def convert_to_saved_dtype(raw_cache_dtype):
    """
    Convert the input cache dtype to the real saved dtype.
    """
    if "int4" in raw_cache_dtype.lower() or "float4" in raw_cache_dtype.lower():
        return "uint8"
    elif "int8" in raw_cache_dtype.lower() or "float8" in raw_cache_dtype.lower():
        return "uint8"
    else:
        return raw_cache_dtype
