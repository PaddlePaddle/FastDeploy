"""
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

"""
metrics
"""
from typing import List


def build_buckets(mantissa_lst: List[int], max_value: int) -> List[int]:
    """
    Generate a list of bucket boundaries using a set of mantissas scaled by powers of 10,
    stopping when the generated value exceeds the specified maximum value.
    """
    exponent = 0
    buckets: List[int] = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


def build_1_2_5_buckets(max_value: int) -> List[int]:
    """
    Generate a bucket list using the common [1, 2, 5] mantissa pattern,
    scaled by powers of 10 up to the specified maximum value.
    """
    return build_buckets([1, 2, 5], max_value)
